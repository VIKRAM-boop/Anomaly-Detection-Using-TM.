"""
Corrected temporal-window features (Phase 5, Task 1).

00_build_temporal_windows.py has two problems found by 29_audit_temporal.py:

  1. Values from flows still in progress. A flow is classified when it
     ends (its own dur, sbytes, ... are only known then), but 00 averages
     the duration, bytes, packets, RTT and final state of every earlier
     flow in the window -- including flows that have not ended yet at that
     moment (41% of flows at the 10 s scale have at least one).
  2. Held-out families shape the training windows. 00 builds the windows
     once over all flows; 09 removes the held-out family's rows afterwards,
     but its training-period flows are still counted inside the windows of
     the known rows that remain.

This script recomputes the same 13 features per scale with:

  membership   unchanged: same source IP, start time in (t - w, t], at or
               before the current row (the current flow included). Count-
               type features (arrival rate, unique destinations and ports,
               destination entropy, protocol count, burstiness, delta
               arrival rate) only need flow starts, which are known.
  values       byte rate, packet rate, mean/std duration, mean TCP RTT and
               failed ratio use only members that have ended by the current
               flow's end (ltime_j <= ltime_i). ltime has whole-second
               resolution, so a member ending in the same second counts.
  --exclude F  family F's training-period flows are removed before any
               window is built, so with F held out the known training rows
               are what they would have been had F never occurred. Test-
               period flows of F stay: in deployment the new attack's
               traffic is really there. The train/test boundary is the one
               00 drew, so test rows are identical.
  --legacy     reproduce 00's features exactly (no value filter); used to
               check this implementation against the stored ones.

Row order is the stored one (00's start-time sort). Reads the existing
train/test_windowed.parquet for the per-flow columns; writes
<data>/temporal[_excl_<family>][_legacy]/train_windowed.parquet and
test_windowed.parquet in the same format.

Usage: 00b_build_windows_v2.py [--exclude FAMILY] [--scales s,m,l] [--legacy]
"""
import argparse
import os
import time
from pathlib import Path

import numba
import numpy as np
import pandas as pd

WINDOWS = {"short": 1.0, "medium": 10.0, "long": 60.0}
FAILED_STATES = {"INT", "REQ"}
SUFFIXES = ["arrival_rate", "byte_rate", "pkt_rate", "mean_dur", "std_dur",
            "mean_tcprtt", "unique_dst", "unique_dst_port", "dst_entropy",
            "failed_ratio", "burstiness", "proto_nunique", "delta_arrival_rate"]

def find_dir(name):
    if name == "data":
        override = os.environ.get("TM_DATA_DIR")
        if override:
            return Path(override)
    here = Path(__file__).resolve().parent
    for c in (here.parent / name, here / name):
        if c.exists():
            return c
    return here / name

def source_dir():
    """The stored 00 output: per-flow columns are read from here."""
    override = os.environ.get("TM_SOURCE_WINDOWS")
    if override:
        return Path(override)
    return find_dir("data") / "temporal"

@numba.njit(cache=True)
def window_kernel(gid, st, lt, dur, byt, pkt, rtt, fail, dst, port, proto, gap,
                  w, completed_only, n_dst, n_port, n_proto):
    n = len(gid)
    out = np.zeros((n, 13))
    mark_d = np.full(n_dst, -1, np.int64)
    mark_p = np.full(n_port, -1, np.int64)
    mark_r = np.full(n_proto, -1, np.int64)
    cnt_d = np.zeros(n_dst, np.int64)
    touched = np.zeros(n + 1, np.int64)
    left = 0
    for k in range(n):
        if k == 0 or gid[k] != gid[k - 1]:
            left = k
        while st[left] <= st[k] - w:
            left += 1
        m = k - left + 1
        # start-based features over all members
        ud = 0
        up = 0
        ur = 0
        nt = 0
        gs = 0.0
        for j in range(left, k + 1):
            c = dst[j]
            if mark_d[c] != k:
                mark_d[c] = k
                cnt_d[c] = 0
                touched[nt] = c
                nt += 1
                ud += 1
            cnt_d[c] += 1
            if mark_p[port[j]] != k:
                mark_p[port[j]] = k
                up += 1
            if mark_r[proto[j]] != k:
                mark_r[proto[j]] = k
                ur += 1
            gs += gap[j]
        ent = 0.0
        for t in range(nt):
            p = cnt_d[touched[t]] / m
            ent -= p * np.log2(p)
        gmean = gs / m
        gstd = 0.0
        if m > 1:
            ss = 0.0
            for j in range(left, k + 1):
                ss += (gap[j] - gmean) ** 2
            gstd = np.sqrt(ss / (m - 1))
        # value features over members that have ended by the current flow's end
        nv = 0
        sb = 0.0
        sp = 0.0
        sd = 0.0
        sr = 0.0
        sf = 0.0
        for j in range(left, k + 1):
            if completed_only and lt[j] > lt[k]:
                continue
            nv += 1
            sb += byt[j]
            sp += pkt[j]
            sd += dur[j]
            sr += rtt[j]
            sf += fail[j]
        dmean = sd / nv
        dstd = 0.0
        if nv > 1:
            ss = 0.0
            for j in range(left, k + 1):
                if completed_only and lt[j] > lt[k]:
                    continue
                ss += (dur[j] - dmean) ** 2
            dstd = np.sqrt(ss / (nv - 1))
        out[k, 0] = m / w
        out[k, 1] = sb / w
        out[k, 2] = sp / w
        out[k, 3] = dmean
        out[k, 4] = dstd
        out[k, 5] = sr / nv
        out[k, 6] = ud
        out[k, 7] = up
        out[k, 8] = ent
        out[k, 9] = sf / nv
        out[k, 10] = gstd / (gmean + 1e-9)
        out[k, 11] = ur
        out[k, 12] = 0.0 if left == k and (k == 0 or gid[k] != gid[k - 1]) else \
            out[k, 0] - out[k - 1, 0]
    return out

def build(df, scales, completed_only):
    codes = pd.factorize(df["srcip"])[0]
    order = np.argsort(codes, kind="stable")          # by source, then stored row order
    g = codes[order]
    ts = df["ts"].to_numpy()
    st = (ts.astype("datetime64[s]").astype(np.int64)).astype(np.float64)[order]
    lt = df["ltime"].to_numpy(np.float64)[order]
    gap = np.zeros(len(df))
    same = np.r_[False, g[1:] == g[:-1]]
    gap[same] = st[1:][same[1:]] - st[:-1][same[1:]]
    dstc = pd.factorize(df["dstip"])[0][order]
    portc = pd.factorize(df["dsport"])[0][order]
    protoc = pd.factorize(df["proto"])[0][order]
    args = (g, st, lt,
            df["dur"].to_numpy(np.float64)[order],
            (df["sbytes"] + df["dbytes"]).to_numpy(np.float64)[order],
            (df["spkts"] + df["dpkts"]).to_numpy(np.float64)[order],
            df["tcprtt"].to_numpy(np.float64)[order],
            df["state"].isin(FAILED_STATES).to_numpy(np.float64)[order],
            dstc, portc, protoc, gap)
    feats = {}
    for scale in scales:
        t0 = time.perf_counter()
        o = window_kernel(*args, WINDOWS[scale], completed_only,
                          dstc.max() + 1, portc.max() + 1, protoc.max() + 1)
        back = np.empty_like(o)
        back[order] = o
        for c, suf in enumerate(SUFFIXES):
            feats[f"{scale}_{suf}"] = back[:, c]
        print(f"  {scale}: {time.perf_counter() - t0:.1f}s", flush=True)
    return pd.DataFrame(feats, index=df.index)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exclude", default=None)
    ap.add_argument("--scales", default="short,medium,long")
    ap.add_argument("--legacy", action="store_true")
    a = ap.parse_args()
    scales = a.scales.split(",")

    src = source_dir()
    window_cols = [f"{s}_{suf}" for s in WINDOWS for suf in SUFFIXES]
    tr = pd.read_parquet(src / "train_windowed.parquet")
    te = pd.read_parquet(src / "test_windowed.parquet")
    keep_cols = [c for c in tr.columns if c not in window_cols and c != "__index_level_0__"]
    stored = {"train": tr[window_cols].copy(), "test": te[window_cols].copy()} if a.legacy else None
    n_tr = len(tr)
    df = pd.concat([tr[keep_cols], te[keep_cols]], ignore_index=True)
    del tr, te
    split_time = df["ts"].iloc[n_tr]
    is_train = np.r_[np.ones(n_tr, bool), np.zeros(len(df) - n_tr, bool)]

    name = "temporal"
    if a.exclude:
        fam = a.exclude
        cats = df["attack_cat"].astype(str).str.strip()
        hit = cats.str.lower().isin({fam.lower()} | ({"backdoors"} if fam.lower() == "backdoor" else set()))
        drop = hit.to_numpy() & is_train
        print(f"excluding {int(drop.sum())} training-period {fam} flows "
              f"({int((hit.to_numpy() & ~is_train).sum())} test-period flows kept)", flush=True)
        df = df.loc[~drop].reset_index(drop=True)
        is_train = is_train[~drop]
        name += f"_excl_{fam.lower()}"
    if a.legacy:
        name += "_legacy"
    print(f"building {scales} windows over {len(df)} flows "
          f"({'legacy' if a.legacy else 'completed-only values'})", flush=True)
    feats = build(df, scales, completed_only=not a.legacy)
    df = pd.concat([df, feats], axis=1)
    assert (df["ts"][~is_train] >= split_time).all() and (df["ts"][is_train] < split_time).all()

    if a.legacy:
        for part, m in (("train", is_train), ("test", ~is_train)):
            ref = stored[part][[c for c in feats.columns]].reset_index(drop=True)
            new = df.loc[m, feats.columns].reset_index(drop=True)
            for c in feats.columns:
                bad = ~np.isclose(new[c].to_numpy(), ref[c].to_numpy(), rtol=1e-6, atol=1e-6)
                if bad.any():
                    print(f"  MISMATCH {part} {c}: {int(bad.sum())} rows, e.g. "
                          f"{new[c][bad].iloc[0]} vs {ref[c][bad].iloc[0]}")
        print("legacy check done (no MISMATCH lines = identical to 00)")
        return

    out = find_dir("data") / name
    out.mkdir(parents=True, exist_ok=True)
    df.loc[is_train].reset_index(drop=True).to_parquet(out / "train_windowed.parquet")
    df.loc[~is_train].reset_index(drop=True).to_parquet(out / "test_windowed.parquet")
    print(f"wrote {out} (train {int(is_train.sum())}, test {int((~is_train).sum())})")

if __name__ == "__main__":
    main()
