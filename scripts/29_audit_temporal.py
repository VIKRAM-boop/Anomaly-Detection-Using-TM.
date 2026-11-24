"""
Audit of the temporal-window construction in 00_build_temporal_windows.py.

00 sorts every flow by start time (stime, whole seconds) and, per source
IP, aggregates a trailing pandas offset window ("1s", "10s", "60s"). An
offset window is closed on the right, (t - w, t], and only reaches rows
positioned at or before the current one. This script recomputes the
windows independently on a random sample of flows and measures:

  alignment      stored arrival_rate * w equals the recomputed count of
                 window members (catches rows attached to the wrong flow)
  self           the current flow is a member of its own window, so its
                 own dur/bytes/pkts/rtt/state enter its window statistics
  in progress    a prior window member j that has not finished when the
                 current flow i does (ltime_j > ltime_i). Per-flow features
                 (dur, sbytes, ...) are only known at a flow's end, so the
                 natural prediction time for i is ltime_i; j's duration,
                 bytes, packets, RTT and state are not yet known then
  same second    a prior member with stime_j == stime_i. stime has whole-
                 second resolution and 00 sorts with a non-stable sort, so
                 which same-second flows count as "prior" is arbitrary
  LOFO           with family f held out, the share of known fit/validation
                 rows whose window still counts a training-period flow of f
                 (windows are built once on all flows, before 09 removes f)
  boundary       no fit/validation flow's window reaches a later partition
                 (windows only look backward; verified on the sample), and
                 the share of validation/test flows whose window reaches
                 back into the previous partition (past data, legitimate)

Writes data/phase5/audit_temporal.json.

Usage: 29_audit_temporal.py [sample_size]
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

WINDOWS = {"short": 1.0, "medium": 10.0, "long": 60.0}
VAL_FRACTION = 0.10
SAMPLE = int(sys.argv[1]) if len(sys.argv) > 1 else 200000
RNG = np.random.default_rng(0)

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

def main():
    src = Path("/tmp/tmwork2")
    if not (src / "train_windowed.parquet").exists():
        src = find_dir("data") / "temporal"
    cols = ["srcip", "stime", "ltime", "attack_cat"] + [f"{s}_arrival_rate" for s in WINDOWS]
    tr = pd.read_parquet(src / "train_windowed.parquet", columns=cols)
    te = pd.read_parquet(src / "test_windowed.parquet", columns=cols)
    n_tr = len(tr)
    df = pd.concat([tr, te], ignore_index=True)
    del tr, te
    stime = df["stime"].to_numpy(np.float64)
    ltime = df["ltime"].to_numpy(np.float64)
    assert np.all(np.diff(stime) >= 0), "rows are not in start-time order"

    # partitions exactly as 01c draws them: fit / validation by time inside train
    tr_t = pd.to_datetime(stime[:n_tr], unit="s")
    val_cut = tr_t[int(n_tr * (1 - VAL_FRACTION))]
    part = np.full(len(df), 2, np.int8)                   # 0 fit, 1 val, 2 test
    part[:n_tr] = np.where(tr_t < val_cut, 0, 1)

    codes = pd.factorize(df["srcip"])[0]
    order = np.argsort(codes, kind="stable")              # group, then global row order
    grp_start = np.r_[0, np.flatnonzero(np.diff(codes[order])) + 1]
    grp_end = np.r_[grp_start[1:], len(order)]
    pos_in_order = np.empty(len(order), np.int64)
    pos_in_order[order] = np.arange(len(order))
    grp_of = np.repeat(np.arange(len(grp_start)), grp_end - grp_start)

    sample = np.sort(RNG.choice(len(df), min(SAMPLE, len(df)), replace=False))
    out = {"sample": int(len(sample)), "rows": int(len(df)), "n_train": n_tr,
           "same_srcip_same_second_rows": float(
               df.duplicated(["srcip", "stime"], keep=False).mean()),
           "flows_longer_than_window": {s: float((ltime - stime > w).mean())
                                        for s, w in WINDOWS.items()},
           "scales": {}}
    cats = df["attack_cat"].to_numpy()
    for scale, w in WINDOWS.items():
        stored = df[f"{scale}_arrival_rate"].to_numpy() * w
        acc = {k: np.zeros(len(sample), bool) for k in
               ("aligned", "self", "in_progress", "same_second", "forward", "back_cross")}
        fams = [c for c in np.unique(cats) if c not in ("Normal", "Backdoors")]
        held = {f: np.zeros(len(sample), bool) for f in fams}
        share_in_progress = np.zeros(len(sample))
        for n, i in enumerate(sample):
            p = pos_in_order[i]
            g = grp_of[p]
            lo = grp_start[g]
            members = order[lo:p + 1]                     # same source, at or before i
            st = stime[members]
            first = np.searchsorted(st, stime[i] - w, side="right")
            win = members[first:]                         # (t - w, t], i last
            prior = win[:-1]
            acc["aligned"][n] = abs(stored[i] - len(win)) < 1e-6
            acc["self"][n] = win[-1] == i
            if len(prior):
                late = ltime[prior] > ltime[i]
                acc["in_progress"][n] = late.any()
                share_in_progress[n] = late.mean()
                acc["same_second"][n] = (stime[prior] == stime[i]).any()
                acc["forward"][n] = (part[prior] > part[i]).any()
                acc["back_cross"][n] = (part[prior] < part[i]).any()
                if part[i] < 2:
                    pc = set(cats[prior][part[prior] < 2])
                    for f in fams:
                        held[f][n] = f in pc or (f == "Backdoor" and "Backdoors" in pc)
        res = {"aligned": float(acc["aligned"].mean()),
               "includes_self": float(acc["self"].mean()),
               "has_in_progress_prior": float(acc["in_progress"].mean()),
               "mean_share_of_prior_in_progress": float(share_in_progress.mean()),
               "has_same_second_prior": float(acc["same_second"].mean()),
               "window_reaches_later_partition": int(acc["forward"].sum()),
               "val_or_test_window_reaches_back": float(
                   acc["back_cross"][part[sample] > 0].mean())}
        by_cat = {}
        for c in np.unique(cats[sample]):
            m = cats[sample] == c
            by_cat[c] = {"n": int(m.sum()),
                         "has_in_progress_prior": float(acc["in_progress"][m].mean()),
                         "has_same_second_prior": float(acc["same_second"][m].mean())}
        res["by_class"] = by_cat
        # LOFO: share of the known fit/validation rows, with family f held
        # out, whose window still counts a training-period flow of f
        trv = part[sample] < 2
        res["lofo_window_contains_held_out"] = {}
        for f in fams:
            known = trv & (cats[sample] != f) & ~((f == "Backdoor") & (cats[sample] == "Backdoors"))
            res["lofo_window_contains_held_out"][f] = {
                "all_known": float(held[f][known].mean()),
                "known_attacks": float(held[f][known & (cats[sample] != "Normal")].mean())}
        out["scales"][scale] = res
        print(f"{scale}: aligned {res['aligned']:.4%}  self {res['includes_self']:.2%}  "
              f"in-progress prior {res['has_in_progress_prior']:.2%}  "
              f"same-second prior {res['has_same_second_prior']:.2%}  "
              f"forward reach {res['window_reaches_later_partition']}", flush=True)
    od = find_dir("data") / "phase5"
    od.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(od / "audit_temporal.json", "w"), indent=2)
    print(json.dumps({k: v for k, v in out.items() if k != "scales"}, indent=2))
    for s in WINDOWS:
        print(s, "LOFO contamination", json.dumps(out["scales"][s]["lofo_window_contains_held_out"]))

if __name__ == "__main__":
    main()
