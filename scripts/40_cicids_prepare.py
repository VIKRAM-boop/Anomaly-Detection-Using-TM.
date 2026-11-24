"""
Phase 5, Task 6: CICIDS2017 prepared under the same leakage-aware protocol.

Source: the corrected CICIDS2017 of Engelen, Rimmer and Joosen (WTMC 2021,
https://intrusion-detection.distrinet-research.be/WTMC2021/), which re-
extracts the flows with a fixed CICFlowMeter and relabels >20% of the
original traces. Five day files, one row per flow, with source/destination
IP and port and a start timestamp (whole seconds).

Dataset-specific choices (UNSW-NB15 labels do not carry over; no family
here is assumed to correspond to a UNSW-NB15 family):

  families     BruteForce  FTP-Patator + SSH-Patator
               DoS         DoS Hulk, GoldenEye, slowloris, Slowhttptest
               DDoS        DDoS (LOIC)
               PortScan    PortScan
               Bot         Bot (Ares)
               WebAttack   Web Attack - Brute Force, XSS, Sql Injection
               Normal      BENIGN
  excluded     Infiltration (32 flows) and Heartbleed (11): too few to
               train on or to evaluate. "... - Attempted" flows: the
               relabelling marks attack traffic that carried no attack
               payload, which is neither benign nor a successful attack.
               Excluded rows are never fitted on or scored, but they are
               real traffic, so they still count in other flows' windows.
  split        each attack occupies its own hours of a single day, so one
               global time cut would put whole families only in train or
               only in test. Each class is split chronologically on its own
               timeline instead: its earliest 72% fit, next 8% validation,
               latest 20% test (UNSW-NB15: 20% test, last 10% of train
               validation). Within every class, train precedes test.
  features     every CICFlowMeter per-flow statistic except identifiers
               (Flow ID, IPs, ports, timestamp), +/-inf read as missing;
               Protocol one-hot; plus the 13 medium-scale (10 s) window
               features of 00b, per source IP, with values only from flows
               that have ended (ltime = start + Flow Duration). CICIDS has
               no connection state or TCP RTT: a flow with no backward
               packets counts as failed, and the RTT feature is dropped.
  LOFO         with family F held out, F's fit- and validation-period
               flows are removed before the windows are built and F leaves
               fit and validation; F's test-period flows are relabelled
               Unknown (99). F's validation-period flows are kept apart as
               the pseudo-unknowns of Task 2's nested selection.
  encoding     5 quantile thresholds per numeric feature fitted on fit rows
               (09's levels), one-hot protocol fitted on fit rows.
  resampling   majorities capped at 40,000, then smote / ros / none, fit
               rows only.

For each family writes <data>/openset_medium_<family>[_<strategy>]/ in 09's
format (X/y fit, val, test, X_val_heldout, meta.json), and for the trees
the continuous rows of the same partitions (C_fit = the capped real rows,
C_val, C_test, C_val_heldout).

Usage: 40_cicids_prepare.py <raw csv dir> [strategy ...]
"""
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import OneHotEncoder

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("w2", HERE / "00b_build_windows_v2.py")
w2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(w2)

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
FAMILY_OF = {"BENIGN": "Normal", "FTP-Patator": "BruteForce", "SSH-Patator": "BruteForce",
             "DoS Hulk": "DoS", "DoS GoldenEye": "DoS", "DoS slowloris": "DoS",
             "DoS Slowhttptest": "DoS", "DDoS": "DDoS", "PortScan": "PortScan", "Bot": "Bot",
             "Web Attack - Brute Force": "WebAttack", "Web Attack - XSS": "WebAttack",
             "Web Attack - Sql Injection": "WebAttack"}
FAMILIES = ["Bot", "BruteForce", "DDoS", "DoS", "PortScan", "WebAttack"]
FIT_END, VAL_END = 0.72, 0.80
QUANTILE_LEVELS = [0.10, 0.25, 0.50, 0.75, 0.90]
PER_CLASS_CAP = 40000
RANDOM_STATE = 42
UNKNOWN_LABEL = 99
ID_COLS = ["Flow ID", "Src IP", "Src Port", "Dst IP", "Dst Port", "Protocol", "Timestamp", "Label"]
WINDOW_DROP = ["medium_mean_tcprtt"]

def find_dir(name):
    o = os.environ.get("TM_DATA_DIR")
    if name == "data" and o:
        p = Path(o)
        p.mkdir(parents=True, exist_ok=True)
        return p
    return HERE.parent / name

WCOLS = ["Src IP", "Dst IP", "Dst Port", "Protocol", "Flow Duration", "Total Bwd packets",
         "Total Fwd Packet", "Total Length of Fwd Packet", "Total Length of Bwd Packet"]

def load(raw):
    """-> (frame of the columns windows/labels need, float32 matrix of every
    per-flow statistic, its column names), all in start-time order. Read
    day by day straight into float32 to keep memory low."""
    frames, mats = [], []
    for d in DAYS:
        df = pd.read_csv(raw / f"{d}-WorkingHours.csv", low_memory=False)
        num_cols = [c for c in df.columns if c not in ID_COLS]
        mats.append(df[num_cols].to_numpy(np.float32))
        f = df[WCOLS].copy()
        f["ts"] = pd.to_datetime(df["Timestamp"], format="%d/%m/%Y %I:%M:%S %p")
        f["family"] = df["Label"].map(FAMILY_OF)          # NaN = excluded (context only)
        frames.append(f)
        del df
    df = pd.concat(frames, ignore_index=True)
    V = np.concatenate(mats)
    del frames, mats
    order = np.argsort(df["ts"].to_numpy(), kind="stable")
    return df.iloc[order].reset_index(drop=True), V[order], num_cols

def window_frame(df):
    """00b's per-flow columns, from CICFlowMeter's."""
    st = df["ts"].astype("datetime64[s]").astype(np.int64).astype(np.float64)
    return pd.DataFrame({
        "srcip": df["Src IP"].astype(str), "ts": df["ts"],
        "ltime": st + df["Flow Duration"].clip(lower=0) / 1e6,
        "dstip": df["Dst IP"].astype(str), "dsport": df["Dst Port"], "proto": df["Protocol"],
        "state": np.where(df["Total Bwd packets"] == 0, "INT", "CON"),
        "dur": df["Flow Duration"].clip(lower=0) / 1e6,
        "sbytes": df["Total Length of Fwd Packet"], "dbytes": df["Total Length of Bwd Packet"],
        "spkts": df["Total Fwd Packet"], "dpkts": df["Total Bwd packets"], "tcprtt": 0.0})

def windows(df, keep):
    feats = w2.build(window_frame(df.loc[keep]).reset_index(drop=True), ["medium"], True)
    out = pd.DataFrame(np.nan, index=df.index, columns=feats.columns)
    out.loc[keep] = feats.to_numpy()
    return out.drop(columns=WINDOW_DROP)

def split_by_class(df):
    """fit / val / test per class, chronologically within the class."""
    part = np.full(len(df), "", dtype=object)
    for c, idx in df.groupby("family").indices.items():
        idx = np.sort(idx)                                  # df is in time order
        n = len(idx)
        part[idx[: int(n * FIT_END)]] = "fit"
        part[idx[int(n * FIT_END): int(n * VAL_END)]] = "val"
        part[idx[int(n * VAL_END):]] = "test"
    return part

def quantile_encode(fit, others):
    thr = np.nanquantile(fit, QUANTILE_LEVELS, axis=0)       # (5, F)
    def enc(V):
        Z = np.nan_to_num(V, nan=-np.inf)
        return (Z[:, :, None] > thr.T[None]).reshape(len(V), -1).astype(np.uint8)
    return enc(fit), [enc(o) for o in others]

def resample(X, y, strategy, rng):
    keep = []
    for c in np.unique(y):
        i = np.where(y == c)[0]
        keep.append(rng.choice(i, PER_CLASS_CAP, replace=False) if len(i) > PER_CLASS_CAP else i)
    keep = np.concatenate(keep)
    Xc, yc = X[keep], y[keep]
    if strategy == "none":
        return Xc, yc, keep
    if strategy == "ros":
        extra = [rng.choice(np.where(yc == c)[0], PER_CLASS_CAP - (yc == c).sum(), replace=True)
                 for c in np.unique(yc) if (yc == c).sum() < PER_CLASS_CAP]
        extra = np.concatenate(extra) if extra else np.zeros(0, int)
        return np.concatenate([Xc, Xc[extra]]), np.concatenate([yc, yc[extra]]), keep
    counts = np.bincount(yc)
    targets = {int(c): PER_CLASS_CAP for c in np.unique(yc)}
    k = min(5, int(counts[counts > 0].min()) - 1)
    Xs, ys = SMOTE(random_state=RANDOM_STATE, sampling_strategy=targets,
                   k_neighbors=k).fit_resample(Xc.astype(np.float32), yc)
    return (Xs >= 0.5).astype(np.uint8), ys, keep

def main():
    raw = Path(sys.argv[1])
    strategies = sys.argv[2:] or ["smote", "none"]
    data = find_dir("data")
    df, V, num_cols = load(raw)
    names = sorted(FAMILIES + ["Normal"])
    cls = {n: i for i, n in enumerate(names)}
    labelled = df["family"].notna().to_numpy()
    part = split_by_class(df)
    y_all = np.full(len(df), -1, np.int64)
    y_all[labelled] = df.loc[labelled, "family"].map(cls).to_numpy()
    print(f"{len(df)} flows, {labelled.sum()} labelled; per class (fit/val/test):", flush=True)
    for n in names:
        m = y_all == cls[n]
        print(f"  {n:11s} {int((m & (part == 'fit')).sum()):8d} {int((m & (part == 'val')).sum()):7d} "
              f"{int((m & (part == 'test')).sum()):7d}", flush=True)

    V[~np.isfinite(V)] = np.nan
    full_win = windows(df, np.ones(len(df), bool))
    Wf = full_win.to_numpy(np.float32)

    for fam in FAMILIES:
        h = cls[fam]
        # windows built as if F's training-period flows never occurred
        keep = ~((y_all == h) & np.isin(part, ["fit", "val"]))
        win = windows(df, keep)
        W = win.to_numpy(np.float32)
        fit = (part == "fit") & (y_all >= 0) & (y_all != h)
        val = (part == "val") & (y_all >= 0) & (y_all != h)
        test = (part == "test") & (y_all >= 0)
        vh = (part == "val") & (y_all == h)
        Xnum = np.concatenate([V, W], 1)
        del W
        Xnum_vh = np.concatenate([V[vh], Wf[vh]], 1)          # pseudo-unknowns: full windows
        lit_fit, (lit_val, lit_test, lit_vh) = quantile_encode(Xnum[fit], [Xnum[val], Xnum[test], Xnum_vh])
        oh = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.uint8)
        oh.fit(df.loc[fit, ["Protocol"]].astype(str))
        P = lambda m: oh.transform(df.loc[m, ["Protocol"]].astype(str))
        X_fit = np.concatenate([lit_fit, P(fit)], 1)
        X_val = np.concatenate([lit_val, P(val)], 1)
        X_test = np.concatenate([lit_test, P(test)], 1)
        X_vh = np.concatenate([lit_vh, P(vh)], 1)
        feat_names = [f"{c}>Q{q*100:g}" for c in num_cols + list(win.columns) for q in QUANTILE_LEVELS] + \
                     [f"proto_{c}" for c in oh.categories_[0]]
        y_fit, y_val = y_all[fit], y_all[val]
        y_test = y_all[test].copy()
        y_test[y_test == h] = UNKNOWN_LABEL
        val_time = str(df.loc[val, "ts"].min())
        base = data / f"openset_medium_{fam.lower()}"
        proto_code = pd.Categorical(df["Protocol"].astype(str)).codes.astype(np.float32)[:, None]
        C = lambda m: np.concatenate([np.nan_to_num(Xnum[m], nan=0.0), proto_code[m]], 1)
        C_vh = np.concatenate([np.nan_to_num(Xnum_vh, nan=0.0), proto_code[vh]], 1)
        for strategy in strategies:
            rng = np.random.default_rng(RANDOM_STATE)
            Xb, yb, keep_idx = resample(X_fit, y_fit, strategy, rng)
            out = base if strategy == "smote" else base.with_name(base.name + f"_{strategy}")
            out.mkdir(parents=True, exist_ok=True)
            np.save(out / "X_fit.npy", Xb)
            np.save(out / "y_fit.npy", yb.astype(np.uint8))
            for nm, arr in (("X_val", X_val), ("y_val", y_val.astype(np.uint8)), ("X_test", X_test),
                            ("y_test", y_test.astype(np.uint8)), ("X_val_heldout", X_vh)):
                if out != base and (base / f"{nm}.npy").exists():
                    (out / f"{nm}.npy").unlink(missing_ok=True)
                    (out / f"{nm}.npy").symlink_to(base / f"{nm}.npy")
                else:
                    np.save(out / f"{nm}.npy", arr)
            if strategy == "none":
                np.save(out / "C_fit.npy", C(np.flatnonzero(fit)[keep_idx]))
                np.save(out / "C_val.npy", C(val))
                np.save(out / "C_test.npy", C(test))
                np.save(out / "C_val_heldout.npy", C_vh)
                json.dump(num_cols + list(win.columns) + ["Protocol"], open(out / "C_names.json", "w"))
            json.dump({"dataset": "CICIDS2017 (Engelen et al. 2021)", "all_class_names": names,
                       "held_out_class": fam, "held_out_idx": h, "unknown_label": UNKNOWN_LABEL,
                       "known_classes": sorted(int(c) for c in np.unique(yb)),
                       "n_fit_real": int(fit.sum()),
                       "fit_class_counts_real": np.bincount(y_fit, minlength=len(names)).tolist(),
                       "n_fit": int(len(yb)), "n_val": int(val.sum()), "n_test": int(test.sum()),
                       "n_val_heldout": int(vh.sum()), "n_literals": int(Xb.shape[1]),
                       "strategy": strategy, "quantile_levels": QUANTILE_LEVELS,
                       "val_split_time": val_time, "scale": "medium", "windows": "excluding held-out",
                       "split": "per-class chronological 72/8/20", "feature_names": feat_names},
                      open(out / "meta.json", "w"), indent=1)
        print(f"{fam}: fit {X_fit.shape} -> {Xb.shape} val {X_val.shape} test {X_test.shape} "
              f"(unknown {int((y_test == UNKNOWN_LABEL).sum())}), pseudo-unknown val {int(vh.sum())}",
              flush=True)

if __name__ == "__main__":
    main()
