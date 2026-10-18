"""
Builds one binarized literal dataset per temporal scale.

Reads the windowed parquet files and converts them into binary literals:
five quantile thresholds per continuous feature, one-hot encoded
proto/service/state, two pass-through binary columns, and four relational
literals. Each scale's dataset combines the shared base literals with that
scale's own window literals.

The training partition is split chronologically into fit and validation
before anything is fitted: quantile thresholds, the one-hot encoder and the
relational thresholds all come from the fit rows alone, and resampling is
applied to the fit rows alone. Validation and test rows are only ever
transformed, never fitted on and never resampled.

Takes the resampling strategy as argv[1] -- one of smote (every class
raised to PER_CLASS_CAP), moderate (minorities raised to at most
MODERATE_MULTIPLE times their real count) or undersample (majorities cut to
UNDERSAMPLE_CAP, no synthetic rows at all). Writes X_fit/y_fit, X_val/y_val,
X_test/y_test and meta.json to data/temporal_<scale>[_<strategy>]/.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from imblearn.over_sampling import SMOTE

RANDOM_STATE = 42
PER_CLASS_CAP = 40000
VAL_FRACTION = 0.10          # chronological tail of the training period
MODERATE_MULTIPLE = 5        # moderate strategy: at most 5x a class's real count
UNDERSAMPLE_CAP = 10000      # undersample strategy: majority ceiling, no synthesis
QUANTILE_LEVELS = [0.10, 0.25, 0.50, 0.75, 0.90]
WINDOW_SCALES = ["short", "medium", "long"]
WINDOW_FEATURE_SUFFIXES = ["arrival_rate", "byte_rate", "pkt_rate", "mean_dur", "std_dur",
                           "mean_tcprtt", "unique_dst", "unique_dst_port", "dst_entropy",
                           "failed_ratio", "burstiness", "proto_nunique", "delta_arrival_rate"]
STRATEGIES = ("smote", "moderate", "undersample")

DROP_COLS = ["srcip", "sport", "dstip", "dsport", "stime", "ltime", "ts", "label"]
CATEGORICAL_COLS = ["proto", "service", "state"]
BINARY_PASSTHROUGH_COLS = ["is_ftp_login", "is_sm_ips_ports"]
LABEL_FIXES = {"Backdoors": "Backdoor"}

def find_dir(name):
    # TM_DATA_DIR redirects the data tree to scratch storage. Writing several
    # GB of .npy into a Dropbox-synced folder intermittently fails with
    # TimeoutError [Errno 60] partway through a write.
    if name == "data":
        override = os.environ.get("TM_DATA_DIR")
        if override:
            p = Path(override)
            p.mkdir(parents=True, exist_ok=True)
            return p
    here = Path(__file__).resolve().parent
    for candidate in (here.parent / name, here / name):
        if candidate.exists():
            return candidate
    return here / name

DATA_DIR = Path("/tmp/tmwork2")
if not (DATA_DIR / "train_windowed.parquet").exists():
    DATA_DIR = find_dir("data") / "temporal"

def quantile_literals(fit_vals, other_vals, feature_name):
    """Thresholds come from fit_vals only; every array in other_vals is
    transformed with those same thresholds."""
    thresholds = np.nanquantile(fit_vals, QUANTILE_LEVELS)
    names = [f"{feature_name}>Q{int(q*100)}" for q in QUANTILE_LEVELS]
    def encode(vals):
        return np.stack([(np.nan_to_num(vals, nan=-np.inf) > t).astype(np.uint32)
                          for t in thresholds], axis=1)
    return encode(fit_vals), [encode(v) for v in other_vals], names

def resample(X, y, strategy, rng):
    """Returns the resampled fit set for one strategy. Only ever called on
    fit rows."""
    counts = np.bincount(y)
    present = np.where(counts > 0)[0]

    if strategy == "undersample":
        # class-aware sampling with no synthesis: majorities come down to the
        # cap, minorities are left at their real size
        keep = []
        for cls in present:
            idx = np.where(y == cls)[0]
            if len(idx) > UNDERSAMPLE_CAP:
                idx = rng.choice(idx, size=UNDERSAMPLE_CAP, replace=False)
            keep.append(idx)
        keep = np.concatenate(keep)
        return X[keep], y[keep]

    # both SMOTE paths first cap the majorities, then synthesise upward
    keep = []
    for cls in present:
        idx = np.where(y == cls)[0]
        if len(idx) > PER_CLASS_CAP:
            idx = rng.choice(idx, size=PER_CLASS_CAP, replace=False)
        keep.append(idx)
    keep = np.concatenate(keep)
    X_capped, y_capped = X[keep], y[keep]
    capped_counts = np.bincount(y_capped)
    print(f"  after cap: {dict(enumerate(capped_counts))}")

    # a class needs at least two rows before anything can be interpolated
    # between them; the chronological split can leave a rare class that small
    oversamplable = [int(c) for c in present if capped_counts[c] >= 2]
    too_small = [int(c) for c in present if capped_counts[c] < 2]
    if too_small:
        print(f"  classes left unsynthesised (fewer than 2 fit rows): {too_small}")

    if strategy == "smote":
        targets = {c: PER_CLASS_CAP for c in oversamplable}
    else:  # moderate
        targets = {c: min(PER_CLASS_CAP, MODERATE_MULTIPLE * int(capped_counts[c]))
                   for c in oversamplable}

    # never ask SMOTE for fewer rows than a class already has
    targets = {c: max(t, int(capped_counts[c])) for c, t in targets.items()}
    # k_neighbors must stay below the smallest class SMOTE is asked to grow
    smallest = min(int(capped_counts[c]) for c in oversamplable)
    k = max(1, min(5, smallest - 1))

    # SMOTE must not see uint32 here. It computes x_i + d*(x_zi - x_i), and
    # on unsigned data the subtraction 0 - 1 wraps to 4294967295 instead of
    # -1, so synthetic rows come out as values in the billions rather than
    # 0/1. Interpolate in float32, then threshold back to binary: a bit is
    # set when the interpolation lands closer to 1 than to 0.
    sampler = SMOTE(random_state=RANDOM_STATE, sampling_strategy=targets, k_neighbors=k)
    X_bal_f, y_bal = sampler.fit_resample(X_capped.astype(np.float32), y_capped)
    X_bal = (X_bal_f >= 0.5).astype(np.uint32)
    assert set(np.unique(X_bal)).issubset({0, 1}), "binarization produced non-binary values"
    return X_bal, y_bal

def main():
    strategy = sys.argv[1] if len(sys.argv) > 1 else "smote"
    if strategy not in STRATEGIES:
        raise SystemExit(f"strategy must be one of {STRATEGIES}, got {strategy!r}")
    print(f"=== resampling strategy: {strategy} ===")

    print("=== loading windowed parquet ===")
    train_df = pd.read_parquet(DATA_DIR / "train_windowed.parquet")
    test_df = pd.read_parquet(DATA_DIR / "test_windowed.parquet")
    train_df["attack_cat"] = train_df["attack_cat"].replace(LABEL_FIXES)
    test_df["attack_cat"] = test_df["attack_cat"].replace(LABEL_FIXES)

    # chronological fit/validation split: validation is the tail of the
    # training period, so it sits after everything trained on and before
    # everything tested on
    train_df = train_df.sort_values("ts").reset_index(drop=True)
    split_idx = int(len(train_df) * (1 - VAL_FRACTION))
    val_split_time = train_df["ts"].iloc[split_idx]
    fit_df = train_df[train_df["ts"] < val_split_time].reset_index(drop=True)
    val_df = train_df[train_df["ts"] >= val_split_time].reset_index(drop=True)
    print(f"fit: {fit_df.shape}, val: {val_df.shape}, test: {test_df.shape}")
    print(f"  fit  < {val_split_time} <= val < {test_df['ts'].min()} <= test")

    print("\n=== encoding target ===")
    # fitted on the full training period so class indices stay stable even if a
    # rare class happens to fall entirely inside the validation tail
    target_encoder = LabelEncoder().fit(train_df["attack_cat"])
    y_fit_full = target_encoder.transform(fit_df["attack_cat"])
    y_val = target_encoder.transform(val_df["attack_cat"])
    y_test = target_encoder.transform(test_df["attack_cat"])
    class_names = list(target_encoder.classes_)
    print(f"classes: {class_names}")
    print(f"fit counts: {dict(zip(class_names, np.bincount(y_fit_full, minlength=len(class_names))))}")
    print(f"val counts: {dict(zip(class_names, np.bincount(y_val, minlength=len(class_names))))}")

    base_cols = [c for c in fit_df.columns
                 if c not in DROP_COLS + CATEGORICAL_COLS + ["attack_cat"]
                 and not c.startswith(tuple(f"{s}_" for s in WINDOW_SCALES))]
    print(f"\n=== base per-flow literals ({len(base_cols)} continuous features) ===")

    fit_base_blocks, val_base_blocks, test_base_blocks, base_names = [], [], [], []
    for col in base_cols:
        fb, (vb, xb), names = quantile_literals(
            fit_df[col].to_numpy(dtype=np.float64),
            [val_df[col].to_numpy(dtype=np.float64),
             test_df[col].to_numpy(dtype=np.float64)], col)
        fit_base_blocks.append(fb); val_base_blocks.append(vb); test_base_blocks.append(xb)
        base_names.extend(names)

    print("=== one-hot encoding proto/service/state ===")
    cat_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.uint32)
    cat_encoder.fit(fit_df[CATEGORICAL_COLS])
    fit_cat = cat_encoder.transform(fit_df[CATEGORICAL_COLS])
    val_cat = cat_encoder.transform(val_df[CATEGORICAL_COLS])
    test_cat = cat_encoder.transform(test_df[CATEGORICAL_COLS])
    cat_names = list(cat_encoder.get_feature_names_out(CATEGORICAL_COLS))
    print(f"  {len(cat_names)} one-hot columns")

    # is_ftp_login is documented as a flag but the raw capture also contains
    # 2 and 4, so these are thresholded rather than passed through. Previously
    # only the training rows were implicitly binarised, as a side effect of
    # SMOTE's >=0.5 threshold, leaving test rows holding raw 2s and 4s.
    def binary_block(df):
        return (df[BINARY_PASSTHROUGH_COLS].fillna(0).to_numpy(dtype=np.float64) > 0
                ).astype(np.uint32)
    fit_binary = binary_block(fit_df)
    val_binary = binary_block(val_df)
    test_binary = binary_block(test_df)

    print("=== relational literals ===")
    rel_names, fit_rel, val_rel, test_rel = [], [], [], []
    for df, store in ((fit_df, fit_rel), (val_df, val_rel), (test_df, test_rel)):
        ratio = (df["sbytes"] + 1) / (df["dbytes"] + 1)
        store.append((ratio > 2).astype(np.uint32).to_numpy())
        store.append((ratio < 0.5).astype(np.uint32).to_numpy())
    rel_names += ["R_outbound_heavy", "R_inbound_heavy"]

    rel_defs = {
        "R_many_conn_same_dst": (("ct_dst_ltm", 0.75), ("ct_srv_dst", 0.75)),
        "R_slow_handshake": (("synack", 0.75), ("tcprtt", 0.75)),
    }
    for name, ((c1, l1), (c2, l2)) in rel_defs.items():
        # thresholds from fit rows only
        t1, t2 = np.nanquantile(fit_df[c1], l1), np.nanquantile(fit_df[c2], l2)
        for df, store in ((fit_df, fit_rel), (val_df, val_rel), (test_df, test_rel)):
            store.append(((df[c1] > t1) & (df[c2] > t2)).astype(np.uint32).to_numpy())
        rel_names.append(name)

    X_fit_base = np.concatenate(fit_base_blocks + [fit_cat, fit_binary,
                                 np.stack(fit_rel, axis=1)], axis=1).astype(np.uint32)
    X_val_base = np.concatenate(val_base_blocks + [val_cat, val_binary,
                                 np.stack(val_rel, axis=1)], axis=1).astype(np.uint32)
    X_test_base = np.concatenate(test_base_blocks + [test_cat, test_binary,
                                  np.stack(test_rel, axis=1)], axis=1).astype(np.uint32)
    base_feature_names = base_names + cat_names + BINARY_PASSTHROUGH_COLS + rel_names
    print(f"\nbase literals: {X_fit_base.shape[1]}")

    for scale in WINDOW_SCALES:
        print(f"\n=== building TM_{scale} dataset ===")
        win_cols = [f"{scale}_{suf}" for suf in WINDOW_FEATURE_SUFFIXES]
        win_fit, win_val, win_test, win_names = [], [], [], []
        for col in win_cols:
            fb, (vb, xb), names = quantile_literals(
                fit_df[col].to_numpy(dtype=np.float64),
                [val_df[col].to_numpy(dtype=np.float64),
                 test_df[col].to_numpy(dtype=np.float64)], col)
            win_fit.append(fb); win_val.append(vb); win_test.append(xb); win_names.extend(names)

        X_fit = np.concatenate([X_fit_base] + win_fit, axis=1).astype(np.uint32)
        X_val = np.concatenate([X_val_base] + win_val, axis=1).astype(np.uint32)
        X_test = np.concatenate([X_test_base] + win_test, axis=1).astype(np.uint32)
        feature_names = base_feature_names + win_names
        print(f"  {scale}: {X_fit.shape[1]} total literals ({len(win_names)} window-specific)")

        print(f"  resampling fit rows only ({strategy})")
        rng = np.random.default_rng(RANDOM_STATE)
        X_fit_bal, y_fit_bal = resample(X_fit, y_fit_full, strategy, rng)
        print(f"  after {strategy}: {dict(enumerate(np.bincount(y_fit_bal)))}")

        suffix = "" if strategy == "smote" else f"_{strategy}"
        out_dir = find_dir("data") / f"temporal_{scale}{suffix}"
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "X_fit.npy", X_fit_bal)
        np.save(out_dir / "y_fit.npy", y_fit_bal.astype(np.uint32))
        np.save(out_dir / "X_val.npy", X_val)
        np.save(out_dir / "y_val.npy", y_val.astype(np.uint32))
        np.save(out_dir / "X_test.npy", X_test)
        np.save(out_dir / "y_test.npy", y_test.astype(np.uint32))
        with open(out_dir / "meta.json", "w") as f:
            json.dump({"class_names": class_names, "feature_names": feature_names,
                       "strategy": strategy,
                       "n_fit": int(X_fit_bal.shape[0]), "n_val": int(X_val.shape[0]),
                       "n_test": int(X_test.shape[0]),
                       "n_fit_real": int(len(y_fit_full)),
                       # real, pre-resampling class counts on the fit partition:
                       # the prior 03/08 calibrate against
                       "fit_class_counts_real": np.bincount(
                           y_fit_full, minlength=len(class_names)).tolist(),
                       "n_binary_features": int(X_fit_bal.shape[1]),
                       "val_split_time": str(val_split_time),
                       "scale": scale, "per_class_cap": PER_CLASS_CAP}, f, indent=2)
        print(f"  saved to {out_dir}")

if __name__ == "__main__":
    main()
