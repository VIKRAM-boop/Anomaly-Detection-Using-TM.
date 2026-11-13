"""
Builds an open-set dataset where one attack class is hidden from training.

Same literal encoding and chronological fit/validation split as
01c_preprocess_temporal.py. Every row belonging to the held-out class is
removed from BOTH the fit and validation partitions before any encoder is
fitted, so the family is unseen during training and model selection alike;
its test rows are kept and relabelled as the unknown class. Quantile
thresholds, one-hot categories and relational thresholds come from fit rows
only, and resampling touches fit rows only.

Takes the held-out class as argv[1] and builds all three temporal scales
(short, medium, long -- Step 4 calls for a separate clause bank per scale) in
one pass, sharing the parquet load and the scale-independent base literals.
Literals are stored as uint8: as uint32 the 27 family x scale datasets
would need about 49 GB. Writes X_fit/y_fit, X_val/y_val, X_test/y_test and
meta.json to data/openset_<scale>_<class>/.

Two optional arguments produce variants for the resampling and encoding
studies (written to data/openset_<scale>_<class>_<strategy>[_q<n>]/; the
defaults reproduce the original dataset and directory name):

  strategy  smote  cap majorities at PER_CLASS_CAP, SMOTE minorities up to
                   it on the binary literals, threshold back at 0.5 (original)
            ros    same cap, minorities raised to the cap by duplicating
                   real rows (random oversampling, nothing synthesised)
            none   same cap only; minorities keep their real size. The
                   banks then balance each binary problem by undersampling
                   (banks.balanced_binary). The capped rows are exactly the
                   rows 20_openset_baselines.py trains on, so the
                   continuous and Boolean baselines see the same flows.
  quantiles number of thresholds per continuous feature: 5 is the original
            {10,25,50,75,90}th percentiles; any other n uses n evenly
            spaced percentiles strictly between 0 and 100

Every variant shares the base dataset's validation and test rows. When the
base directory exists, a variant's X_val/X_test are compared with it and,
if byte-identical, stored as symlinks, so identical partitions are both
guaranteed and cheap. fit_synthetic.npy marks the fit rows that are not
real observations (SMOTE output or ROS duplicates).

Usage: 09_openset_preprocess.py <family> [scales] [strategy] [quantiles]
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
VAL_FRACTION = 0.10
PER_CLASS_CAP = 40000
QUANTILE_LEVELS = [0.10, 0.25, 0.50, 0.75, 0.90]
STRATEGIES = ("smote", "ros", "none")

def quantile_levels(n):
    if n == len(QUANTILE_LEVELS):
        return list(QUANTILE_LEVELS)
    return [round(i / (n + 1), 4) for i in range(1, n + 1)]
UNKNOWN_LABEL = 99          # marks held-out rows in y_test; never appears in y_train

WINDOW_FEATURE_SUFFIXES = ["arrival_rate", "byte_rate", "pkt_rate", "mean_dur", "std_dur",
                           "mean_tcprtt", "unique_dst", "unique_dst_port", "dst_entropy",
                           "failed_ratio", "burstiness", "proto_nunique", "delta_arrival_rate"]
WINDOW_SCALES = ["short", "medium", "long"]
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

def quantile_literals(fit_vals, other_vals, feature_name, levels=QUANTILE_LEVELS):
    """Thresholds from fit rows only; every array in other_vals is
    transformed with those same thresholds."""
    thresholds = np.nanquantile(fit_vals, levels)
    names = [f"{feature_name}>Q{q*100:g}" for q in levels]
    def encode(v):
        v = np.nan_to_num(v, nan=-np.inf)
        return np.stack([(v > t) for t in thresholds], axis=1).astype(np.uint8)
    return encode(fit_vals), [encode(v) for v in other_vals], names

def resample_fit(X_fit, y_fit_full, strategy="smote"):
    """Cap majorities at PER_CLASS_CAP, then (smote) synthesise or (ros)
    duplicate minority rows up to it, or (none) stop at the cap. Returns the
    rows, labels and a mask of rows that are not real observations."""
    rng = np.random.default_rng(RANDOM_STATE)
    keep_idx = []
    for cls in np.unique(y_fit_full):
        idx = np.where(y_fit_full == cls)[0]
        keep_idx.append(rng.choice(idx, PER_CLASS_CAP, replace=False)
                        if len(idx) > PER_CLASS_CAP else idx)
    keep_idx = np.concatenate(keep_idx)
    X_capped, y_capped = X_fit[keep_idx], y_fit_full[keep_idx]
    if strategy == "none":
        return X_capped, y_capped, np.zeros(len(y_capped), dtype=bool)
    if strategy == "ros":
        extra = []
        for cls in np.unique(y_capped):
            idx = np.where(y_capped == cls)[0]
            if len(idx) < PER_CLASS_CAP:
                extra.append(rng.choice(idx, PER_CLASS_CAP - len(idx), replace=True))
        extra = np.concatenate(extra) if extra else np.zeros(0, dtype=int)
        synth = np.r_[np.zeros(len(y_capped), bool), np.ones(len(extra), bool)]
        return (np.concatenate([X_capped, X_capped[extra]]),
                np.concatenate([y_capped, y_capped[extra]]), synth)
    counts = np.bincount(y_capped)
    present = [int(c) for c in np.unique(y_capped)]
    oversamplable = [c for c in present if counts[c] >= 2]
    targets = {c: max(PER_CLASS_CAP, int(counts[c])) for c in oversamplable}
    k = max(1, min(5, min(int(counts[c]) for c in oversamplable) - 1))
    # float32 then threshold: interpolating unsigned ints wraps 0-1 around
    X_bal_f, y_bal = SMOTE(random_state=RANDOM_STATE, sampling_strategy=targets,
                            k_neighbors=k).fit_resample(X_capped.astype(np.float32), y_capped)
    X_bal = (X_bal_f >= 0.5).astype(np.uint8)
    # imblearn returns the original rows first, synthetic rows after them
    synth = np.arange(len(y_bal)) >= len(y_capped)
    return X_bal, y_bal, synth

def main():
    held_out_class = sys.argv[1] if len(sys.argv) > 1 else "DoS"
    scales = sys.argv[2].split(",") if len(sys.argv) > 2 else WINDOW_SCALES
    strategy = sys.argv[3] if len(sys.argv) > 3 else "smote"
    n_q = int(sys.argv[4]) if len(sys.argv) > 4 else len(QUANTILE_LEVELS)
    if strategy not in STRATEGIES:
        raise SystemExit(f"strategy must be one of {STRATEGIES}")
    levels = quantile_levels(n_q)
    suffix = ("" if strategy == "smote" else f"_{strategy}") + \
             ("" if n_q == len(QUANTILE_LEVELS) else f"_q{n_q}")
    print(f"=== held out: {held_out_class} | scales: {scales} | {strategy} | "
          f"{n_q} quantiles {levels} ===", flush=True)

    train_df = pd.read_parquet(DATA_DIR / "train_windowed.parquet")
    test_df = pd.read_parquet(DATA_DIR / "test_windowed.parquet")
    train_df["attack_cat"] = train_df["attack_cat"].replace(LABEL_FIXES)
    test_df["attack_cat"] = test_df["attack_cat"].replace(LABEL_FIXES)

    # fitted on the full label set so class indices stay identical across hold-outs
    target_encoder = LabelEncoder().fit(train_df["attack_cat"])
    all_class_names = list(target_encoder.classes_)
    held_out_idx = all_class_names.index(held_out_class)
    y_test_orig = target_encoder.transform(test_df["attack_cat"])

    # the held-out family leaves both fit and validation, so it is unseen
    # during training, calibration and hyperparameter selection alike
    train_df = train_df.sort_values("ts").reset_index(drop=True)
    y_train_full = target_encoder.transform(train_df["attack_cat"])
    keep = y_train_full != held_out_idx
    train_df = train_df.loc[keep].reset_index(drop=True)
    y_known = y_train_full[keep]
    print(f"  train rows removed: {int((~keep).sum())} | test unknown rows: "
          f"{int((y_test_orig == held_out_idx).sum())}", flush=True)

    split_idx = int(len(train_df) * (1 - VAL_FRACTION))
    val_time = train_df["ts"].iloc[split_idx]
    is_fit = (train_df["ts"] < val_time).to_numpy()
    fit_df = train_df.loc[is_fit].reset_index(drop=True)
    val_df = train_df.loc[~is_fit].reset_index(drop=True)
    y_fit_full, y_val = y_known[is_fit], y_known[~is_fit]
    del train_df
    print(f"  fit={len(fit_df)} < {val_time} <= val={len(val_df)}", flush=True)

    # --- scale-independent base literals, thresholds from fit rows only ---
    base_cols = [c for c in fit_df.columns
                 if c not in DROP_COLS + CATEGORICAL_COLS + ["attack_cat"]
                 and not c.startswith(tuple(f"{sc}_" for sc in WINDOW_SCALES))]
    fit_blocks, val_blocks, test_blocks, names = [], [], [], []
    for col in base_cols:
        fb, (vb, xb), nm = quantile_literals(
            fit_df[col].to_numpy(dtype=np.float64),
            [val_df[col].to_numpy(dtype=np.float64),
             test_df[col].to_numpy(dtype=np.float64)], col, levels)
        fit_blocks.append(fb); val_blocks.append(vb); test_blocks.append(xb)
        names.extend(nm)

    cat_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.uint8)
    cat_encoder.fit(fit_df[CATEGORICAL_COLS])
    fit_blocks.append(cat_encoder.transform(fit_df[CATEGORICAL_COLS]))
    val_blocks.append(cat_encoder.transform(val_df[CATEGORICAL_COLS]))
    test_blocks.append(cat_encoder.transform(test_df[CATEGORICAL_COLS]))
    names.extend(cat_encoder.get_feature_names_out(CATEGORICAL_COLS))

    # is_ftp_login holds 0/1/2/4 in the raw capture, so it is thresholded
    def binary_block(df):
        return (df[BINARY_PASSTHROUGH_COLS].fillna(0).to_numpy(dtype=np.float64) > 0
                ).astype(np.uint8)
    fit_blocks.append(binary_block(fit_df))
    val_blocks.append(binary_block(val_df))
    test_blocks.append(binary_block(test_df))
    names.extend(BINARY_PASSTHROUGH_COLS)

    rel_fit, rel_val, rel_test = [], [], []
    for df, store in ((fit_df, rel_fit), (val_df, rel_val), (test_df, rel_test)):
        ratio = (df["sbytes"] + 1) / (df["dbytes"] + 1)
        store.append((ratio > 2).to_numpy().astype(np.uint8))
        store.append((ratio < 0.5).to_numpy().astype(np.uint8))
    rel_names = ["R_outbound_heavy", "R_inbound_heavy"]
    for nm, ((c1, l1), (c2, l2)) in {
        "R_many_conn_same_dst": (("ct_dst_ltm", 0.75), ("ct_srv_dst", 0.75)),
        "R_slow_handshake": (("synack", 0.75), ("tcprtt", 0.75)),
    }.items():
        t1, t2 = np.nanquantile(fit_df[c1], l1), np.nanquantile(fit_df[c2], l2)
        for df, store in ((fit_df, rel_fit), (val_df, rel_val), (test_df, rel_test)):
            store.append(((df[c1] > t1) & (df[c2] > t2)).to_numpy().astype(np.uint8))
        rel_names.append(nm)
    fit_blocks.append(np.stack(rel_fit, axis=1))
    val_blocks.append(np.stack(rel_val, axis=1))
    test_blocks.append(np.stack(rel_test, axis=1))
    names.extend(rel_names)

    base_fit = np.concatenate(fit_blocks, axis=1)
    base_val = np.concatenate(val_blocks, axis=1)
    base_test = np.concatenate(test_blocks, axis=1)
    del fit_blocks, val_blocks, test_blocks
    base_names = list(names)

    y_test = y_test_orig.copy()
    y_test[y_test_orig == held_out_idx] = UNKNOWN_LABEL

    for scale in scales:
        wf, wv, wt, wn = [], [], [], []
        for suf in WINDOW_FEATURE_SUFFIXES:
            col = f"{scale}_{suf}"
            fb, (vb, xb), nm = quantile_literals(
                fit_df[col].to_numpy(dtype=np.float64),
                [val_df[col].to_numpy(dtype=np.float64),
                 test_df[col].to_numpy(dtype=np.float64)], col, levels)
            wf.append(fb); wv.append(vb); wt.append(xb); wn.extend(nm)
        X_fit = np.concatenate([base_fit] + wf, axis=1)
        X_val = np.concatenate([base_val] + wv, axis=1)
        X_test = np.concatenate([base_test] + wt, axis=1)
        X_bal, y_bal, synth = resample_fit(X_fit, y_fit_full, strategy)
        del X_fit
        assert X_bal.max() <= 1 and X_val.max() <= 1 and X_test.max() <= 1

        base = find_dir("data") / f"openset_{scale}_{held_out_class.lower()}"
        out = base.with_name(base.name + suffix)
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "X_fit.npy", X_bal)
        np.save(out / "y_fit.npy", y_bal.astype(np.uint8))
        np.save(out / "fit_synthetic.npy", synth)
        for nm, arr in (("X_val", X_val), ("y_val", y_val.astype(np.uint8)),
                        ("X_test", X_test), ("y_test", y_test.astype(np.uint8))):
            src = base / f"{nm}.npy"
            target = out / f"{nm}.npy"
            if out != base and n_q == len(QUANTILE_LEVELS) and src.exists():
                # identical partitions across resampling variants, checked
                assert np.array_equal(np.load(src, mmap_mode="r"), arr), \
                    f"{nm} differs from the base dataset"
                target.unlink(missing_ok=True)
                target.symlink_to(src)
            else:
                np.save(target, arr)
        with open(out / "meta.json", "w") as f:
            json.dump({"all_class_names": all_class_names,
                       "held_out_class": held_out_class,
                       "held_out_idx": int(held_out_idx),
                       "unknown_label": UNKNOWN_LABEL,
                       "known_classes": sorted(int(c) for c in np.unique(y_bal)),
                       "n_fit_real": int(len(y_fit_full)),
                       "fit_class_counts_real": np.bincount(
                           y_fit_full, minlength=len(all_class_names)).tolist(),
                       "n_fit": int(len(y_bal)), "n_val": int(len(y_val)),
                       "n_test": int(len(y_test)),
                       "n_literals": int(X_bal.shape[1]),
                       "strategy": strategy, "quantile_levels": levels,
                       "n_synthetic": int(synth.sum()),
                       "val_split_time": str(val_time),
                       "scale": scale, "feature_names": base_names + wn}, f, indent=2)
        print(f"  {scale}: fit {X_bal.shape} val {X_val.shape} test {X_test.shape} "
              f"-> {out.name}", flush=True)

if __name__ == "__main__":
    main()
