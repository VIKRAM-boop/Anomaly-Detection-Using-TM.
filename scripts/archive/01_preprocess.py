"""
Stage 1 of the new proposed-methodology system (Steps 1-3 of the spec):
feature engineering, leakage-free preprocessing, and Boolean literal
construction -- deliberately WITHOUT temporal windowing (Step 4) or the
benign/known-attack/unknown-anomaly hierarchy (Step 5), which come later
once this flat baseline is validated.

Column choices vs. paper 2 (Salvi et al.) are deliberately different --
this is a separate spec, not a reproduction:
  - proto/service/state are KEPT (one-hot encoded), not dropped: the spec's
    Step 1 candidate feature list explicitly includes protocol, and neither
    service nor state are "identifiers" or "collection artifacts" per
    Step 2's removal criteria.
  - Only id, label are dropped (identifier + duplicate of attack_cat).
  - The official train/test split is used as-is (no swap) -- this is a
    fresh implementation, not tied to paper 2's confirmed swap finding.

Step 3 (Boolean literals): rather than KBinsDiscretizer's automatic
n_bins quantile binning (paper 2's approach), this follows the spec's
literal exactly: for each continuous feature, compute Q10/Q25/Q50/Q75/Q90
from training data, and emit five threshold literals per feature
(x > Q10, ..., x > Q90). TM's built-in literal negation (feature_negation)
gives the complementary "<=" literals for free, so "x < Q10"-style
literals from the spec's examples don't need to be materialized separately.

Relational literals (spec's R1-R3): hand-crafted AND-combinations using
the closest real UNSW-NB15 analogs to the spec's generic examples
(SYN/ACK dynamics -> tcprtt/synack/ackdat; outbound/inbound byte
asymmetry -> sbytes/dbytes; many-connections-same-destination -> the
dataset's own ct_dst_ltm/ct_srv_dst windowed connection-count features).
These are added as EXTRA binary columns alongside the per-feature
threshold literals, not a replacement for them -- the TM's clause
learning can already discover such conjunctions on its own given the
individual literals, so these are a head start, not a requirement.

Resampling: the spec doesn't mandate a specific balancing technique, just
that "any resampling is restricted to training data." SMOTE is applied
here as our own choice, consistent with paper 2's established finding
that TM benefits substantially from class balancing on this dataset.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from imblearn.over_sampling import SMOTE

RANDOM_STATE = 42
DROP_COLS = ["id", "label"]
CATEGORICAL_COLS = ["proto", "service", "state"]
BINARY_PASSTHROUGH_COLS = ["is_ftp_login", "is_sm_ips_ports"]
QUANTILE_LEVELS = [0.10, 0.25, 0.50, 0.75, 0.90]

def find_dir(name):
    here = Path(__file__).resolve().parent
    for candidate in (here.parent / name, here / name):
        if candidate.exists():
            return candidate
    return here / name

RAW_DIR = find_dir("raw_data")
OUT_DIR = find_dir("data") / "stage1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def load_and_clean(path):
    df = pd.read_csv(path)
    df.columns = [c.strip().lstrip("﻿") for c in df.columns]
    df["attack_cat"] = df["attack_cat"].fillna("Normal")
    y = df["attack_cat"].astype(str)
    X = df.drop(columns=[c for c in DROP_COLS + ["attack_cat"] if c in df.columns])
    return X, y

def quantile_literals(train_vals, *other_vals, feature_name):
    """For one continuous feature column, return (n_samples, 5) binary
    arrays (per dataset) marking x > Q10/Q25/Q50/Q75/Q90, thresholds fit
    on train_vals only."""
    thresholds = np.quantile(train_vals, QUANTILE_LEVELS)
    names = [f"{feature_name}>Q{int(q*100)}" for q in QUANTILE_LEVELS]
    out = [np.stack([(vals > t).astype(np.uint32) for t in thresholds], axis=1)
           for vals in (train_vals, *other_vals)]
    return out, names

def main():
    print("=== loading (official split: 175,341 train / 82,332 test) ===")
    X_train, y_train_raw = load_and_clean(RAW_DIR / "UNSW_NB15_training-set.csv")
    X_test, y_test_raw = load_and_clean(RAW_DIR / "UNSW_NB15_testing-set.csv")
    print(f"train: {X_train.shape}, test: {X_test.shape}")

    print("\n=== encoding target (attack_cat) ===")
    target_encoder = LabelEncoder().fit(y_train_raw)
    y_train = target_encoder.transform(y_train_raw)
    y_test = target_encoder.transform(y_test_raw)
    class_names = list(target_encoder.classes_)
    print(f"classes: {class_names}")

    # relational literals computed from raw (unscaled) values, quantiles fit on train
    print("\n=== relational literals (train-fit quantiles) ===")
    def q_train(col, level):
        return np.quantile(X_train[col], level)
    rel_names = []
    train_rel_cols, test_rel_cols = [], []

    # byte-direction asymmetry: a row-wise RATIO between sbytes/dbytes,
    # not two independently-thresholded marginal quantiles -- sbytes and
    # dbytes are correlated (bigger flows are bigger in both directions),
    # so ANDing independent global-quantile conditions on each came out
    # completely empty (0 positives in train and test). Comparing the two
    # features to each other per-sample is what "relational" should mean.
    ratio = (X_train["sbytes"] + 1) / (X_train["dbytes"] + 1)
    ratio_test = (X_test["sbytes"] + 1) / (X_test["dbytes"] + 1)
    train_rel_cols.append((ratio > 2).astype(np.uint32).to_numpy())
    test_rel_cols.append((ratio_test > 2).astype(np.uint32).to_numpy())
    rel_names.append("R_outbound_heavy")
    train_rel_cols.append((ratio < 0.5).astype(np.uint32).to_numpy())
    test_rel_cols.append((ratio_test < 0.5).astype(np.uint32).to_numpy())
    rel_names.append("R_inbound_heavy")

    rel_defs = {
        "R_many_conn_same_dst": (("ct_dst_ltm", 0.75, ">"), ("ct_srv_dst", 0.75, ">")),
        "R_slow_handshake": (("synack", 0.75, ">"), ("tcprtt", 0.75, ">")),
    }
    for name, ((c1, l1, op1), (c2, l2, op2)) in rel_defs.items():
        t1, t2 = q_train(c1, l1), q_train(c2, l2)
        def cond(df, c, t, op):
            return (df[c] > t) if op == ">" else (df[c] < t)
        train_rel_cols.append((cond(X_train, c1, t1, op1) & cond(X_train, c2, t2, op2)).astype(np.uint32).to_numpy())
        test_rel_cols.append((cond(X_test, c1, t1, op1) & cond(X_test, c2, t2, op2)).astype(np.uint32).to_numpy())
        rel_names.append(name)

    for name, train_col, test_col in zip(rel_names, train_rel_cols, test_rel_cols):
        print(f"  {name}: train positives={train_col.sum()}, test positives={test_col.sum()}")

    # categorical columns: one-hot, fit on train only, unseen test categories -> all-zero row
    print("\n=== one-hot encoding proto/service/state ===")
    cat_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.uint32)
    cat_encoder.fit(X_train[CATEGORICAL_COLS])
    train_cat = cat_encoder.transform(X_train[CATEGORICAL_COLS])
    test_cat = cat_encoder.transform(X_test[CATEGORICAL_COLS])
    cat_names = list(cat_encoder.get_feature_names_out(CATEGORICAL_COLS))
    print(f"  {len(cat_names)} one-hot columns from {CATEGORICAL_COLS}")

    # continuous columns: everything else numeric, excluding binary-passthrough
    continuous_cols = [c for c in X_train.columns
                        if c not in CATEGORICAL_COLS + BINARY_PASSTHROUGH_COLS]
    print(f"\n=== quantile literals for {len(continuous_cols)} continuous features ===")
    # spec's robust scaling (median/IQR) step is skipped here: it's a
    # monotonic transform, so it can't change which side of a per-feature
    # quantile threshold a value falls on -- with literals as the only
    # downstream representation, scaling first vs. not is mathematically
    # identical, so computing it would be pure overhead.

    train_lit_blocks, test_lit_blocks, lit_names = [], [], []
    for col in continuous_cols:
        train_vals = X_train[col].to_numpy(dtype=np.float64)
        test_vals = X_test[col].to_numpy(dtype=np.float64)
        (train_block, test_block), names = quantile_literals(train_vals, test_vals, feature_name=col)
        train_lit_blocks.append(train_block)
        test_lit_blocks.append(test_block)
        lit_names.extend(names)

    train_quantile_lits = np.concatenate(train_lit_blocks, axis=1)
    test_quantile_lits = np.concatenate(test_lit_blocks, axis=1)
    print(f"  {train_quantile_lits.shape[1]} quantile-threshold literals")

    # binary passthrough columns (already 0/1 in the raw data)
    train_binary = X_train[BINARY_PASSTHROUGH_COLS].to_numpy(dtype=np.uint32)
    test_binary = X_test[BINARY_PASSTHROUGH_COLS].to_numpy(dtype=np.uint32)

    train_rel = np.stack(train_rel_cols, axis=1)
    test_rel = np.stack(test_rel_cols, axis=1)

    X_train_bin = np.concatenate(
        [train_quantile_lits, train_cat, train_binary, train_rel], axis=1
    ).astype(np.uint32)
    X_test_bin = np.concatenate(
        [test_quantile_lits, test_cat, test_binary, test_rel], axis=1
    ).astype(np.uint32)
    all_feature_names = lit_names + cat_names + BINARY_PASSTHROUGH_COLS + rel_names
    print(f"\ntotal binary literals: {X_train_bin.shape[1]}")

    print(f"\n=== balancing (train only): SMOTE ===")
    print(f"before: {np.bincount(y_train)}")
    sampler = SMOTE(random_state=RANDOM_STATE)
    X_train_bal, y_train_bal = sampler.fit_resample(X_train_bin, y_train)
    print(f"after:  {np.bincount(y_train_bal)}")

    np.save(OUT_DIR / "X_train.npy", X_train_bal.astype(np.uint32))
    np.save(OUT_DIR / "y_train.npy", y_train_bal.astype(np.uint32))
    np.save(OUT_DIR / "X_test.npy", X_test_bin)
    np.save(OUT_DIR / "y_test.npy", y_test.astype(np.uint32))

    meta = {
        "class_names": class_names,
        "feature_names": all_feature_names,
        "n_train": int(X_train_bal.shape[0]),
        "n_test": int(X_test_bin.shape[0]),
        "n_binary_features": int(X_train_bal.shape[1]),
        "quantile_levels": QUANTILE_LEVELS,
        "relational_literals": rel_names,
    }
    with open(OUT_DIR / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n=== DONE -- saved to {OUT_DIR} ===")

if __name__ == "__main__":
    main()
