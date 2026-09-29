"""
Builds an open-set dataset where one attack class is hidden from training.

Same literal encoding as 01c_preprocess_temporal.py, with one change: every
training row belonging to the held-out class is removed before any encoder is
fitted, so quantile thresholds, one-hot categories and relational thresholds
are all derived only from data the model is allowed to see. Its test rows are
kept and relabelled as the unknown class, giving genuinely novel data to
evaluate Step 5's unknown-anomaly branch against.

Takes the held-out class name as argv[1]. Writes X_train/y_train/X_test/y_test
and meta.json to data/openset_<scale>_<class>/.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from imblearn.over_sampling import SMOTE

RANDOM_STATE = 42
PER_CLASS_CAP = 40000
QUANTILE_LEVELS = [0.10, 0.25, 0.50, 0.75, 0.90]
SCALE = "medium"
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
    here = Path(__file__).resolve().parent
    for candidate in (here.parent / name, here / name):
        if candidate.exists():
            return candidate
    return here / name

DATA_DIR = Path("/tmp/tmwork2")
if not (DATA_DIR / "train_windowed.parquet").exists():
    DATA_DIR = find_dir("data") / "temporal"

def quantile_literals(train_vals, test_vals, feature_name):
    thresholds = np.nanquantile(train_vals, QUANTILE_LEVELS)
    names = [f"{feature_name}>Q{int(q*100)}" for q in QUANTILE_LEVELS]
    tb = np.stack([(np.nan_to_num(train_vals, nan=-np.inf) > t).astype(np.uint32)
                    for t in thresholds], axis=1)
    xb = np.stack([(np.nan_to_num(test_vals, nan=-np.inf) > t).astype(np.uint32)
                    for t in thresholds], axis=1)
    return tb, xb, names

def main():
    held_out_class = sys.argv[1] if len(sys.argv) > 1 else "DoS"

    print("=== loading windowed parquet ===")
    train_df = pd.read_parquet(DATA_DIR / "train_windowed.parquet")
    test_df = pd.read_parquet(DATA_DIR / "test_windowed.parquet")
    train_df["attack_cat"] = train_df["attack_cat"].replace(LABEL_FIXES)
    test_df["attack_cat"] = test_df["attack_cat"].replace(LABEL_FIXES)

    # fitted on the full label set so class indices stay identical across hold-outs
    target_encoder = LabelEncoder().fit(train_df["attack_cat"])
    all_class_names = list(target_encoder.classes_)
    held_out_idx = all_class_names.index(held_out_class)
    y_train_full = target_encoder.transform(train_df["attack_cat"])
    y_test_orig = target_encoder.transform(test_df["attack_cat"])
    print(f"all classes: {all_class_names}")
    print(f"holding out '{held_out_class}' (index {held_out_idx}) from training")

    # dropped before any encoder is fitted, so the held-out class informs nothing
    keep_mask = y_train_full != held_out_idx
    train_df = train_df.loc[keep_mask].reset_index(drop=True)
    y_known_full = y_train_full[keep_mask]
    print(f"  train rows removed: {int((~keep_mask).sum())} "
          f"(remaining {len(train_df)})")
    print(f"  test rows kept as unknown: {int((y_test_orig == held_out_idx).sum())}")

    # --- literal construction, identical to 01c ---
    base_cols = [c for c in train_df.columns
                 if c not in DROP_COLS + CATEGORICAL_COLS + ["attack_cat"]
                 and not c.startswith(tuple(f"{s}_" for s in WINDOW_SCALES))]
    train_blocks, test_blocks, names = [], [], []
    for col in base_cols:
        tb, xb, nm = quantile_literals(train_df[col].to_numpy(dtype=np.float64),
                                        test_df[col].to_numpy(dtype=np.float64), col)
        train_blocks.append(tb); test_blocks.append(xb); names.extend(nm)

    cat_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.uint32)
    cat_encoder.fit(train_df[CATEGORICAL_COLS])
    train_blocks.append(cat_encoder.transform(train_df[CATEGORICAL_COLS]))
    test_blocks.append(cat_encoder.transform(test_df[CATEGORICAL_COLS]))
    names.extend(cat_encoder.get_feature_names_out(CATEGORICAL_COLS))

    train_blocks.append(train_df[BINARY_PASSTHROUGH_COLS].fillna(0).to_numpy(dtype=np.uint32))
    test_blocks.append(test_df[BINARY_PASSTHROUGH_COLS].fillna(0).to_numpy(dtype=np.uint32))
    names.extend(BINARY_PASSTHROUGH_COLS)

    ratio = (train_df["sbytes"] + 1) / (train_df["dbytes"] + 1)
    ratio_te = (test_df["sbytes"] + 1) / (test_df["dbytes"] + 1)
    rel_tr = [(ratio > 2).astype(np.uint32).to_numpy(), (ratio < 0.5).astype(np.uint32).to_numpy()]
    rel_te = [(ratio_te > 2).astype(np.uint32).to_numpy(), (ratio_te < 0.5).astype(np.uint32).to_numpy()]
    rel_names = ["R_outbound_heavy", "R_inbound_heavy"]
    for nm, ((c1, l1), (c2, l2)) in {
        "R_many_conn_same_dst": (("ct_dst_ltm", 0.75), ("ct_srv_dst", 0.75)),
        "R_slow_handshake": (("synack", 0.75), ("tcprtt", 0.75)),
    }.items():
        t1, t2 = np.nanquantile(train_df[c1], l1), np.nanquantile(train_df[c2], l2)
        rel_tr.append(((train_df[c1] > t1) & (train_df[c2] > t2)).astype(np.uint32).to_numpy())
        rel_te.append(((test_df[c1] > t1) & (test_df[c2] > t2)).astype(np.uint32).to_numpy())
        rel_names.append(nm)
    train_blocks.append(np.stack(rel_tr, axis=1)); test_blocks.append(np.stack(rel_te, axis=1))
    names.extend(rel_names)

    for suf in WINDOW_FEATURE_SUFFIXES:
        col = f"{SCALE}_{suf}"
        tb, xb, nm = quantile_literals(train_df[col].to_numpy(dtype=np.float64),
                                        test_df[col].to_numpy(dtype=np.float64), col)
        train_blocks.append(tb); test_blocks.append(xb); names.extend(nm)

    X_known = np.concatenate(train_blocks, axis=1).astype(np.uint32)
    X_test = np.concatenate(test_blocks, axis=1).astype(np.uint32)
    y_known = y_known_full
    print(f"\nliterals: {X_known.shape[1]}")

    rng = np.random.default_rng(RANDOM_STATE)
    keep_idx = []
    for cls in np.unique(y_known):
        idx = np.where(y_known == cls)[0]
        keep_idx.append(rng.choice(idx, PER_CLASS_CAP, replace=False)
                        if len(idx) > PER_CLASS_CAP else idx)
    keep_idx = np.concatenate(keep_idx)
    X_capped, y_capped = X_known[keep_idx], y_known[keep_idx]
    print(f"after cap: {dict(zip(*np.unique(y_capped, return_counts=True)))}")

    # float32 then threshold: SMOTE on uint32 underflows (0-1 wraps to 4294967295)
    X_bal_f, y_bal = SMOTE(random_state=RANDOM_STATE).fit_resample(
        X_capped.astype(np.float32), y_capped)
    X_bal = (X_bal_f >= 0.5).astype(np.uint32)
    assert set(np.unique(X_bal)).issubset({0, 1}), "non-binary values after SMOTE"
    print(f"after SMOTE: {dict(zip(*np.unique(y_bal, return_counts=True)))}")

    # held-out rows in the test set become the unknown class
    y_test = y_test_orig.copy()
    y_test[y_test_orig == held_out_idx] = UNKNOWN_LABEL

    out = find_dir("data") / f"openset_{SCALE}_{held_out_class.lower()}"
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "X_train.npy", X_bal)
    np.save(out / "y_train.npy", y_bal.astype(np.uint32))
    np.save(out / "X_test.npy", X_test)
    np.save(out / "y_test.npy", y_test.astype(np.uint32))
    n_real = len(y_capped)
    with open(out / "meta.json", "w") as f:
        json.dump({"all_class_names": all_class_names,
                   "held_out_class": held_out_class,
                   "held_out_idx": int(held_out_idx),
                   "unknown_label": UNKNOWN_LABEL,
                   "known_classes": sorted(int(c) for c in np.unique(y_bal)),
                   "n_real_train_rows": int(n_real),
                   "n_train": int(len(y_bal)), "n_test": int(len(y_test)),
                   "n_literals": int(X_bal.shape[1]),
                   "scale": SCALE, "feature_names": list(names)}, f, indent=2)
    print(f"\nreal rows precede synthetic ones; boundary at {n_real}")
    print(f"saved to {out}")

if __name__ == "__main__":
    main()
