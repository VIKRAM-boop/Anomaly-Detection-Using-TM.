"""
Builds one binarized literal dataset per temporal scale.

Reads the windowed parquet files and converts them into binary literals:
five quantile thresholds per continuous feature, one-hot encoded
proto/service/state, two pass-through binary columns, and four relational
literals. Each scale's dataset combines the shared base literals with that
scale's own window literals.

Training rows are balanced by capping each class at PER_CLASS_CAP and
applying SMOTE to the remainder; the test set is left untouched. Writes
X_train/y_train/X_test/y_test and meta.json to data/temporal_<scale>/.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from imblearn.over_sampling import SMOTE

RANDOM_STATE = 42
PER_CLASS_CAP = 40000
QUANTILE_LEVELS = [0.10, 0.25, 0.50, 0.75, 0.90]
WINDOW_SCALES = ["short", "medium", "long"]
WINDOW_FEATURE_SUFFIXES = ["arrival_rate", "byte_rate", "pkt_rate", "mean_dur", "std_dur",
                           "mean_tcprtt", "unique_dst", "unique_dst_port", "dst_entropy",
                           "failed_ratio", "burstiness", "proto_nunique", "delta_arrival_rate"]

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

def quantile_literals(train_vals, test_vals, feature_name):
    thresholds = np.nanquantile(train_vals, QUANTILE_LEVELS)
    names = [f"{feature_name}>Q{int(q*100)}" for q in QUANTILE_LEVELS]
    train_block = np.stack([(np.nan_to_num(train_vals, nan=-np.inf) > t).astype(np.uint32)
                             for t in thresholds], axis=1)
    test_block = np.stack([(np.nan_to_num(test_vals, nan=-np.inf) > t).astype(np.uint32)
                            for t in thresholds], axis=1)
    return train_block, test_block, names

def main():
    print("=== loading windowed parquet ===")
    train_df = pd.read_parquet(DATA_DIR / "train_windowed.parquet")
    test_df = pd.read_parquet(DATA_DIR / "test_windowed.parquet")
    train_df["attack_cat"] = train_df["attack_cat"].replace(LABEL_FIXES)
    test_df["attack_cat"] = test_df["attack_cat"].replace(LABEL_FIXES)
    print(f"train: {train_df.shape}, test: {test_df.shape}")

    print("\n=== encoding target ===")
    target_encoder = LabelEncoder().fit(train_df["attack_cat"])
    y_train_full = target_encoder.transform(train_df["attack_cat"])
    y_test = target_encoder.transform(test_df["attack_cat"])
    class_names = list(target_encoder.classes_)
    print(f"classes: {class_names}")

    base_cols = [c for c in train_df.columns
                 if c not in DROP_COLS + CATEGORICAL_COLS + ["attack_cat"]
                 and not c.startswith(tuple(f"{s}_" for s in WINDOW_SCALES))]
    print(f"\n=== base per-flow literals ({len(base_cols)} continuous features) ===")

    train_base_blocks, test_base_blocks, base_names = [], [], []
    for col in base_cols:
        tb, xb, names = quantile_literals(
            train_df[col].to_numpy(dtype=np.float64),
            test_df[col].to_numpy(dtype=np.float64), col)
        train_base_blocks.append(tb); test_base_blocks.append(xb); base_names.extend(names)

    print("=== one-hot encoding proto/service/state ===")
    cat_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.uint32)
    cat_encoder.fit(train_df[CATEGORICAL_COLS])
    train_cat = cat_encoder.transform(train_df[CATEGORICAL_COLS])
    test_cat = cat_encoder.transform(test_df[CATEGORICAL_COLS])
    cat_names = list(cat_encoder.get_feature_names_out(CATEGORICAL_COLS))
    print(f"  {len(cat_names)} one-hot columns")

    train_binary = train_df[BINARY_PASSTHROUGH_COLS].fillna(0).to_numpy(dtype=np.uint32)
    test_binary = test_df[BINARY_PASSTHROUGH_COLS].fillna(0).to_numpy(dtype=np.uint32)

    print("=== relational literals ===")
    def q_train(df, col, level):
        return np.nanquantile(df[col], level)
    rel_names, train_rel_cols, test_rel_cols = [], [], []
    ratio = (train_df["sbytes"] + 1) / (train_df["dbytes"] + 1)
    ratio_test = (test_df["sbytes"] + 1) / (test_df["dbytes"] + 1)
    train_rel_cols += [(ratio > 2).astype(np.uint32).to_numpy(), (ratio < 0.5).astype(np.uint32).to_numpy()]
    test_rel_cols += [(ratio_test > 2).astype(np.uint32).to_numpy(), (ratio_test < 0.5).astype(np.uint32).to_numpy()]
    rel_names += ["R_outbound_heavy", "R_inbound_heavy"]
    rel_defs = {
        "R_many_conn_same_dst": (("ct_dst_ltm", 0.75, ">"), ("ct_srv_dst", 0.75, ">")),
        "R_slow_handshake": (("synack", 0.75, ">"), ("tcprtt", 0.75, ">")),
    }
    for name, ((c1, l1, op1), (c2, l2, op2)) in rel_defs.items():
        t1, t2 = q_train(train_df, c1, l1), q_train(train_df, c2, l2)
        def cond(df, c, t, op):
            return (df[c] > t) if op == ">" else (df[c] < t)
        train_rel_cols.append((cond(train_df, c1, t1, op1) & cond(train_df, c2, t2, op2)).astype(np.uint32).to_numpy())
        test_rel_cols.append((cond(test_df, c1, t1, op1) & cond(test_df, c2, t2, op2)).astype(np.uint32).to_numpy())
        rel_names.append(name)

    X_train_base = np.concatenate(train_base_blocks + [train_cat, train_binary,
                                   np.stack(train_rel_cols, axis=1)], axis=1).astype(np.uint32)
    X_test_base = np.concatenate(test_base_blocks + [test_cat, test_binary,
                                  np.stack(test_rel_cols, axis=1)], axis=1).astype(np.uint32)
    base_feature_names = base_names + cat_names + BINARY_PASSTHROUGH_COLS + rel_names
    print(f"\nbase literals: {X_train_base.shape[1]}")

    for scale in WINDOW_SCALES:
        print(f"\n=== building TM_{scale} dataset ===")
        win_cols = [f"{scale}_{suf}" for suf in WINDOW_FEATURE_SUFFIXES]
        win_train_blocks, win_test_blocks, win_names = [], [], []
        for col in win_cols:
            tb, xb, names = quantile_literals(
                train_df[col].to_numpy(dtype=np.float64),
                test_df[col].to_numpy(dtype=np.float64), col)
            win_train_blocks.append(tb); win_test_blocks.append(xb); win_names.extend(names)

        X_train = np.concatenate([X_train_base] + win_train_blocks, axis=1).astype(np.uint32)
        X_test = np.concatenate([X_test_base] + win_test_blocks, axis=1).astype(np.uint32)
        feature_names = base_feature_names + win_names
        print(f"  {scale}: {X_train.shape[1]} total literals ({len(win_names)} window-specific)")

        print(f"  balancing (cap={PER_CLASS_CAP}, downsample above / SMOTE below)")
        rng = np.random.default_rng(RANDOM_STATE)
        keep_idx = []
        for cls in np.unique(y_train_full):
            cls_idx = np.where(y_train_full == cls)[0]
            if len(cls_idx) > PER_CLASS_CAP:
                cls_idx = rng.choice(cls_idx, size=PER_CLASS_CAP, replace=False)
            keep_idx.append(cls_idx)
        keep_idx = np.concatenate(keep_idx)
        X_capped, y_capped = X_train[keep_idx], y_train_full[keep_idx]
        print(f"  after cap: {np.bincount(y_capped)}")

        # SMOTE must not see uint32 here. It computes x_i + d*(x_zi - x_i), and
        # on unsigned data the subtraction 0 - 1 wraps to 4294967295 instead of
        # -1, so synthetic rows come out as values in the billions rather than
        # 0/1. Interpolate in float32, then threshold back to binary: a bit is
        # set when the interpolation lands closer to 1 than to 0.
        sampler = SMOTE(random_state=RANDOM_STATE)
        X_bal_f, y_train_bal = sampler.fit_resample(X_capped.astype(np.float32), y_capped)
        X_train_bal = (X_bal_f >= 0.5).astype(np.uint32)
        print(f"  after SMOTE: {np.bincount(y_train_bal)}")
        assert set(np.unique(X_train_bal)).issubset({0, 1}), "binarization produced non-binary values"

        out_dir = find_dir("data") / f"temporal_{scale}"
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "X_train.npy", X_train_bal)
        np.save(out_dir / "y_train.npy", y_train_bal.astype(np.uint32))
        np.save(out_dir / "X_test.npy", X_test)
        np.save(out_dir / "y_test.npy", y_test.astype(np.uint32))
        with open(out_dir / "meta.json", "w") as f:
            json.dump({"class_names": class_names, "feature_names": feature_names,
                       "n_train": int(X_train_bal.shape[0]), "n_test": int(X_test.shape[0]),
                       "n_binary_features": int(X_train_bal.shape[1]),
                       "scale": scale, "per_class_cap": PER_CLASS_CAP}, f, indent=2)
        print(f"  saved to {out_dir}")

if __name__ == "__main__":
    main()
