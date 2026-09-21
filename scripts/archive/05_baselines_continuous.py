"""
Fair-fight baselines: RF and DT on CONTINUOUS features rather than the
binarized literals the TM requires.

Why this matters: the TM can only consume binary input, so our pipeline
quantile-bins every feature into 5 threshold literals. Trees do not need
that and are actively hurt by it -- they normally choose their own split
points at whatever resolution the data supports. Scoring RF/DT on the
binarized matrix therefore understates them, and any "TM is competitive"
claim drawn from it would be unfair.

Everything else is held identical to the TM run: same chronological
split, same rows (the same per-class cap, same seed), same test set. The
only difference is the feature representation each model is given, which
is the thing being tested. Trees get class_weight='balanced' instead of
SMOTE, that being the natural tree-side equivalent of the TM's balancing.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    precision_recall_fscore_support, accuracy_score, classification_report,
)

RANDOM_STATE = 42
PER_CLASS_CAP = 40000          # identical to 01c_preprocess_temporal.py
DROP_COLS = ["srcip", "sport", "dstip", "dsport", "stime", "ltime", "ts", "label"]
CATEGORICAL_COLS = ["proto", "service", "state"]
LABEL_FIXES = {"Backdoors": "Backdoor"}

def find_dir(name):
    here = Path(__file__).resolve().parent
    for candidate in (here.parent / name, here / name):
        if candidate.exists():
            return candidate
    return here / name

def main():
    # read from a local copy: reading these full frames (71 cols x 2M rows) out of
    # the Dropbox-synced project folder times out under sustained I/O
    d = Path("/tmp/tm_work")
    if not (d / "train_windowed.parquet").exists():
        d = find_dir("data") / "temporal"
    train_df = pd.read_parquet(d / "train_windowed.parquet")
    test_df = pd.read_parquet(d / "test_windowed.parquet")
    for df in (train_df, test_df):
        df["attack_cat"] = df["attack_cat"].replace(LABEL_FIXES)

    le = LabelEncoder().fit(train_df["attack_cat"])
    y_train_full = le.transform(train_df["attack_cat"])
    y_test = le.transform(test_df["attack_cat"])
    class_names = list(le.classes_)

    feat_cols = [c for c in train_df.columns if c not in DROP_COLS + ["attack_cat"]]
    Xtr = train_df[feat_cols].copy()
    Xte = test_df[feat_cols].copy()
    for c in CATEGORICAL_COLS:
        cats = pd.Categorical(Xtr[c])
        Xtr[c] = cats.codes
        Xte[c] = pd.Categorical(Xte[c], categories=cats.categories).codes
    Xtr = Xtr.to_numpy(dtype=np.float32)
    Xte = Xte.to_numpy(dtype=np.float32)
    Xtr = np.nan_to_num(Xtr, nan=0.0, posinf=0.0, neginf=0.0)
    Xte = np.nan_to_num(Xte, nan=0.0, posinf=0.0, neginf=0.0)
    print(f"continuous features: {Xtr.shape[1]} (vs 425 binary literals for the TM)")

    # same per-class cap and seed as the TM's training set
    rng = np.random.default_rng(RANDOM_STATE)
    keep = []
    for cls in np.unique(y_train_full):
        idx = np.where(y_train_full == cls)[0]
        if len(idx) > PER_CLASS_CAP:
            idx = rng.choice(idx, PER_CLASS_CAP, replace=False)
        keep.append(idx)
    keep = np.concatenate(keep)
    Xc, yc = Xtr[keep], y_train_full[keep]
    print(f"train rows: {len(yc)}  test rows: {len(y_test)}\n")

    for name, model in [
        ("Random Forest", RandomForestClassifier(n_estimators=100, class_weight="balanced",
                                                  n_jobs=-1, random_state=RANDOM_STATE)),
        ("Decision Tree", DecisionTreeClassifier(class_weight="balanced",
                                                  random_state=RANDOM_STATE)),
    ]:
        model.fit(Xc, yc)
        p = model.predict(Xte)
        print(f"=== {name} on CONTINUOUS features ===")
        print(f"  accuracy = {accuracy_score(y_test, p):.4f}")
        for avg in ("macro", "weighted"):
            pr, rc, f1, _ = precision_recall_fscore_support(y_test, p, average=avg, zero_division=0)
            print(f"  {avg:>8s}: precision={pr:.4f} recall={rc:.4f} f1={f1:.4f}")
        print(classification_report(y_test, p, target_names=class_names, zero_division=0))

if __name__ == "__main__":
    main()
