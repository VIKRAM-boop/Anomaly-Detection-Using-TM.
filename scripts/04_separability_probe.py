"""
Tests whether attack families are separable in the literal features.

Trains a Random Forest twice on the same binary literals the TM uses:
once over all 10 classes, and once on attack rows only with Normal
excluded. Prints macro metrics and a per-class report for each.

The first run doubles as a non-TM baseline; the second shows what a model
achieves when its whole capacity goes to distinguishing attack families.
"""
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support, classification_report

SCALE = "medium"
N_REAL_ROWS = 155144        # real rows precede synthetic ones in the balanced pool
NORMAL_CLASS = 6
RANDOM_STATE = 42
TRAIN_SUBSAMPLE = 120000    # keep RF training tractable

def find_dir(name):
    here = Path(__file__).resolve().parent
    for candidate in (here.parent / name, here / name):
        if candidate.exists():
            return candidate
    return here / name

def macro(y, p):
    pr, rc, f1, _ = precision_recall_fscore_support(y, p, average="macro", zero_division=0)
    return pr, rc, f1

def main():
    data_dir = find_dir("data") / f"temporal_{SCALE}"
    X = np.load(data_dir / "X_train.npy")[:N_REAL_ROWS]   # real rows only, no synthetic
    y = np.load(data_dir / "y_train.npy")[:N_REAL_ROWS]
    X_test = np.load(data_dir / "X_test.npy")
    y_test = np.load(data_dir / "y_test.npy")
    class_names = json.load(open(data_dir / "meta.json"))["class_names"]

    rng = np.random.default_rng(RANDOM_STATE)

    print("=" * 62)
    print("PROBE 1: full 10-class Random Forest (non-TM baseline)")
    print("=" * 62)
    idx = rng.choice(len(X), min(TRAIN_SUBSAMPLE, len(X)), replace=False)
    rf = RandomForestClassifier(n_estimators=100, class_weight="balanced",
                                 n_jobs=-1, random_state=RANDOM_STATE)
    rf.fit(X[idx], y[idx])
    pred = rf.predict(X_test)
    pr, rc, f1 = macro(y_test, pred)
    print(f"RF 10-class on our test set: precision={pr:.4f} recall={rc:.4f} macro_f1={f1:.4f}")
    print(classification_report(y_test, pred, target_names=class_names, zero_division=0))

    print("=" * 62)
    print("PROBE 2: attack-family separability (Normal excluded entirely)")
    print("=" * 62)
    atk = y != NORMAL_CLASS
    atk_test = y_test != NORMAL_CLASS
    Xa, ya = X[atk], y[atk]
    idx = rng.choice(len(Xa), min(TRAIN_SUBSAMPLE, len(Xa)), replace=False)
    rf2 = RandomForestClassifier(n_estimators=100, class_weight="balanced",
                                  n_jobs=-1, random_state=RANDOM_STATE)
    rf2.fit(Xa[idx], ya[idx])
    pred2 = rf2.predict(X_test[atk_test])
    yt2 = y_test[atk_test]
    pr2, rc2, f12 = macro(yt2, pred2)
    print(f"RF attack-only ({atk.sum()} train / {atk_test.sum()} test rows):")
    print(f"  precision={pr2:.4f} recall={rc2:.4f} macro_f1={f12:.4f}")
    present = sorted(set(yt2) | set(pred2))
    print(classification_report(yt2, pred2, labels=present,
                                 target_names=[class_names[i] for i in present],
                                 zero_division=0))
    print("READ: if the hard classes (Analysis/Backdoor/DoS/Worms) score well here,")
    print("the boundary exists and a two-stage TM can chase it. If RF also fails,")
    print("the overlap is in the data and no architecture recovers it.")

if __name__ == "__main__":
    main()
