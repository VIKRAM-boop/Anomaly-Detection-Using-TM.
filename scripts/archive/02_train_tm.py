"""
Stage 1 baseline TM training: flat 10-class TMClassifier on the
literal-based binary representation from 01_preprocess.py (quantile
threshold literals + one-hot categoricals + relational literals). No
temporal windowing (Step 4) or benign/known/unknown hierarchy (Step 5)
yet -- this validates the flat literal representation first.

The spec doesn't prescribe TM hyperparameters (m, T, s), so these are
starting values to sweep from, not claimed-correct settings.
"""
import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix,
)
from tmu.tsetlin_machine import TMClassifier

NUMBER_OF_CLAUSES = 1000
T = 300
S = 3.0
WEIGHTED_CLAUSES = False
EPOCHS = 15
RANDOM_STATE = 42

SUBSAMPLE_FOR_TESTING = None

def find_dir(name):
    here = Path(__file__).resolve().parent
    for candidate in (here.parent / name, here / name):
        if candidate.exists():
            return candidate
    return here / name

DATA_DIR = find_dir("data") / "stage1"

def main():
    X_train = np.load(DATA_DIR / "X_train.npy")
    y_train = np.load(DATA_DIR / "y_train.npy")
    X_test = np.load(DATA_DIR / "X_test.npy")
    y_test = np.load(DATA_DIR / "y_test.npy")
    with open(DATA_DIR / "meta.json") as f:
        meta = json.load(f)
    class_names = meta["class_names"]

    if SUBSAMPLE_FOR_TESTING is not None:
        rng = np.random.default_rng(RANDOM_STATE)
        if SUBSAMPLE_FOR_TESTING < len(X_train):
            idx = rng.choice(len(X_train), size=SUBSAMPLE_FOR_TESTING, replace=False)
            X_train, y_train = X_train[idx], y_train[idx]
        test_size = SUBSAMPLE_FOR_TESTING // 3
        if test_size < len(X_test):
            idx = rng.choice(len(X_test), size=test_size, replace=False)
            X_test, y_test = X_test[idx], y_test[idx]
        print(f"(subsampled train/test for quick testing: "
              f"train={len(X_train)}, test={len(X_test)})")

    print(f"train: {X_train.shape}, test: {X_test.shape}")
    print(f"params: clauses={NUMBER_OF_CLAUSES}, T={T}, s={S}, "
          f"weighted_clauses={WEIGHTED_CLAUSES}, epochs={EPOCHS}\n")

    tm = TMClassifier(
        number_of_clauses=NUMBER_OF_CLAUSES,
        T=T,
        s=S,
        weighted_clauses=WEIGHTED_CLAUSES,
    )

    for epoch in range(1, EPOCHS + 1):
        tm.fit(X_train, y_train)
        train_acc = accuracy_score(y_train, tm.predict(X_train))
        test_acc = accuracy_score(y_test, tm.predict(X_test))
        print(f"epoch {epoch:2d}/{EPOCHS}  train_acc={train_acc:.4f}  test_acc={test_acc:.4f}")

    t0 = time.perf_counter()
    y_pred = tm.predict(X_test)
    elapsed = time.perf_counter() - t0
    per_sample_us = (elapsed / len(X_test)) * 1e6

    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )

    print("\n=== FINAL TEST METRICS (held-out test set) ===")
    print(f"accuracy:  {acc:.4f}")
    print(f"precision: {precision:.4f} (macro)")
    print(f"recall:    {recall:.4f} (macro)")
    print(f"f1-score:  {f1:.4f} (macro)")
    print(f"inference time: {per_sample_us:.3f} microseconds/sample")

    print("\n=== PER-CLASS REPORT ===")
    print(classification_report(
        y_test, y_pred, target_names=class_names, zero_division=0
    ))

    print("=== CONFUSION MATRIX (rows=true, cols=predicted) ===")
    cm = confusion_matrix(y_test, y_pred)
    header = "        " + " ".join(f"{n[:8]:>8s}" for n in class_names)
    print(header)
    for name, row in zip(class_names, cm):
        print(f"{name[:8]:>8s} " + " ".join(f"{v:8d}" for v in row))

if __name__ == "__main__":
    main()
