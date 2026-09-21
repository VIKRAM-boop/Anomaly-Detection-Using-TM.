"""
Step 4: trains three separate TMClassifiers, one per temporal scale
(short ~1s, medium ~10s, long ~60s), each on its own literal set from
01c_preprocess_temporal.py (shared base per-flow literals + that scale's
own window-derived literals). Per spec: "short windows target bursts and
abrupt attacks; medium windows target repeated attempts and scans; long
windows target slow scans, low-rate attacks, persistent bot behavior."

This trains and reports each branch independently -- combining their
outputs into a single decision (spec's Step 5 hierarchy) is a separate
step once each branch's standalone behavior is validated.
"""
import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, classification_report,
)
from tmu.tsetlin_machine import TMClassifier

NUMBER_OF_CLAUSES = 1000
T = 300
S = 3.0
WEIGHTED_CLAUSES = False
EPOCHS = 15
RANDOM_STATE = 42
SCALES = ["medium", "long"]  # short already completed successfully in the prior run
SUBSAMPLE_FOR_TESTING = None  # set an int to subsample in-memory for a quick smoke test

def find_dir(name):
    here = Path(__file__).resolve().parent
    for candidate in (here.parent / name, here / name):
        if candidate.exists():
            return candidate
    return here / name

def train_one_scale(scale):
    data_dir = find_dir("data") / f"temporal_{scale}"
    X_train = np.load(data_dir / "X_train.npy")
    y_train = np.load(data_dir / "y_train.npy")
    X_test = np.load(data_dir / "X_test.npy")
    y_test = np.load(data_dir / "y_test.npy")
    with open(data_dir / "meta.json") as f:
        meta = json.load(f)
    class_names = meta["class_names"]

    if SUBSAMPLE_FOR_TESTING is not None:
        rng = np.random.default_rng(RANDOM_STATE)
        idx = rng.choice(len(X_train), size=min(SUBSAMPLE_FOR_TESTING, len(X_train)), replace=False)
        X_train, y_train = X_train[idx], y_train[idx]
        test_size = min(SUBSAMPLE_FOR_TESTING // 3, len(X_test))
        idx = rng.choice(len(X_test), size=test_size, replace=False)
        X_test, y_test = X_test[idx], y_test[idx]

    print(f"\n{'='*60}\nTM_{scale}  (train={X_train.shape}, test={X_test.shape})\n{'='*60}")

    tm = TMClassifier(number_of_clauses=NUMBER_OF_CLAUSES, T=T, s=S,
                       weighted_clauses=WEIGHTED_CLAUSES)

    for epoch in range(1, EPOCHS + 1):
        tm.fit(X_train, y_train)
        train_acc = accuracy_score(y_train, tm.predict(X_train))
        test_acc = accuracy_score(y_test, tm.predict(X_test))
        print(f"TM_{scale} epoch {epoch:2d}/{EPOCHS}  train_acc={train_acc:.4f}  test_acc={test_acc:.4f}")

    y_pred = tm.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0)

    print(f"\n--- TM_{scale} FINAL: acc={acc:.4f} precision={precision:.4f} "
          f"recall={recall:.4f} f1={f1:.4f} ---")
    print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))

    return {"scale": scale, "accuracy": acc, "precision": precision,
            "recall": recall, "f1": f1, "tm": tm, "class_names": class_names}

def main():
    results = []
    for scale in SCALES:
        t0 = time.perf_counter()
        r = train_one_scale(scale)
        r["elapsed"] = time.perf_counter() - t0
        results.append(r)

    print(f"\n{'='*60}\nSUMMARY -- all three temporal branches\n{'='*60}")
    for r in results:
        print(f"TM_{r['scale']:<7s} acc={r['accuracy']:.4f}  precision={r['precision']:.4f}  "
              f"recall={r['recall']:.4f}  f1={r['f1']:.4f}  ({r['elapsed']:.0f}s)")

if __name__ == "__main__":
    main()
