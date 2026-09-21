"""
Trains a Tsetlin Machine on one temporal scale.

Takes a scale name and seed as arguments. Splits the balanced training set
into fit and validation portions, trains for EPOCHS epochs, and reports
train accuracy plus a per-class test report.

Saves the raw per-class vote scores for the validation and test splits,
which TMClassifier.predict() computes internally but discards. Writes to
data/scores_<scale>_seed<seed>/.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, classification_report,
)
from tmu.tsetlin_machine import TMClassifier

# Chosen by validation-only sweep (see 02d/02e/02f). The inherited paper-2
# config (1000/300/3.0) scored 0.6397 val macro F1; this scores 0.7004.
# The controlling factor turned out to be the T/clauses ratio (~0.10 peak),
# not either value alone -- consistent with class_sum being clipped to [-T, T].
NUMBER_OF_CLAUSES = 2000
T = 200
S = 5.0
WEIGHTED_CLAUSES = False
# Prior full runs plateaued by epoch 3-5 with no movement through epoch 15,
# so 6 is past the plateau with margin and keeps a multi-seed sweep affordable.
EPOCHS = 6
RANDOM_STATE = 42
VAL_FRACTION = 0.10

# TMClassifier exposes no random_state; it draws from numpy's global RNG
# (clause feedback, negative-target selection), so seeding globally is how
# independent runs are obtained.
TM_SEED = 42

def find_dir(name):
    here = Path(__file__).resolve().parent
    for candidate in (here.parent / name, here / name):
        if candidate.exists():
            return candidate
    return here / name

def predict_scores(tm, X, batch_report=None):
    """Per-class vote sums for every row -- the values predict() computes
    then discards. Mirrors TMClassifier.predict()'s inner loop exactly,
    including the clip to [-T, T]."""
    if not np.array_equal(tm.X_test, X):
        tm.encoded_X_test = tm.clause_banks[0].prepare_X(X)
        tm.X_test = X.copy()
    scores = np.zeros((X.shape[0], tm.number_of_classes), dtype=np.int32)
    for e in range(X.shape[0]):
        for i in range(tm.number_of_classes):
            class_sum = np.dot(
                tm.weight_banks[i].get_weights(),
                tm.clause_banks[i].calculate_clause_outputs_predict(tm.encoded_X_test, e),
            ).astype(np.int32)
            scores[e, i] = np.clip(class_sum, -tm.T, tm.T)
        if batch_report and e and e % batch_report == 0:
            print(f"    scored {e}/{X.shape[0]}", flush=True)
    return scores

def main(scale, tm_seed=TM_SEED):
    np.random.seed(tm_seed)   # TMClassifier draws from the global RNG
    data_dir = find_dir("data") / f"temporal_{scale}"
    X_all = np.load(data_dir / "X_train.npy")
    y_all = np.load(data_dir / "y_train.npy")
    X_test = np.load(data_dir / "X_test.npy")
    y_test = np.load(data_dir / "y_test.npy")
    with open(data_dir / "meta.json") as f:
        class_names = json.load(f)["class_names"]

    rng = np.random.default_rng(RANDOM_STATE)
    perm = rng.permutation(len(X_all))
    n_val = int(len(X_all) * VAL_FRACTION)
    val_idx, fit_idx = perm[:n_val], perm[n_val:]
    X_fit, y_fit = X_all[fit_idx], y_all[fit_idx]
    X_val, y_val = X_all[val_idx], y_all[val_idx]

    print(f"TM_{scale} (seed {tm_seed}): fit={X_fit.shape} val={X_val.shape} test={X_test.shape}", flush=True)
    print(f"params: clauses={NUMBER_OF_CLAUSES}, T={T}, s={S}, epochs={EPOCHS}", flush=True)

    tm = TMClassifier(number_of_clauses=NUMBER_OF_CLAUSES, T=T, s=S,
                       weighted_clauses=WEIGHTED_CLAUSES)
    for epoch in range(1, EPOCHS + 1):
        t0 = time.perf_counter()
        tm.fit(X_fit, y_fit)
        test_acc = accuracy_score(y_test, tm.predict(X_test))
        print(f"TM_{scale} epoch {epoch:2d}/{EPOCHS}  test_acc={test_acc:.4f}  "
              f"({time.perf_counter()-t0:.0f}s)", flush=True)

    train_acc = accuracy_score(y_fit, tm.predict(X_fit))
    print(f"TM_{scale} TRAIN accuracy = {train_acc:.4f}", flush=True)

    print(f"TM_{scale}: scoring val split...", flush=True)
    val_scores = predict_scores(tm, X_val)
    print(f"TM_{scale}: scoring test split...", flush=True)
    test_scores = predict_scores(tm, X_test, batch_report=100000)

    out_dir = find_dir("data") / f"scores_{scale}_seed{tm_seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "val_scores.npy", val_scores)
    np.save(out_dir / "y_val.npy", y_val)
    np.save(out_dir / "test_scores.npy", test_scores)
    np.save(out_dir / "y_test.npy", y_test)

    y_pred = test_scores.argmax(axis=1)
    acc = accuracy_score(y_test, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="macro", zero_division=0)
    print(f"\n--- TM_{scale} seed{tm_seed} (plain argmax): train_acc={train_acc:.4f} "
          f"acc={acc:.4f} precision={p:.4f} recall={r:.4f} f1={f1:.4f} ---", flush=True)
    print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0), flush=True)
    print(f"scores saved to {out_dir}", flush=True)

if __name__ == "__main__":
    scale = sys.argv[1] if len(sys.argv) > 1 else "medium"
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else TM_SEED
    main(scale, seed)
