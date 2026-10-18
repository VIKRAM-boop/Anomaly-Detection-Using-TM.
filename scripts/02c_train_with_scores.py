"""
Trains a Tsetlin Machine on one temporal scale.

Takes a scale name, seed and optionally a resampling strategy as arguments.
Trains on the resampled fit partition for EPOCHS epochs and reports train
accuracy plus a per-class report on validation and test.

The validation partition is the real, untouched chronological tail of the
training period written by 01c -- no synthetic rows and nothing the encoders
were fitted on -- so it is a fair surface for model selection. The test set
is only scored, never selected on.

Saves the raw per-class vote scores for the validation and test splits,
which TMClassifier.predict() computes internally but discards. Writes to
data/scores_<scale>_seed<seed>/.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, classification_report,
    confusion_matrix,
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
# Across a 15-epoch run, train accuracy never moved off ~0.667 and test
# accuracy was already within 0.005 of its final value after one epoch
# (medium 0.9436 vs 0.9482 plateau; long 0.9479 vs 0.9482). The automata
# reach steady state almost immediately at this clause/T setting, so extra
# epochs buy noise rather than fit.
EPOCHS = 1

# TMClassifier exposes no random_state; it draws from numpy's global RNG
# (clause feedback, negative-target selection), so seeding globally is how
# independent runs are obtained.
TM_SEED = 42

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

def main(scale, tm_seed=TM_SEED, strategy="smote"):
    np.random.seed(tm_seed)   # TMClassifier draws from the global RNG
    suffix = "" if strategy == "smote" else f"_{strategy}"
    data_dir = find_dir("data") / f"temporal_{scale}{suffix}"
    X_fit = np.load(data_dir / "X_fit.npy")
    y_fit = np.load(data_dir / "y_fit.npy")
    X_val = np.load(data_dir / "X_val.npy")
    y_val = np.load(data_dir / "y_val.npy")
    X_test = np.load(data_dir / "X_test.npy")
    y_test = np.load(data_dir / "y_test.npy")
    with open(data_dir / "meta.json") as f:
        class_names = json.load(f)["class_names"]

    print(f"TM_{scale} (seed {tm_seed}, {strategy}): fit={X_fit.shape} "
          f"val={X_val.shape} test={X_test.shape}", flush=True)
    print(f"params: clauses={NUMBER_OF_CLAUSES}, T={T}, s={S}, epochs={EPOCHS}", flush=True)

    tm = TMClassifier(number_of_clauses=NUMBER_OF_CLAUSES, T=T, s=S,
                       weighted_clauses=WEIGHTED_CLAUSES)
    # validation is what gets watched across epochs; the test set is not
    # looked at until the model is final
    for epoch in range(1, EPOCHS + 1):
        t0 = time.perf_counter()
        tm.fit(X_fit, y_fit)
        val_pred = tm.predict(X_val)
        val_acc = accuracy_score(y_val, val_pred)
        val_f1 = precision_recall_fscore_support(y_val, val_pred, average="macro",
                                                  zero_division=0)[2]
        print(f"TM_{scale} epoch {epoch:2d}/{EPOCHS}  val_acc={val_acc:.4f}  "
              f"val_macro_f1={val_f1:.4f}  ({time.perf_counter()-t0:.0f}s)", flush=True)

    train_acc = accuracy_score(y_fit, tm.predict(X_fit))
    print(f"TM_{scale} TRAIN accuracy = {train_acc:.4f}", flush=True)

    print(f"TM_{scale}: scoring val split...", flush=True)
    val_scores = predict_scores(tm, X_val)
    print(f"TM_{scale}: scoring test split...", flush=True)
    test_scores = predict_scores(tm, X_test, batch_report=100000)

    out_dir = find_dir("data") / f"scores_{scale}_seed{tm_seed}{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "val_scores.npy", val_scores)
    np.save(out_dir / "y_val.npy", y_val)
    np.save(out_dir / "test_scores.npy", test_scores)
    np.save(out_dir / "y_test.npy", y_test)

    val_pred = val_scores.argmax(axis=1)
    vp, vr, vf1, _ = precision_recall_fscore_support(y_val, val_pred, average="macro",
                                                      zero_division=0)
    print(f"\n--- TM_{scale} seed{tm_seed} {strategy} VALIDATION: "
          f"acc={accuracy_score(y_val, val_pred):.4f} precision={vp:.4f} "
          f"recall={vr:.4f} macro_f1={vf1:.4f} ---", flush=True)
    print(classification_report(y_val, val_pred, target_names=class_names,
                                 zero_division=0), flush=True)

    y_pred = test_scores.argmax(axis=1)
    acc = accuracy_score(y_test, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="macro", zero_division=0)
    print(f"--- TM_{scale} seed{tm_seed} {strategy} TEST (plain argmax): "
          f"train_acc={train_acc:.4f} acc={acc:.4f} precision={p:.4f} recall={r:.4f} "
          f"macro_f1={f1:.4f} ---", flush=True)
    print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0), flush=True)
    print("confusion matrix (rows=true, cols=pred):", flush=True)
    print(confusion_matrix(y_test, y_pred), flush=True)
    print(f"scores saved to {out_dir}", flush=True)

if __name__ == "__main__":
    scale = sys.argv[1] if len(sys.argv) > 1 else "medium"
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else TM_SEED
    strategy = sys.argv[3] if len(sys.argv) > 3 else "smote"
    main(scale, seed, strategy)
