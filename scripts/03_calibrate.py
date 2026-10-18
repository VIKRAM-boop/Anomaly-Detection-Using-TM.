"""
Applies prior calibration to saved Tsetlin Machine scores.

Sweeps a temperature over per-class log priors and selects the value giving
the best macro F1 on the validation partition. Applies the chosen
temperature to the test scores once and prints accuracy, precision, recall,
macro F1, a per-class report and a confusion matrix, alongside the
uncalibrated baseline.

The validation partition written by 01c is real and untouched, and the
priors come from the real pre-resampling class counts of the fit partition,
so neither carries test information.

Takes the scale, seed and resampling strategy as argv[1..3].
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, classification_report,
    confusion_matrix,
)

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

def macro_f1(y_true, y_pred):
    _, _, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return f1

def main(scale="medium", seed=42, strategy="smote"):
    suffix = "" if strategy == "smote" else f"_{strategy}"
    score_dir = find_dir("data") / f"scores_{scale}_seed{seed}{suffix}"
    val_scores = np.load(score_dir / "val_scores.npy")
    y_val = np.load(score_dir / "y_val.npy")
    test_scores = np.load(score_dir / "test_scores.npy")
    y_test = np.load(score_dir / "y_test.npy")
    with open(find_dir("data") / f"temporal_{scale}{suffix}" / "meta.json") as f:
        meta = json.load(f)
    class_names = meta["class_names"]

    # every validation row is real and untouched now, so no reconstruction
    rv_scores, rv_y = val_scores, y_val
    print(f"validation rows: {len(y_val)} (all real, chronological tail of train)")
    print(f"val class counts: {np.bincount(rv_y, minlength=len(class_names))}\n")

    # prior from the fit partition's real, pre-resampling counts
    counts = np.array(meta["fit_class_counts_real"], dtype=np.float64)
    counts = np.maximum(counts, 1.0)        # a class absent from fit gets a floor
    log_prior = np.log(counts / counts.sum())

    print("=== selecting on validation rows (test untouched) ===")
    results = {}
    results["raw"] = (macro_f1(rv_y, rv_scores.argmax(axis=1)), None)
    print(f"  raw argmax                  val_macro_f1={results['raw'][0]:.4f}")

    for tau in [0.5, 1, 2, 5, 10, 20, 30, 50, 80, 120]:
        pred = (rv_scores + tau * log_prior).argmax(axis=1)
        f1 = macro_f1(rv_y, pred)
        results[f"prior_tau{tau}"] = (f1, tau)
        print(f"  prior correction tau={tau:<5g}  val_macro_f1={f1:.4f}")

    best_name = max(results, key=lambda k: results[k][0])
    best_f1, best_tau = results[best_name]
    print(f"\nchosen: {best_name} (real-val macro F1 {best_f1:.4f})")

    print("\n=== applying to TEST, once ===")
    base_pred = test_scores.argmax(axis=1)
    b_acc = accuracy_score(y_test, base_pred)
    b_p, b_r, b_f1, _ = precision_recall_fscore_support(y_test, base_pred, average="macro", zero_division=0)

    cal_pred = base_pred if best_tau is None else (test_scores + best_tau * log_prior).argmax(axis=1)
    c_acc = accuracy_score(y_test, cal_pred)
    c_p, c_r, c_f1, _ = precision_recall_fscore_support(y_test, cal_pred, average="macro", zero_division=0)

    print(f"  raw argmax  : acc={b_acc:.4f} precision={b_p:.4f} recall={b_r:.4f} f1={b_f1:.4f}")
    print(f"  calibrated  : acc={c_acc:.4f} precision={c_p:.4f} recall={c_r:.4f} f1={c_f1:.4f}")
    print(f"  delta       : acc={c_acc-b_acc:+.4f}  f1={c_f1-b_f1:+.4f}")

    print("\n=== PER-CLASS, calibrated ===")
    print(classification_report(y_test, cal_pred, target_names=class_names, zero_division=0))
    print("confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_test, cal_pred))

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "medium",
         int(sys.argv[2]) if len(sys.argv) > 2 else 42,
         sys.argv[3] if len(sys.argv) > 3 else "smote")
