"""
Applies prior calibration to saved Tsetlin Machine scores.

Reconstructs which validation rows are real rather than SMOTE-synthetic,
then sweeps a temperature over per-class log priors and selects the value
giving the best macro F1 on those rows. Applies the chosen temperature to
the test scores once and prints accuracy, precision, recall, macro F1 and
a per-class report, alongside the uncalibrated baseline.
"""
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, classification_report,
)

SCALE = "medium"
SEED = 42                 # which 02c_train_with_scores.py run to calibrate
N_REAL_ROWS = 155144      # verified boundary between real and synthetic rows
TRAIN_SPLIT_SEED = 42     # must match 02c_train_with_scores.py
VAL_FRACTION = 0.10
BALANCED_POOL = 400000

# real (pre-cap, pre-SMOTE) training class counts -- training-set information,
# not test information, so legitimate to use for prior correction
REAL_TRAIN_COUNTS = np.array([1999, 1627, 11192, 31675, 17535,
                               147021, 1809834, 9923, 1071, 122], dtype=np.float64)

def find_dir(name):
    here = Path(__file__).resolve().parent
    for candidate in (here.parent / name, here / name):
        if candidate.exists():
            return candidate
    return here / name

def macro_f1(y_true, y_pred):
    _, _, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return f1

def main():
    score_dir = find_dir("data") / f"scores_{SCALE}_seed{SEED}"
    val_scores = np.load(score_dir / "val_scores.npy")
    y_val = np.load(score_dir / "y_val.npy")
    test_scores = np.load(score_dir / "test_scores.npy")
    y_test = np.load(score_dir / "y_test.npy")
    with open(find_dir("data") / f"temporal_{SCALE}" / "meta.json") as f:
        class_names = json.load(f)["class_names"]

    # reconstruct which validation rows were real rather than SMOTE-synthesised
    rng = np.random.default_rng(TRAIN_SPLIT_SEED)
    perm = rng.permutation(BALANCED_POOL)
    val_idx = perm[:int(BALANCED_POOL * VAL_FRACTION)]
    is_real = val_idx < N_REAL_ROWS
    rv_scores, rv_y = val_scores[is_real], y_val[is_real]
    print(f"validation rows: {len(val_idx)} total, {is_real.sum()} real "
          f"({is_real.sum()/len(val_idx):.1%})")
    print(f"real-val class counts: {np.bincount(rv_y, minlength=10)}\n")

    log_prior = np.log(REAL_TRAIN_COUNTS / REAL_TRAIN_COUNTS.sum())

    print("=== selecting on REAL validation rows (test untouched) ===")
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

if __name__ == "__main__":
    main()
