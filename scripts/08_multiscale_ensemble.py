"""
Combines the short, medium and long Tsetlin Machine branches.

Loads the saved per-class scores for all three scales and compares three
combination rules -- summing scores, averaging z-normalised scores, and
taking the per-class maximum -- crossed with a calibration temperature.
Selection uses macro F1 on the validation partition, which 01c writes real
and untouched. Priors come from the fit partition's real pre-resampling
class counts.

Applies the winning combination to the test set once and prints its
metrics next to the single-branch result, plus a per-class report and a
confusion matrix. Takes the resampling strategy as argv[1].
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

SEED = 42
SCALES = ["short", "medium", "long"]
TAU_GRID = [0, 5, 10, 20, 30, 50, 80]

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

def macro_f1(y, p):
    return precision_recall_fscore_support(y, p, average="macro", zero_division=0)[2]

def zscore(a):
    mu, sd = a.mean(axis=0), a.std(axis=0)
    sd[sd == 0] = 1.0
    return (a - mu) / sd

COMBINERS = {
    "sum": lambda mats: np.sum(mats, axis=0),
    "mean_z": lambda mats: np.mean([zscore(m.astype(np.float64)) for m in mats], axis=0),
    "max": lambda mats: np.max(mats, axis=0),
}

def main(strategy="smote"):
    suffix = "" if strategy == "smote" else f"_{strategy}"
    data = find_dir("data")
    val, test = {}, {}
    for s in SCALES:
        d = data / f"scores_{s}_seed{SEED}{suffix}"
        if not d.exists():
            raise SystemExit(f"missing {d} -- run 02c_train_with_scores.py for '{s}' first")
        val[s] = np.load(d / "val_scores.npy")
        test[s] = np.load(d / "test_scores.npy")
    y_val = np.load(data / f"scores_medium_seed{SEED}{suffix}" / "y_val.npy")
    y_test = np.load(data / f"scores_medium_seed{SEED}{suffix}" / "y_test.npy")
    meta = json.load(open(data / f"temporal_medium{suffix}" / "meta.json"))
    class_names = meta["class_names"]

    # validation is real and untouched now, so every row counts
    y_val_real = y_val
    counts = np.maximum(np.array(meta["fit_class_counts_real"], dtype=np.float64), 1.0)
    log_prior = np.log(counts / counts.sum())
    print(f"validation: {len(y_val)} real rows (chronological tail of train)\n")

    print("=== single branches (validation rows) ===")
    for s in SCALES:
        print(f"  {s:<7s} val_macro_f1={macro_f1(y_val_real, val[s].argmax(1)):.4f}")

    print("\n=== combination rules x calibration (validation only) ===")
    best = (None, None, -1.0)
    for cname, fn in COMBINERS.items():
        combined = fn([val[s] for s in SCALES])
        for tau in TAU_GRID:
            f1 = macro_f1(y_val_real, (combined + tau * log_prior).argmax(1))
            if f1 > best[2]:
                best = (cname, tau, f1)
                print(f"  {cname:<7s} tau={tau:<3d} val_macro_f1={f1:.4f}  <- best")
    cname, tau, _ = best
    print(f"\nchosen: combiner={cname}, tau={tau}")

    print("\n=== applied to TEST, once ===")
    single = (test["medium"] + tau * log_prior).argmax(1)
    ens = (COMBINERS[cname]([test[s] for s in SCALES]) + tau * log_prior).argmax(1)
    for tag, pred in (("medium branch alone", single), (f"multi-scale ensemble ({cname})", ens)):
        acc = accuracy_score(y_test, pred)
        wp, wr, wf, _ = precision_recall_fscore_support(y_test, pred, average="weighted", zero_division=0)
        mp, mr, mf, _ = precision_recall_fscore_support(y_test, pred, average="macro", zero_division=0)
        print(f"  {tag:<32s} acc={acc:.4f} weighted_f1={wf:.4f} macro_f1={mf:.4f}")
        print(f"  {'':<32s} weighted P={wp:.4f} R={wr:.4f} | macro P={mp:.4f} R={mr:.4f}")

    print(f"\n=== PER-CLASS, multi-scale ensemble ===")
    print(classification_report(y_test, ens, target_names=class_names, zero_division=0))

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "smote")
