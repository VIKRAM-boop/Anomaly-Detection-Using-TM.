"""
Hyperparameter sweep for paper 3's data. The clauses=1000 / T=300 / s=3.0
config we have been using was inherited from paper 2's context and never
tuned for this setup (395+ literals, chronological split, capped+SMOTE
balancing), so it may be well off.

Selection is by macro F1 on a validation slice carved out of the training
set -- NOT on the test set. Test is never touched here. Macro F1 rather
than accuracy because accuracy is saturated by Normal/Generic and cannot
distinguish configs meaningfully.
"""
import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from tmu.tsetlin_machine import TMClassifier

SCALE = "medium"
SUBSAMPLE_FIT = 40000
SUBSAMPLE_VAL = 15000
EPOCHS = 8
RANDOM_STATE = 42

CONFIGS = [
    {"clauses": 500,  "T": 100, "s": 3.0},
    {"clauses": 1000, "T": 100, "s": 3.0},
    {"clauses": 1000, "T": 300, "s": 3.0},   # current default
    {"clauses": 1000, "T": 300, "s": 5.0},
    {"clauses": 1000, "T": 600, "s": 3.0},
    {"clauses": 2000, "T": 300, "s": 3.0},
    {"clauses": 2000, "T": 600, "s": 5.0},
    {"clauses": 2000, "T": 1000, "s": 8.0},
]

def find_dir(name):
    here = Path(__file__).resolve().parent
    for candidate in (here.parent / name, here / name):
        if candidate.exists():
            return candidate
    return here / name

def main():
    data_dir = find_dir("data") / f"temporal_{SCALE}"
    X_all = np.load(data_dir / "X_train.npy")
    y_all = np.load(data_dir / "y_train.npy")

    rng = np.random.default_rng(RANDOM_STATE)
    perm = rng.permutation(len(X_all))
    val_idx = perm[:SUBSAMPLE_VAL]
    fit_idx = perm[SUBSAMPLE_VAL:SUBSAMPLE_VAL + SUBSAMPLE_FIT]
    X_fit, y_fit = X_all[fit_idx], y_all[fit_idx]
    X_val, y_val = X_all[val_idx], y_all[val_idx]

    print(f"sweep on TM_{SCALE}: fit={X_fit.shape} val={X_val.shape}, "
          f"{len(CONFIGS)} configs x {EPOCHS} epochs", flush=True)
    print("selection metric: macro F1 on VALIDATION (test never touched)\n", flush=True)

    results = []
    for i, cfg in enumerate(CONFIGS, 1):
        tm = TMClassifier(number_of_clauses=cfg["clauses"], T=cfg["T"], s=cfg["s"],
                           weighted_clauses=False)
        t0 = time.perf_counter()
        for _ in range(EPOCHS):
            tm.fit(X_fit, y_fit)
        y_pred = tm.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        p, r, f1, _ = precision_recall_fscore_support(y_val, y_pred, average="macro", zero_division=0)
        elapsed = time.perf_counter() - t0
        print(f"[{i}/{len(CONFIGS)}] clauses={cfg['clauses']:<5d} T={cfg['T']:<5d} s={cfg['s']:<4.1f}  "
              f"val_acc={acc:.4f}  val_macro_f1={f1:.4f}  ({elapsed:.0f}s)", flush=True)
        results.append({**cfg, "val_acc": acc, "val_f1": f1, "val_precision": p, "val_recall": r})

    results.sort(key=lambda r: r["val_f1"], reverse=True)
    print("\n=== SWEEP SUMMARY (ranked by validation macro F1) ===", flush=True)
    for r in results:
        print(f"clauses={r['clauses']:<5d} T={r['T']:<5d} s={r['s']:<4.1f}  "
              f"f1={r['val_f1']:.4f}  acc={r['val_acc']:.4f}", flush=True)
    best = results[0]
    print(f"\nBEST: clauses={best['clauses']}, T={best['T']}, s={best['s']} "
          f"-> val macro F1 {best['val_f1']:.4f}", flush=True)
    with open(find_dir("data") / "best_hyperparams.json", "w") as f:
        json.dump(best, f, indent=2)

if __name__ == "__main__":
    main()
