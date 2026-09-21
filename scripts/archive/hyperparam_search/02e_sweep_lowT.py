"""
Follow-up sweep. The first sweep showed macro F1 rising monotonically as T
falls (T=600: 0.612, T=300: 0.640, T=100: 0.691 at 1000 clauses) and rising
with clause count (1000->2000 at T=300: 0.640 -> 0.657). The original grid
never tested those two together, so the likely optimum -- high clauses with
low T -- went unmeasured. This probes that corner, plus T below 100 to find
where the trend turns.

Same discipline as the first sweep: selection by macro F1 on a validation
slice from the training set; test is never touched.
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
    {"clauses": 1000, "T": 50,  "s": 3.0},
    {"clauses": 2000, "T": 50,  "s": 3.0},
    {"clauses": 2000, "T": 100, "s": 3.0},
    {"clauses": 2000, "T": 100, "s": 5.0},
    {"clauses": 4000, "T": 100, "s": 3.0},
    {"clauses": 4000, "T": 200, "s": 3.0},
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

    print(f"low-T follow-up on TM_{SCALE}: fit={X_fit.shape} val={X_val.shape}\n", flush=True)

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
        print(f"[{i}/{len(CONFIGS)}] clauses={cfg['clauses']:<5d} T={cfg['T']:<4d} s={cfg['s']:<4.1f}  "
              f"val_acc={acc:.4f}  val_macro_f1={f1:.4f}  ({time.perf_counter()-t0:.0f}s)", flush=True)
        results.append({**cfg, "val_acc": acc, "val_f1": f1})

    results.sort(key=lambda r: r["val_f1"], reverse=True)
    print("\n=== LOW-T SWEEP SUMMARY (ranked by validation macro F1) ===", flush=True)
    for r in results:
        print(f"clauses={r['clauses']:<5d} T={r['T']:<4d} s={r['s']:<4.1f}  "
              f"f1={r['val_f1']:.4f}  acc={r['val_acc']:.4f}", flush=True)
    best = results[0]
    print(f"\nBEST: clauses={best['clauses']}, T={best['T']}, s={best['s']} "
          f"-> val macro F1 {best['val_f1']:.4f}", flush=True)
    with open(find_dir("data") / "best_hyperparams_lowT.json", "w") as f:
        json.dump(best, f, indent=2)

if __name__ == "__main__":
    main()
