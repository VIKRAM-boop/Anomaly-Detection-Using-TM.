"""
Tunes and evaluates Random Forest and Decision Tree baselines.

Runs on the continuous features. Splits real training rows into fit and
validation before applying SMOTE, grid-searches each model by macro F1 on
the validation rows, then retrains the best configuration over three seeds
and scores the test set.

Prints train accuracy, test accuracy, precision, recall and F1 in both
weighted and macro form, with standard deviations across seeds. Writes
data/tuned_baselines.json.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from imblearn.over_sampling import SMOTE

RANDOM_STATE = 42
PER_CLASS_CAP = 40000
VAL_FRACTION = 0.10
SEARCH_SUBSAMPLE = 100000     # search on a subsample; winner retrains on everything
SEEDS = [42, 7, 1234]
DROP = ["srcip", "sport", "dstip", "dsport", "stime", "ltime", "ts", "label"]
CAT = ["proto", "service", "state"]
PARQUET = Path("/tmp/tmwork2")

RF_GRID = [
    {"n_estimators": n, "max_depth": d, "min_samples_leaf": l, "max_features": f}
    for n in (100, 300) for d in (None, 30) for l in (1, 5, 20) for f in ("sqrt", 0.3)
]
DT_GRID = [
    {"max_depth": d, "min_samples_leaf": l, "criterion": c}
    for d in (None, 10, 20, 30, 50) for l in (1, 5, 20, 50) for c in ("gini", "entropy")
]

def find_dir(name):
    here = Path(__file__).resolve().parent
    for candidate in (here.parent / name, here / name):
        if candidate.exists():
            return candidate
    return here / name

def macro_f1(y, p):
    return precision_recall_fscore_support(y, p, average="macro", zero_division=0)[2]

def load_continuous():
    tr = pd.read_parquet(PARQUET / "train_windowed.parquet")
    te = pd.read_parquet(PARQUET / "test_windowed.parquet")
    for f in (tr, te):
        f["attack_cat"] = f["attack_cat"].replace({"Backdoors": "Backdoor"})
    le = LabelEncoder().fit(tr["attack_cat"])
    y_full, y_test = le.transform(tr["attack_cat"]), le.transform(te["attack_cat"])
    cols = [c for c in tr.columns if c not in DROP + ["attack_cat"]]
    Xtr, Xte = tr[cols].copy(), te[cols].copy()
    for c in CAT:
        cats = pd.Categorical(Xtr[c])
        Xtr[c] = cats.codes
        Xte[c] = pd.Categorical(Xte[c], categories=cats.categories).codes
    f32 = lambda d: np.nan_to_num(d.to_numpy(np.float32), nan=0, posinf=0, neginf=0)
    return f32(Xtr), y_full, f32(Xte), y_test

def cap(X, y, rng):
    keep = []
    for c in np.unique(y):
        i = np.where(y == c)[0]
        keep.append(rng.choice(i, PER_CLASS_CAP, replace=False) if len(i) > PER_CLASS_CAP else i)
    keep = np.concatenate(keep)
    return X[keep], y[keep]

def run(tag, X_train_all, y_train_all, X_test, y_test, already_capped):
    print(f"\n{'='*70}\n{tag}\n{'='*70}", flush=True)
    rng = np.random.default_rng(RANDOM_STATE)
    Xc, yc = (X_train_all, y_train_all) if already_capped else cap(X_train_all, y_train_all, rng)

    # split real rows into fit/val BEFORE any synthesis, so val stays real
    perm = rng.permutation(len(Xc))
    n_val = int(len(Xc) * VAL_FRACTION)
    val_i, fit_i = perm[:n_val], perm[n_val:]
    X_val, y_val = Xc[val_i], yc[val_i]
    X_fit, y_fit = Xc[fit_i], yc[fit_i]
    Xb, yb = SMOTE(random_state=RANDOM_STATE).fit_resample(X_fit, y_fit)
    print(f"fit(real)={X_fit.shape} -> SMOTE {Xb.shape} | val(real)={X_val.shape} | test={X_test.shape}", flush=True)

    sub = rng.choice(len(Xb), min(SEARCH_SUBSAMPLE, len(Xb)), replace=False)
    Xs, ys = Xb[sub], yb[sub]

    results = {}
    for name, grid, make in [
        ("Random Forest", RF_GRID, lambda p, s: RandomForestClassifier(random_state=s, n_jobs=-1, **p)),
        ("Decision Tree", DT_GRID, lambda p, s: DecisionTreeClassifier(random_state=s, **p)),
    ]:
        print(f"\n-- tuning {name}: {len(grid)} configs on {len(Xs)} rows --", flush=True)
        best, best_f1 = None, -1.0
        for i, params in enumerate(grid, 1):
            t0 = time.perf_counter()
            f1 = macro_f1(y_val, make(params, RANDOM_STATE).fit(Xs, ys).predict(X_val))
            if f1 > best_f1:
                best, best_f1 = params, f1
                print(f"   [{i}/{len(grid)}] val_macro_f1={f1:.4f}  <- best  {params}  ({time.perf_counter()-t0:.0f}s)", flush=True)
        print(f"   BEST {name}: {best}  val_macro_f1={best_f1:.4f}", flush=True)

        runs = []
        for s in SEEDS:
            m = make(best, s).fit(Xb, yb)
            p = m.predict(X_test)
            wp, wr, wf, _ = precision_recall_fscore_support(y_test, p, average="weighted", zero_division=0)
            runs.append({"train_acc": accuracy_score(yb, m.predict(Xb)),
                          "test_acc": accuracy_score(y_test, p),
                          "w_p": wp, "w_r": wr, "w_f1": wf, "m_f1": macro_f1(y_test, p)})
            print(f"   seed {s}: test_acc={runs[-1]['test_acc']:.4f} "
                  f"weighted_f1={wf:.4f} macro_f1={runs[-1]['m_f1']:.4f}", flush=True)
        agg = {k: (float(np.mean([r[k] for r in runs])), float(np.std([r[k] for r in runs])))
               for k in runs[0]}
        results[name] = {"params": best, "val_macro_f1": best_f1, "agg": agg}
        print(f"   {name} over {len(SEEDS)} seeds: "
              f"train={agg['train_acc'][0]:.4f} test={agg['test_acc'][0]:.4f}+/-{agg['test_acc'][1]:.4f} "
              f"P={agg['w_p'][0]:.4f} R={agg['w_r'][0]:.4f} "
              f"weighted_F1={agg['w_f1'][0]:.4f}+/-{agg['w_f1'][1]:.4f} "
              f"macro_F1={agg['m_f1'][0]:.4f}+/-{agg['m_f1'][1]:.4f}", flush=True)
    return results

def main():
    out = {}
    Xtr, ytr, Xte, yte = load_continuous()
    out["continuous"] = run("CONTINUOUS FEATURES", Xtr, ytr, Xte, yte, already_capped=False)
    with open(find_dir("data") / "tuned_baselines.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nsaved -> data/tuned_baselines.json", flush=True)

if __name__ == "__main__":
    main()
