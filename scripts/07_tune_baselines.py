"""
Tunes and evaluates Random Forest and Decision Tree baselines.

Runs on the continuous features. The default strategy is "none": trees get
every real training row with no synthesis and no downsampling, and handle
imbalance through class_weight instead. The TM's resampling is part of the
TM pipeline, not something a standard baseline should inherit -- capping
Normal at 40k would discard 97.6% of the real majority rows for no reason
that applies to trees. Splits real training rows into fit and
validation before applying SMOTE, grid-searches each model by macro F1 on
the validation rows, then retrains the best configuration over three seeds
and scores the test set.

Prints train accuracy, test accuracy, precision, recall and F1 in both
weighted and macro form, with standard deviations across seeds. Writes
data/tuned_baselines.json.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (precision_recall_fscore_support, accuracy_score,
                             classification_report, confusion_matrix)
from imblearn.over_sampling import SMOTE

RANDOM_STATE = 42
PER_CLASS_CAP = 40000
MODERATE_MULTIPLE = 5        # moderate strategy: at most 5x a class's real count
UNDERSAMPLE_CAP = 10000      # undersample strategy: majority ceiling, no synthesis
VAL_FRACTION = 0.10
SEARCH_SUBSAMPLE = 100000     # search on a subsample; winner retrains on everything
SEEDS = [42, 7, 1234]
DROP = ["srcip", "sport", "dstip", "dsport", "stime", "ltime", "ts", "label"]
CAT = ["proto", "service", "state"]
PARQUET = Path("/tmp/tmwork2")

# class_weight is how trees are normally made imbalance-aware, so it belongs
# in the search rather than being substituted for by resampling
RF_GRID = [
    {"n_estimators": n, "max_depth": d, "min_samples_leaf": l, "max_features": f,
     "class_weight": w}
    for n in (100, 300) for d in (None, 30) for l in (1, 5, 20) for f in ("sqrt", 0.3)
    for w in (None, "balanced")
]
DT_GRID = [
    {"max_depth": d, "min_samples_leaf": l, "criterion": c, "class_weight": w}
    for d in (None, 10, 20, 30, 50) for l in (1, 5, 20, 50) for c in ("gini", "entropy")
    for w in (None, "balanced")
]

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

if not (PARQUET / "train_windowed.parquet").exists():
    PARQUET = find_dir("data") / "temporal"

def macro_f1(y, p):
    return precision_recall_fscore_support(y, p, average="macro", zero_division=0)[2]

def load_continuous():
    """Chronological fit/val/test partitions. Validation is the tail of the
    training period, so it sits after everything fitted on and before
    everything tested on. Category codes come from the fit rows only."""
    tr = pd.read_parquet(PARQUET / "train_windowed.parquet")
    te = pd.read_parquet(PARQUET / "test_windowed.parquet")
    for f in (tr, te):
        f["attack_cat"] = f["attack_cat"].replace({"Backdoors": "Backdoor"})
    tr = tr.sort_values("ts").reset_index(drop=True)

    split_idx = int(len(tr) * (1 - VAL_FRACTION))
    val_time = tr["ts"].iloc[split_idx]
    fit = tr[tr["ts"] < val_time].reset_index(drop=True)
    val = tr[tr["ts"] >= val_time].reset_index(drop=True)
    print(f"chronological split: fit={len(fit)} < {val_time} <= val={len(val)}", flush=True)

    le = LabelEncoder().fit(tr["attack_cat"])
    y_fit = le.transform(fit["attack_cat"])
    y_val = le.transform(val["attack_cat"])
    y_test = le.transform(te["attack_cat"])

    cols = [c for c in tr.columns if c not in DROP + ["attack_cat"]]
    Xf, Xv, Xt = fit[cols].copy(), val[cols].copy(), te[cols].copy()
    for c in CAT:
        cats = pd.Categorical(Xf[c])           # categories from fit rows only
        Xf[c] = cats.codes
        Xv[c] = pd.Categorical(Xv[c], categories=cats.categories).codes
        Xt[c] = pd.Categorical(Xt[c], categories=cats.categories).codes
    f32 = lambda d: np.nan_to_num(d.to_numpy(np.float32), nan=0, posinf=0, neginf=0)
    return (f32(Xf), y_fit, f32(Xv), y_val, f32(Xt), y_test,
            list(le.classes_))

def cap(X, y, rng):
    keep = []
    for c in np.unique(y):
        i = np.where(y == c)[0]
        keep.append(rng.choice(i, PER_CLASS_CAP, replace=False) if len(i) > PER_CLASS_CAP else i)
    keep = np.concatenate(keep)
    return X[keep], y[keep]

def resample(X, y, strategy, rng):
    """Resamples fit rows only. Mirrors 01c's strategies, plus "none".

    "none" is the default for the baselines: every real fit row, untouched.
    Imbalance is left to class_weight."""
    if strategy == "none":
        # Baseline setup: majorities capped at PER_CLASS_CAP for tractability,
        # nothing synthesised and no minority class touched. Remaining
        # imbalance is left to class_weight, which is how trees normally
        # handle it -- the TM's SMOTE is part of the TM pipeline, not
        # something a standard baseline should inherit.
        Xc, yc = cap(X, y, rng)
        print(f"  capped, no synthesis: {dict(enumerate(np.bincount(yc)))} "
              f"({len(yc)} rows)", flush=True)
        return Xc, yc

    if strategy == "undersample":
        keep = []
        for c in np.unique(y):
            i = np.where(y == c)[0]
            keep.append(rng.choice(i, UNDERSAMPLE_CAP, replace=False)
                        if len(i) > UNDERSAMPLE_CAP else i)
        keep = np.concatenate(keep)
        return X[keep], y[keep]

    Xc, yc = cap(X, y, rng)
    counts = np.bincount(yc)
    present = [int(c) for c in np.unique(yc)]
    oversamplable = [c for c in present if counts[c] >= 2]
    if strategy == "smote":
        targets = {c: PER_CLASS_CAP for c in oversamplable}
    else:  # moderate
        targets = {c: min(PER_CLASS_CAP, MODERATE_MULTIPLE * int(counts[c]))
                   for c in oversamplable}
    targets = {c: max(t, int(counts[c])) for c, t in targets.items()}
    k = max(1, min(5, min(int(counts[c]) for c in oversamplable) - 1))
    return SMOTE(random_state=RANDOM_STATE, sampling_strategy=targets,
                  k_neighbors=k).fit_resample(Xc, yc)

def run(tag, X_fit, y_fit, X_val, y_val, X_test, y_test, class_names, strategy):
    print(f"\n{'='*70}\n{tag} | strategy={strategy}\n{'='*70}", flush=True)
    rng = np.random.default_rng(RANDOM_STATE)
    Xb, yb = resample(X_fit, y_fit, strategy, rng)
    print(f"fit(real)={X_fit.shape} -> {strategy} {Xb.shape} | "
          f"val(real)={X_val.shape} | test={X_test.shape}", flush=True)
    print(f"fit class counts after {strategy}: {dict(enumerate(np.bincount(yb)))}", flush=True)

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
        final = make(best, SEEDS[0]).fit(Xb, yb)
        pred = final.predict(X_test)
        print(f"\n   {name} per-class test report ({strategy}):", flush=True)
        print(classification_report(y_test, pred, target_names=class_names,
                                     zero_division=0), flush=True)
        print(f"   confusion matrix (rows=true, cols=pred):\n{confusion_matrix(y_test, pred)}",
              flush=True)
        pc_p, pc_r, pc_f1, pc_s = precision_recall_fscore_support(
            y_test, pred, zero_division=0, labels=list(range(len(class_names))))
        results[name] = {"params": best, "val_macro_f1": best_f1, "agg": agg,
                          "per_class": {class_names[i]: {"precision": float(pc_p[i]),
                                                          "recall": float(pc_r[i]),
                                                          "f1": float(pc_f1[i]),
                                                          "support": int(pc_s[i])}
                                         for i in range(len(class_names))},
                          "confusion_matrix": confusion_matrix(y_test, pred).tolist()}
        print(f"   {name} over {len(SEEDS)} seeds: "
              f"train={agg['train_acc'][0]:.4f} test={agg['test_acc'][0]:.4f}+/-{agg['test_acc'][1]:.4f} "
              f"P={agg['w_p'][0]:.4f} R={agg['w_r'][0]:.4f} "
              f"weighted_F1={agg['w_f1'][0]:.4f}+/-{agg['w_f1'][1]:.4f} "
              f"macro_F1={agg['m_f1'][0]:.4f}+/-{agg['m_f1'][1]:.4f}", flush=True)
    return results

def main():
    strategy = sys.argv[1] if len(sys.argv) > 1 else "none"
    if strategy not in ("none", "smote", "moderate", "undersample"):
        raise SystemExit(f"unknown strategy {strategy!r}")
    out = {}
    Xf, yf, Xv, yv, Xt, yt, class_names = load_continuous()
    out["continuous"] = run("CONTINUOUS FEATURES", Xf, yf, Xv, yv, Xt, yt,
                             class_names, strategy)
    suffix = "" if strategy == "none" else f"_{strategy}"
    path = find_dir("data") / f"tuned_baselines{suffix}.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved -> {path}", flush=True)

if __name__ == "__main__":
    main()
