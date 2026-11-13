"""
Closed-set study of resampling (Task 1) and feature representation (Task 2).

All models use the same chronological fit/validation/test partition and
the same information: the per-flow features plus the medium-scale window
features (what TM_medium sees). Test is scored once per seed.

Task 1 -- TM trained on each resampling variant of 01c's dataset:
  smote  original (SMOTE on binary literals, thresholded at 0.5)
  ros    majorities capped, minorities duplicated up to the cap
  none   capped real rows only (tmu has no class weights, so the natural
         imbalance remains; the prior-calibrated decision corrects for it)

Task 2 -- representation, on identical real rows (01c's "none" rows):
  TM       Boolean literals
  RF, XGB  the same Boolean literals
  RF, XGB  the original continuous features (rebuilt from the parquet
           exactly as 01c partitions it; asserted row for row by label)
  and Boolean encodings with 3, 5 (original), 9 and 19 quantile thresholds
  per continuous feature, every threshold fitted on fit rows only (01c).

For each encoding it also reports dimensionality (literals, and distinct
literal columns once duplicated thresholds collapse), compute (TM train and
scoring time from 02c's run_meta.json, RF/XGB fit time), and an information
check: the share of each attack family's test rows whose Boolean pattern is
identical to some Normal test row's -- rows no model on that encoding can
separate from Normal.

TM decisions: raw argmax of class sums, and the prior-calibrated decision
of 22_closed_summary.py (tau chosen on validation). RF/XGB use class
weights ("balanced") and one small grid shared by every representation (two
RF depths at min_samples_leaf 5, two XGBoost depths), selected on
validation macro F1 per representation, then refitted over TREE_SEEDS.

Writes data/phase4/encoding_study.json.

Usage: 27_encoding_study.py [part ...]   parts: tm, trees, info (default all)
"""
import importlib.util
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

TM_SEEDS = [42, 7, 1234, 2024, 99]
TREE_SEEDS = [42, 7, 1234]
RESAMPLING = ["smote", "ros", "none"]
ENCODINGS = [3, 5, 9, 19]
N_JOBS = int(os.environ.get("BASELINE_JOBS", "6"))
PER_CLASS_CAP = 40000
VAL_FRACTION = 0.10
DROP = ["srcip", "sport", "dstip", "dsport", "stime", "ltime", "ts", "label"]
CAT = ["proto", "service", "state"]
RF_GRID = [{"n_estimators": 100, "max_depth": d, "min_samples_leaf": 5,
            "max_features": 0.3, "class_weight": "balanced"} for d in (None, 30)]
XGB_GRID = [{"n_estimators": 300, "max_depth": d, "learning_rate": 0.1,
             "subsample": 0.9, "colsample_bytree": 0.8} for d in (6, 10)]
CHUNK = 100000

_spec = importlib.util.spec_from_file_location(
    "cs", Path(__file__).resolve().parent / "22_closed_summary.py")
cs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cs)
find_dir = cs.find_dir

def tm_dir(strategy, q):
    sfx = ("" if strategy == "smote" else f"_{strategy}") + ("" if q == 5 else f"_q{q}")
    return sfx

def per_class(y, p, names):
    pr, rc, f1, sup = precision_recall_fscore_support(y, p, labels=range(len(names)), zero_division=0)
    return {n: {"precision": float(pr[i]), "recall": float(rc[i]), "f1": float(f1[i]),
                "support": int(sup[i])} for i, n in enumerate(names)}

def summary(y, p, names):
    out = cs.metrics(y, p)
    out["per_class"] = per_class(y, p, names)
    return out

def tie(x, seed=0):
    return x + np.random.default_rng(seed).uniform(0, 1e-3, x.shape)

def agg(runs):
    """mean/std over seeds of the scalar metrics and of per-class scores."""
    keys = ["acc", "macro_f1", "w_f1"]
    out = {k: cs.ms([r[k] for r in runs]) for k in keys}
    names = list(runs[0]["per_class"])
    out["per_class"] = {n: {m: cs.ms([r["per_class"][n][m] for r in runs])
                            for m in ("precision", "recall", "f1")} for n in names}
    return out

# ---------------- TM ----------------

def tm_part(data, names, log_prior):
    res = {}
    variants = [(s, 5) for s in RESAMPLING] + [("none", q) for q in ENCODINGS if q != 5]
    for strategy, q in variants:
        sfx = tm_dir(strategy, q)
        raw, cal, times = [], [], []
        for seed in TM_SEEDS:
            d = data / f"scores_medium_seed{seed}{sfx}"
            if not (d / "test_scores.npy").exists():
                continue
            vs, ts = tie(np.load(d / "val_scores.npy").astype(np.float64)), \
                     tie(np.load(d / "test_scores.npy").astype(np.float64), 1)
            yv, yt = np.load(d / "y_val.npy"), np.load(d / "y_test.npy")
            raw.append(summary(yt, ts.argmax(1), names))
            p, tau = cs.calibrated(vs, yv, ts, log_prior)
            c = summary(yt, p, names); c["tau"] = tau
            cal.append(c)
            rm = d / "run_meta.json"
            if rm.exists():
                times.append(json.load(open(rm)))
            else:      # runs from before 02c recorded timings: read its log
                log = Path.home() / "tm_scratch" / "logs" / f"closed_medium_s{seed}.log"
                m = re.search(r"\((\d+)s\)", log.read_text()) if log.exists() else None
                if m:
                    times.append({"train_seconds": float(m.group(1))})
        if not raw:
            continue
        key = f"{strategy}_q{q}"
        res[key] = {"strategy": strategy, "quantiles": q, "seeds": len(raw),
                    "raw": agg(raw), "calibrated": agg(cal),
                    "train_seconds": cs.ms([t["train_seconds"] for t in times]) if times else None,
                    "score_seconds": cs.ms([t["score_seconds"] for t in times if "score_seconds" in t])
                    if any("score_seconds" in t for t in times) else None}
        print(f"TM {key:10s} seeds {len(raw)}  macro F1 raw {res[key]['raw']['macro_f1'][0]:.4f}"
              f"±{res[key]['raw']['macro_f1'][1]:.4f}  calibrated "
              f"{res[key]['calibrated']['macro_f1'][0]:.4f}±{res[key]['calibrated']['macro_f1'][1]:.4f}",
              flush=True)
    return res

# ---------------- RF / XGBoost ----------------

def continuous_partition(data, y_ref):
    """01c's partition and capping, on the continuous features TM_medium
    is built from (per-flow plus medium-window)."""
    tr = pd.read_parquet(data / "temporal" / "train_windowed.parquet")
    te = pd.read_parquet(data / "temporal" / "test_windowed.parquet")
    for f in (tr, te):
        f["attack_cat"] = f["attack_cat"].replace({"Backdoors": "Backdoor"})
    tr = tr.sort_values("ts").reset_index(drop=True)
    le = LabelEncoder().fit(tr["attack_cat"])
    split = tr["ts"].iloc[int(len(tr) * (1 - VAL_FRACTION))]
    fit, val = tr[tr["ts"] < split].reset_index(drop=True), tr[tr["ts"] >= split].reset_index(drop=True)
    cols = [c for c in tr.columns if c not in DROP + ["attack_cat"]
            and not c.startswith(("short_", "long_"))]
    y_fit = le.transform(fit["attack_cat"])
    rng = np.random.default_rng(42)
    keep = []
    for c in np.where(np.bincount(y_fit) > 0)[0]:
        i = np.where(y_fit == c)[0]
        keep.append(rng.choice(i, PER_CLASS_CAP, replace=False) if len(i) > PER_CLASS_CAP else i)
    keep = np.concatenate(keep)
    fit = fit.iloc[keep].reset_index(drop=True)
    assert np.array_equal(le.transform(fit["attack_cat"]), y_ref), "rows differ from 01c's"
    out = []
    for df in (fit, val, te):
        df = df[cols].copy()
        for c in CAT:
            df[c] = pd.Categorical(df[c], categories=pd.Categorical(fit[c]).categories).codes
        out.append(np.nan_to_num(df.to_numpy(np.float32), nan=0, posinf=0, neginf=0))
    return out, cols, le.transform(val["attack_cat"]), le.transform(te["attack_cat"])

def predict_chunked(model, X):
    return np.concatenate([model.predict(X[i:i + CHUNK]) for i in range(0, len(X), CHUNK)])

def fit_trees(Xf, yf, Xv, yv, Xt, yt, names, label):
    w = len(yf) / (len(names) * np.bincount(yf, minlength=len(names)).clip(1))
    rf_make = lambda p, s: RandomForestClassifier(random_state=s, n_jobs=N_JOBS, **p)
    xgb_make = lambda p, s: XGBClassifier(random_state=s, n_jobs=N_JOBS, tree_method="hist",
                                          eval_metric="mlogloss", **p)
    out = {}
    for mname, grid, make, kw in (("rf", RF_GRID, rf_make, {}),
                                  ("xgb", XGB_GRID, xgb_make, {"sample_weight": w[yf]})):
        best, best_f1, best_model, best_t = None, -1, None, None
        for p in grid:
            t0 = time.perf_counter()
            m = make(p, TREE_SEEDS[0]).fit(Xf, yf, **kw)
            t = time.perf_counter() - t0
            f1 = cs.metrics(yv, predict_chunked(m, Xv))["macro_f1"]
            if f1 > best_f1:
                best, best_f1, best_model, best_t = p, f1, m, t
        runs, times = [], [best_t]
        for s in TREE_SEEDS:
            if s == TREE_SEEDS[0]:
                m = best_model
            else:
                t0 = time.perf_counter()
                m = make(best, s).fit(Xf, yf, **kw)
                times.append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            runs.append(summary(yt, predict_chunked(m, Xt), names))
            runs[-1]["predict_seconds"] = time.perf_counter() - t0
        out[mname] = {"selected": best, "val_macro_f1": best_f1, **agg(runs),
                      "train_seconds": cs.ms(times),
                      "predict_seconds": cs.ms([r["predict_seconds"] for r in runs])}
        print(f"{mname.upper():4s}{label:16s} macro F1 {out[mname]['macro_f1'][0]:.4f}"
              f"±{out[mname]['macro_f1'][1]:.4f}  (selected {best}, fit {np.mean(times):.0f}s)",
              flush=True)
    return out

def trees_part(data, names):
    res = {}
    for q in ENCODINGS:
        d = data / f"temporal_medium_none{'' if q == 5 else f'_q{q}'}"
        if not (d / "X_fit.npy").exists():
            continue
        Xf, yf = np.load(d / "X_fit.npy"), np.load(d / "y_fit.npy").astype(np.int64)
        Xv, yv = np.load(d / "X_val.npy"), np.load(d / "y_val.npy").astype(np.int64)
        Xt, yt = np.load(d / "X_test.npy"), np.load(d / "y_test.npy").astype(np.int64)
        res[f"boolean_q{q}"] = fit_trees(Xf, yf, Xv, yv, Xt, yt, names, f" boolean q{q}")
        res[f"boolean_q{q}"]["n_features"] = int(Xf.shape[1])
        if q == 5:
            y_ref = yf
        del Xf, Xv, Xt
    (Xf, Xv, Xt), cols, yv, yt = continuous_partition(data, y_ref)
    res["continuous"] = fit_trees(Xf, y_ref, Xv, yv, Xt, yt, names, " continuous")
    res["continuous"]["n_features"] = len(cols)
    return res

# ---------------- information and dimensionality ----------------

def keys(X):
    P = np.packbits(X, axis=1)
    return np.array([r.tobytes() for r in P], dtype=object)

def info_part(data, names):
    ni = names.index("Normal")
    res = {}
    for q in ENCODINGS:
        d = data / f"temporal_medium_none{'' if q == 5 else f'_q{q}'}"
        if not (d / "X_test.npy").exists():
            continue
        meta = json.load(open(d / "meta.json"))
        Xf = np.load(d / "X_fit.npy")
        cols = np.packbits(Xf, axis=0).T
        distinct_cols = len({c.tobytes() for c in cols})
        Xt, yt = np.load(d / "X_test.npy"), np.load(d / "y_test.npy").astype(np.int64)
        k = keys(Xt)
        normal_patterns = set(k[yt == ni].tolist())
        by_pattern = pd.Series(yt).groupby(pd.Series(k)).nunique()
        shared = set(by_pattern[by_pattern > 1].index)
        fam = {}
        for c, n in enumerate(names):
            if c == ni or not (yt == c).any():
                continue
            kc = k[yt == c]
            fam[n] = {"test_rows": int(len(kc)),
                      "identical_to_a_normal_row": float(np.mean([x in normal_patterns for x in kc])),
                      "distinct_patterns": int(len(set(kc.tolist())))}
        sizes = {nm: (d / f"{nm}.npy").stat().st_size for nm in ("X_fit", "X_val", "X_test")}
        res[f"q{q}"] = {"quantiles": q, "levels": meta.get("quantile_levels"),
                        "n_literals": int(Xf.shape[1]), "distinct_literal_columns": distinct_cols,
                        "distinct_test_patterns": int(len(set(k.tolist()))),
                        "test_rows_with_pattern_shared_across_classes":
                            float(np.mean([x in shared for x in k])),
                        "families": fam, "bytes_on_disk": sizes}
        worst = max(fam.items(), key=lambda kv: kv[1]["identical_to_a_normal_row"])
        print(f"q{q:<3d} literals {Xf.shape[1]:5d} (distinct {distinct_cols:5d})  test rows sharing a "
              f"pattern across classes {res[f'q{q}']['test_rows_with_pattern_shared_across_classes']:.1%}"
              f"  most Normal-identical family: {worst[0]} {worst[1]['identical_to_a_normal_row']:.1%}",
              flush=True)
        del Xf, Xt
    return res

def main():
    parts = sys.argv[1:] or ["tm", "trees", "info"]
    data = find_dir("data")
    meta = json.load(open(data / "temporal_medium" / "meta.json"))
    names = meta["class_names"]
    counts = np.maximum(np.array(meta["fit_class_counts_real"], dtype=np.float64), 1)
    log_prior = np.log(counts / counts.sum())
    path = data / "phase4" / "encoding_study.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    out = json.load(open(path)) if path.exists() else {}
    out["class_names"] = names
    if "tm" in parts:
        out["tm"] = tm_part(data, names, log_prior)
    if "info" in parts:
        out["info"] = info_part(data, names)
    if "trees" in parts:
        out["trees"] = trees_part(data, names)
    json.dump(out, open(path, "w"), indent=2, default=float)
    print(f"written to {path}")

if __name__ == "__main__":
    main()
