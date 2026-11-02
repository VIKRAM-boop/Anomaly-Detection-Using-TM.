"""
Open-set baselines on the same partitions and protocol as the TM (Task 6).

For each held-out family, the data are rebuilt exactly as 09 builds them:
same chronological order, the family removed from fit and validation before
the split, the same validation cut-off (asserted against the TM dataset's
meta.json) and held-out test rows relabelled unknown. Baselines use the
continuous features, their native form; majorities in fit are capped at
PER_CLASS_CAP for tractability and nothing is synthesised.

Methods:

  rf_msp        Random Forest; novelty = 1 - max class probability
  xgb_msp       XGBoost; novelty = 1 - max class probability
  iforest_rf    the TM's two-stage structure with standard parts: an
                Isolation Forest trained on fit Normal rows is the benign
                gate (threshold at a 1% missed-attack budget on validation,
                as tau_B is for the TM), and a Random Forest over the known
                attack families supplies the attack-novelty score
  mahalanobis   distance-based: shared-covariance Mahalanobis distance to
                each known class mean in log-scaled standardised space;
                novelty = distance to the nearest class

RF and XGBoost hyperparameters are re-selected per held-out family on
validation (family absent) by known-class macro F1. Every method's novelty
threshold is calibrated on validation to the same false-rejection budgets as
18_openset_eval.py -- the share of all known validation rows predicted
Unknown -- and the same metrics are reported. Stochastic methods are run
over the same five seeds as the TM.

Writes data/phase3/baselines_openset.json.

Usage: 20_openset_baselines.py [family ...]
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             precision_recall_fscore_support, roc_auc_score)
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

SEEDS = [42, 7, 1234, 2024, 99]
BUDGETS = [0.01, 0.05, 0.10]
REPORT_BUDGET = 0.05
MISSED_ATTACK_BUDGET = 0.01
VAL_FRACTION = 0.10
PER_CLASS_CAP = 40000
UNKNOWN_LABEL = 99
N_JOBS = int(os.environ.get("BASELINE_JOBS", "4"))
DROP = ["srcip", "sport", "dstip", "dsport", "stime", "ltime", "ts", "label"]
CAT = ["proto", "service", "state"]
FAMILIES = ["Analysis", "Backdoor", "DoS", "Exploits", "Fuzzers", "Generic",
            "Reconnaissance", "Shellcode", "Worms"]
RF_GRID = [{"n_estimators": 100, "max_depth": d, "min_samples_leaf": l,
            "max_features": 0.3, "class_weight": "balanced"} for d in (None, 30) for l in (1, 5)]
XGB_GRID = [{"n_estimators": 300, "max_depth": d, "learning_rate": 0.1,
             "subsample": 0.9, "colsample_bytree": 0.8} for d in (6, 10)]
if os.environ.get("BASELINE_QUICK"):          # smoke test only
    SEEDS, RF_GRID, XGB_GRID = SEEDS[:1], RF_GRID[:1], XGB_GRID[:1]

def find_dir(name):
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

def load_partitions(family, tr_all, te_all):
    """Mirrors 09_openset_preprocess.py's partitioning exactly."""
    le = LabelEncoder().fit(tr_all["attack_cat"])
    names = list(le.classes_)
    h = names.index(family)
    tr = tr_all.sort_values("ts").reset_index(drop=True)
    y_tr = le.transform(tr["attack_cat"])
    keep = y_tr != h
    tr, y_tr = tr.loc[keep].reset_index(drop=True), y_tr[keep]
    val_time = tr["ts"].iloc[int(len(tr) * (1 - VAL_FRACTION))]
    is_fit = (tr["ts"] < val_time).to_numpy()
    meta = json.load(open(find_dir("data") / f"openset_medium_{family.lower()}" / "meta.json"))
    assert str(val_time) == meta["val_split_time"], "partition differs from the TM's"
    cols = [c for c in tr.columns if c not in DROP + ["attack_cat"]]
    fit, val = tr.loc[is_fit, cols].copy(), tr.loc[~is_fit, cols].copy()
    test = te_all[cols].copy()
    for c in CAT:
        cats = pd.Categorical(fit[c])
        fit[c] = cats.codes
        val[c] = pd.Categorical(val[c], categories=cats.categories).codes
        test[c] = pd.Categorical(test[c], categories=cats.categories).codes
    f32 = lambda d: np.nan_to_num(d.to_numpy(np.float32), nan=0, posinf=0, neginf=0)
    y_te = le.transform(te_all["attack_cat"])
    y_te[y_te == h] = UNKNOWN_LABEL
    return (f32(fit), y_tr[is_fit], f32(val), y_tr[~is_fit], f32(test), y_te,
            names, names.index("Normal"))

def cap(X, y, rng):
    keep = []
    for c in np.unique(y):
        i = np.where(y == c)[0]
        keep.append(rng.choice(i, PER_CLASS_CAP, replace=False) if len(i) > PER_CLASS_CAP else i)
    keep = np.concatenate(keep)
    return X[keep], y[keep]

def macro_f1(y, p, labels=None):
    return float(precision_recall_fscore_support(y, p, labels=labels, average="macro",
                                                  zero_division=0)[2])

def calibrate_single(nov_val, beta):
    return float(np.quantile(nov_val, 1.0 - beta))

def metrics(y, pred, nov, known_classes, fams, ni):
    unk = y == UNKNOWN_LABEL
    known = ~unk
    p_unk = pred[unk]
    out = {"unknown_recall": float((p_unk == UNKNOWN_LABEL).mean()),
           "held_to_known_family": float(np.isin(p_unk, fams).mean()),
           "held_to_normal": float((p_unk == ni).mean()),
           "false_unknown": float((pred[known] == UNKNOWN_LABEL).mean()),
           "benign_fpr": float((pred[y == ni] != ni).mean()),
           "known_macro_f1": macro_f1(y[known], pred[known], known_classes),
           "known_attack_macro_f1": macro_f1(y[known], pred[known], fams),
           "auroc": float(roc_auc_score(unk, nov)),
           "auprc": float(average_precision_score(unk, nov))}
    return out

def tiebreak(x, seed):
    """Seeded jitter below the scores' resolution: tree probabilities come in
    discrete steps, so without it a threshold lands on a tied value and the
    method rejects fewer validation rows than the shared budget allows."""
    return x + np.random.default_rng(seed).uniform(0, 1e-7, x.shape)

def run_single_stage(name, proba_val, proba_test, classes, y_val, y_te, known_classes, fams, ni):
    """Methods with one novelty score over all rows (Normal is just a class)."""
    nov_v = tiebreak(1.0 - proba_val.max(1), 1); nov_t = tiebreak(1.0 - proba_test.max(1), 2)
    arg_v = classes[proba_val.argmax(1)]; arg_t = classes[proba_test.argmax(1)]
    res = {"closed": metrics(y_te, arg_t, nov_t, known_classes, fams, ni)}
    for b in BUDGETS:
        thr = calibrate_single(nov_v, b)
        pred = np.where(nov_t > thr, UNKNOWN_LABEL, arg_t)
        m = metrics(y_te, pred, nov_t, known_classes, fams, ni)
        m["val_false_reject"] = float((nov_v > thr).mean())
        if abs(b - REPORT_BUDGET) < 1e-9:
            labels = known_classes + [UNKNOWN_LABEL]
            m["confusion"] = {"labels": labels,
                              "matrix": confusion_matrix(y_te, pred, labels=labels).tolist()}
        res[f"{b:.2f}"] = m
    return res

def select(grid, make, X, y, Xv, yv):
    best, best_f1 = None, -1
    for p in grid:
        f1 = macro_f1(yv, make(p, 42).fit(X, y).predict(Xv))
        if f1 > best_f1:
            best, best_f1 = p, f1
    return best, best_f1

def log_std(X, ref):
    Z = np.sign(X) * np.log1p(np.abs(X))
    R = np.sign(ref) * np.log1p(np.abs(ref))
    mu, sd = R.mean(0), R.std(0)
    sd[sd == 0] = 1
    return (Z - mu) / sd

def main():
    families = sys.argv[1:] or FAMILIES
    data = find_dir("data")
    parquet = data / "temporal"
    tr_all = pd.read_parquet(parquet / "train_windowed.parquet")
    te_all = pd.read_parquet(parquet / "test_windowed.parquet")
    for f in (tr_all, te_all):
        f["attack_cat"] = f["attack_cat"].replace({"Backdoors": "Backdoor"})

    out_path = data / "phase3" / "baselines_openset.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results = json.load(open(out_path)) if out_path.exists() else {}

    for family in families:
        t0 = time.perf_counter()
        Xf, yf, Xv, yv, Xt, yt, names, ni = load_partitions(family, tr_all, te_all)
        known_classes = sorted(set(np.unique(yv).tolist()) | set(np.unique(yf).tolist()))
        fams = [c for c in known_classes if c != ni]
        rng = np.random.default_rng(42)
        Xc, yc = cap(Xf, yf, rng)
        print(f"\n=== held out {family}: fit {Xc.shape} val {Xv.shape} test {Xt.shape} ===", flush=True)
        fam_res = {"methods": {}}

        rf_make = lambda p, s: RandomForestClassifier(random_state=s, n_jobs=N_JOBS, **p)
        rf_best, rf_f1 = select(RF_GRID, rf_make, Xc, yc, Xv, yv)
        enc = {c: i for i, c in enumerate(known_classes)}
        yc_enc = np.array([enc[c] for c in yc]); yv_enc = np.array([enc[c] for c in yv])
        w = len(yc) / (len(known_classes) * np.bincount(yc_enc))
        xgb_make = lambda p, s: XGBClassifier(random_state=s, n_jobs=N_JOBS, tree_method="hist",
                                              eval_metric="mlogloss", **p)
        xgb_best, xgb_f1 = None, -1
        for p in XGB_GRID:
            m = xgb_make(p, 42).fit(Xc, yc_enc, sample_weight=w[yc_enc])
            f1 = macro_f1(yv_enc, m.predict(Xv))
            if f1 > xgb_f1:
                xgb_best, xgb_f1 = p, f1
        print(f"  selected RF {rf_best} (val F1 {rf_f1:.4f}); XGB {xgb_best} (val F1 {xgb_f1:.4f})",
              flush=True)

        classes = np.array(known_classes)
        per_seed = {m: {} for m in ("rf_msp", "xgb_msp", "iforest_rf")}
        for s in SEEDS:
            rf = rf_make(rf_best, s).fit(Xc, yc)
            per_seed["rf_msp"][s] = run_single_stage("rf", rf.predict_proba(Xv), rf.predict_proba(Xt),
                                                     rf.classes_, yv, yt, known_classes, fams, ni)
            xg = xgb_make(xgb_best, s).fit(Xc, yc_enc, sample_weight=w[yc_enc])
            per_seed["xgb_msp"][s] = run_single_stage("xgb", xg.predict_proba(Xv), xg.predict_proba(Xt),
                                                      classes, yv, yt, known_classes, fams, ni)
            # two-stage: Isolation Forest benign gate + RF attack novelty
            iso = IsolationForest(n_estimators=200, random_state=s, n_jobs=N_JOBS).fit(Xc[yc == ni])
            bv, bt = iso.score_samples(Xv), iso.score_samples(Xt)       # higher = more normal
            tau_b = np.quantile(bv[yv != ni], 1.0 - MISSED_ATTACK_BUDGET)
            atk = yc != ni
            rfa = rf_make(rf_best, s).fit(Xc[atk], yc[atk])
            pa_v, pa_t = rfa.predict_proba(Xv), rfa.predict_proba(Xt)
            nov_v, nov_t = tiebreak(1 - pa_v.max(1), 1), tiebreak(1 - pa_t.max(1), 2)
            bv, bt = tiebreak(bv, 3), tiebreak(bt, 4)
            arg_v, arg_t = rfa.classes_[pa_v.argmax(1)], rfa.classes_[pa_t.argmax(1)]
            gv, gt = bv < tau_b, bt < tau_b
            sys_t = np.where(gt, nov_t, nov_t - 1e6)
            res = {"closed": metrics(yt, np.where(gt, arg_t, ni), sys_t, known_classes, fams, ni)}
            for b in BUDGETS:
                k = int(round(b * len(yv)))
                cand = np.sort(nov_v[gv])[::-1]
                thr = cand[k] if k < len(cand) else -np.inf
                pred = np.where(gt, np.where(nov_t > thr, UNKNOWN_LABEL, arg_t), ni)
                m = metrics(yt, pred, sys_t, known_classes, fams, ni)
                m["val_false_reject"] = float((gv & (nov_v > thr)).mean())
                if abs(b - REPORT_BUDGET) < 1e-9:
                    labels = known_classes + [UNKNOWN_LABEL]
                    m["confusion"] = {"labels": labels,
                                      "matrix": confusion_matrix(yt, pred, labels=labels).tolist()}
                res[f"{b:.2f}"] = m
            per_seed["iforest_rf"][s] = res
            print(f"  seed {s}: rf {per_seed['rf_msp'][s]['0.05']['unknown_recall']:.1%}  "
                  f"xgb {per_seed['xgb_msp'][s]['0.05']['unknown_recall']:.1%}  "
                  f"iforest_rf {res['0.05']['unknown_recall']:.1%}", flush=True)

        # deterministic distance-based baseline
        Zf, Zv, Zt = log_std(Xc, Xc), log_std(Xv, Xc), log_std(Xt, Xc)
        means = np.stack([Zf[yc == c].mean(0) for c in known_classes])
        centred = np.concatenate([Zf[yc == c] - means[i] for i, c in enumerate(known_classes)])
        prec = LedoitWolf().fit(centred).precision_
        def dists(Z):
            out = np.empty((len(Z), len(known_classes)))
            for i in range(len(known_classes)):
                d = Z - means[i]
                out[:, i] = np.einsum("ij,jk,ik->i", d, prec, d)
            return out
        dv, dt = dists(Zv), dists(Zt)
        # expressed as pseudo-probabilities so run_single_stage can be reused:
        # 1 - max "proba" = normalised distance to the nearest class
        scale = np.quantile(dv.min(1), 0.999)
        pv = 1 - np.clip(dv / scale, 0, 1); pt = 1 - np.clip(dt / scale, 0, 1)
        fam_res["methods"]["mahalanobis"] = {"seeds": {42: run_single_stage(
            "maha", pv, pt, classes, yv, yt, known_classes, fams, ni)}}
        for m in per_seed:
            fam_res["methods"][m] = {"seeds": per_seed[m]}
        fam_res["selected"] = {"rf": rf_best, "rf_val_f1": rf_f1, "xgb": xgb_best, "xgb_val_f1": xgb_f1}
        fam_res["n_unknown"] = int((yt == UNKNOWN_LABEL).sum())
        results[family.lower()] = fam_res
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=float)
        md = fam_res["methods"]["mahalanobis"]["seeds"][42]["0.05"]["unknown_recall"]
        print(f"  mahalanobis {md:.1%} | {family} done in {time.perf_counter()-t0:.0f}s", flush=True)
    print(f"written to {out_path}")

if __name__ == "__main__":
    main()
