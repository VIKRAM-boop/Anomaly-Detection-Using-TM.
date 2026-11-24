"""
Phase 5, Task 6: RF and XGBoost on CICIDS2017 (40_cicids_prepare.py),
with 20_openset_baselines.py's grids, selection and saved scores.

Per held-out family and representation (boolean: the TM's literals on the
capped real rows; continuous_medium: the same rows' per-flow statistics and
medium windows), hyperparameters are selected on validation known-class
macro F1 (family absent), then refitted over the seeds. Validation, test
and pseudo-unknown validation scores are saved as 20 saves them, so
30_task2_compare.py evaluates trees and TM with the same code.

Writes <data>/phase5/tree_scores/<repr>/ and <data>/phase3/baselines_openset_<repr>.json
(selected hyperparameters).

Usage: 41_cicids_trees.py [family ...]
"""
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("bl", HERE / "20_openset_baselines.py")
bl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bl)

FAMILIES = ["Bot", "BruteForce", "DDoS", "DoS", "PortScan", "WebAttack"]
SEEDS = [int(s) for s in os.environ.get("BASELINE_SEEDS", "42,7,1234").split(",")]
N_JOBS = int(os.environ.get("BASELINE_JOBS", "6"))

def main():
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier
    data = bl.find_dir("data")
    fams = sys.argv[1:] or FAMILIES
    for rep, pre in (("boolean", "X"), ("continuous_medium", "C")):
        sel_path = data / "phase3" / f"baselines_openset_{rep}.json"
        sel_path.parent.mkdir(parents=True, exist_ok=True)
        sel_all = json.load(open(sel_path)) if sel_path.exists() else {}
        sd = data / "phase5" / "tree_scores" / rep
        sd.mkdir(parents=True, exist_ok=True)
        for fam in fams:
            t0 = time.perf_counter()
            d = data / f"openset_medium_{fam.lower()}_none"
            Xf = np.load(d / f"{pre}_fit.npy").astype(np.float32)
            Xv = np.load(d / f"{pre}_val.npy").astype(np.float32)
            Xt = np.load(d / f"{pre}_test.npy").astype(np.float32)
            Xh = np.load(d / f"{pre}_val_heldout.npy").astype(np.float32)
            yf = np.load(d / "y_fit.npy").astype(np.int64)
            yv = np.load(d / "y_val.npy").astype(np.int64)
            yt = np.load(d / "y_test.npy").astype(np.int64)
            classes = sorted(set(np.unique(yf).tolist()))
            enc = {c: i for i, c in enumerate(classes)}
            yfe, yve = np.array([enc[c] for c in yf]), np.array([enc[c] for c in yv])
            w = len(yfe) / (len(classes) * np.bincount(yfe))
            rf_make = lambda p, s: RandomForestClassifier(random_state=s, n_jobs=N_JOBS, **p)
            rf_best, rf_f1 = bl.select(bl.RF_GRID, rf_make, Xf, yf, Xv, yv)
            xgb_make = lambda p, s: XGBClassifier(random_state=s, n_jobs=N_JOBS, tree_method="hist",
                                                  eval_metric="mlogloss", **p)
            xgb_best, xgb_f1 = None, -1
            for p in bl.XGB_GRID:
                f1 = bl.macro_f1(yve, xgb_make(p, 42).fit(Xf, yfe, sample_weight=w[yfe]).predict(Xv))
                if f1 > xgb_f1:
                    xgb_best, xgb_f1 = p, f1
            np.save(sd / f"{fam.lower()}_y_val.npy", yv.astype(np.int16))
            np.save(sd / f"{fam.lower()}_y_test.npy", yt.astype(np.int16))
            for s in SEEDS:
                rf = rf_make(rf_best, s).fit(Xf, yf)
                bl.save_scores(sd / f"{fam.lower()}_rf_s{s}.npz", rf.classes_, rf.predict_proba(Xv),
                               rf.predict_proba(Xt), rf.predict_proba(Xh))
                xg = xgb_make(xgb_best, s).fit(Xf, yfe, sample_weight=w[yfe])
                bl.save_scores(sd / f"{fam.lower()}_xgb_s{s}.npz", np.array(classes), xg.predict_proba(Xv),
                               xg.predict_proba(Xt), xg.predict_proba(Xh))
            sel_all[fam.lower()] = {"selected": {"rf": rf_best, "rf_val_f1": rf_f1, "xgb": xgb_best,
                                                 "xgb_val_f1": xgb_f1}, "n_features": int(Xf.shape[1])}
            json.dump(sel_all, open(sel_path, "w"), indent=1, default=float)
            print(f"{rep} {fam}: RF {rf_best} ({rf_f1:.4f}), XGB {xgb_best} ({xgb_f1:.4f}) "
                  f"in {time.perf_counter()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    main()
