"""
Closed-set (10-class) TM results over seeds, with the tuned hyperparameters.

Reads data/scores_<scale>_seed<seed>/ written by 02c for every scale and
seed present and reports, as mean +/- std over seeds:

  raw         argmax of the per-class vote sums
  calibrated  vote sums plus tau * log class prior, tau chosen per seed on
              validation by macro F1 (priors are the fit partition's real,
              pre-resampling class counts)
  ensemble    per-class maximum across the three scales, then the same
              validation-selected prior calibration

Test is scored once per seed with whatever validation selected.

Writes data/phase3/closed_summary.json.
"""
import json
import os
import re
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

SCALES = ["short", "medium", "long"]
TAU_GRID = [0, 0.5, 1, 2, 5, 10, 20, 30, 50, 80, 120]

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

def metrics(y, p):
    wp, wr, wf, _ = precision_recall_fscore_support(y, p, average="weighted", zero_division=0)
    return {"acc": float(accuracy_score(y, p)), "w_precision": float(wp), "w_recall": float(wr),
            "w_f1": float(wf),
            "macro_f1": float(precision_recall_fscore_support(y, p, average="macro", zero_division=0)[2])}

def calibrated(val, yv, test, log_prior):
    best_tau, best = 0, -1
    for t in TAU_GRID:
        f = precision_recall_fscore_support(yv, (val + t * log_prior).argmax(1), average="macro",
                                            zero_division=0)[2]
        if f > best:
            best, best_tau = f, t
    return (test + best_tau * log_prior).argmax(1), best_tau

def ms(xs):
    return (float(np.mean(xs)), float(np.std(xs)))

def main():
    data = find_dir("data")
    runs = {}
    for d in data.glob("scores_*_seed*"):
        m = re.match(r"scores_(short|medium|long)_seed(\d+)$", d.name)
        if m:
            runs.setdefault(int(m.group(2)), {})[m.group(1)] = d
    seeds = sorted(s for s, v in runs.items() if len(v) == 3)
    meta = json.load(open(data / "temporal_medium" / "meta.json"))
    counts = np.maximum(np.array(meta["fit_class_counts_real"], dtype=np.float64), 1)
    log_prior = np.log(counts / counts.sum())
    tuned = data / "tuned" / "closed.json"
    cfg = json.load(open(tuned))["best"] if tuned.exists() else None

    per = {s: {} for s in seeds}
    for s in seeds:
        vals, tests = {}, {}
        for sc in SCALES:
            d = runs[s][sc]
            vals[sc] = np.load(d / "val_scores.npy").astype(np.float64)
            tests[sc] = np.load(d / "test_scores.npy").astype(np.float64)
            yv = np.load(d / "y_val.npy"); yt = np.load(d / "y_test.npy")
            per[s][f"{sc}_raw"] = metrics(yt, tests[sc].argmax(1))
            pred, tau = calibrated(vals[sc], yv, tests[sc], log_prior)
            per[s][f"{sc}_calibrated"] = {**metrics(yt, pred), "tau": tau}
        ev = np.max([vals[sc] for sc in SCALES], axis=0)
        et = np.max([tests[sc] for sc in SCALES], axis=0)
        pred, tau = calibrated(ev, yv, et, log_prior)
        per[s]["ensemble_max_calibrated"] = {**metrics(yt, pred), "tau": tau}

    keys = list(per[seeds[0]].keys())
    summary = {k: {m: ms([per[s][k][m] for s in seeds])
                   for m in ("acc", "w_precision", "w_recall", "w_f1", "macro_f1")} for k in keys}
    print(f"closed-set TM, config {cfg}, seeds {seeds}\n")
    print(f"{'configuration':<26}{'acc':>16}{'weighted F1':>18}{'macro F1':>18}")
    for k in keys:
        v = summary[k]
        print(f"{k:<26}{v['acc'][0]:>9.4f}±{v['acc'][1]:.4f}{v['w_f1'][0]:>11.4f}±{v['w_f1'][1]:.4f}"
              f"{v['macro_f1'][0]:>11.4f}±{v['macro_f1'][1]:.4f}")
    out = data / "phase3"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "closed_summary.json", "w") as f:
        json.dump({"config": cfg, "seeds": seeds, "per_seed": per, "summary": summary}, f, indent=2)
    print(f"\nwritten to {out / 'closed_summary.json'}")

if __name__ == "__main__":
    main()
