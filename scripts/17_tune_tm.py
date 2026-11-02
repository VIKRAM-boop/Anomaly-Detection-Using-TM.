"""
Selects Tsetlin Machine hyperparameters on the corrected validation data.

Two modes:

  open <family>   tunes Step 5's clause banks for one held-out family. Data
                  come from 09, which removed that family from fit and
                  validation, so it plays no part in selection. Objective:
                  validation known-class macro F1 of the Step 5 decision
                  without an unknown option (banks.closed_macro_f1), since
                  the held-out family must not be used to score novelty.
  closed          tunes the 10-class TM used in the main pipeline (02c) on
                  the medium-scale data from 01c. Objective: validation
                  macro F1 of the argmax prediction.

Search is staged rather than a full grid, to stay within a laptop's budget:

  stage 1  clauses in {1000, 2000, 4000} x T/clauses in {0.05, 0.10, 0.20}
           (the T/clauses ratio is what mattered in earlier sweeps, because
           class sums are clipped to [-T, T]); s = 5, 1 epoch
  stage 2  s in {2.5, 5, 10} at the stage-1 winner
  stage 3  epochs in {1, 3, 5} at the stage-2 winner, read off a single
           5-epoch run checkpointed after epochs 1, 3 and 5

Training uses a subsample (banks: at most SUBSAMPLE_POS positives per bank;
closed: SUBSAMPLE_PER_CLASS rows per class); validation is scored in full.
Where two configurations are within TIE of each other the cheaper is kept.
Every model trains in a fresh process so results are reproducible (tmu keeps
random state in compiled code that a reused worker would carry over).
Writes every trial and the winner to data/tuned/open_<family>.json or
data/tuned/closed.json.

Usage: 17_tune_tm.py open <family> | 17_tune_tm.py closed
"""
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import precision_recall_fscore_support

import banks
import tm_fast

SEED = 42
SUBSAMPLE_POS = 20000
SUBSAMPLE_PER_CLASS = 10000
# Differences below ~0.005 are within validation noise: Worms has 19
# validation rows, so one row moves macro F1 by roughly 0.003. Within that
# margin the cheaper configuration wins.
TIE = 0.005
CLAUSES = [1000, 2000, 4000]
T_RATIOS = [0.05, 0.10, 0.20]
S_VALUES = [2.5, 5.0, 10.0]
EPOCH_VALUES = [1, 3, 5]

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

def cost(cfg):
    return cfg["clauses"] * cfg["epochs"]

def pick(trials):
    """Best validation score; among those within TIE of it, the cheapest."""
    top = max(t["val_macro_f1"] for t in trials)
    near = [t for t in trials if t["val_macro_f1"] >= top - TIE]
    return min(near, key=lambda t: cost(t["config"]))

# ---------------- open (Step 5 banks) ----------------

def eval_open(cfg, epochs_list, X_fit, y_fit, X_val, y_val, families, normal_idx):
    trained = banks.train_banks(X_fit, y_fit, families, normal_idx, cfg, SEED,
                                max_pos=SUBSAMPLE_POS, epochs_to_report=epochs_list)
    out = []
    for ep in epochs_list:
        models = {k: v[ep] for k, v in trained.items()}
        f1 = banks.closed_macro_f1(banks.score_banks(models, X_val), y_val, families, normal_idx)
        out.append({"config": {**cfg, "epochs": ep}, "val_macro_f1": f1})
    return out

def tune_open(family):
    d = find_dir("data") / f"openset_medium_{family.lower()}"
    meta = json.load(open(d / "meta.json"))
    names = meta["all_class_names"]
    normal_idx = names.index("Normal")
    families = [c for c in meta["known_classes"] if c != normal_idx]
    X_fit = np.load(d / "X_fit.npy"); y_fit = np.load(d / "y_fit.npy").astype(np.int64)
    X_val = np.load(d / "X_val.npy"); y_val = np.load(d / "y_val.npy").astype(np.int64)
    held = meta["held_out_idx"]
    assert held not in set(np.unique(y_fit)) | set(np.unique(y_val)), "held-out family leaked"
    print(f"=== tuning banks, held out {family} (absent from fit and validation) ===", flush=True)
    return run_stages(lambda cfg, eps: eval_open(cfg, eps, X_fit, y_fit, X_val, y_val,
                                                 families, normal_idx))

# ---------------- closed (10-class TM) ----------------

def _closed_worker(job):
    from tmu.tsetlin_machine import TMClassifier
    cfg, epochs_list, X, y = job
    np.random.seed(SEED)
    tm = TMClassifier(number_of_clauses=int(cfg["clauses"]), T=int(cfg["T"]),
                      s=float(cfg["s"]), weighted_clauses=False)
    X = np.ascontiguousarray(X, dtype=np.uint32)
    ckpt, done = {}, 0
    for ep in sorted(epochs_list):
        for _ in range(ep - done):
            tm.fit(X, y)
        done = ep
        ckpt[ep] = tm_fast.extract_tm(tm)
    return cfg, ckpt

def tune_closed():
    d = find_dir("data") / "temporal_medium"
    X_fit = np.load(d / "X_fit.npy"); y_fit = np.load(d / "y_fit.npy").astype(np.int64)
    X_val = np.load(d / "X_val.npy"); y_val = np.load(d / "y_val.npy").astype(np.int64)
    rng = np.random.default_rng(SEED)
    idx = np.concatenate([rng.choice(np.where(y_fit == c)[0],
                                     min(SUBSAMPLE_PER_CLASS, int((y_fit == c).sum())),
                                     replace=False) for c in np.unique(y_fit)])
    Xs, ys = X_fit[idx], y_fit[idx].astype(np.uint32)
    print(f"=== tuning 10-class TM on {len(idx)} fit rows, full validation ===", flush=True)

    def evaluate_many(cfgs, epochs_list):
        ctx = mp.get_context("spawn")
        with ctx.Pool(min(banks.N_WORKERS, len(cfgs)), maxtasksperchild=1) as pool:
            res = pool.map(_closed_worker, [(c, epochs_list, Xs, ys) for c in cfgs], chunksize=1)
        out = []
        for cfg, ckpt in res:
            for ep in epochs_list:
                sums = tm_fast.class_sums(ckpt[ep], X_val).astype(np.float64)
                sums += np.random.default_rng(0).uniform(0, 1e-3, sums.shape)   # tie-break
                pred = sums.argmax(axis=1)
                f1 = float(precision_recall_fscore_support(y_val, pred, average="macro",
                                                            zero_division=0)[2])
                out.append({"config": {**cfg, "epochs": ep}, "val_macro_f1": f1})
        return out
    return run_stages(None, evaluate_many)

# ---------------- staged search ----------------

def run_stages(eval_one, eval_many=None):
    """eval_one(cfg, epochs_list) -> trials, or eval_many(cfgs, epochs_list) -> trials."""
    def evaluate(cfgs, eps):
        if eval_many is not None:
            return eval_many(cfgs, eps)
        out = []
        for c in cfgs:
            out += eval_one(c, eps)
        return out

    trials = []
    t0 = time.perf_counter()
    s1 = [{"clauses": c, "T": int(round(c * r)), "s": 5.0, "epochs": 1}
          for c in CLAUSES for r in T_RATIOS]
    r1 = evaluate(s1, [1]); trials += r1
    for t in r1:
        print(f"  stage1 {t['config']} val_macro_f1={t['val_macro_f1']:.4f}", flush=True)
    b1 = pick(r1)["config"]

    s2 = [{**b1, "s": s} for s in S_VALUES if s != b1["s"]]
    r2 = evaluate(s2, [1]); trials += r2
    for t in r2:
        print(f"  stage2 {t['config']} val_macro_f1={t['val_macro_f1']:.4f}", flush=True)
    b2 = pick([t for t in trials if t["config"]["clauses"] == b1["clauses"]
               and t["config"]["T"] == b1["T"] and t["config"]["epochs"] == 1])["config"]

    r3 = evaluate([{**b2, "epochs": max(EPOCH_VALUES)}], EPOCH_VALUES)
    trials += r3
    for t in r3:
        print(f"  stage3 {t['config']} val_macro_f1={t['val_macro_f1']:.4f}", flush=True)
    best = pick(r3)
    print(f"  BEST {best['config']} val_macro_f1={best['val_macro_f1']:.4f} "
          f"({time.perf_counter()-t0:.0f}s)", flush=True)
    return best, trials

def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "closed"
    out_dir = find_dir("data") / "tuned"
    out_dir.mkdir(parents=True, exist_ok=True)
    if mode == "open":
        family = sys.argv[2]
        best, trials = tune_open(family)
        path = out_dir / f"open_{family.lower()}.json"
    else:
        best, trials = tune_closed()
        path = out_dir / "closed.json"
    with open(path, "w") as f:
        json.dump({"best": best["config"], "best_val_macro_f1": best["val_macro_f1"],
                   "seed": SEED, "tie_tolerance": TIE, "trials": trials}, f, indent=2)
    print(f"written to {path}")

if __name__ == "__main__":
    main()
