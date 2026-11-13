"""
Trains the single-stage TM for one held-out family (Task 3, version 2).

One multi-class TMClassifier over Normal and every known attack family --
no separate normality branch. Its per-class sums (clipped to [-T, T], the
same range for every class) are the class-support scores that
24_single_stage.py turns into a known-class prediction or Unknown.

Data are 09's open-set datasets, so the held-out family is absent from fit
and validation. Hyperparameters are the family's own bank configuration
(data/tuned/open_<family>.json), which 17_tune_tm.py selected with that
family excluded; the 10-class configuration (tuned/closed.json) is not
used, because its validation set contained every family, including the one
held out here.

The resampling strategy selects 09's variant (smote, ros or none), which
is how Task 1's alternatives reach the open-set evaluation. Multi-class
tmu has no class weights, so "none" trains on the capped real rows with
their natural imbalance.

Jobs run in parallel worker processes, one fresh process per model (tmu
keeps random state in compiled code; see banks.py). Each saves validation
and test class sums and the model to
data/single_<scale>_<family>[_<strategy>]_s<seed>/.

Usage: 25_train_single_stage.py <strategy> <seed,seed,...> [family ...]
"""
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

import tm_fast

SCALE = "medium"
N_WORKERS = int(os.environ.get("SINGLE_WORKERS", "3"))
FAMILIES = ["Analysis", "Backdoor", "DoS", "Exploits", "Fuzzers", "Generic",
            "Reconnaissance", "Shellcode", "Worms"]

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

def suffix(strategy):
    return "" if strategy == "smote" else f"_{strategy}"

def train_one(job):
    from tmu.tsetlin_machine import TMClassifier
    family, strategy, seed = job
    data = find_dir("data")
    src = data / f"openset_{SCALE}_{family.lower()}{suffix(strategy)}"
    out = data / f"single_{SCALE}_{family.lower()}{suffix(strategy)}_s{seed}"
    if (out / "meta.json").exists():
        return f"{out.name}: exists, skipped"
    meta = json.load(open(src / "meta.json"))
    cfg = json.load(open(data / "tuned" / f"open_{family.lower()}.json"))["best"]
    y_fit = np.load(src / "y_fit.npy").astype(np.int64)
    y_val = np.load(src / "y_val.npy").astype(np.int64)
    held = meta["held_out_idx"]
    assert held not in set(np.unique(y_fit)) | set(np.unique(y_val)), "held-out family leaked"
    classes = sorted(int(c) for c in np.unique(y_fit))
    enc = {c: i for i, c in enumerate(classes)}
    X_fit = np.ascontiguousarray(np.load(src / "X_fit.npy"), dtype=np.uint32)
    t0 = time.perf_counter()
    np.random.seed(seed)                     # TMClassifier draws from the global RNG
    tm = TMClassifier(number_of_clauses=int(cfg["clauses"]), T=int(cfg["T"]),
                      s=float(cfg["s"]), weighted_clauses=False)
    yf = np.array([enc[c] for c in y_fit], dtype=np.uint32)
    for _ in range(int(cfg["epochs"])):
        tm.fit(X_fit, yf)
    t_train = time.perf_counter() - t0
    del X_fit
    X_val = np.load(src / "X_val.npy")
    tm_fast.verify_against_tmu(tm, X_val[:500], n=500)
    model = tm_fast.extract_tm(tm)
    t0 = time.perf_counter()
    val_s = tm_fast.class_sums(model, X_val).astype(np.int32)
    test_s = tm_fast.class_sums(model, np.load(src / "X_test.npy")).astype(np.int32)
    t_score = time.perf_counter() - t0
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "val_scores.npy", val_s)
    np.save(out / "test_scores.npy", test_s)
    tm_fast.save_model(model, out / "model.npz")
    pred = np.asarray(classes)[val_s.argmax(1)]
    from sklearn.metrics import precision_recall_fscore_support
    f1 = float(precision_recall_fscore_support(y_val, pred, average="macro", zero_division=0)[2])
    json.dump({"classes": classes, "normal_idx": meta["all_class_names"].index("Normal"),
               "unknown_label": meta["unknown_label"], "held_out_idx": held,
               "held_out_class": meta["held_out_class"], "all_class_names": meta["all_class_names"],
               "scale": SCALE, "strategy": strategy, "seed": seed, "config": cfg,
               "n_fit": int(len(y_fit)), "fit_class_counts": np.bincount(y_fit).tolist(),
               "train_seconds": t_train, "score_seconds": t_score,
               "val_closed_macro_f1": f1}, open(out / "meta.json", "w"), indent=2)
    return (f"{out.name}: val closed macro F1 {f1:.4f}, trained in {t_train:.0f}s, "
            f"scored in {t_score:.0f}s")

def main():
    strategy = sys.argv[1]
    seeds = [int(s) for s in sys.argv[2].split(",")]
    families = sys.argv[3:] or FAMILIES
    jobs = [(f, strategy, s) for f in families for s in seeds]
    ctx = mp.get_context("spawn")
    with ctx.Pool(min(N_WORKERS, len(jobs)), maxtasksperchild=1) as pool:
        for msg in pool.imap_unordered(train_one, jobs, chunksize=1):
            print(msg, flush=True)

if __name__ == "__main__":
    main()
