"""
Trains Step 5's clause banks for one held-out family, scale and seed.

One normality branch (Normal vs every known attack family) and one detector
per known attack family (that family vs the rest), each a binary
TMClassifier on a balanced sample. The held-out family never appears: 09
removes it from fit and validation before anything is built.

Hyperparameters come from data/tuned/open_<family>.json, written by
17_tune_tm.py from a search in which that family was also excluded; the
defaults are used when no tuned file exists. The seed drives both the
balanced negative sampling and the TM's own randomness.

Banks are trained in parallel (banks.py) and scored with tm_fast, which is
checked against tmu's own prediction path on every run. Saves validation and
test margins plus each bank's clause masks and weights -- enough to rescore
and to explain predictions without retraining -- to
data/banks_<scale>_<family>_s<seed>/.

An optional fourth argument picks 09's resampling variant (smote, the
original, or ros / none for the Task 1 comparison); its banks are written
to data/banks_<scale>_<family>_<strategy>_s<seed>/.

Usage: 10_train_family_banks.py <family> <scale> <seed> [strategy]
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

import banks
import tm_fast

NORMAL_NAME = "Normal"

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

def load_config(family):
    f = find_dir("data") / "tuned" / f"open_{family.lower()}.json"
    if f.exists():
        return json.load(open(f))["best"], str(f)
    return dict(banks.DEFAULT_CONFIG), "default"

def main():
    family = sys.argv[1] if len(sys.argv) > 1 else "Fuzzers"
    scale = sys.argv[2] if len(sys.argv) > 2 else "medium"
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 42
    strategy = sys.argv[4] if len(sys.argv) > 4 else "smote"
    sfx = "" if strategy == "smote" else f"_{strategy}"

    data_dir = find_dir("data") / f"openset_{scale}_{family.lower()}{sfx}"
    meta = json.load(open(data_dir / "meta.json"))
    names = meta["all_class_names"]
    normal_idx = names.index(NORMAL_NAME)
    families = [c for c in meta["known_classes"] if c != normal_idx]
    cfg, cfg_src = load_config(family)

    X_fit = np.load(data_dir / "X_fit.npy")
    y_fit = np.load(data_dir / "y_fit.npy").astype(np.int64)
    X_val = np.load(data_dir / "X_val.npy")
    y_val = np.load(data_dir / "y_val.npy").astype(np.int64)
    X_test = np.load(data_dir / "X_test.npy")
    print(f"held out {family} | scale {scale} | seed {seed} | config {cfg} ({cfg_src})")
    print(f"fit {X_fit.shape} val {X_val.shape} test {X_test.shape}", flush=True)
    assert meta["held_out_idx"] not in set(np.unique(y_fit)) | set(np.unique(y_val))

    t0 = time.perf_counter()
    trained = banks.train_banks(X_fit, y_fit, families, normal_idx, cfg, seed,
                                verify_X=X_val[:500])
    models = {k: v[int(cfg["epochs"])] for k, v in trained.items()}
    print(f"trained {len(models)} banks in {time.perf_counter()-t0:.0f}s "
          f"(fast scorer verified against tmu)", flush=True)

    t0 = time.perf_counter()
    val_s = banks.score_banks(models, X_val)
    test_s = banks.score_banks(models, X_test)
    print(f"scored val + test in {time.perf_counter()-t0:.0f}s", flush=True)
    val_f1 = banks.closed_macro_f1(val_s, y_val, families, normal_idx)
    print(f"validation known-class macro F1: {val_f1:.4f}", flush=True)

    out = find_dir("data") / f"banks_{scale}_{family.lower()}{sfx}_s{seed}"
    (out / "models").mkdir(parents=True, exist_ok=True)
    np.save(out / "y_val.npy", y_val)
    for k in models:
        np.save(out / f"val_{k}.npy", val_s[k])
        np.save(out / f"test_{k}.npy", test_s[k])
        tm_fast.save_model(models[k], out / "models" / f"{k}.npz")
    with open(out / "banks_meta.json", "w") as f:
        json.dump({"families": families, "normal_idx": normal_idx,
                   "all_class_names": names, "scale": scale, "seed": seed,
                   "config": cfg, "config_source": cfg_src, "strategy": strategy,
                   "held_out_class": meta["held_out_class"],
                   "held_out_idx": meta["held_out_idx"],
                   "unknown_label": meta["unknown_label"],
                   "fit_class_counts_real": meta["fit_class_counts_real"],
                   "val_known_macro_f1": val_f1}, f, indent=2)
    print(f"saved to {out}")

if __name__ == "__main__":
    main()
