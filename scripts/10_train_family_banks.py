"""
Trains Step 5's separate clause banks: one normality branch and one
detector per known attack family.

The normality branch learns benign versus everything else, producing the
benign support N(X). Each attack-family bank learns its own family versus
everything else, producing S_k(X). Every bank is a binary TMClassifier
trained on a balanced sample, so their scores are comparable.

Spec deviation recorded: the normality branch is trained against real
attack rows rather than the spec's self-supervised perturbed windows,
since labelled attack data is available here.

Takes the held-out class name as argv[1] and reads the matching open-set
dataset. Saves per-bank validation and test scores to
data/banks_<scale>_<class>/.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score
from tmu.tsetlin_machine import TMClassifier

SCALE = "medium"
NUMBER_OF_CLAUSES = 2000
T = 200
S = 5.0
EPOCHS = 6
TM_SEED = 42
VAL_FRACTION = 0.10
RANDOM_STATE = 42
NORMAL_NAME = "Normal"

def find_dir(name):
    here = Path(__file__).resolve().parent
    for candidate in (here.parent / name, here / name):
        if candidate.exists():
            return candidate
    return here / name

def predict_margin(tm, X, chunk=50000):
    """Score for the positive class minus score for the negative class.

    A binary TMClassifier keeps one clause bank per class; the difference
    between their vote sums is how strongly this bank claims the row, and
    is directly comparable across banks because every bank is trained on a
    balanced sample with the same T."""
    if not np.array_equal(tm.X_test, X):
        tm.encoded_X_test = tm.clause_banks[0].prepare_X(X)
        tm.X_test = X.copy()
    out = np.zeros(X.shape[0], dtype=np.float64)
    for e in range(X.shape[0]):
        sums = []
        for i in range(tm.number_of_classes):
            cs = np.dot(tm.weight_banks[i].get_weights(),
                        tm.clause_banks[i].calculate_clause_outputs_predict(tm.encoded_X_test, e)
                        ).astype(np.int32)
            sums.append(float(np.clip(cs, -tm.T, tm.T)))
        out[e] = sums[1] - sums[0]
        if chunk and e and e % chunk == 0:
            print(f"      scored {e}/{X.shape[0]}", flush=True)
    return out

def balanced_binary(X, y, positive_labels, rng):
    """All rows matching positive_labels, plus an equal number sampled from
    the rest, so no bank is biased by how common its family happens to be."""
    pos = np.where(np.isin(y, positive_labels))[0]
    neg_pool = np.where(~np.isin(y, positive_labels))[0]
    neg = rng.choice(neg_pool, size=min(len(pos), len(neg_pool)), replace=False)
    idx = np.concatenate([pos, neg])
    rng.shuffle(idx)
    return X[idx], np.isin(y[idx], positive_labels).astype(np.uint32)

def train_bank(name, X_fit, y_fit, X_val, X_test):
    print(f"\n=== bank: {name} ===", flush=True)
    print(f"  train={X_fit.shape} positives={int(y_fit.sum())}", flush=True)
    np.random.seed(TM_SEED)
    tm = TMClassifier(number_of_clauses=NUMBER_OF_CLAUSES, T=T, s=S, weighted_clauses=False)
    for epoch in range(1, EPOCHS + 1):
        t0 = time.perf_counter()
        tm.fit(X_fit, y_fit)
        acc = accuracy_score(y_fit, tm.predict(X_fit))
        print(f"  epoch {epoch}/{EPOCHS} train_acc={acc:.4f} ({time.perf_counter()-t0:.0f}s)", flush=True)
    print(f"  scoring val...", flush=True)
    val_margin = predict_margin(tm, X_val, chunk=None)
    print(f"  scoring test...", flush=True)
    test_margin = predict_margin(tm, X_test)
    return val_margin, test_margin

def main():
    held = sys.argv[1] if len(sys.argv) > 1 else "DoS"
    data_dir = find_dir("data") / f"openset_{SCALE}_{held.lower()}"
    X_all = np.load(data_dir / "X_train.npy")
    y_all = np.load(data_dir / "y_train.npy")
    X_test = np.load(data_dir / "X_test.npy")
    meta = json.load(open(data_dir / "meta.json"))
    names = meta["all_class_names"]
    normal_idx = names.index(NORMAL_NAME)
    known = [c for c in meta["known_classes"]]
    families = [c for c in known if c != normal_idx]
    print(f"known classes: {[names[c] for c in known]}")
    print(f"attack families to model: {[names[c] for c in families]}")
    print(f"held out as unknown: {meta['held_out_class']}")

    rng = np.random.default_rng(RANDOM_STATE)
    perm = rng.permutation(len(X_all))
    n_val = int(len(X_all) * VAL_FRACTION)
    val_idx, fit_idx = perm[:n_val], perm[n_val:]
    X_fit_all, y_fit_all = X_all[fit_idx], y_all[fit_idx]
    X_val, y_val = X_all[val_idx], y_all[val_idx]
    print(f"\nfit={X_fit_all.shape} val={X_val.shape} test={X_test.shape}")

    out = find_dir("data") / f"banks_{SCALE}_{held.lower()}"
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "y_val.npy", y_val)
    np.save(out / "val_idx.npy", val_idx)

    # normality branch: benign versus everything else
    Xb, yb = balanced_binary(X_fit_all, y_fit_all, [normal_idx], rng)
    v, t = train_bank("normality (Normal vs rest)", Xb, yb, X_val, X_test)
    np.save(out / "val_normal.npy", v)
    np.save(out / "test_normal.npy", t)

    # one bank per known attack family
    for c in families:
        Xb, yb = balanced_binary(X_fit_all, y_fit_all, [c], rng)
        v, t = train_bank(f"{names[c]} vs rest", Xb, yb, X_val, X_test)
        np.save(out / f"val_fam{c}.npy", v)
        np.save(out / f"test_fam{c}.npy", t)

    with open(out / "banks_meta.json", "w") as f:
        json.dump({"families": families, "normal_idx": normal_idx,
                   "all_class_names": names, "scale": SCALE,
                   "held_out_class": meta["held_out_class"],
                   "held_out_idx": meta["held_out_idx"],
                   "unknown_label": meta["unknown_label"],
                   "n_real_train_rows": meta["n_real_train_rows"]}, f, indent=2)
    print(f"\nall banks saved to {out}")

if __name__ == "__main__":
    main()
