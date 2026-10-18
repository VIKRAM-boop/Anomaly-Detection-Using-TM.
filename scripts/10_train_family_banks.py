"""
Trains Step 5's separate clause banks: one normality branch and one
detector per known attack family.

The normality branch learns benign versus everything else, producing the
benign support N(X). Each attack-family bank learns its own family versus
everything else, producing S_k(X). Every bank is a binary TMClassifier
trained on a balanced sample, so their scores are comparable.

Step 5 asks for the normality branch to be trained "primarily on benign
windows and self-supervised perturbed windows", so three modes are
available via argv[3]:

  supervised  benign vs real attack rows (what was used previously)
  selfsup     benign vs perturbed copies of benign rows, no attack labels
  combined    benign vs both

Perturbations respect the literal encoding. A feature-level perturbation
re-draws a feature's quantile level, keeping the five monotone threshold
literals consistent; a temporal perturbation transplants another row's
window-literal block, severing the link between a flow and its temporal
context. Flipping literals at random would instead produce rows that cannot
occur (x > Q90 while x <= Q10) and teach the branch nothing useful.

Takes the held-out class as argv[1], the temporal scale as argv[2] and the
normality mode as argv[3]. Saves per-bank validation and test scores to
data/banks_<scale>_<class>[_<mode>]/.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score
from tmu.tsetlin_machine import TMClassifier

NUMBER_OF_CLAUSES = 2000
T = 200
S = 5.0
EPOCHS = 1
TM_SEED = 42
VAL_FRACTION = 0.10
RANDOM_STATE = 42
NORMAL_NAME = "Normal"
QUANTILE_LEVELS = 5          # literals per continuous feature, monotone
PERTURB_RATIO = 1.0          # perturbed rows generated per benign row
TEMPORAL_SHARE = 0.5         # share of perturbations that are temporal
FEATURES_PER_ROW = 0.15      # share of a row's features perturbed

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

def quantile_groups(feature_names):
    """Index groups for the five monotone threshold literals of each feature.

    A continuous feature j contributes [j>Q10, j>Q25, j>Q50, j>Q75, j>Q90],
    which are monotone: setting a higher threshold implies all lower ones.
    Perturbing within a group means moving the crossing point, which keeps
    the row realisable."""
    groups, current, base = [], [], None
    for i, n in enumerate(feature_names):
        if ">" not in n:
            continue
        stem = n.rsplit(">", 1)[0]
        if stem != base:
            if len(current) == QUANTILE_LEVELS:
                groups.append(current)
            current, base = [], stem
        current.append(i)
    if len(current) == QUANTILE_LEVELS:
        groups.append(current)
    return groups

def make_pseudo_anomalies(X_benign, feature_names, scale, rng):
    """Perturbed copies of benign rows, used as negatives for the
    self-supervised normality branch."""
    n = int(len(X_benign) * PERTURB_RATIO)
    src = rng.choice(len(X_benign), size=n, replace=n > len(X_benign))
    out = X_benign[src].copy()

    groups = quantile_groups(feature_names)
    window_cols = np.array([i for i, nm in enumerate(feature_names)
                            if nm.startswith(f"{scale}_")], dtype=np.int64)
    n_temporal = int(n * TEMPORAL_SHARE)

    # temporal: transplant a different row's window block, so the flow no
    # longer matches the traffic context it sits in
    if n_temporal and len(window_cols):
        donor = rng.choice(len(X_benign), size=n_temporal, replace=True)
        out[np.arange(n_temporal)[:, None], window_cols] = \
            X_benign[donor][:, window_cols]

    # feature-level: re-draw the quantile level of a few features per row
    if groups:
        per_row = max(1, int(len(groups) * FEATURES_PER_ROW))
        for r in range(n_temporal, n):
            for g in rng.choice(len(groups), size=per_row, replace=False):
                idx = groups[g]
                level = rng.integers(0, QUANTILE_LEVELS + 1)   # 0..5 thresholds passed
                out[r, idx] = 0
                if level:
                    out[r, idx[:level]] = 1
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
    scale = sys.argv[2] if len(sys.argv) > 2 else "medium"
    mode = sys.argv[3] if len(sys.argv) > 3 else "supervised"
    if mode not in ("supervised", "selfsup", "combined"):
        raise SystemExit(f"normality mode must be supervised/selfsup/combined, got {mode!r}")
    data_dir = find_dir("data") / f"openset_{scale}_{held.lower()}"
    X_test = np.load(data_dir / "X_test.npy")
    meta = json.load(open(data_dir / "meta.json"))
    names = meta["all_class_names"]
    feature_names = meta["feature_names"]
    normal_idx = names.index(NORMAL_NAME)
    known = [c for c in meta["known_classes"]]
    families = [c for c in known if c != normal_idx]
    print(f"known classes: {[names[c] for c in known]}")
    print(f"attack families to model: {[names[c] for c in families]}")
    print(f"held out as unknown: {meta['held_out_class']}")
    print(f"scale={scale}  normality mode={mode}")

    rng = np.random.default_rng(RANDOM_STATE)
    X_fit_all = np.load(data_dir / "X_fit.npy")
    y_fit_all = np.load(data_dir / "y_fit.npy")
    X_val = np.load(data_dir / "X_val.npy")
    y_val = np.load(data_dir / "y_val.npy")
    print(f"\nfit={X_fit_all.shape} val={X_val.shape} test={X_test.shape}")

    suffix = "" if mode == "supervised" else f"_{mode}"
    out = find_dir("data") / f"banks_{scale}_{held.lower()}{suffix}"
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "y_val.npy", y_val)

    # The family banks do not depend on the normality mode, so when another
    # mode has already trained them for this scale and hold-out they are
    # reused rather than retrained.
    share_from = None
    for other in ("", "_selfsup", "_combined"):
        cand = find_dir("data") / f"banks_{scale}_{held.lower()}{other}"
        if cand != out and (cand / f"test_fam{families[0]}.npy").exists():
            share_from = cand
            break
    if share_from:
        print(f"reusing family banks from {share_from.name}")

    # normality branch, per Step 5
    X_benign = X_fit_all[y_fit_all == normal_idx]
    if mode == "supervised":
        Xb, yb = balanced_binary(X_fit_all, y_fit_all, [normal_idx], rng)
    else:
        pseudo = make_pseudo_anomalies(X_benign, feature_names, scale, rng)
        print(f"  generated {len(pseudo)} pseudo-anomalies from {len(X_benign)} benign rows")
        if mode == "selfsup":
            neg = pseudo
        else:
            attacks = X_fit_all[y_fit_all != normal_idx]
            take = min(len(attacks), len(pseudo))
            neg = np.concatenate([pseudo[:take],
                                   attacks[rng.choice(len(attacks), take, replace=False)]])
        k = min(len(X_benign), len(neg))
        Xb = np.concatenate([X_benign[rng.choice(len(X_benign), k, replace=False)],
                              neg[rng.choice(len(neg), k, replace=False)]])
        yb = np.concatenate([np.ones(k, dtype=np.uint32), np.zeros(k, dtype=np.uint32)])
        shuf = rng.permutation(len(Xb))
        Xb, yb = Xb[shuf], yb[shuf]
    v, t = train_bank(f"normality ({mode})", Xb, yb, X_val, X_test)
    np.save(out / "val_normal.npy", v)
    np.save(out / "test_normal.npy", t)

    # one bank per known attack family
    for c in families:
        if share_from and (share_from / f"test_fam{c}.npy").exists():
            np.save(out / f"val_fam{c}.npy", np.load(share_from / f"val_fam{c}.npy"))
            np.save(out / f"test_fam{c}.npy", np.load(share_from / f"test_fam{c}.npy"))
            continue
        Xb, yb = balanced_binary(X_fit_all, y_fit_all, [c], rng)
        v, t = train_bank(f"{names[c]} vs rest", Xb, yb, X_val, X_test)
        np.save(out / f"val_fam{c}.npy", v)
        np.save(out / f"test_fam{c}.npy", t)

    with open(out / "banks_meta.json", "w") as f:
        json.dump({"families": families, "normal_idx": normal_idx,
                   "all_class_names": names, "scale": scale,
                   "normality_mode": mode,
                   "held_out_class": meta["held_out_class"],
                   "held_out_idx": meta["held_out_idx"],
                   "unknown_label": meta["unknown_label"],
                   "n_fit_real": meta["n_fit_real"],
                   "fit_class_counts_real": meta["fit_class_counts_real"]}, f, indent=2)
    print(f"\nall banks saved to {out}")

if __name__ == "__main__":
    main()
