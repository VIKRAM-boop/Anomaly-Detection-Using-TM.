"""
Extracts the clauses behind individual predictions and measures the quality
of the resulting explanations.

A Tsetlin Machine prediction is a weighted vote over clauses, and every
clause is a conjunction of literals that each name a threshold on a real
feature. An explanation for one row is therefore the set of clauses that
fired on it, written out as those conjunctions.

The methodology asks for explanations to be assessed but does not define the
measures, so these are the definitions used here:

  fidelity    the share of rows where a surrogate built from only the top-k
              highest-weight firing clauses per class reproduces the full
              model's prediction. A faithful explanation should let you
              recover the decision from the clauses alone.

  stability   mean Jaccard overlap between the firing-clause set of a row
              and that of a minimally perturbed copy of the same row (one
              literal flipped, repeated over several draws). An explanation
              that reshuffles completely under a one-bit change is not
              describing anything stable.

  complexity  mean number of included literals per firing clause, plus the
              mean number of clauses firing per row. Short conjunctions and
              few of them are what makes a rule readable.

Takes the scale as argv[1], the sampling strategy as argv[2] and the number
of rows to explain as argv[3]. Trains one TM on the matching dataset, then
reports the three measures and prints a worked example.
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
RANDOM_STATE = 42
TOP_K = [1, 3, 5, 10, 25, 50]      # surrogate sizes for fidelity
N_PERTURB = 5                       # perturbed copies per row for stability
N_EXAMPLE_CLAUSES = 5               # clauses printed in the worked example

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

def clause_literals(tm, cls, clause, n_features):
    """Literals included in one clause, as (feature_index, negated) pairs.

    A TM clause holds two automata per feature -- one for the plain literal
    and one for its negation -- and a literal is part of the conjunction
    when its automaton is in an including state."""
    out = []
    bank = tm.clause_banks[cls]
    for f in range(n_features):
        if bank.get_ta_action(clause, f):
            out.append((f, False))
        if bank.get_ta_action(clause, f + n_features):
            out.append((f, True))
    return out

def firing_clauses(tm, encoded_X, e, cls):
    """Indices of clauses of one class that evaluate true on example e."""
    outputs = tm.clause_banks[cls].calculate_clause_outputs_predict(encoded_X, e)
    return np.where(np.asarray(outputs) > 0)[0]

def class_sums(tm, encoded_X, e):
    sums = np.zeros(tm.number_of_classes)
    for i in range(tm.number_of_classes):
        w = tm.weight_banks[i].get_weights()
        o = tm.clause_banks[i].calculate_clause_outputs_predict(encoded_X, e)
        sums[i] = np.clip(np.dot(w, o).astype(np.int32), -tm.T, tm.T)
    return sums

def topk_sums(tm, encoded_X, e, k):
    """Class sums restricted to each class's k highest-weight firing clauses."""
    sums = np.zeros(tm.number_of_classes)
    for i in range(tm.number_of_classes):
        w = np.asarray(tm.weight_banks[i].get_weights())
        o = np.asarray(tm.clause_banks[i].calculate_clause_outputs_predict(encoded_X, e))
        fired = np.where(o > 0)[0]
        if len(fired) == 0:
            continue
        top = fired[np.argsort(-w[fired])[:k]]
        sums[i] = w[top].sum()
    return sums

def main():
    scale = sys.argv[1] if len(sys.argv) > 1 else "medium"
    strategy = sys.argv[2] if len(sys.argv) > 2 else "smote"
    n_rows = int(sys.argv[3]) if len(sys.argv) > 3 else 300

    suffix = "" if strategy == "smote" else f"_{strategy}"
    d = find_dir("data") / f"temporal_{scale}{suffix}"
    X_fit = np.load(d / "X_fit.npy")
    y_fit = np.load(d / "y_fit.npy")
    X_test = np.load(d / "X_test.npy")
    y_test = np.load(d / "y_test.npy")
    meta = json.load(open(d / "meta.json"))
    names = meta["class_names"]
    feature_names = meta["feature_names"]
    n_features = X_fit.shape[1]

    print(f"scale={scale} strategy={strategy} | fit={X_fit.shape} test={X_test.shape}")
    np.random.seed(TM_SEED)
    tm = TMClassifier(number_of_clauses=NUMBER_OF_CLAUSES, T=T, s=S, weighted_clauses=False)
    for epoch in range(1, EPOCHS + 1):
        t0 = time.perf_counter()
        tm.fit(X_fit, y_fit)
        print(f"  epoch {epoch}/{EPOCHS} ({time.perf_counter()-t0:.0f}s)", flush=True)

    rng = np.random.default_rng(RANDOM_STATE)
    rows = rng.choice(len(X_test), size=min(n_rows, len(X_test)), replace=False)
    X = np.ascontiguousarray(X_test[rows])
    y = y_test[rows]

    tm.encoded_X_test = tm.clause_banks[0].prepare_X(X)
    tm.X_test = X.copy()
    enc = tm.encoded_X_test

    # --- full-model predictions on the sample ---
    full_pred = np.array([class_sums(tm, enc, e).argmax() for e in range(len(X))])
    print(f"\nsample of {len(X)} test rows, full-model accuracy "
          f"{accuracy_score(y, full_pred):.4f}")

    # --- complexity ---
    n_fired, lits_per_clause = [], []
    for e in range(len(X)):
        c = full_pred[e]
        fired = firing_clauses(tm, enc, e, c)
        n_fired.append(len(fired))
        # number_of_include_actions is one C call per clause; walking the
        # automata in Python would be 2 * n_features calls instead
        bank = tm.clause_banks[c]
        for cl in fired[:20]:
            lits_per_clause.append(int(bank.number_of_include_actions(int(cl))))
    print("\n=== complexity ===")
    print(f"  clauses firing per row (predicted class): mean {np.mean(n_fired):.1f}, "
          f"median {np.median(n_fired):.0f}, max {np.max(n_fired)}")
    print(f"  literals per firing clause: mean {np.mean(lits_per_clause):.1f}, "
          f"median {np.median(lits_per_clause):.0f}, max {np.max(lits_per_clause)}")

    # --- fidelity ---
    print("\n=== fidelity (surrogate from top-k firing clauses per class) ===")
    fid = {}
    for k in TOP_K:
        agree = np.array([topk_sums(tm, enc, e, k).argmax() for e in range(len(X))])
        fid[k] = float((agree == full_pred).mean())
        print(f"  top-{k:<3d} reproduces the full prediction on {fid[k]:.1%} of rows")

    # --- stability ---
    print("\n=== stability (one literal flipped, "
          f"{N_PERTURB} draws per row) ===")
    jac = []
    sub = rows[:min(60, len(rows))]          # perturbation needs a re-encode per draw
    for i in range(len(sub)):
        base_row = X[i:i+1]
        c = full_pred[i]
        base_fired = set(firing_clauses(tm, enc, i, c).tolist())
        for _ in range(N_PERTURB):
            pert = base_row.copy()
            j = rng.integers(0, n_features)
            pert[0, j] ^= 1
            tm.encoded_X_test = tm.clause_banks[0].prepare_X(pert)
            tm.X_test = pert.copy()
            f2 = set(firing_clauses(tm, tm.encoded_X_test, 0, c).tolist())
            union = base_fired | f2
            jac.append(len(base_fired & f2) / len(union) if union else 1.0)
        tm.encoded_X_test, tm.X_test = enc, X.copy()
    print(f"  mean Jaccard overlap of firing-clause sets: {np.mean(jac):.4f} "
          f"(1.0 = unchanged)")
    print(f"  rows sampled: {len(sub)}, comparisons: {len(jac)}")

    # --- worked example ---
    print(f"\n=== worked example ===")
    e = 0
    c = int(full_pred[e])
    fired = firing_clauses(tm, enc, e, c)
    w = np.asarray(tm.weight_banks[c].get_weights())
    order = fired[np.argsort(-w[fired])][:N_EXAMPLE_CLAUSES]
    print(f"row {rows[e]}: true={names[y[e]]}  predicted={names[c]}  "
          f"{len(fired)} clauses firing")
    for cl in order:
        lits = clause_literals(tm, c, int(cl), n_features)
        txt = " AND ".join(("NOT " if neg else "") + feature_names[f] for f, neg in lits[:8])
        more = "" if len(lits) <= 8 else f" ... (+{len(lits)-8} literals)"
        print(f"  clause {int(cl):4d} w={w[cl]:>4d}: {txt}{more}")

    out = find_dir("data") / f"explain_{scale}{suffix}.json"
    with open(out, "w") as f:
        json.dump({"scale": scale, "strategy": strategy, "n_rows": int(len(X)),
                   "complexity": {
                       "clauses_per_row_mean": float(np.mean(n_fired)),
                       "clauses_per_row_median": float(np.median(n_fired)),
                       "literals_per_clause_mean": float(np.mean(lits_per_clause)),
                       "literals_per_clause_median": float(np.median(lits_per_clause))},
                   "fidelity": fid,
                   "stability_jaccard": float(np.mean(jac))}, f, indent=2)
    print(f"\nwritten to {out}")

if __name__ == "__main__":
    main()
