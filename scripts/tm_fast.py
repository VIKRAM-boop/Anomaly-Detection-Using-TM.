"""
Fast, tmu-independent evaluation of trained Tsetlin Machine clause banks.

tmu scores one example at a time through a Python loop, which made scoring
roughly two thirds of every run's cost. A trained clause is just a set of
included literals, and it fires when every included literal is true, so a
whole matrix of examples can be scored at once:

    violations = |plain includes| + x @ (negated - plain includes)
    fires      = (violations == 0) & (clause has at least one include)

The product is a dense float32 matrix multiply, whose cost does not depend
on how many literals each clause includes.

extract_tm() pulls each class's include mask and clause weights out of a
trained TMClassifier into plain numpy arrays. Those arrays are the whole
model as far as prediction is concerned, so they can be saved and reused
for scoring and for clause-level explanations without tmu.

verify_against_tmu() checks clause outputs and class sums against tmu's own
prediction path; every script that uses this module calls it once.
"""
import numpy as np

def extract_tm(tm):
    """Returns {"T", "n_features", "include": [bool (C, 2F)], "weights": [int (C,)]}."""
    banks = tm.clause_banks
    n_features = banks[0].number_of_features
    out = {"T": int(tm.T), "n_features": int(n_features), "include": [], "weights": []}
    for i in range(tm.number_of_classes):
        cb = banks[i]
        state = cb.clause_bank.reshape(cb.number_of_clauses, cb.number_of_ta_chunks,
                                       cb.number_of_state_bits_ta)
        action = state[:, :, -1]                                    # top state bit = include
        bits = (action[:, :, None] >> np.arange(32, dtype=np.uint32)) & 1
        bits = bits.reshape(cb.number_of_clauses, -1)[:, :2 * n_features].astype(bool)
        out["include"].append(bits)
        out["weights"].append(np.asarray(tm.weight_banks[i].get_weights(), dtype=np.int64).copy())
    return out

def _violation_operator(include):
    """Precompute (D, npos) so that violations = npos + x @ D.

    A clause includes plain literals P and negated literals N. A plain
    literal is violated when x = 0 and a negated one when x = 1, so

        violations = sum_P (1 - x) + sum_N x = |P| + x @ (N - P).

    One product over the F features instead of the 2F literals halves the
    work, and every term is a small integer, so float32 is exact."""
    F = include.shape[1] // 2
    pos = include[:, :F].astype(np.float32)
    neg = include[:, F:].astype(np.float32)
    D = np.ascontiguousarray((neg - pos).T)                        # (F, C)
    npos = pos.sum(axis=1)                                         # (C,)
    return D, npos

def _ops(model, cls):
    """Cached (D, npos, nonempty) for one class; rebuilt only once per model."""
    cache = model.setdefault("_ops", {})
    if cls not in cache:
        inc = model["include"][cls]
        D, npos = _violation_operator(inc)
        cache[cls] = (D, npos, inc.any(axis=1))
    return cache[cls]

def clause_outputs(model, X, cls, chunk=20000):
    """Boolean (N, C) firing matrix for one class's clauses.

    Dense BLAS rather than a sparse product: after short training clauses
    include around 40 literals on average, which makes a sparse product far
    slower, while the dense product's cost does not depend on that."""
    D, npos, nonempty = _ops(model, cls)
    out = np.empty((X.shape[0], D.shape[1]), dtype=bool)
    for s in range(0, X.shape[0], chunk):
        x = np.asarray(X[s:s + chunk], dtype=np.float32)
        out[s:s + chunk] = ((x @ D) == -npos) & nonempty     # exact: small integers
    return out

def class_sums(model, X, chunk=20000):
    """(N, n_classes) clipped vote sums, matching TMClassifier.predict's scores."""
    T = model["T"]
    n_cls = len(model["include"])
    sums = np.empty((X.shape[0], n_cls), dtype=np.int64)
    ops = [_ops(model, i)[:2] for i in range(n_cls)]
    nonempty = [_ops(model, i)[2] for i in range(n_cls)]
    ws = [w.astype(np.float32) for w in model["weights"]]
    for s in range(0, X.shape[0], chunk):
        x = np.asarray(X[s:s + chunk], dtype=np.float32)
        for i in range(n_cls):
            D, npos = ops[i]
            fires = ((x @ D) == -npos) & nonempty[i]          # exact: small integers
            sums[s:s + chunk, i] = np.clip(np.rint(fires.astype(np.float32) @ ws[i]), -T, T)
    return sums

def binary_margin(model, X, chunk=50000):
    """Positive-class minus negative-class score for a two-class bank."""
    s = class_sums(model, X, chunk)
    return (s[:, 1] - s[:, 0]).astype(np.float64)

def save_model(model, path):
    np.savez_compressed(path, T=model["T"], n_features=model["n_features"],
                        n_classes=len(model["include"]),
                        **{f"inc{i}": np.packbits(m, axis=1) for i, m in enumerate(model["include"])},
                        **{f"inc_shape{i}": np.array(m.shape) for i, m in enumerate(model["include"])},
                        **{f"w{i}": w for i, w in enumerate(model["weights"])})

def load_model(path):
    z = np.load(path)
    n = int(z["n_classes"])
    inc = []
    for i in range(n):
        shape = tuple(z[f"inc_shape{i}"])
        inc.append(np.unpackbits(z[f"inc{i}"], axis=1)[:, :shape[1]].astype(bool))
    return {"T": int(z["T"]), "n_features": int(z["n_features"]),
            "include": inc, "weights": [z[f"w{i}"] for i in range(n)]}

def verify_against_tmu(tm, X, n=2000):
    """Asserts clause outputs and class sums match tmu on n rows."""
    X = np.ascontiguousarray(X[:n]).astype(np.uint32)
    model = extract_tm(tm)
    enc = tm.clause_banks[0].prepare_X(X)
    for i in range(tm.number_of_classes):
        # tmu's incremental evaluator keeps state between calls; a fresh
        # bank evaluation per class avoids carrying it across classes
        ref = np.array([np.asarray(tm.clause_banks[i].calculate_clause_outputs_predict(enc, e)).copy()
                        for e in range(len(X))]) > 0
        mine = clause_outputs(model, X, i)
        mism = int((ref != mine).sum())
        assert mism == 0, f"class {i}: {mism} clause outputs differ from tmu"
    ref_pred = tm.predict(X)
    my_pred = class_sums(model, X).argmax(axis=1)
    agree = float((ref_pred == my_pred).mean())
    assert agree == 1.0, f"argmax agreement with tmu only {agree:.4f}"
    return model
