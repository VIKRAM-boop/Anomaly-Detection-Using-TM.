"""
Phase 5, Task 4: TM explanations against SHAP for RF and XGBoost, under
one set of criteria.

Every model is wrapped the same way: a class-score function S(X) (n, C) in
the model's own additive space, and per-row, per-class feature attributions
A(X) (n, C, G) with a base value b (C,) such that S is approximately b + sum_g A.

  TM       S = clipped class sums (tm_fast). A clause fires or not, so a
           clause's weight is exact; at feature level each firing clause's
           weight is shared equally among its included literals and literals
           are summed per original feature. b = 0; the only gap between S and
           b + sum A is the clip at +/-T.
  RF       S = class probability, A = TreeSHAP (path-dependent; exact for
           the forest's averaged probability).
  XGBoost  S = class log-odds (raw margin), A = XGBoost's own TreeSHAP
           (pred_contribs, path-dependent as for RF); exact in that space.
           Its rejection here is the global rule on the top class's
           log-odds (5% budget), not the probability rule of Task 2:
           probabilities are not additive in the trees' contributions.

Features are compared at the level of original features: a quantile ladder
or a one-hot group is one feature, as is each relational literal (Boolean
input only).

Two explanation targets, kept apart:

  known    why class k was predicted: q = S_k - S_runner-up. q > 0 is the
           decision.
  unknown  why the row was rejected under the global rule: q = S_top - theta,
           theta calibrated on known validation rows at a 5% budget. q < 0
           is the rejection; attributions say which features failed to build
           support for even the best class.

Rows (test, held-out fold) are grouped by the model's own decision:
known_correct, known_rejected (a known row rejected as Unknown),
unknown_rejected (held-out row caught) and unknown_missed (held-out row
given a known label). Measures per group:

  completeness    |q - (b_q + sum_g A_q)| / |q|
  fidelity@k      share of rows where b_q + the k largest-|A_q| features
                  alone reproduces the sign of q (the decision); for the TM
                  also with the k largest clauses instead of features
  sparsity        smallest k reproducing the decision (features; TM clauses
                  too), and the number of features carrying 90% of sum |A_q|
  faithfulness    the k features most supporting the decision are reset to
                  a reference value (the fit set's most common value of that
                  feature), and the model is re-run: the share of decisions
                  that flip, against k random features
  stability       (a) seeds: Jaccard overlap of the top-5 features for the
                  same row between models trained with different seeds;
                  (b) perturbation: one random feature reset to its
                  reference, top-5 overlap with the original explanation and
                  the share of decisions that change

Writes <data>/phase5/explain_compare.json.

Usage: 32_explain_compare.py [--folds dos,fuzzers,reconnaissance] [--models tm,rf_boolean,...]
"""
import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import tm_fast

SEEDS = [7, 42, 1234]
REF_SEED = 42
PER_GROUP = 200
STAB_ROWS = 300
K_VALUES = [1, 3, 5, 10]
TOP = 5
BACKGROUND = 100
BUDGET = 0.05
TIE = 1e-4
UNK = 99
GROUPS_ORDER = ["known_correct", "known_rejected", "unknown_rejected", "unknown_missed"]

def find_dir(name):
    if name == "data":
        o = os.environ.get("TM_DATA_DIR")
        if o:
            return Path(o)
    for c in (HERE.parent / name, HERE / name):
        if c.exists():
            return c
    return HERE / name

def load_module(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

def group_of(name):
    if ">Q" in name:
        return name.split(">Q")[0]
    for p in ("proto_", "service_", "state_"):
        if name.startswith(p):
            return p[:-1]
    return name

# ---------------- models ----------------

class TMModel:
    def __init__(self, path, meta, names):
        self.m = tm_fast.load_model(path)
        self.classes = list(meta["classes"])
        self.T = self.m["T"]
        F = self.m["n_features"]
        self.groups = sorted({group_of(n) for n in names})
        gi = {g: i for i, g in enumerate(self.groups)}
        M = np.zeros((2 * F, len(self.groups)), np.float32)
        for f, n in enumerate(names):
            M[f, gi[group_of(n)]] = 1
            M[F + f, gi[group_of(n)]] = 1
        self.M = M
    def scores(self, X):
        return tm_fast.class_sums(self.m, X).astype(np.float64)
    def clause_contrib(self, X, c):
        """(n, J) signed clause contributions to class c's (unclipped) sum."""
        f = tm_fast.clause_outputs(self.m, X, c)
        return f * self.m["weights"][c][None, :].astype(np.float64)
    def attributions(self, X):
        out = np.zeros((len(X), len(self.classes), len(self.groups)))
        for c in range(len(self.classes)):
            inc = self.m["include"][c].astype(np.float32)
            size = np.maximum(inc.sum(1), 1)
            share = self.clause_contrib(X, c) / size[None, :]
            out[:, c] = (share.astype(np.float32) @ inc) @ self.M
        return out, np.zeros(len(self.classes))

class TreeModel:
    def __init__(self, model, kind, classes, groups_of_cols, Xbg):
        import shap
        self.mdl, self.kind, self.classes = model, kind, list(classes)
        self.groups = sorted(set(groups_of_cols))
        gi = {g: i for i, g in enumerate(self.groups)}
        self.M = np.zeros((len(groups_of_cols), len(self.groups)), np.float32)
        for j, g in enumerate(groups_of_cols):
            self.M[j, gi[g]] = 1
        if kind == "rf":
            self.ex = shap.TreeExplainer(model)
    def scores(self, X):
        if self.kind == "xgb":
            import xgboost
            return self.mdl.get_booster().predict(xgboost.DMatrix(X), output_margin=True).astype(np.float64)
        return self.mdl.predict_proba(X).astype(np.float64)
    def attributions(self, X):
        if self.kind == "xgb":
            # XGBoost's own TreeSHAP (path-dependent, as for RF), log-odds space
            import xgboost
            c = self.mdl.get_booster().predict(xgboost.DMatrix(X), pred_contribs=True)  # (n, C, F+1)
            return c[:, :, :-1] @ self.M, c[0, :, -1].astype(np.float64)
        sv = self.ex.shap_values(X, check_additivity=False)
        sv = np.stack(sv, 1) if isinstance(sv, list) else np.transpose(sv, (0, 2, 1))  # (n, C, F)
        base = np.asarray(self.ex.expected_value, dtype=np.float64).reshape(-1)
        return sv @ self.M, base

# ---------------- decisions and targets ----------------

def decide(S, theta, classes):
    top = S.max(1)
    pred = np.asarray(classes)[S.argmax(1)].astype(np.int64)
    pred[top < theta] = UNK
    return pred

def target(S, A, base, theta, rejected):
    """q, its attribution (n, G) and base, per row: known -> S_k - S_runner,
    rejected -> S_top - theta."""
    o = np.argsort(S, 1)
    i1, i2 = o[:, -1], o[:, -2]
    r = np.arange(len(S))
    q = np.where(rejected, S[r, i1] - theta, S[r, i1] - S[r, i2])
    Aq = np.where(rejected[:, None], A[r, i1], A[r, i1] - A[r, i2])
    bq = np.where(rejected, base[i1] - theta, base[i1] - base[i2])
    return q, Aq, bq, i1, i2

def fidelity_sparsity(q, Aq, bq):
    """Rows with |q| < TIE (settled by the tie-break jitter: the TM's integer
    scores often sit exactly on its integer threshold) have no sign to
    reproduce; they are reported as a share and left out."""
    tie = np.abs(q) < TIE
    tie_share = float(tie.mean())
    q, Aq, bq = q[~tie], Aq[~tie], bq[~tie]
    if not len(q):
        return {"tie_share": tie_share}
    order = np.argsort(-np.abs(Aq), 1)
    cs = np.cumsum(np.take_along_axis(Aq, order, 1), 1) + bq[:, None]
    ok = np.sign(cs) == np.sign(q)[:, None]
    fid = {k: float(ok[:, min(k, ok.shape[1]) - 1].mean()) for k in K_VALUES}
    # smallest k from which the reconstruction stays on the right side
    stays = np.flip(np.cumprod(np.flip(ok, 1), 1), 1).astype(bool)
    kmin = np.where(stays.any(1), stays.argmax(1) + 1, ok.shape[1] + 1)
    mass = np.cumsum(np.take_along_axis(np.abs(Aq), order, 1), 1)
    k90 = (mass < 0.9 * mass[:, -1:]).sum(1) + 1
    comp = np.abs(q - (bq + Aq.sum(1))) / np.maximum(np.abs(q), 1e-9)
    return {"tie_share": tie_share, "fidelity_at_k": fid, "k_min_median": float(np.median(kmin)),
            "k_min_mean": float(kmin.mean()), "k90_median": float(np.median(k90)),
            "completeness_err_median": float(np.median(comp)),
            "completeness_exact_share": float((comp < 1e-6).mean())}

def tm_clause_fidelity(tm, X, q, i1, i2, rejected, theta):
    """The TM's native unit: the k largest firing clauses of the classes
    involved, everything else dropped (unclipped sums)."""
    keep = np.abs(q) >= TIE
    X, q, i1, i2, rejected = X[keep], q[keep], i1[keep], i2[keep], rejected[keep]
    n = len(X)
    out_ok = {k: np.zeros(n, bool) for k in K_VALUES}
    kmin, lits = np.zeros(n), np.zeros(n)
    by_c = {}
    for c in np.unique(np.r_[i1, i2]):
        by_c[c] = tm.clause_contrib(X, c)
    for r in range(n):
        a = by_c[i1[r]][r]
        sizes = tm.m["include"][i1[r]].sum(1)
        if rejected[r]:
            contrib, sz, const = a, sizes, -theta
        else:
            b = by_c[i2[r]][r]
            contrib = np.r_[a, -b]
            sz = np.r_[sizes, tm.m["include"][i2[r]].sum(1)]
            const = 0.0
        nz = np.flatnonzero(contrib)
        order = nz[np.argsort(-np.abs(contrib[nz]))]
        cs = np.cumsum(contrib[order]) + const
        good = np.sign(cs) == np.sign(q[r])
        for k in K_VALUES:
            out_ok[k][r] = good[min(k, len(good)) - 1] if len(good) else np.sign(const) == np.sign(q[r])
        if len(good):
            stays = np.flip(np.cumprod(np.flip(good)))
            km = int(stays.argmax()) + 1 if stays.any() else len(good) + 1
        else:
            km = 0
        kmin[r] = km
        lits[r] = float(sz[order[:km]].mean()) if km and km <= len(order) else 0.0
    return {"clause_fidelity_at_k": {k: float(v.mean()) for k, v in out_ok.items()},
            "clauses_min_median": float(np.median(kmin)),
            "literals_per_cited_clause_mean": float(lits[kmin > 0].mean()) if (kmin > 0).any() else 0.0}

def reset(X, groups_idx, ref, cols_of_group):
    X2 = X.copy()
    for r, gs in enumerate(groups_idx):
        for g in gs:
            c = cols_of_group[g]
            X2[r, c] = ref[c]
    return X2

# ---------------- per fold ----------------

def tree_fold_data(fold, rep, data):
    if rep == "boolean":
        d = data / f"openset_medium_{fold}_none"
        meta = json.load(open(d / "meta.json"))
        Xf = np.load(d / "X_fit.npy").astype(np.float32)
        return (Xf, np.load(d / "y_fit.npy").astype(np.int64), np.load(d / "X_val.npy").astype(np.float32),
                np.load(d / "X_test.npy").astype(np.float32), np.load(d / "y_test.npy").astype(np.int64),
                [group_of(n) for n in meta["feature_names"]])
    os.environ["BASELINE_REPR"] = "continuous_medium"
    b = load_module("bl", "20_openset_baselines.py")
    import pandas as pd
    tr_all = pd.read_parquet(data / "temporal" / "train_windowed.parquet")
    te_all = pd.read_parquet(data / "temporal" / "test_windowed.parquet")
    excl = data / f"temporal_excl_{fold}"
    tr_f = pd.read_parquet(excl / "train_windowed.parquet") if excl.exists() else tr_all
    te_f = pd.read_parquet(excl / "test_windowed.parquet") if excl.exists() else te_all
    for f in (tr_all, te_all, tr_f, te_f):
        f["attack_cat"] = f["attack_cat"].replace({"Backdoors": "Backdoor"})
    name = {f.lower(): f for f in b.FAMILIES}[fold]
    cols = [c for c in tr_f.columns if c not in b.DROP + ["attack_cat"]
            and not c.startswith(("short_", "long_"))]
    Xf, yf, Xv, yv, Xt, yt, names, ni, _ = b.load_partitions(name, tr_f, te_f, tr_all)
    Xc, yc = b.cap(Xf, yf, np.random.default_rng(42))
    return Xc, yc, Xv, Xt, yt, cols

def build_models(kind, fold, data, seed):
    """-> (model, X_val, X_test, y_test, fit X for references/background)."""
    if kind == "tm":
        d = data / f"single_medium_{fold}_s{seed}"
        src = data / f"openset_medium_{fold}"
        meta = json.load(open(d / "meta.json"))
        names = json.load(open(src / "meta.json"))["feature_names"]
        m = TMModel(d / "model.npz", meta, names)
        return m, np.load(src / "X_val.npy"), np.load(src / "X_test.npy"), \
            np.load(src / "y_test.npy").astype(np.int64), np.load(src / "X_fit.npy", mmap_mode="r"), \
            [group_of(n) for n in names]
    model, rep = kind.split("_", 1)
    rep = "continuous_medium" if rep == "continuous" else rep
    key = (fold, rep)
    if key not in _CACHE:
        _CACHE.clear()
        _CACHE[key] = tree_fold_data(fold, rep, data)
    Xf, yf, Xv, Xt, yt, gcols = _CACHE[key]
    p = data / "phase3" / f"baselines_openset_{rep}.json"
    sel = json.load(open(p))[fold]["selected"]
    classes = sorted(set(np.unique(yf).tolist()))
    enc = {c: i for i, c in enumerate(classes)}
    y = np.array([enc[c] for c in yf])
    if model == "rf":
        from sklearn.ensemble import RandomForestClassifier
        mdl = RandomForestClassifier(random_state=seed, n_jobs=8, **sel["rf"]).fit(Xf, y)
    else:
        from xgboost import XGBClassifier
        w = len(y) / (len(classes) * np.bincount(y))
        mdl = XGBClassifier(random_state=seed, n_jobs=8, tree_method="hist", eval_metric="mlogloss",
                            **sel["xgb"]).fit(Xf, y, sample_weight=w[y])
    bg = Xf[np.random.default_rng(0).choice(len(Xf), BACKGROUND, replace=False)]
    tm_like = TreeModel(mdl, model, classes, [group_of(c) for c in gcols], bg)
    return tm_like, Xv, Xt, yt, Xf, [group_of(c) for c in gcols]

_CACHE = {}

def reference_values(Xfit, gcols, groups):
    """Per column: the value of the fit set's most common pattern of that
    column's feature (mode over rows of the feature's columns)."""
    Xs = np.asarray(Xfit[np.random.default_rng(1).choice(len(Xfit), min(50000, len(Xfit)), replace=False)])
    ref = np.zeros(Xs.shape[1], dtype=Xs.dtype)
    cols_of = {}
    for j, g in enumerate(gcols):
        cols_of.setdefault(g, []).append(j)
    for g, cs in cols_of.items():
        pats, cnt = np.unique(Xs[:, cs], axis=0, return_counts=True)
        ref[cs] = pats[cnt.argmax()]
    return ref, [cols_of[g] for g in groups]

def explain_fold(kind, fold, data, rng):
    t0 = time.perf_counter()
    m, Xv, Xt, yt, Xfit, gcols = build_models(kind, fold, data, REF_SEED)
    Sv = m.scores(Xv)
    tops = np.sort(Sv.max(1) + rng.uniform(0, 1e-6, len(Sv)))
    theta = float(tops[int(round(BUDGET * len(tops)))])     # 5% of known validation rows below
    St = m.scores(Xt)
    pred = decide(St, theta, m.classes)
    known = yt != UNK
    grp = np.full(len(yt), "", dtype=object)
    grp[known & (pred == yt)] = "known_correct"
    grp[known & (pred == UNK)] = "known_rejected"
    grp[~known & (pred == UNK)] = "unknown_rejected"
    grp[~known & (pred != UNK)] = "unknown_missed"
    idx = np.concatenate([rng.choice(np.flatnonzero(grp == g), min(PER_GROUP, (grp == g).sum()),
                                     replace=False) for g in GROUPS_ORDER if (grp == g).any()])
    X = np.asarray(Xt[idx])
    S = St[idx]
    A, base = m.attributions(X)
    rejected = pred[idx] == UNK
    q, Aq, bq, i1, i2 = target(S, A, base, theta, rejected)
    ref, cols_of_group = reference_values(Xfit, gcols, m.groups)
    res = {"n_test": int(len(yt)), "theta": theta,
           "group_sizes": {g: int((grp == g).sum()) for g in GROUPS_ORDER}, "groups": {}}
    # faithfulness: reset top-k supporting features vs random k, re-run
    support = Aq * np.sign(q)[:, None]
    order = np.argsort(-support, 1)
    G = Aq.shape[1]
    faith = {}
    for k in K_VALUES:
        top = [list(order[r, :k]) for r in range(len(X))]
        rnd = [list(rng.choice(G, k, replace=False)) for _ in range(len(X))]
        Pt = decide(m.scores(reset(X, top, ref, cols_of_group)), theta, m.classes)
        Pr = decide(m.scores(reset(X, rnd, ref, cols_of_group)), theta, m.classes)
        faith[k] = (Pt != pred[idx], Pr != pred[idx])
    # stability under one reset feature
    s_rows = rng.choice(len(X), min(STAB_ROWS, len(X)), replace=False)
    pert = [[int(rng.integers(G))] for _ in s_rows]
    Xp = reset(X[s_rows], pert, ref, cols_of_group)
    Sp = m.scores(Xp)
    Ap, bp = m.attributions(Xp)
    rej_p = decide(Sp, theta, m.classes) == UNK
    qp, Aqp, _, _, _ = target(Sp, Ap, bp, theta, rej_p)
    def topset(a):
        return [set(np.argsort(-np.abs(r))[:TOP].tolist()) for r in a]
    jac = lambda a, b: [len(x & y) / len(x | y) for x, y in zip(a, b)]
    pert_j = np.array(jac(topset(Aq[s_rows]), topset(Aqp)))
    pert_change = decide(Sp, theta, m.classes) != pred[idx][s_rows]
    # stability across seeds (same rows, each seed's own decision)
    seed_sets = {REF_SEED: topset(Aq)}
    for sd in SEEDS:
        if sd == REF_SEED:
            continue
        ms = build_models(kind, fold, data, sd)[0]
        Svs = ms.scores(Xv)
        tps = np.sort(Svs.max(1) + rng.uniform(0, 1e-6, len(Svs)))
        th_s = float(tps[int(round(BUDGET * len(tps)))])
        Ss = ms.scores(X)
        As, bs = ms.attributions(X)
        rj = decide(Ss, th_s, ms.classes) == UNK
        seed_sets[sd] = topset(target(Ss, As, bs, th_s, rj)[1])
    pairs = [(a, b) for i, a in enumerate(SEEDS) for b in SEEDS[i + 1:]]
    seed_j = np.mean([jac(seed_sets[a], seed_sets[b]) for a, b in pairs], 0)

    gl = grp[idx]
    for g in GROUPS_ORDER:
        sel = gl == g
        if not sel.any():
            continue
        e = {"rows": int(sel.sum()), "target": "unknown" if g.endswith("rejected") else "known",
             **fidelity_sparsity(q[sel], Aq[sel], bq[sel]),
             "faithfulness_flip_top": {k: float(faith[k][0][sel].mean()) for k in K_VALUES},
             "faithfulness_flip_random": {k: float(faith[k][1][sel].mean()) for k in K_VALUES},
             "stability_seed_jaccard_top5": float(seed_j[sel].mean())}
        ss_ = np.isin(np.arange(len(X))[s_rows], np.flatnonzero(sel))
        if ss_.any():
            e["stability_perturb_jaccard_top5"] = float(pert_j[ss_].mean())
            e["stability_perturb_decision_change"] = float(pert_change[ss_].mean())
        if kind == "tm":
            e.update(tm_clause_fidelity(m, X[sel], q[sel], i1[sel], i2[sel], rejected[sel], theta))
            Xs, top_c = X[sel], i1[sel]
            firing = np.zeros(len(Xs))
            for c in np.unique(top_c):
                rows = top_c == c
                firing[rows] = (m.clause_contrib(Xs[rows], c) != 0).sum(1)
            e["clauses_firing_top_class_median"] = float(np.median(firing))
        res["groups"][g] = e
    res["features"] = len(m.groups)
    res["seconds"] = time.perf_counter() - t0
    return res

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="dos,fuzzers,reconnaissance")
    ap.add_argument("--models", default="tm,rf_boolean,xgb_boolean,rf_continuous,xgb_continuous")
    a = ap.parse_args()
    data = find_dir("data")
    path = data / "phase5" / "explain_compare.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    out = json.load(open(path)) if path.exists() else {}
    for fold in a.folds.split(","):
        for kind in a.models.split(","):
            rng = np.random.default_rng(0)
            r = explain_fold(kind, fold, data, rng)
            out.setdefault(kind, {})[fold] = r
            json.dump(out, open(path, "w"), indent=1, default=float)
            g = r["groups"]
            print(f"{kind:16s}{fold:15s}" + "  ".join(
                f"{n}: fid@5 {v['fidelity_at_k'][5]:.2f} kmin {v['k_min_median']:.0f} "
                f"faith {v['faithfulness_flip_top'][5]:.2f}/{v['faithfulness_flip_random'][5]:.2f} "
                f"seedJ {v['stability_seed_jaccard_top5']:.2f}" for n, v in g.items())
                + f"  ({r['seconds']:.0f}s)", flush=True)
    print(f"written to {path}")

if __name__ == "__main__":
    main()
