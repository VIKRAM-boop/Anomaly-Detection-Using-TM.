"""
Clause-level explanations for the complete open-set decision (Task 6).

A Step 5 decision is made by the normality bank and the family banks, each a
vote over clauses. For one test row the explanation names the clauses that
carried each part of the decision:

  Normal      the normality bank's clauses supporting "benign", and the
              benign margin clearing tau_B
  known k     the normality bank's clauses voting "not benign", and family
              k's clauses supporting k, with S_k clearing tau_A
  Unknown     the clauses voting "not benign", plus the best-placed family
              bank's evidence for and against it, showing why it fell short
              of tau_A

Within a bank, clause j of class i contributes sign_i * w_j * fire_j to the
bank's margin (sign +1 for the positive class, -1 for the negative). Its
direction and strength are measured on validation (held-out family absent)
as c_j = sign_i * w_j * (fire rate on the bank's own class - fire rate on the
rest). |c_j| ranks clauses by how much they separate the bank's class from
the rest; when a clause fires on a row, the direction it pushes that row's
margin is sign_i * w_j, which is what "for"/"against" reports.

Measures, all computed per subgroup -- Normal rows, known-attack rows,
held-out rows caught as Unknown, and held-out rows missed (sent to a known
family or to Normal):

  fidelity    the three-way decision is recomputed with every bank reduced
              to its k highest-|c| clauses, thresholds re-calibrated on
              validation for the reduced model at the same budgets; fidelity
              is agreement with the full model's decision
  stability   Jaccard overlap of the cited clause set between a row and
              copies with one literal flipped, plus how often the decision
              itself changes
  complexity  clauses cited, literals per cited clause, and clauses firing
              in the banks involved

Medium scale, seed 42, the "global" decision strategy at a 5% validation
false-rejection budget -- the same calibration as 18_openset_eval.py.

Writes data/phase3/explain_openset.json and prints worked examples.

Usage: 21_openset_explain.py [family ...]
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

import tm_fast

here = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("ev", here / "18_openset_eval.py")
ev = importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)
find_dir = ev.find_dir

SCALE, SEED, STRAT, BUDGET = "medium", 42, "global", 0.05
K_VALUES = [1, 3, 5, 10, 25, 50, 100]
K_MAX = max(K_VALUES)
CITE_PER_BANK = 5
FIDELITY_PER_GROUP = 5000
STABILITY_PER_GROUP = 150
FLIPS = 5
FAMILIES = ["Analysis", "Backdoor", "DoS", "Exploits", "Fuzzers", "Generic",
            "Reconnaissance", "Shellcode", "Worms"]

def subset(model, cls, idx):
    return {"T": model["T"], "n_features": model["n_features"],
            "include": [model["include"][cls][idx]], "weights": [model["weights"][cls][idx]]}

def contributions(model, X, pos_mask, chunk=40000):
    """c_j for every clause of both classes of one binary bank."""
    out = []
    for i, sign in ((0, -1.0), (1, 1.0)):
        rp = np.zeros(model["include"][i].shape[0]); rn = np.zeros_like(rp)
        for s in range(0, len(X), chunk):
            f = tm_fast.clause_outputs(model, X[s:s + chunk], i)
            pm = pos_mask[s:s + chunk]
            rp += f[pm].sum(0); rn += f[~pm].sum(0)
        rp /= max(pos_mask.sum(), 1); rn /= max((~pos_mask).sum(), 1)
        out.append(sign * model["weights"][i] * (rp - rn))
    return out

def top_clauses(contrib, k):
    """(class, clause) pairs of the k largest |c_j| across both classes."""
    flat = [(abs(c), i, j) for i in (0, 1) for j, c in enumerate(contrib[i])]
    flat.sort(reverse=True)
    return [(i, j) for _, i, j in flat[:k]]

def reduced_margin(model, X, top, k):
    """Bank margin using only the first k clauses of `top`."""
    m = np.zeros(len(X))
    for cls in (0, 1):
        idx = np.array([j for i, j in top[:k] if i == cls], dtype=np.int64)
        if len(idx) == 0:
            continue
        sm = subset(model, cls, idx)
        fire = tm_fast.clause_outputs(sm, X, 0)
        sign = 1.0 if cls == 1 else -1.0
        m += sign * (fire.astype(np.float64) @ model["weights"][cls][idx].astype(np.float64))
    return m

def clause_text(model, cls, j, names, max_lits=6):
    inc = model["include"][cls][j]
    F = model["n_features"]
    lits = [names[f] for f in np.where(inc[:F])[0]] + \
           ["NOT " + names[f] for f in np.where(inc[F:])[0]]
    more = "" if len(lits) <= max_lits else f" ... (+{len(lits) - max_lits} more)"
    return " AND ".join(lits[:max_lits]) + more, len(lits)

def cite(models, contrib, x, bank_names, cal_names):
    """Cited clause set for one row: top-|c| clauses that fire, per bank."""
    cited = set()
    for b in bank_names:
        m = models[b]
        firing = []
        for cls in (0, 1):
            f = tm_fast.clause_outputs(m, x[None, :], cls)[0]
            for j in np.where(f)[0]:
                firing.append((abs(contrib[b][cls][j]), cls, int(j)))
        firing.sort(reverse=True)
        for _, cls, j in firing[:CITE_PER_BANK]:
            cited.add((b, cls, j))
    return cited

def involved(pred_label, best_fam, ni, unk):
    banks = ["normal"]
    if pred_label != ni:
        banks.append(f"fam{pred_label if pred_label != unk else best_fam}")
    return banks

def explain_family(family):
    data = find_dir("data")
    bdir = data / f"banks_{SCALE}_{family.lower()}_s{SEED}"
    ddir = data / f"openset_{SCALE}_{family.lower()}"
    meta = json.load(open(bdir / "banks_meta.json"))
    names = json.load(open(ddir / "meta.json"))["feature_names"]
    fams, ni, unk = meta["families"], meta["normal_idx"], meta["unknown_label"]
    cls_names = meta["all_class_names"]
    models = {"normal": tm_fast.load_model(bdir / "models" / "normal.npz")}
    for c in fams:
        models[f"fam{c}"] = tm_fast.load_model(bdir / "models" / f"fam{c}.npz")
    X_val = np.load(ddir / "X_val.npy"); y_val = np.load(ddir / "y_val.npy").astype(np.int64)
    X_test = np.load(ddir / "X_test.npy"); y_test = np.load(ddir / "y_test.npy").astype(np.int64)

    # full-model decision, exactly as 18 computes it
    run = ev.load_run(bdir, data)
    cal = ev.Calibrated(run)
    thr = cal.calibrate(STRAT, BUDGET)
    full_pred, _ = cal.predict(STRAT, run["tb"], run["tf"], thr)
    best_fam = np.asarray(fams)[run["tf"].argmax(1)]
    # load_run adds a seeded tie-break jitter to every score; perturbed copies
    # must carry their own row's offsets, otherwise a row sitting exactly at a
    # tied threshold value flips decision with no change in its real score
    jit_tb = run["tb"] - np.load(bdir / "test_normal.npy").astype(np.float64)
    jit_tf = run["tf"] - np.stack([np.load(bdir / f"test_fam{c}.npy") for c in fams], 1).astype(np.float64)

    # clause directions and strengths on validation
    contrib = {"normal": contributions(models["normal"], X_val, y_val == ni)}
    for c in fams:
        contrib[f"fam{c}"] = contributions(models[f"fam{c}"], X_val, y_val == c)
    tops = {b: top_clauses(contrib[b], K_MAX) for b in models}

    # subgroups
    rng = np.random.default_rng(SEED)
    held = y_test == unk
    groups = {"normal": y_test == ni,
              "known_attack": (~held) & (y_test != ni),
              "unknown_caught": held & (full_pred == unk),
              "unknown_to_family": held & np.isin(full_pred, fams),
              "unknown_to_normal": held & (full_pred == ni)}
    def sample(mask, n):
        idx = np.where(mask)[0]
        return rng.choice(idx, min(n, len(idx)), replace=False) if len(idx) else idx

    fid_idx = {g: sample(m, FIDELITY_PER_GROUP) for g, m in groups.items()}
    all_fid = np.concatenate(list(fid_idx.values()))

    # ---- fidelity ----
    fidelity = {g: {} for g in groups}
    fidelity["all_weighted"] = {}
    for k in K_VALUES:
        vb = reduced_margin(models["normal"], X_val, tops["normal"], k)
        tb = reduced_margin(models["normal"], X_test[all_fid], tops["normal"], k)
        vf = np.stack([reduced_margin(models[f"fam{c}"], X_val, tops[f"fam{c}"], k) for c in fams], 1)
        tf = np.stack([reduced_margin(models[f"fam{c}"], X_test[all_fid], tops[f"fam{c}"], k) for c in fams], 1)
        # reduced margins are small integers with heavy ties, and a quantile
        # threshold lands on a tied value; a seeded jitter below the integer
        # gap splits ties at random without reordering distinct scores
        jit = np.random.default_rng(k)
        vb = vb + jit.uniform(0, 1e-3, vb.shape); tb = tb + jit.uniform(0, 1e-3, tb.shape)
        vf = vf + jit.uniform(0, 1e-3, vf.shape); tf = tf + jit.uniform(0, 1e-3, tf.shape)
        rr = dict(run); rr.update(vb=vb, vf=vf, tb=tb, tf=tf)
        rcal = ev.Calibrated(rr)
        rpred, _ = rcal.predict(STRAT, tb, tf, rcal.calibrate(STRAT, BUDGET))
        agree = rpred == full_pred[all_fid]
        pos = 0
        weighted = 0.0
        for g, idx in fid_idx.items():
            a = agree[pos:pos + len(idx)]; pos += len(idx)
            fidelity[g][k] = float(a.mean()) if len(a) else None
            if len(a):
                weighted += a.mean() * groups[g].mean()
        fidelity["all_weighted"][k] = float(weighted)

    # ---- stability and complexity ----
    stab_idx = {g: sample(m, STABILITY_PER_GROUP) for g, m in groups.items()}
    stability, complexity = {}, {}
    for g, idx in stab_idx.items():
        if len(idx) == 0:
            stability[g] = complexity[g] = None
            continue
        base_x = X_test[idx]
        pert = np.repeat(base_x, FLIPS, axis=0).copy()
        flip_cols = rng.integers(0, pert.shape[1], len(pert))
        pert[np.arange(len(pert)), flip_cols] ^= 1
        pb = tm_fast.binary_margin(models["normal"], pert) + np.repeat(jit_tb[idx], FLIPS)
        pf = np.stack([tm_fast.binary_margin(models[f"fam{c}"], pert) for c in fams], 1) \
            + np.repeat(jit_tf[idx], FLIPS, axis=0)
        ppred, _ = cal.predict(STRAT, pb, pf, thr)
        pbest = np.asarray(fams)[pf.argmax(1)]
        jac, flips, n_cited, lits, n_firing = [], [], [], [], []
        for r, row in enumerate(idx):
            banks0 = involved(full_pred[row], best_fam[row], ni, unk)
            e0 = cite(models, contrib, X_test[row], banks0, cls_names)
            n_cited.append(len(e0))
            for (b, cls, j) in e0:
                lits.append(int(models[b]["include"][cls][j].sum()))
            n_firing.append(sum(int(tm_fast.clause_outputs(models[b], X_test[row][None, :], c)[0].sum())
                                for b in banks0 for c in (0, 1)))
            for t in range(FLIPS):
                q = r * FLIPS + t
                banks1 = involved(ppred[q], pbest[q], ni, unk)
                e1 = cite(models, contrib, pert[q], banks1, cls_names)
                u = e0 | e1
                jac.append(len(e0 & e1) / len(u) if u else 1.0)
                flips.append(ppred[q] != full_pred[row])
        stability[g] = {"jaccard_mean": float(np.mean(jac)), "decision_change_rate": float(np.mean(flips)),
                        "rows": int(len(idx))}
        complexity[g] = {"cited_clauses_mean": float(np.mean(n_cited)),
                         "literals_per_cited_clause_mean": float(np.mean(lits)) if lits else None,
                         "literals_per_cited_clause_median": float(np.median(lits)) if lits else None,
                         "firing_clauses_in_involved_banks_mean": float(np.mean(n_firing))}

    # ---- worked examples ----
    examples = {}
    for g, m in groups.items():
        idx = np.where(m)[0]
        if not len(idx):
            continue
        row = int(rng.choice(idx))
        p = full_pred[row]
        lines = [f"true={cls_names[y_test[row]] if y_test[row] != unk else family + ' (held out)'}  "
                 f"predicted={'UNKNOWN' if p == unk else cls_names[p]}",
                 f"benign margin {run['tb'][row]:.0f} vs tau_B {cal.tau_b:.0f}"]
        for b in involved(p, best_fam[row], ni, unk):
            label = "normality bank" if b == "normal" else f"{cls_names[int(b[3:])]} bank"
            score = run["tb"][row] if b == "normal" else run["tf"][row][fams.index(int(b[3:]))]
            lines.append(f"{label}: margin {score:.0f}" +
                         ("" if b == "normal" else f" vs tau_A {-thr.get('theta', np.nan):.0f}"))
            firing = []
            for cls in (0, 1):
                f = tm_fast.clause_outputs(models[b], X_test[row][None, :], cls)[0]
                push = (1.0 if cls == 1 else -1.0) * models[b]["weights"][cls]
                for j in np.where(f)[0]:
                    # importance is |c_j|; direction is which way the clause
                    # moves this row's margin when it fires: sign_i * w_j
                    firing.append((abs(contrib[b][cls][j]), cls, int(j), push[j]))
            firing.sort(reverse=True)
            pos_name = "benign" if b == "normal" else cls_names[int(b[3:])]
            for _, cls, j, cval in firing[:3]:
                txt, n = clause_text(models[b], cls, j, names)
                lines.append(f"   {'for' if cval > 0 else 'against'} {pos_name}: {txt}")
        examples[g] = lines
    return {"n_unknown": int(held.sum()),
            "group_sizes": {g: int(m.sum()) for g, m in groups.items()},
            "fidelity": fidelity, "stability": stability, "complexity": complexity,
            "examples": examples}

def main():
    families = sys.argv[1:] or FAMILIES
    out_path = find_dir("data") / "phase3" / "explain_openset.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results = json.load(open(out_path)) if out_path.exists() else {}
    for fam in families:
        r = explain_family(fam)
        results[fam.lower()] = r
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=float)
        print(f"\n=== {fam} (held out, {r['n_unknown']} test rows) ===")
        print("fidelity (agreement with full decision) by k:")
        for g in ("all_weighted", "normal", "known_attack", "unknown_caught",
                  "unknown_to_family", "unknown_to_normal"):
            v = r["fidelity"].get(g, {})
            if v and any(x is not None for x in v.values()):
                print(f"  {g:<18}" + "  ".join(f"k={k}:{v[k]:.1%}" if v[k] is not None else f"k={k}:-"
                                               for k in K_VALUES))
        for g, s in r["stability"].items():
            if s:
                c = r["complexity"][g]
                print(f"  {g:<18} stability J={s['jaccard_mean']:.3f} decision-change={s['decision_change_rate']:.1%}"
                      f" | cited {c['cited_clauses_mean']:.1f} clauses, "
                      f"{c['literals_per_cited_clause_median']:.0f} literals (median)")
        for g, lines in r["examples"].items():
            print(f"  -- example: {g}")
            for ln in lines:
                print(f"     {ln}")
    print(f"\nwritten to {out_path}")

if __name__ == "__main__":
    main()
