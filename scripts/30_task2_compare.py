"""
Phase 5, Task 2: validation-only selection of the TM configuration and a
like-for-like comparison with Random Forest and XGBoost.

Candidates (all at 5 quantile thresholds, medium scale):

  TM     architecture  two-stage (gate + family banks), single-stage over
                       the same banks, single-stage multi-class TM
         rule          global, class-conditional, margin
         resampling    smote, none
  RF / XGBoost         input Boolean (the TM's literals) or continuous
                       (base + medium windows); rule global (1 - max class
                       probability) or margin (top-1 minus top-2)

Selection uses validation data only. Validation never contains the held-
out family, so known-class validation scores alone cannot tell a rule
that detects novelty from one that never rejects. Each fold v instead
has the held-out family's own validation-period rows (09: X_val_heldout),
which fold v never used. They are validation-period data, not test data.
For fold v and configuration c the objective is

  J(c, v) = macro F1 over the known classes + Unknown on validation
            (known rows + those pseudo-unknown rows), with c's thresholds
            calibrated on the known validation rows at a 5% budget

and the configuration used for outer fold u is the one maximising the
mean of J(c, v) over the other folds v != u and over seeds. Fold u's own
held-out rows play no part in choosing u's configuration. The same nested
rule chooses the trees' rule and the TM's architecture/rule/resampling.

Test metrics per fold, seed and budget (1%, 5%, 10% of known validation
rows rejected): unknown recall and precision, known-attack macro F1 after
rejection, known-sample rejection rate, Normal false-alarm rate, AUROC and
AUPRC of the decision's novelty score, held-out -> Normal / known family.
Summaries: family average (mean +/- std over seeds of the 9-fold mean) and
micro-pooled over folds (counts summed across the nine folds, then the
rate; known-attack F1 from pooled per-class counts). AUROC/AUPRC are
family-average only: novelty scores of different folds' models are not on
one scale. Per family: the number of test unknowns and a Wilson 95%
interval on recall.

Writes <data>/phase5/task2[_<tag>].json.

Usage: 30_task2_compare.py [--tag v1] [--seeds 7,42,1234] [--no-select]
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score

_spec = importlib.util.spec_from_file_location(
    "ss", Path(__file__).resolve().parent / "24_single_stage.py")
ss = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ss)
ev = ss.ev

BUDGETS = [0.01, 0.05, 0.10]
SELECT_BUDGET = 0.05
# TM_FAMILIES (comma-separated) evaluates another dataset's folds (Task 6)
FAMILIES = [f.lower() for f in (os.environ["TM_FAMILIES"].split(",") if os.environ.get("TM_FAMILIES")
                                 else ev.FAMILY_ORDER)]
UNK = 99
TM_RULES = ["global", "classcond", "margin"]
TREE_RULES = ["global", "margin"]
TREE_REPRS = ["boolean", "continuous_medium"]

# ---------------- decisions ----------------

def bkey(b):
    return f"{b:.2f}"

def tm_two_stage(r, vh):
    cal = ev.Calibrated(r)
    out = {}
    for rule in TM_RULES:
        for b in BUDGETS:
            thr = cal.calibrate(rule, b)
            vp, _ = cal.predict(rule, r["vb"], r["vf"], thr)
            tp, nov = cal.predict(rule, r["tb"], r["tf"], thr)
            hp = cal.predict(rule, vh[0], vh[1], thr)[0] if vh is not None else None
            out[(rule, bkey(b))] = (vp, hp, tp, nov)
    return out, cal.ni, list(cal.fams)

def tm_single(classes, vs, ts, hs, y_val):
    cal = ss.SingleStage(classes, vs, y_val, UNK)
    out = {}
    for rule in TM_RULES:
        for b in BUDGETS:
            thr = cal.calibrate(rule, b)
            vp, _ = cal.predict(rule, vs, thr)
            tp, nov = cal.predict(rule, ts, thr)
            hp = cal.predict(rule, hs, thr)[0] if hs is not None else None
            out[(rule, bkey(b))] = (vp, hp, tp, nov)
    return out

def tree_rules(z, seed):
    rng = np.random.default_rng(seed)
    def nov(part, rule):
        top = z[f"{part}_top"].astype(np.float64)
        x = -top[:, 0] if rule == "global" else -(top[:, 0] - top[:, 1])
        return x + rng.uniform(0, 1e-7, len(x))
    out = {}
    for rule in TREE_RULES:
        nv, nt = nov("val", rule), nov("test", rule)
        nh = nov("valheld", rule) if "valheld_top" in z else None
        for b in BUDGETS:
            k = int(round(b * len(nv)))
            theta = float(np.sort(nv)[::-1][k])
            f = lambda n, a: np.where(n > theta, UNK, a).astype(np.int64)
            out[(rule, bkey(b))] = (f(nv, z["val_arg"]), f(nh, z["valheld_arg"]) if nh is not None else None,
                                    f(nt, z["test_arg"]), nt)
    return out

# ---------------- metrics ----------------

def counts(y, pred, nov, ni, fams):
    is_unk = y == UNK
    known = ~is_unk
    pu = pred[is_unk]
    c = {"n_unknown": int(is_unk.sum()), "tp_unknown": int((pu == UNK).sum()),
         "pred_unknown": int((pred == UNK).sum()),
         "unk_to_normal": int((pu == ni).sum()), "unk_to_family": int(np.isin(pu, fams).sum()),
         "n_known": int(known.sum()), "known_rejected": int((pred[known] == UNK).sum()),
         "n_known_attack": int((known & (y != ni)).sum()),
         "known_attack_rejected": int((pred[known & (y != ni)] == UNK).sum()),
         "n_normal": int((y == ni).sum()), "normal_flagged": int((pred[y == ni] != ni).sum())}
    yk, pk = y[known], pred[known]
    c["class_tp_fp_fn"] = {int(f): [int(((yk == f) & (pk == f)).sum()), int(((yk != f) & (pk == f)).sum()),
                                    int(((yk == f) & (pk != f)).sum())] for f in fams}
    return c

def rates(c):
    f1s = [2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
           for tp, fp, fn in c["class_tp_fp_fn"].values()]
    return {"unknown_recall": c["tp_unknown"] / max(c["n_unknown"], 1),
            "unknown_precision": c["tp_unknown"] / c["pred_unknown"] if c["pred_unknown"] else 0.0,
            "known_attack_macro_f1": float(np.mean(f1s)),
            "known_rejection": c["known_rejected"] / c["n_known"],
            "known_attack_rejection": c["known_attack_rejected"] / max(c["n_known_attack"], 1),
            "normal_false_alarm": c["normal_flagged"] / c["n_normal"],
            "unknown_to_normal": c["unk_to_normal"] / max(c["n_unknown"], 1),
            "unknown_to_known_family": c["unk_to_family"] / max(c["n_unknown"], 1)}

def pool(cs):
    out = {k: sum(c[k] for c in cs) for k in cs[0] if k != "class_tp_fp_fn"}
    per = {}
    for c in cs:
        for f, v in c["class_tp_fp_fn"].items():
            per.setdefault(f, [0, 0, 0])
            per[f] = [a + b for a, b in zip(per[f], v)]
    out["class_tp_fp_fn"] = per
    return out

def open_f1(y_val, vp, hp, classes):
    if hp is None or len(hp) == 0:
        return float("nan")
    y = np.r_[y_val, np.full(len(hp), UNK)]
    p = np.r_[vp, hp]
    labels = list(classes) + [UNK]
    return float(precision_recall_fscore_support(y, p, labels=labels, average="macro",
                                                 zero_division=0)[2])

def evaluate(dec, y_val, y_test, ni, fams, classes):
    """dec: {(rule, budget): (val_pred, valheld_pred, test_pred, test_nov)}"""
    out = {}
    is_unk = y_test == UNK
    auc = {}
    for (rule, b), (vp, hp, tp, nov) in dec.items():
        if rule not in auc:
            auc[rule] = (float(roc_auc_score(is_unk, nov)), float(average_precision_score(is_unk, nov)))
        c = counts(y_test, tp, nov, ni, fams)
        out[f"{rule}|{b}"] = {"counts": c, **rates(c), "auroc": auc[rule][0], "auprc": auc[rule][1],
                              "val_open_f1": open_f1(y_val, vp, hp, classes),
                              "val_unknown_recall": float((hp == UNK).mean()) if hp is not None and len(hp)
                              else float("nan"),
                              "val_known_f1": float(precision_recall_fscore_support(
                                  y_val, vp, labels=list(classes), average="macro", zero_division=0)[2]),
                              "val_reject": float((vp == UNK).mean()),
                              "n_val_heldout": 0 if hp is None else int(len(hp))}
    return out

# ---------------- sources ----------------

def jitter(x, seed):
    return x + np.random.default_rng(seed).uniform(0, 1e-3, x.shape)

def tm_runs(data, strategy, seeds):
    sfx = "" if strategy == "smote" else f"_{strategy}"
    res = {}
    for fam in FAMILIES:
        for seed in seeds:
            d = data / f"banks_medium_{fam}{sfx}_s{seed}"
            if (d / "banks_meta.json").exists():
                r = ev.load_run(d, data)
                fams_ = r["families"]
                vh = None
                if (d / "valheld_normal.npy").exists():
                    vb = jitter(np.load(d / "valheld_normal.npy").astype(np.float64), 11)
                    vf = jitter(np.stack([np.load(d / f"valheld_fam{c}.npy") for c in fams_], 1)
                                .astype(np.float64), 12)
                    vh = (vb, vf)
                dec, ni, fl = tm_two_stage(r, vh)
                classes = [ni] + fl
                res.setdefault(("two_stage", strategy), {}).setdefault(fam, {})[seed] = \
                    evaluate(dec, r["y_val"], r["y_test"], ni, fl, classes)
                vs = np.column_stack([r["vb"], r["vf"]])
                ts = np.column_stack([r["tb"], r["tf"]])
                hs = np.column_stack(vh) if vh is not None else None
                dec = tm_single(classes, vs, ts, hs, r["y_val"])
                res.setdefault(("single_banks", strategy), {}).setdefault(fam, {})[seed] = \
                    evaluate(dec, r["y_val"], r["y_test"], ni, fl, classes)
            d = data / f"single_medium_{fam}{sfx}_s{seed}"
            if (d / "meta.json").exists():
                r, classes, vs, ts = ss.load_mc(d, data)
                m = r["meta"]
                hs = jitter(np.load(d / "valheld_scores.npy").astype(np.float64), 13) \
                    if (d / "valheld_scores.npy").exists() else None
                ni = m["normal_idx"]
                fl = [c for c in classes if c != ni]
                dec = tm_single(classes, vs, ts, hs, r["y_val"])
                res.setdefault(("single_mc", strategy), {}).setdefault(fam, {})[seed] = \
                    evaluate(dec, r["y_val"], r["y_test"], ni, fl, classes)
        print(f"  TM {strategy} {fam} done", flush=True)
    return res

def tree_runs(data, seeds):
    res = {}
    for rep in TREE_REPRS:
        sd = data / "phase5" / "tree_scores" / rep
        for fam in FAMILIES:
            if not (sd / f"{fam}_y_test.npy").exists():
                continue
            yv = np.load(sd / f"{fam}_y_val.npy").astype(np.int64)
            yt = np.load(sd / f"{fam}_y_test.npy").astype(np.int64)
            meta = json.load(open(data / f"openset_medium_{fam}" / "meta.json"))
            ni = meta["all_class_names"].index("Normal")
            classes = sorted(set(np.unique(yv).tolist()))
            fl = [c for c in classes if c != ni]
            for model in ("rf", "xgb"):
                for seed in seeds:
                    p = sd / f"{fam}_{model}_s{seed}.npz"
                    if p.exists():
                        dec = tree_rules(np.load(p), seed)
                        res.setdefault((model, rep), {}).setdefault(fam, {})[seed] = \
                            evaluate(dec, yv, yt, ni, fl, classes)
    return res

# ---------------- selection and summaries ----------------

def configs_of(res, rules):
    return [(a, s, r) for (a, s) in res for r in rules]

OBJECTIVES = {
    # pre-specified: macro F1 over known classes + Unknown
    "open_f1": lambda m: m["val_open_f1"],
    # sensitivity analysis, chosen after the primary result was seen: equal
    # weight on pseudo-unknown recall and known-class macro F1
    "balanced": lambda m: 0.5 * (m["val_unknown_recall"] + m["val_known_f1"]),
}

def select(res, rules, seeds, objective="open_f1"):
    """{outer fold: chosen (arch, variant, rule)}, plus the J table."""
    key = bkey(SELECT_BUDGET)
    obj = OBJECTIVES[objective]
    J = {}
    for a, s, r in configs_of(res, rules):
        for fam, by_seed in res[(a, s)].items():
            vals = [obj(by_seed[sd][f"{r}|{key}"]) for sd in seeds if sd in by_seed]
            J.setdefault((a, s, r), {})[fam] = float(np.mean(vals))
    chosen = {}
    for u in FAMILIES:
        best, best_j = None, -1
        for c, per in J.items():
            others = [per[v] for v in per if v != u and np.isfinite(per[v])]
            if len(others) < len(FAMILIES) - 1:
                continue
            j = float(np.mean(others))
            if j > best_j:
                best, best_j = c, j
        chosen[u] = {"config": list(best) if best else None, "J_other_folds": best_j}
    return chosen, {"|".join(k): v for k, v in J.items()}

def summarise(res, picks, seeds):
    """picks: {family: (arch, variant, rule)} -- one system, possibly nested."""
    out = {}
    for b in BUDGETS:
        key = bkey(b)
        per_seed_avg, per_fam, pooled_seed = [], {}, []
        for sd in seeds:
            rows, cs = [], []
            for fam in FAMILIES:
                a, v, r = picks[fam]
                e = res.get((a, v), {}).get(fam, {}).get(sd)
                if e is None:
                    break
                m = e[f"{r}|{key}"]
                rows.append(m); cs.append(m["counts"])
                per_fam.setdefault(fam, []).append(m)
            if len(rows) != len(FAMILIES):
                continue
            per_seed_avg.append({k: float(np.mean([m[k] for m in rows]))
                                 for k in list(rates(cs[0])) + ["auroc", "auprc"]})
            pooled_seed.append(rates(pool(cs)))
        if not per_seed_avg:
            continue
        ms = lambda xs: [float(np.mean(xs)), float(np.std(xs))]
        out[key] = {
            "seeds": len(per_seed_avg),
            "family_average": {k: ms([s[k] for s in per_seed_avg]) for k in per_seed_avg[0]},
            "micro": {k: ms([s[k] for s in pooled_seed]) for k in pooled_seed[0]},
            "per_family": {f: {**{k: ms([m[k] for m in ms_]) for k in
                                  list(rates(ms_[0]["counts"])) + ["auroc", "auprc"]},
                               "n_test_unknown": ms_[0]["counts"]["n_unknown"],
                               "n_val_heldout": ms_[0]["n_val_heldout"],
                               "recall_ci95_pooled_seeds": ev.wilson(
                                   int(round(np.mean([m["counts"]["tp_unknown"] for m in ms_]))),
                                   ms_[0]["counts"]["n_unknown"])}
                           for f, ms_ in per_fam.items()}}
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    ap.add_argument("--seeds", default="7,42,1234")
    ap.add_argument("--no-select", action="store_true")
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    data = ev.find_dir("data")

    tm = {}
    for st in ("smote", "none"):
        tm.update(tm_runs(data, st, seeds))
    trees = tree_runs(data, seeds)

    systems = {}
    for (arch, st) in tm:
        for r in TM_RULES:
            systems[f"TM {arch} {st} {r}"] = (tm, {f: (arch, st, r) for f in FAMILIES})
    for (model, rep) in trees:
        for r in TREE_RULES:
            systems[f"{model.upper()} {rep} {r}"] = (trees, {f: (model, rep, r) for f in FAMILIES})

    selection = {}
    if not a.no_select:
        ch, J = select(tm, TM_RULES, seeds)
        selection["TM"] = {"chosen": ch, "J": J}
        systems["TM selected"] = (tm, {f: tuple(ch[f]["config"]) for f in FAMILIES})
        ch2, J2 = select(tm, TM_RULES, seeds, "balanced")
        selection["TM (balanced objective, sensitivity)"] = {"chosen": ch2, "J": J2}
        systems["TM selected, balanced objective"] = (tm, {f: tuple(ch2[f]["config"]) for f in FAMILIES})
        for model in ("rf", "xgb"):
            for rep in TREE_REPRS:
                sub = {k: v for k, v in trees.items() if k == (model, rep)}
                if not sub:
                    continue
                ch, J = select(sub, TREE_RULES, seeds)
                selection[f"{model.upper()} {rep}"] = {"chosen": ch, "J": J}
                systems[f"{model.upper()} {rep} selected"] = (trees, {f: tuple(ch[f]["config"])
                                                                       for f in FAMILIES})
    summary = {name: summarise(res, picks, seeds) for name, (res, picks) in systems.items()}
    out = data / "phase5" / f"task2{'_' + a.tag if a.tag else ''}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"seeds": seeds, "budgets": BUDGETS, "selection": selection, "summary": summary},
              open(out, "w"), indent=1, default=float)

    print(f"\n{'system':<44}{'unk.rec':>9}{'unk.prec':>9}{'atkF1':>8}{'knownRej':>9}{'NormFA':>8}"
          f"{'AUROC':>7}   (5%, family average; micro recall)")
    for name, s in summary.items():
        if "0.05" not in s:
            continue
        fa, mi = s["0.05"]["family_average"], s["0.05"]["micro"]
        print(f"{name:<44}{fa['unknown_recall'][0]:>9.1%}{fa['unknown_precision'][0]:>9.1%}"
              f"{fa['known_attack_macro_f1'][0]:>8.3f}{fa['known_rejection'][0]:>9.2%}"
              f"{fa['normal_false_alarm'][0]:>8.2%}{fa['auroc'][0]:>7.3f}   {mi['unknown_recall'][0]:.1%}")
    for k, v in selection.items():
        print(k, {f: v["chosen"][f]["config"] for f in FAMILIES})
    print(f"written to {out}")

if __name__ == "__main__":
    main()
