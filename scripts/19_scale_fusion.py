"""
Temporal-scale contributions and fusion (Task 4).

For every held-out family and seed that has banks at all three scales:

  contribution     unknown recall per scale for the held-out family, and the
                   closed-set per-class F1 per scale for each known family
  complementarity  overlap of the held-out rows each scale catches (union
                   versus best single scale), and rank correlation of the
                   three scales' novelty scores
  fusion           single scales, mean, max, the previously tested
                   margin-weighted adaptive rule, and two revised rules

The revised adaptive rules use a per-family notion of reliability rather than
a per-row margin. On validation (held-out family absent), each bank's score
at each scale gets an AUROC for separating its own class from the other
known rows. Then:

  adaptive_rel     each scale's standardised score is weighted by
                   (AUROC - 0.5), normalised across scales, per bank
  select           each bank uses only its single most reliable scale
  *_raw            the same two rules on native margins; the first versions
                   standardised over validation (mostly Normal rows), which
                   distorted attack scores, and are kept for the record

Every fused score set is evaluated through 18_openset_eval.py's decision and
calibration code, so the comparison runs under identical thresholds and
budgets. A revised rule is recommended over max only if its family-averaged
gain in unknown recall exceeds twice the seed-to-seed spread of the paired
difference and known-class macro F1 does not fall.

Writes data/phase3/scale_fusion_<strategy>.json.

Usage: 19_scale_fusion.py [strategy]   (decision strategy, default "global")
"""
import importlib.util
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

SCALES = ["short", "medium", "long"]
BUDGET = "0.05"

here = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("ev", here / "18_openset_eval.py")
ev = importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)
find_dir = ev.find_dir

def standardise(val, test):
    mu, sd = val.mean(), val.std() or 1.0
    return (val - mu) / sd, (test - mu) / sd

def reliability(score, y, pos_labels):
    pos = np.isin(y, pos_labels)
    if pos.all() or not pos.any():
        return 0.5
    return float(roc_auc_score(pos, score))

def fuse(runs, rule):
    """runs: {scale: run dict}. Returns a run dict with fused scores."""
    base = runs["medium"]
    fams = base["families"]
    ni = base["meta"]["normal_idx"]
    vy = base["y_val"]
    if rule in SCALES:
        return runs[rule]
    vb = np.stack([runs[s]["vb"] for s in SCALES]); tb = np.stack([runs[s]["tb"] for s in SCALES])
    vf = np.stack([runs[s]["vf"] for s in SCALES]); tf = np.stack([runs[s]["tf"] for s in SCALES])
    out = dict(base)
    if rule == "mean":
        out.update(vb=vb.mean(0), tb=tb.mean(0), vf=vf.mean(0), tf=tf.mean(0))
    elif rule == "max":
        out.update(vb=vb.max(0), tb=tb.max(0), vf=vf.max(0), tf=tf.max(0))
    elif rule == "adaptive_margin":
        # the rule tested in Phase 2: per-row weights from each scale's
        # top-1 minus top-2 family margin
        def w_of(f):
            p = np.partition(f, -2, axis=2)
            m = np.clip(p[:, :, -1] - p[:, :, -2], 0, None)
            t = m.sum(0, keepdims=True)
            return np.where(t == 0, 1 / 3, m / np.where(t == 0, 1, t))
        wv, wt = w_of(vf), w_of(tf)
        out.update(vb=(vb * wv).sum(0), tb=(tb * wt).sum(0),
                   vf=(vf * wv[:, :, None]).sum(0), tf=(tf * wt[:, :, None]).sum(0))
    elif rule in ("adaptive_rel", "select", "adaptive_rel_raw", "select_raw"):
        raw = rule.endswith("_raw")
        def combine(vstack, tstack, pos):
            aucs = np.array([reliability(vstack[s], vy, pos) for s in range(3)])
            # *_raw keeps the margins on their native scale (all three scales
            # share a family's T); standardising over validation, ~78% Normal,
            # distorts attack scores and is kept only as the first attempt
            z = ([(vstack[s], tstack[s]) for s in range(3)] if raw
                 else [standardise(vstack[s], tstack[s]) for s in range(3)])
            if rule.startswith("select"):
                k = int(aucs.argmax())
                return z[k][0], z[k][1], aucs
            w = np.clip(aucs - 0.5, 1e-3, None); w = w / w.sum()
            return (sum(w[s] * z[s][0] for s in range(3)),
                    sum(w[s] * z[s][1] for s in range(3)), aucs)
        nvb, ntb, _ = combine(vb, tb, [ni])
        nvf, ntf = [], []
        for j, c in enumerate(fams):
            a, b, _ = combine(vf[:, :, j], tf[:, :, j], [c])
            nvf.append(a); ntf.append(b)
        out.update(vb=nvb, tb=ntb, vf=np.stack(nvf, 1), tf=np.stack(ntf, 1))
    return out

def eval_strategy(run, strat):
    """The metrics this script reports, for one strategy at BUDGET only.
    Same calibration and decision code as 18_openset_eval.evaluate_run, which
    computes every strategy and budget and is ~15x slower here."""
    from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score
    cal = ev.Calibrated(run)
    thr = cal.calibrate(strat, float(BUDGET))
    pred, nov = cal.predict(strat, run["tb"], run["tf"], thr)
    y = run["y_test"]; unk = y == cal.unk; known = ~unk
    labels = sorted(set(np.unique(y[known]).tolist()))
    return {"unknown_recall": float((pred[unk] == cal.unk).mean()),
            "held_to_normal": float((pred[unk] == cal.ni).mean()),
            "false_unknown": float((pred[known] == cal.unk).mean()),
            "benign_fpr": float((pred[y == cal.ni] != cal.ni).mean()),
            "known_macro_f1": float(precision_recall_fscore_support(
                y[known], pred[known], labels=labels, average="macro", zero_division=0)[2]),
            "auroc": float(roc_auc_score(unk, nov)),
            "auprc": float(average_precision_score(unk, nov))}

def per_class_f1(run):
    cal = ev.Calibrated(run)
    pred, _ = cal.predict("closed", run["tb"], run["tf"], {})
    y = run["y_test"]
    known = y != cal.unk
    labels = sorted(set(np.unique(y[known]).tolist()))
    f1 = precision_recall_fscore_support(y[known], pred[known], labels=labels, zero_division=0)[2]
    names = run["meta"]["all_class_names"]
    return {names[c]: float(v) for c, v in zip(labels, f1)}

def caught(run, strat):
    cal = ev.Calibrated(run)
    thr = cal.calibrate(strat, float(BUDGET))
    pred, nov = cal.predict(strat, run["tb"], run["tf"], thr)
    return pred == cal.unk, nov

def main():
    strat = sys.argv[1] if len(sys.argv) > 1 else "global"
    data = find_dir("data")
    found = {}
    for d in data.glob("banks_*_*_s*"):
        m = re.match(r"banks_(short|medium|long)_(.+)_s(\d+)$", d.name)
        if m and (d / "banks_meta.json").exists():
            found.setdefault((m.group(2), int(m.group(3))), {})[m.group(1)] = d
    combos = sorted(k for k, v in found.items() if len(v) == 3)
    if not combos:
        raise SystemExit("no family/seed has banks at all three scales")

    rules = SCALES + ["mean", "max", "adaptive_margin", "adaptive_rel", "select",
                      "adaptive_rel_raw", "select_raw"]
    results = {}
    for fam, seed in combos:
        runs = {s: ev.load_run(found[(fam, seed)][s], data) for s in SCALES}
        y = runs["medium"]["y_test"]; unk = runs["medium"]["meta"]["unknown_label"]
        held = y == unk
        entry = {"rules": {}, "per_class_f1": {}, "complementarity": {}}
        masks, novs = {}, {}
        for s in SCALES:
            masks[s], novs[s] = caught(runs[s], strat)
            entry["per_class_f1"][s] = per_class_f1(runs[s])
        best = max(SCALES, key=lambda s: masks[s][held].mean())
        union = np.any([masks[s][held] for s in SCALES], axis=0)
        inter = np.all([masks[s][held] for s in SCALES], axis=0)
        entry["complementarity"] = {
            "recall_per_scale": {s: float(masks[s][held].mean()) for s in SCALES},
            "best_single_scale": best,
            "union_recall": float(union.mean()),
            "all_three_recall": float(inter.mean()),
            "oracle_gain_over_best": float(union.mean() - masks[best][held].mean()),
            "spearman_novelty_held_out": {
                f"{a}-{b}": float(spearmanr(novs[a][held], novs[b][held])[0])
                for a, b in (("short", "medium"), ("short", "long"), ("medium", "long"))},
        }
        for rule in rules:
            entry["rules"][rule] = eval_strategy(fuse(runs, rule), strat)
        results[f"{fam}_s{seed}"] = entry
        print(f"{fam} s{seed}: " + "  ".join(
            f"{r}={entry['rules'][r]['unknown_recall']:.1%}" for r in rules)
            + f"  | union={entry['complementarity']['union_recall']:.1%}", flush=True)

    # family-averaged comparison over seeds
    fams = sorted({k.rsplit("_s", 1)[0] for k in results})
    seeds = sorted({int(k.rsplit("_s", 1)[1]) for k in results})
    def fam_avg(rule, metric, seed):
        vals = [results[f"{f}_s{seed}"]["rules"][rule][metric] for f in fams
                if f"{f}_s{seed}" in results]
        return float(np.mean(vals))
    table = {}
    for rule in rules:
        table[rule] = {m: ev.mean_std([fam_avg(rule, m, s) for s in seeds])
                       for m in ("unknown_recall", "held_to_normal", "false_unknown",
                                 "benign_fpr", "known_macro_f1", "auroc", "auprc")}
    verdict = {}
    for cand in ("adaptive_rel", "select", "adaptive_rel_raw", "select_raw"):
        d_rec = [fam_avg(cand, "unknown_recall", s) - fam_avg("max", "unknown_recall", s) for s in seeds]
        d_f1 = [fam_avg(cand, "known_macro_f1", s) - fam_avg("max", "known_macro_f1", s) for s in seeds]
        mean_d, sd_d = float(np.mean(d_rec)), float(np.std(d_rec))
        improved_fams = sum(
            np.mean([results[f"{f}_s{s}"]["rules"][cand]["unknown_recall"] -
                     results[f"{f}_s{s}"]["rules"]["max"]["unknown_recall"] for s in seeds
                     if f"{f}_s{s}" in results]) > 0 for f in fams)
        measurable = (mean_d > 2 * sd_d if len(seeds) > 1 else mean_d > 0.01) and np.mean(d_f1) >= -0.005
        verdict[cand] = {"recall_gain_vs_max": (mean_d, sd_d),
                         "known_f1_change_vs_max": (float(np.mean(d_f1)), float(np.std(d_f1))),
                         "families_improved": f"{improved_fams}/{len(fams)}",
                         "measurable_improvement": bool(measurable)}

    print(f"\n=== family average, strategy={strat}, {float(BUDGET):.0%} budget, seeds {seeds} ===")
    print(f"{'rule':<17}{'unk.recall':>16}{'->Normal':>10}{'false-unk':>11}{'benignFPR':>11}"
          f"{'knownF1':>9}{'AUROC':>8}{'AUPRC':>8}")
    for rule in rules:
        t = table[rule]
        print(f"{rule:<17}{t['unknown_recall'][0]:>9.1%}±{t['unknown_recall'][1]:<5.1%}"
              f"{t['held_to_normal'][0]:>10.1%}{t['false_unknown'][0]:>11.2%}"
              f"{t['benign_fpr'][0]:>11.2%}{t['known_macro_f1'][0]:>9.4f}"
              f"{t['auroc'][0]:>8.3f}{t['auprc'][0]:>8.3f}")
    for cand, v in verdict.items():
        print(f"\n{cand} vs max: recall {v['recall_gain_vs_max'][0]:+.2%} "
              f"(seed sd {v['recall_gain_vs_max'][1]:.2%}), known F1 "
              f"{v['known_f1_change_vs_max'][0]:+.4f}, families improved {v['families_improved']} "
              f"-> {'measurable improvement' if v['measurable_improvement'] else 'no measurable improvement; keep max'}")

    out = data / "phase3"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"scale_fusion_{strat}.json", "w") as f:
        json.dump({"strategy": strat, "budget": BUDGET, "seeds": seeds, "families": fams,
                   "per_run": results, "family_average": table, "verdict": verdict}, f, indent=2)
    print(f"\nwritten to {out / f'scale_fusion_{strat}.json'}")

if __name__ == "__main__":
    main()
