"""
Compares fusion rules for combining the short, medium and long clause banks.

Step 4 asks for a separate bank per temporal scale but does not state how
their scores should be combined, so three rules are compared on the same
bank outputs:

  mean       unweighted average across scales
  max        per-class maximum (what 08_multiscale_ensemble.py uses)
  adaptive   per-row weights proportional to each scale's confidence

NOTE ON "adaptive": the supplied methodology does not define a
confidence-adaptive rule, so this is an interpretation, not a transcription.
Confidence for a scale on a row is the margin between its best and
second-best family score, floored at zero and normalised across scales; a
scale that cannot separate its top two candidates on a given row is given
little say in that row's decision. Uniform weights are used where every
scale has zero margin.

Thresholds are selected on validation and applied once to test, matching
14_openset_redecide.py: tau_B is pinned to a missed-attack budget and tau_A
to a false-unknown budget.

Takes the held-out class as argv[1] and the normality mode as argv[2].
Reads data/banks_<scale>_<class>[_<mode>]/ for all three scales.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             roc_auc_score)

SCALES = ["short", "medium", "long"]
MISSED_ATTACK_BUDGET = 0.01
FALSE_UNKNOWN_BUDGETS = [0.01, 0.05, 0.10, 0.20]

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

def confidence_weights(family_stack):
    """Per-row, per-scale weights from the top-1 minus top-2 family margin.

    family_stack has shape (n_scales, n_rows, n_families). A scale whose two
    best candidates are nearly tied is expressing no preference on that row,
    so it is down-weighted there."""
    part = np.partition(family_stack, -2, axis=2)
    margin = np.clip(part[:, :, -1] - part[:, :, -2], 0, None)   # (scales, rows)
    total = margin.sum(axis=0, keepdims=True)
    flat = total[0] == 0
    w = np.where(total == 0, 1.0 / len(family_stack), margin / np.where(total == 0, 1, total))
    return w, int(flat.sum())

def fuse(benign_stack, family_stack, rule):
    """Returns fused (benign, family) scores from per-scale stacks."""
    if rule == "mean":
        return benign_stack.mean(axis=0), family_stack.mean(axis=0)
    if rule == "max":
        return benign_stack.max(axis=0), family_stack.max(axis=0)
    if rule == "adaptive":
        w, n_flat = confidence_weights(family_stack)
        fam = (family_stack * w[:, :, None]).sum(axis=0)
        ben = (benign_stack * w).sum(axis=0)
        return ben, fam
    raise ValueError(rule)

def decide(s_benign, s_family, families, tau_b, tau_a, normal_idx, unknown_label):
    s_known = s_family.max(axis=1)
    best = np.asarray(families)[s_family.argmax(axis=1)]
    pred = np.full(len(s_benign), unknown_label, dtype=np.int64)
    is_attack = s_benign < tau_b
    pred[~is_attack] = normal_idx
    take = is_attack & (s_known >= tau_a)
    pred[take] = best[take]
    return pred

def load_scale(data, scale, held, suffix, families):
    d = data / f"banks_{scale}_{held.lower()}{suffix}"
    if not (d / "banks_meta.json").exists():
        raise SystemExit(f"missing {d} -- run 10_train_family_banks.py for '{scale}' first")
    vb = np.load(d / "val_normal.npy")
    tb = np.load(d / "test_normal.npy")
    vf = np.stack([np.load(d / f"val_fam{c}.npy") for c in families], axis=1)
    tf = np.stack([np.load(d / f"test_fam{c}.npy") for c in families], axis=1)
    return vb, tb, vf, tf

def main():
    held = sys.argv[1] if len(sys.argv) > 1 else "Fuzzers"
    mode = sys.argv[2] if len(sys.argv) > 2 else "supervised"
    suffix = "" if mode == "supervised" else f"_{mode}"
    data = find_dir("data")

    ref = data / f"banks_medium_{held.lower()}{suffix}"
    meta = json.load(open(ref / "banks_meta.json"))
    families = meta["families"]
    normal_idx = meta["normal_idx"]
    names = meta["all_class_names"]
    unknown_label = meta["unknown_label"]

    y_val = np.load(ref / "y_val.npy")
    y_test = np.load(data / f"openset_medium_{held.lower()}" / "y_test.npy").astype(np.int64)

    vb, tb, vf, tf = [], [], [], []
    for sc in SCALES:
        a, b, c, d_ = load_scale(data, sc, held, suffix, families)
        vb.append(a); tb.append(b); vf.append(c); tf.append(d_)
    vb, tb = np.stack(vb), np.stack(tb)
    vf, tf = np.stack(vf), np.stack(tf)

    print(f"held out: {held} | normality mode: {mode}")
    print(f"scales: {SCALES} | val rows: {len(y_val)} | test rows: {len(y_test)}")
    _, n_flat = confidence_weights(vf)
    print(f"validation rows where every scale had zero margin "
          f"(adaptive falls back to uniform): {n_flat} ({n_flat/vf.shape[1]:.2%})\n")

    is_unknown = y_test == unknown_label
    known_mask = ~is_unknown
    known_attack = known_mask & (y_test != normal_idx)

    head = f"{'rule':<10}{'AUROC':>8}{'tau_B':>9}"
    for b in FALSE_UNKNOWN_BUDGETS:
        head += f"{'@' + format(b, '.0%'):>9}"
    head += f"{'known acc':>11}{'macroF1@5%':>12}{'novel->Normal':>15}"
    print(head)
    print("-" * len(head))

    results = {}
    for rule in ("mean", "max", "adaptive"):
        v_ben, v_fam = fuse(vb, vf, rule)
        t_ben, t_fam = fuse(tb, tf, rule)

        # tau_B at a fixed missed-attack budget on known validation rows
        val_attack = y_val != normal_idx
        tau_b = float(np.quantile(v_ben[val_attack], 1.0 - MISSED_ATTACK_BUDGET))

        gate = t_ben < tau_b
        a = t_fam.max(axis=1)[gate & is_unknown]
        k = t_fam.max(axis=1)[gate & known_attack]
        auroc = (float(roc_auc_score(np.r_[np.zeros(len(a)), np.ones(len(k))], np.r_[a, k]))
                 if len(a) and len(k) else float("nan"))

        val_known_max = v_fam.max(axis=1)[v_ben < tau_b]
        row, curve = {}, []
        for budget in FALSE_UNKNOWN_BUDGETS:
            tau_a = float(np.quantile(val_known_max, budget)) if len(val_known_max) else 0.0
            pred = decide(t_ben, t_fam, families, tau_b, tau_a, normal_idx, unknown_label)
            caught = float((pred[is_unknown] == unknown_label).mean())
            kept = known_mask & (pred != unknown_label)
            curve.append({
                "budget": budget, "tau_A": tau_a, "unknown_caught": caught,
                "false_unknown": float((pred[known_mask] == unknown_label).mean()),
                "novel_called_benign": float((pred[is_unknown] == normal_idx).mean()),
                "known_acc": float(accuracy_score(y_test[kept], pred[kept])) if kept.sum() else 0.0,
                "macro_f1": float(precision_recall_fscore_support(
                    y_test, pred, average="macro", zero_division=0)[2]),
            })
            row[budget] = caught

        at5 = [c for c in curve if c["budget"] == 0.05][0]
        at20 = [c for c in curve if c["budget"] == 0.20][0]
        line = f"{rule:<10}{auroc:>8.3f}{tau_b:>9.0f}"
        for b in FALSE_UNKNOWN_BUDGETS:
            line += f"{row[b]:>9.1%}"
        line += f"{at20['known_acc']:>11.4f}{at5['macro_f1']:>12.4f}{at5['novel_called_benign']:>15.1%}"
        print(line)
        results[rule] = {"auroc": auroc, "tau_B": tau_b, "curve": curve}

    print("\n@N% is the share of held-out rows flagged UNKNOWN at an N% "
          "false-unknown budget.\nknown acc is at the 20% budget; macroF1 and "
          "novel->Normal are at the 5% budget.")

    out = data / f"fusion_compare_{held.lower()}{suffix}.json"
    with open(out, "w") as f:
        json.dump({"held_out_class": held, "normality_mode": mode,
                   "scales": SCALES, "results": results}, f, indent=2)
    print(f"\nwritten to {out}")

if __name__ == "__main__":
    main()
