"""
Re-derives Step 5's decision layer for every hold-out under a comparable
benign threshold, and prints the leave-one-class-out table.

11_openset_decision.py picks tau_B by maximising balanced accuracy, which is
unstable when the normality bank is very accurate: many thresholds score alike,
so the argmax wanders and swung from -154 to +333 across hold-outs. At the low
end it silently routed novel attack rows to Normal, where they can never reach
the unknown branch at all.

Here tau_B is pinned instead to a missed-attack budget: the threshold at which
at most MISSED_ATTACK_BUDGET of known attack validation rows are called benign.
That is identical in meaning across hold-outs, puts the safety-critical error
under explicit control, and is reported alongside the benign recall it costs.

Reads the saved bank scores, so it needs no retraining.
"""
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             roc_auc_score)

SCALE = "medium"
MISSED_ATTACK_BUDGET = 0.01     # share of known attack val rows allowed to look benign
FALSE_UNKNOWN_BUDGETS = [0.01, 0.02, 0.05, 0.10, 0.20]
REPORT_BUDGETS = [0.01, 0.05, 0.10, 0.20]

def find_dir(name):
    here = Path(__file__).resolve().parent
    for candidate in (here.parent / name, here / name):
        if candidate.exists():
            return candidate
    return here / name

def decide(s_benign, s_family, families, tau_b, tau_a, normal_idx, unknown_label):
    """Spec Step 5's three-way rule, vectorised over rows."""
    s_known = s_family.max(axis=1)
    best_family = np.asarray(families)[s_family.argmax(axis=1)]
    pred = np.full(len(s_benign), unknown_label, dtype=np.int64)
    is_attack = s_benign < tau_b
    pred[~is_attack] = normal_idx
    take_known = is_attack & (s_known >= tau_a)
    pred[take_known] = best_family[take_known]
    return pred

def evaluate(banks):
    meta = json.load(open(banks / "banks_meta.json"))
    families = meta["families"]
    normal_idx = meta["normal_idx"]
    names = meta["all_class_names"]
    held = names[meta["held_out_idx"]]
    unknown_label = meta["unknown_label"]
    n_real = meta["n_real_train_rows"]

    y_val = np.load(banks / "y_val.npy")
    val_idx = np.load(banks / "val_idx.npy")
    y_test = np.load(find_dir("data") / f"openset_{SCALE}_{held.lower()}" / "y_test.npy")
    y_test = y_test.astype(np.int64)

    val_b = np.load(banks / "val_normal.npy")
    test_b = np.load(banks / "test_normal.npy")
    val_f = np.stack([np.load(banks / f"val_fam{c}.npy") for c in families], axis=1)
    test_f = np.stack([np.load(banks / f"test_fam{c}.npy") for c in families], axis=1)

    real = val_idx < n_real
    vb, vf, vy = val_b[real], val_f[real], y_val[real]

    # tau_B at a fixed missed-attack budget, so every hold-out uses the same rule
    val_attack_rows = vy != normal_idx
    tau_b = float(np.quantile(vb[val_attack_rows], 1.0 - MISSED_ATTACK_BUDGET))
    benign_recall = float((vb[~val_attack_rows] >= tau_b).mean())

    is_unknown = y_test == unknown_label
    known_mask = ~is_unknown
    known_attack = known_mask & (y_test != normal_idx)

    gate = test_b < tau_b
    a, k = test_f.max(axis=1)[gate & is_unknown], test_f.max(axis=1)[gate & known_attack]
    auroc = (float(roc_auc_score(np.r_[np.zeros(len(a)), np.ones(len(k))], np.r_[a, k]))
             if len(a) and len(k) else float("nan"))

    val_known_max = vf.max(axis=1)[vb < tau_b]
    curve = []
    for budget in FALSE_UNKNOWN_BUDGETS:
        tau_a = float(np.quantile(val_known_max, budget)) if len(val_known_max) else 0.0
        pred = decide(test_b, test_f, families, tau_b, tau_a, normal_idx, unknown_label)
        kept = known_mask & (pred != unknown_label)
        curve.append({
            "budget": budget, "tau_A": tau_a,
            "unknown_caught": float((pred[is_unknown] == unknown_label).mean()),
            "false_unknown": float((pred[known_mask] == unknown_label).mean()),
            "novel_called_benign": float((pred[is_unknown] == normal_idx).mean()),
            "known_acc": float(accuracy_score(y_test[kept], pred[kept])) if kept.sum() else 0.0,
            "macro_f1": float(precision_recall_fscore_support(
                y_test, pred, average="macro", zero_division=0)[2]),
        })

    closed = np.asarray(families)[test_f.argmax(axis=1)]
    closed[test_b >= tau_b] = normal_idx
    return {
        "held_out_class": held,
        "n_unknown_test_rows": int(is_unknown.sum()),
        "tau_B": tau_b,
        "benign_recall_at_tau_B": benign_recall,
        "auroc_known_vs_unknown": auroc,
        "closed_set": {
            "known_acc": float(accuracy_score(y_test[known_mask], closed[known_mask])),
            "macro_f1": float(precision_recall_fscore_support(
                y_test, closed, average="macro", zero_division=0)[2]),
        },
        "curve": curve,
    }

def main():
    data = find_dir("data")
    results = []
    for d in sorted(data.glob(f"banks_{SCALE}_*")):
        if (d / "banks_meta.json").exists() and (d / "test_normal.npy").exists():
            try:
                results.append(evaluate(d))
            except FileNotFoundError as e:
                print(f"skipping {d.name}: {e}")
    if not results:
        print("no completed hold-outs found")
        return

    results.sort(key=lambda r: -r["auroc_known_vs_unknown"])
    print(f"leave-one-class-out open-set results, {SCALE} scale, "
          f"{len(results)} hold-outs")
    print(f"tau_B pinned at a {MISSED_ATTACK_BUDGET:.0%} missed-attack budget "
          f"on known validation rows\n")

    head = f"{'held out':<16}{'rows':>7}{'AUROC':>8}{'tau_B':>8}{'ben.rec':>9}"
    for b in REPORT_BUDGETS:
        head += f"{'@' + format(b, '.0%'):>9}"
    head += f"{'known acc':>11}{'closed':>9}"
    print(head)
    print("-" * len(head))
    for r in results:
        by = {round(c["budget"], 4): c for c in r["curve"]}
        line = (f"{r['held_out_class']:<16}{r['n_unknown_test_rows']:>7d}"
                f"{r['auroc_known_vs_unknown']:>8.3f}{r['tau_B']:>8.0f}"
                f"{r['benign_recall_at_tau_B']:>9.3f}")
        for b in REPORT_BUDGETS:
            line += f"{by[round(b, 4)]['unknown_caught']:>9.1%}"
        line += f"{by[0.2]['known_acc']:>11.4f}{r['closed_set']['known_acc']:>9.4f}"
        print(line)

    print("\n@N% columns are the share of held-out rows flagged UNKNOWN at an N% "
          "false-unknown budget.")
    print("ben.rec is benign recall at tau_B; closed is the same model with no "
          "unknown option, which\ncatches 0% of the held-out class by construction.\n")

    print("novel rows wrongly routed to Normal (these never reach the unknown branch):")
    for r in results:
        by = {round(c["budget"], 4): c for c in r["curve"]}
        print(f"  {r['held_out_class']:<16} {by[0.05]['novel_called_benign']:>6.1%}")

    with open(data / f"openset_sweep_{SCALE}.json", "w") as f:
        json.dump({"missed_attack_budget": MISSED_ATTACK_BUDGET,
                   "results": results}, f, indent=2)
    print(f"\nwritten to {data / f'openset_sweep_{SCALE}.json'}")

if __name__ == "__main__":
    main()
