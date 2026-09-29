"""
Applies Step 5's decision hierarchy and evaluates unknown detection.

Implements the spec's rule over the bank scores from 10_train_family_banks.py:

    S_B  >= tau_B                  -> Benign
    S_known >= tau_A               -> Known attack, arg max_k S_k
    S_B < tau_B and S_known < tau_A -> Unknown anomaly

Both thresholds are chosen on real validation rows, which contain only
known classes, so the held-out attack's test rows are never used to pick
an operating point. Because a stricter threshold trades false unknowns
against caught unknowns, results are reported as a curve over several
operating points rather than a single number.

The comparison point is a closed-set model, which has no unknown option
and therefore misclassifies every held-out row by construction.

Takes the held-out class name as argv[1] and writes summary.json alongside
the bank scores so 13_openset_summary.py can collect every hold-out.
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             roc_auc_score)

SCALE = "medium"
RANDOM_STATE = 42
VAL_FRACTION = 0.10
# operating points, expressed as the share of known validation rows we are
# willing to see wrongly flagged unknown
FALSE_UNKNOWN_BUDGETS = [0.01, 0.02, 0.05, 0.10, 0.20]

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

def main():
    held = sys.argv[1] if len(sys.argv) > 1 else "DoS"
    banks = find_dir("data") / f"banks_{SCALE}_{held.lower()}"
    meta = json.load(open(banks / "banks_meta.json"))
    families = meta["families"]
    normal_idx = meta["normal_idx"]
    names = meta["all_class_names"]
    held_out_idx = meta["held_out_idx"]
    unknown_label = meta["unknown_label"]
    n_real = meta["n_real_train_rows"]

    y_val = np.load(banks / "y_val.npy")
    val_idx = np.load(banks / "val_idx.npy")
    y_test = np.load(find_dir("data") / f"openset_{SCALE}_{held.lower()}" / "y_test.npy")

    val_b = np.load(banks / "val_normal.npy")
    test_b = np.load(banks / "test_normal.npy")
    val_f = np.stack([np.load(banks / f"val_fam{c}.npy") for c in families], axis=1)
    test_f = np.stack([np.load(banks / f"test_fam{c}.npy") for c in families], axis=1)

    # only real validation rows are a fair surface for choosing thresholds
    real = val_idx < n_real
    vb, vf, vy = val_b[real], val_f[real], y_val[real]
    print(f"validation: {real.sum()} real rows of {len(val_idx)}")
    print(f"held out as unknown: {names[held_out_idx]} "
          f"({(y_test == unknown_label).sum()} test rows)\n")

    is_unknown = y_test == unknown_label
    known_mask = ~is_unknown

    # tau_b separates benign from attack; fix it on validation by balanced accuracy
    cand_b = np.quantile(vb, np.linspace(0.01, 0.99, 99))
    truth_normal = vy == normal_idx
    best_tau_b, best_bal = None, -1.0
    for t in cand_b:
        pred_normal = vb >= t
        tpr = (pred_normal & truth_normal).sum() / max(truth_normal.sum(), 1)
        tnr = ((~pred_normal) & (~truth_normal)).sum() / max((~truth_normal).sum(), 1)
        if (tpr + tnr) / 2 > best_bal:
            best_bal, best_tau_b = (tpr + tnr) / 2, t
    print(f"tau_B = {best_tau_b:.1f} (balanced accuracy {best_bal:.4f} on validation)\n")

    # tau_A sets how confident a family bank must be; sweep the budget
    val_attack = vb < best_tau_b
    val_known_max = vf.max(axis=1)[val_attack]

    print(f"{'budget':>8}{'tau_A':>10}{'unknown caught':>16}{'false unknown':>15}"
          f"{'known acc':>12}{'macro F1':>10}")
    print("-" * 71)
    rows = []
    for budget in FALSE_UNKNOWN_BUDGETS:
        # threshold that flags at most `budget` of attack-looking known val rows
        tau_a = np.quantile(val_known_max, budget) if len(val_known_max) else 0.0
        pred = decide(test_b, test_f, families, best_tau_b, tau_a, normal_idx, unknown_label)

        caught = (pred[is_unknown] == unknown_label).mean()
        false_unknown = (pred[known_mask] == unknown_label).mean()
        kept = known_mask & (pred != unknown_label)
        known_acc = accuracy_score(y_test[kept], pred[kept]) if kept.sum() else 0.0
        f1 = precision_recall_fscore_support(y_test, pred, average="macro",
                                              zero_division=0)[2]
        rows.append((budget, tau_a, caught, false_unknown, known_acc, f1))
        print(f"{budget:>8.0%}{tau_a:>10.1f}{caught:>15.1%}{false_unknown:>15.1%}"
              f"{known_acc:>12.4f}{f1:>10.4f}")

    print("\n=== closed-set comparison ===")
    closed = np.asarray(families)[test_f.argmax(axis=1)]
    closed[test_b >= best_tau_b] = normal_idx
    closed_known_acc = accuracy_score(y_test[known_mask], closed[known_mask])
    closed_f1 = precision_recall_fscore_support(y_test, closed, average="macro",
                                                 zero_division=0)[2]
    print(f"  a closed-set model has no unknown option, so all "
          f"{is_unknown.sum()} {names[held_out_idx]} rows are misclassified")
    print(f"  unknown caught: 0.0%  (by construction)")
    print(f"  known-class accuracy: {closed_known_acc:.4f}")
    print(f"  macro F1 over the same labels: {closed_f1:.4f}")

    # threshold-free measure of how separable the novel class is at all
    gate = test_b < best_tau_b
    known_attack = known_mask & (y_test != normal_idx)
    a, k = test_f.max(axis=1)[gate & is_unknown], test_f.max(axis=1)[gate & known_attack]
    auroc = (roc_auc_score(np.r_[np.zeros(len(a)), np.ones(len(k))], np.r_[a, k])
             if len(a) and len(k) else float("nan"))
    print(f"  AUROC, known-attack vs unknown by max family score: {auroc:.4f}")

    print(f"\n=== where {names[held_out_idx]} rows land at the 5% budget ===")
    tau_a = np.quantile(val_known_max, 0.05)
    pred = decide(test_b, test_f, families, best_tau_b, tau_a, normal_idx, unknown_label)
    vals, cnts = np.unique(pred[is_unknown], return_counts=True)
    landing = {}
    for v, c in sorted(zip(vals, cnts), key=lambda t: -t[1]):
        label = "UNKNOWN" if v == unknown_label else names[v]
        landing[label] = int(c)
        print(f"  {label:<16s} {c:>6d}  ({c/is_unknown.sum():.1%})")

    with open(banks / "summary.json", "w") as f:
        json.dump({"held_out_class": names[held_out_idx],
                   "n_unknown_test_rows": int(is_unknown.sum()),
                   "tau_B": float(best_tau_b),
                   "tau_B_balanced_acc": float(best_bal),
                   "auroc_known_vs_unknown": float(auroc),
                   "closed_set": {"known_acc": float(closed_known_acc),
                                  "macro_f1": float(closed_f1),
                                  "unknown_caught": 0.0},
                   "curve": [{"budget": float(b), "tau_A": float(t),
                              "unknown_caught": float(c), "false_unknown": float(fu),
                              "known_acc": float(ka), "macro_f1": float(f1)}
                             for b, t, c, fu, ka, f1 in rows],
                   "landing_at_5pct": landing}, f, indent=2)
    print(f"\nsummary written to {banks / 'summary.json'}")

if __name__ == "__main__":
    main()
