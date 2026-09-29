"""
Collects every hold-out's summary.json into the leave-one-class-out table.

Reads data/banks_<scale>_*/summary.json and prints one row per held-out class:
how separable that class was (AUROC), how much of it the unknown branch caught
at each false-unknown budget, and the closed-set numbers it is measured
against. Classes are ordered by AUROC so the easy and hard cases are visible
as a spread rather than a single headline number.
"""
import json
from pathlib import Path

SCALE = "medium"
BUDGETS = [0.01, 0.05, 0.10, 0.20]

def find_dir(name):
    here = Path(__file__).resolve().parent
    for candidate in (here.parent / name, here / name):
        if candidate.exists():
            return candidate
    return here / name

def main():
    data = find_dir("data")
    summaries = []
    for d in sorted(data.glob(f"banks_{SCALE}_*")):
        f = d / "summary.json"
        if f.exists():
            summaries.append(json.load(open(f)))
    if not summaries:
        print("no summary.json found; run 12_openset_sweep.sh first")
        return

    summaries.sort(key=lambda s: -s["auroc_known_vs_unknown"])
    caught = {b: f"caught@{int(b*100)}%" for b in BUDGETS}

    print(f"leave-one-class-out open-set results, {SCALE} scale, "
          f"{len(summaries)} classes\n")
    head = f"{'held out':<16}{'rows':>7}{'AUROC':>8}"
    for b in BUDGETS:
        head += f"{caught[b]:>12}"
    head += f"{'known acc':>11}{'closed acc':>12}"
    print(head)
    print("-" * len(head))
    for s in summaries:
        by_budget = {round(r["budget"], 4): r for r in s["curve"]}
        line = (f"{s['held_out_class']:<16}{s['n_unknown_test_rows']:>7d}"
                f"{s['auroc_known_vs_unknown']:>8.3f}")
        for b in BUDGETS:
            r = by_budget.get(round(b, 4))
            line += f"{r['unknown_caught']:>11.1%} " if r else f"{'-':>12}"
        r20 = by_budget.get(0.2)
        line += f"{r20['known_acc']:>11.4f}" if r20 else f"{'-':>11}"
        line += f"{s['closed_set']['known_acc']:>12.4f}"
        print(line)

    print("\nknown acc is at the 20% budget; closed acc is the same model with no "
          "unknown option.\nEvery closed-set model catches 0% of the held-out class "
          "by construction.\n")
    print("where the held-out rows go at the 5% budget:")
    for s in summaries:
        top = sorted(s["landing_at_5pct"].items(), key=lambda t: -t[1])[:3]
        n = s["n_unknown_test_rows"]
        parts = ", ".join(f"{k} {v/n:.0%}" for k, v in top)
        print(f"  {s['held_out_class']:<16} {parts}")

if __name__ == "__main__":
    main()
