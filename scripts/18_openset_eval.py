"""
Open-set evaluation across every held-out family and seed (Tasks 2, 3, 5).

Reads data/banks_<scale>_<family>_s<seed>/ for every family and seed present
and evaluates four novelty-decision strategies on identical bank scores:

  global       existing approach: benign gate tau_B, then Unknown when the
               strongest family score falls below one global threshold
  classcond    benign gate, then Unknown when the predicted family's score
               falls below that family's own threshold; per-family
               thresholds sit at a common quantile q of each family's
               validation scores, with q chosen to meet the budget
  margin       benign gate, then Unknown when the gap between the best and
               second-best family scores is small
  combined     no separate gate: each row gets a benign-typicality p_B (ECDF
               of benign support among validation Normal rows) and an
               attack-typicality p_A (ECDF of the strongest family score
               among validation attack rows); novelty = 1 - max(p_B, p_A),
               and non-novel rows go to Normal or to the strongest family,
               whichever is more typical

Every strategy is calibrated on validation only, and validation never
contains the held-out family (09 removes it). All strategies share the same
benign gate where they use one (tau_B at a 1% missed-attack budget on known
validation attacks) and are held to the same false-rejection budget: the
share of ALL known validation rows predicted Unknown. A closed-set row (gate
plus argmax, no Unknown) is reported as the reference for whether better
novelty detection costs benign or known-attack accuracy.

Threshold-free metrics use a system-level novelty score consistent with each
decision rule: rows the benign gate admits as Normal rank below every gated
row, so AUROC/AUPRC describe the whole pipeline rather than only the rows
that clear the gate.

Writes data/phase3/openset_eval_<scale>.json and score-distribution plots
(Task 2) to data/phase3/plots/.

Usage: 18_openset_eval.py [scale]
"""
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             precision_recall_fscore_support, roc_auc_score)

MISSED_ATTACK_BUDGET = 0.01
BUDGETS = [0.01, 0.05, 0.10]
REPORT_BUDGET = 0.05
STRATEGIES = ["closed", "global", "classcond", "margin", "combined"]
BIG = 1e6
MIN_ROWS_CLASSCOND = 20
FAMILY_ORDER = ["Analysis", "Backdoor", "DoS", "Exploits", "Fuzzers", "Generic",
                "Reconnaissance", "Shellcode", "Worms"]

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

def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (float(c - h), float(c + h))

def load_run(d, data):
    meta = json.load(open(d / "banks_meta.json"))
    fams = meta["families"]
    r = {"meta": meta, "families": fams}
    r["vb"] = np.load(d / "val_normal.npy").astype(np.float64)
    r["tb"] = np.load(d / "test_normal.npy").astype(np.float64)
    r["vf"] = np.stack([np.load(d / f"val_fam{c}.npy") for c in fams], 1).astype(np.float64)
    r["tf"] = np.stack([np.load(d / f"test_fam{c}.npy") for c in fams], 1).astype(np.float64)
    r["y_val"] = np.load(d / "y_val.npy").astype(np.int64)
    fam = meta["held_out_class"].lower()
    r["y_test"] = np.load(data / f"openset_{meta['scale']}_{fam}" / "y_test.npy").astype(np.int64)
    # Bank margins are integers, so a quantile threshold usually lands on a
    # tied value and a strategy rejects fewer rows than its budget allows.
    # A seeded jitter far below the integer gap breaks ties at random without
    # reordering distinct scores, so every strategy meets its budget exactly.
    rng = np.random.default_rng(0)
    for k in ("vb", "tb", "vf", "tf"):
        r[k] = r[k] + rng.uniform(0, 1e-3, r[k].shape)
    return r

def ecdf(ref):
    ref = np.sort(ref)
    return lambda x: np.searchsorted(ref, x, side="right") / max(len(ref), 1)

class Calibrated:
    """Per-run calibration of every strategy on validation."""
    def __init__(self, r):
        self.r = r
        m = r["meta"]
        self.fams = np.asarray(r["families"])
        self.ni = m["normal_idx"]
        self.unk = m["unknown_label"]
        vb, vf, vy = r["vb"], r["vf"], r["y_val"]
        atk = vy != self.ni
        self.tau_b = float(np.quantile(vb[atk], 1.0 - MISSED_ATTACK_BUDGET))
        self.pB = ecdf(vb[~atk])
        self.pA = ecdf(vf.max(1)[atk])

    def parts(self, b, f):
        top = np.sort(f, axis=1)
        s_known = top[:, -1]
        margin = top[:, -1] - top[:, -2]
        best = self.fams[f.argmax(1)]
        gated = b < self.tau_b
        return s_known, margin, best, gated

    def predict(self, strat, b, f, thr):
        s_known, margin, best, gated = self.parts(b, f)
        pred = np.full(len(b), self.ni, dtype=np.int64)
        if strat == "closed":
            pred[gated] = best[gated]
            return pred, np.where(gated, -s_known, -s_known - BIG)
        if strat == "global":
            nov = -s_known
        elif strat == "margin":
            nov = -margin
        elif strat == "classcond":
            tau_k = thr["tau_k"]
            own = f[np.arange(len(b)), f.argmax(1)]
            nov = np.array([tau_k[k] for k in best]) - own if len(b) else own
        elif strat == "combined":
            pb, pa = self.pB(b), self.pA(s_known)
            nov = 1.0 - np.maximum(pb, pa)
            pred = np.where(pb >= pa, self.ni, best)
            pred[nov > thr["theta"]] = self.unk
            return pred, nov
        pred[gated] = best[gated]
        cut = thr["theta"] if strat != "classcond" else 0.0
        pred[gated & (nov > cut)] = self.unk
        return pred, np.where(gated, nov, nov - BIG)

    def calibrate(self, strat, beta):
        """Threshold giving a validation false-rejection rate of beta over all
        known validation rows."""
        r = self.r
        vb, vf = r["vb"], r["vf"]
        n = len(vb)
        k = int(round(beta * n))
        if strat == "closed":
            return {}
        s_known, margin, best, gated = self.parts(vb, vf)
        if strat in ("global", "margin"):
            nov = (-s_known if strat == "global" else -margin)[gated]
            if k >= len(nov):
                return {"theta": -np.inf, "saturated": True}
            return {"theta": float(np.sort(nov)[::-1][k]) if k > 0 else float(nov.max())}
        if strat == "combined":
            pb, pa = self.pB(vb), self.pA(s_known)
            nov = 1.0 - np.maximum(pb, pa)
            return {"theta": float(np.quantile(nov, 1.0 - beta))}
        # classcond: common quantile q across families, bisected to meet beta
        vy = r["y_val"]
        own = vf[np.arange(n), vf.argmax(1)]
        pooled = s_known[gated & (vy != self.ni)]
        per = {}
        for j, c in enumerate(self.fams):
            sel = gated & (vy == c)
            per[c] = vf[sel, j] if sel.sum() >= MIN_ROWS_CLASSCOND else pooled
        def rate(q):
            tau_k = {c: np.quantile(v, q) for c, v in per.items()}
            nov = np.array([tau_k[c] for c in best]) - own
            return float((gated & (nov > 0)).mean()), tau_k
        lo, hi = 0.0, 1.0
        for _ in range(40):
            mid = (lo + hi) / 2
            if rate(mid)[0] < beta:
                lo = mid
            else:
                hi = mid
        return {"q": hi, "tau_k": rate(hi)[1]}

def evaluate_run(r):
    cal = Calibrated(r)
    ni, unk = cal.ni, cal.unk
    y = r["y_test"]
    is_unk = y == unk
    known = ~is_unk
    known_classes = sorted(set(np.unique(y[known]).tolist()))
    fams = list(cal.fams)
    out = {"tau_B": cal.tau_b, "n_unknown": int(is_unk.sum()), "strategies": {}}
    for strat in STRATEGIES:
        out["strategies"][strat] = {}
        for beta in (BUDGETS if strat != "closed" else [0.0]):
            thr = cal.calibrate(strat, beta)
            vpred, _ = cal.predict(strat, r["vb"], r["vf"], thr)
            pred, nov = cal.predict(strat, r["tb"], r["tf"], thr)
            p_unk = pred[is_unk]
            res = {
                "val_false_reject": float((vpred == unk).mean()),
                "unknown_recall": float((p_unk == unk).mean()),
                "held_to_known_family": float(np.isin(p_unk, fams).mean()),
                "held_to_normal": float((p_unk == ni).mean()),
                "false_unknown": float((pred[known] == unk).mean()),
                "benign_fpr": float((pred[y == ni] != ni).mean()),
                "known_macro_f1": float(precision_recall_fscore_support(
                    y[known], pred[known], labels=known_classes, average="macro",
                    zero_division=0)[2]),
                "known_attack_macro_f1": float(precision_recall_fscore_support(
                    y[known], pred[known], labels=fams, average="macro", zero_division=0)[2]),
                "auroc": float(roc_auc_score(is_unk, nov)),
                "auprc": float(average_precision_score(is_unk, nov)),
                "prevalence": float(is_unk.mean()),
                "recall_ci95": wilson(int((p_unk == unk).sum()), int(is_unk.sum())),
            }
            if strat == "classcond":
                res["q"] = thr["q"]
            if abs(beta - REPORT_BUDGET) < 1e-9 or strat == "closed":
                labels = known_classes + [unk]
                res["confusion"] = {"labels": labels,
                                    "matrix": confusion_matrix(y, pred, labels=labels).tolist()}
            out["strategies"][strat][f"{beta:.2f}"] = res
    out["distributions"] = distributions(r, cal)
    return out

def distributions(r, cal):
    """Summary statistics behind the Task 2 plots."""
    y, ni, unk = r["y_test"], cal.ni, cal.unk
    s_known, margin, best, gated = cal.parts(r["tb"], r["tf"])
    groups = {"held_out": y == unk, "known_attack": (y != unk) & (y != ni), "normal": y == ni}
    d = {}
    for name, score in (("benign", r["tb"]), ("known_support", s_known), ("top2_margin", margin)):
        d[name] = {g: {"median": float(np.median(score[m])) if m.any() else None,
                       "p10": float(np.quantile(score[m], 0.1)) if m.any() else None,
                       "p90": float(np.quantile(score[m], 0.9)) if m.any() else None}
                   for g, m in groups.items()}
        h, ka = groups["held_out"], groups["known_attack"]
        if h.any() and ka.any():
            lab = np.r_[np.ones(h.sum()), np.zeros(ka.sum())]
            d[name]["auroc_held_vs_known_attack"] = float(roc_auc_score(lab, np.r_[score[h], score[ka]]))
    d["held_out_above_tau_B"] = float((r["tb"][groups["held_out"]] >= cal.tau_b).mean())
    d["known_attack_above_tau_B"] = float((r["tb"][groups["known_attack"]] >= cal.tau_b).mean())
    return d

def plot_distributions(runs_by_family, out_dir):
    """Three panels per family: benign support, strongest family support and
    top-1 minus top-2 margin, for held-out, known-attack and Normal rows."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
    COL = {"held_out": "#2a78d6", "known_attack": "#eb6834", "normal": "#1baf7a"}
    LAB = {"held_out": "Held-out family (unknown)", "known_attack": "Known attacks",
           "normal": "Normal"}
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "text.color": INK})
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for fam, r in runs_by_family.items():
        cal = Calibrated(r)
        y = r["y_test"]
        s_known, margin, _, gated = cal.parts(r["tb"], r["tf"])
        groups = {"held_out": y == cal.unk, "known_attack": (y != cal.unk) & (y != cal.ni),
                  "normal": y == cal.ni}
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), facecolor=SURF)
        g_thr = cal.calibrate("global", REPORT_BUDGET).get("theta", np.nan)
        m_thr = cal.calibrate("margin", REPORT_BUDGET).get("theta", np.nan)
        panels = [("Benign support N(X)", r["tb"], None, cal.tau_b, "tau_B"),
                  ("Strongest family support (gated rows)", s_known, gated,
                   -g_thr, f"tau_A ({REPORT_BUDGET:.0%})"),
                  ("Top-1 minus top-2 family score (gated rows)", margin, gated,
                   -m_thr, f"margin cut ({REPORT_BUDGET:.0%})")]
        for ax, (title, score, mask, line, lname) in zip(axes, panels):
            ax.set_facecolor(SURF)
            lo, hi = np.quantile(score, [0.001, 0.999])
            bins = np.linspace(lo, hi, 60)
            for g in ("normal", "known_attack", "held_out"):
                m = groups[g] if mask is None else groups[g] & mask
                if g == "normal" and mask is not None:
                    continue
                if m.sum() < 5:
                    continue
                ax.hist(score[m], bins=bins, density=True, histtype="step", linewidth=2,
                        color=COL[g], label=LAB[g])
            ax.set_yscale("log")
            if line is not None and np.isfinite(line):
                ax.axvline(line, color=INK2, linestyle="--", linewidth=1)
                ax.text(line, ax.get_ylim()[1], f" {lname}", color=INK2, va="top", fontsize=8)
            ax.set_title(title, fontsize=9, color=INK, loc="left")
            ax.set_yticks([]); ax.minorticks_off()
            ax.set_ylabel("density (log)", fontsize=8)
            for sp_ in ("top", "right", "left"):
                ax.spines[sp_].set_visible(False)
            ax.grid(axis="x", color=GRID, linewidth=0.6)
        h, l = axes[0].get_legend_handles_labels()
        fig.legend(h, l, frameon=False, fontsize=8, loc="upper right", ncol=3)
        fig.suptitle(f"Held out: {fam}  (n = {int(groups['held_out'].sum())} test rows)",
                     x=0.01, ha="left", fontsize=10, color=INK)
        fig.tight_layout()
        p = out_dir / f"scores_{fam.lower()}.png"
        fig.savefig(p, dpi=150, facecolor=SURF)
        plt.close(fig)
        paths.append(str(p))
    return paths

def mean_std(vals):
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
    return (float(np.mean(vals)), float(np.std(vals))) if vals else (float("nan"), float("nan"))

def main():
    scale = sys.argv[1] if len(sys.argv) > 1 else "medium"
    data = find_dir("data")
    runs = {}
    for d in sorted(data.glob(f"banks_{scale}_*_s*")):
        m = re.match(rf"banks_{scale}_(.+)_s(\d+)$", d.name)
        # resampling variants (banks_<scale>_<family>_<strategy>_s<seed>)
        # belong to 24_single_stage.py's Task 1 study, not to this one
        if m and m.group(1) in [x.lower() for x in FAMILY_ORDER] and (d / "banks_meta.json").exists():
            runs.setdefault(m.group(1), {})[int(m.group(2))] = d
    if not runs:
        raise SystemExit(f"no bank runs for scale {scale}")

    results, plot_runs = {}, {}
    for fam_l in sorted(runs, key=lambda f: [x.lower() for x in FAMILY_ORDER].index(f)
                         if f in [x.lower() for x in FAMILY_ORDER] else 99):
        per_seed = {}
        for seed, d in sorted(runs[fam_l].items()):
            r = load_run(d, data)
            per_seed[seed] = evaluate_run(r)
            if seed == min(runs[fam_l]):
                plot_runs[r["meta"]["held_out_class"]] = r
        fam = next(iter(per_seed.values()))
        results[fam_l] = {"seeds": per_seed}
        print(f"evaluated {fam_l}: seeds {sorted(per_seed)}", flush=True)

    # aggregate mean/std over seeds, and the average across families
    metrics = ["unknown_recall", "held_to_known_family", "held_to_normal", "false_unknown",
               "benign_fpr", "known_macro_f1", "known_attack_macro_f1", "auroc", "auprc"]
    summary = {}
    for fam_l, res in results.items():
        summary[fam_l] = {}
        for strat in STRATEGIES:
            summary[fam_l][strat] = {}
            for bkey in res["seeds"][next(iter(res["seeds"]))]["strategies"][strat]:
                summary[fam_l][strat][bkey] = {
                    mt: mean_std([res["seeds"][s]["strategies"][strat][bkey][mt]
                                  for s in res["seeds"]]) for mt in metrics}
    avg = {}
    for strat in STRATEGIES:
        avg[strat] = {}
        bkeys = summary[next(iter(summary))][strat].keys()
        for bkey in bkeys:
            avg[strat][bkey] = {}
            for mt in metrics:
                seeds = sorted(set.intersection(*[set(results[f]["seeds"]) for f in results]))
                per_seed_avg = [np.mean([results[f]["seeds"][s]["strategies"][strat][bkey][mt]
                                         for f in results]) for s in seeds]
                avg[strat][bkey][mt] = mean_std(per_seed_avg)

    out_dir = data / "phase3"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots = plot_distributions(plot_runs, out_dir / "plots")
    with open(out_dir / f"openset_eval_{scale}.json", "w") as f:
        json.dump({"scale": scale, "budgets": BUDGETS, "missed_attack_budget": MISSED_ATTACK_BUDGET,
                   "results": results, "summary": summary, "family_average": avg,
                   "plots": plots}, f, indent=2, default=float)

    # ---- console report ----
    b = f"{REPORT_BUDGET:.2f}"
    print(f"\n=== {scale}: per family at {REPORT_BUDGET:.0%} validation false-rejection "
          f"(mean +/- std over seeds) ===")
    print(f"{'family':<15}{'strategy':<11}{'unk.recall':>14}{'->family':>11}{'->Normal':>11}"
          f"{'false-unk':>11}{'benignFPR':>11}{'knownF1':>9}{'AUROC':>8}{'AUPRC':>8}")
    for fam_l in summary:
        for strat in STRATEGIES:
            key = "0.00" if strat == "closed" else b
            s = summary[fam_l][strat][key]
            print(f"{fam_l:<15}{strat:<11}{s['unknown_recall'][0]:>8.1%}±{s['unknown_recall'][1]:<5.1%}"
                  f"{s['held_to_known_family'][0]:>11.1%}{s['held_to_normal'][0]:>11.1%}"
                  f"{s['false_unknown'][0]:>11.2%}{s['benign_fpr'][0]:>11.2%}"
                  f"{s['known_macro_f1'][0]:>9.4f}{s['auroc'][0]:>8.3f}{s['auprc'][0]:>8.3f}")
    print("\n=== average across held-out families ===")
    for strat in STRATEGIES:
        key = "0.00" if strat == "closed" else b
        s = avg[strat][key]
        print(f"  {strat:<10} unknown recall {s['unknown_recall'][0]:.1%}±{s['unknown_recall'][1]:.1%}  "
              f"->Normal {s['held_to_normal'][0]:.1%}  false-unknown {s['false_unknown'][0]:.2%}  "
              f"benign FPR {s['benign_fpr'][0]:.2%}  known F1 {s['known_macro_f1'][0]:.4f}  "
              f"AUROC {s['auroc'][0]:.3f}  AUPRC {s['auprc'][0]:.3f}")
    print(f"\nplots: {out_dir / 'plots'}\nwritten to {out_dir / f'openset_eval_{scale}.json'}")

if __name__ == "__main__":
    main()
