"""
Builds the Phase 3 results tables directly from the experiment outputs.

Reads data/tuned/*.json, data/phase3/closed_summary.json,
openset_eval_<scale>.json, scale_fusion_<strategy>.json,
baselines_openset.json and explain_openset.json, and writes markdown tables
to data/phase3/tables.md, so that no number in the write-up is transcribed by
hand. Copies the score-distribution plots alongside.
"""
import json
import os
import shutil
from pathlib import Path

import numpy as np

FAMS = ["analysis", "backdoor", "dos", "exploits", "fuzzers", "generic",
        "reconnaissance", "shellcode", "worms"]
STRATS = ["closed", "global", "classcond", "margin", "combined"]

def find_dir(name):
    if name == "data":
        override = os.environ.get("TM_DATA_DIR")
        if override:
            return Path(override)
    here = Path(__file__).resolve().parent
    for candidate in (here.parent / name, here / name):
        if candidate.exists():
            return candidate
    return here / name

def pm(v, pct=True, digits=1):
    m, s = v
    if m is None or (isinstance(m, float) and np.isnan(m)):
        return "–"
    return f"{m:.{digits}%} ± {s:.{digits}%}" if pct else f"{m:.{digits}f} ± {s:.{digits}f}"

def load(p):
    return json.load(open(p)) if p.exists() else None

def main():
    data = find_dir("data")
    p3 = data / "phase3"
    out = []

    # ---- Task 1: tuned hyperparameters ----
    out.append("## Task 1 — tuned hyperparameters\n")
    out.append("| model | clauses | T | s | epochs | val macro F1 |\n|---|---|---|---|---|---|")
    c = load(data / "tuned" / "closed.json")
    if c:
        b = c["best"]
        out.append(f"| closed-set 10-class TM | {b['clauses']} | {b['T']} | {b['s']} | {b['epochs']} | {c['best_val_macro_f1']:.4f} |")
    for f in FAMS:
        t = load(data / "tuned" / f"open_{f}.json")
        if t:
            b = t["best"]
            out.append(f"| banks, {f} held out | {b['clauses']} | {b['T']} | {b['s']} | {b['epochs']} | {t['best_val_macro_f1']:.4f} |")
    cs = load(p3 / "closed_summary.json")
    if cs:
        out.append(f"\n### Closed-set TM over seeds {cs['seeds']}\n")
        out.append("| configuration | accuracy | weighted F1 | macro F1 |\n|---|---|---|---|")
        for k, v in cs["summary"].items():
            out.append(f"| {k} | {pm(v['acc'], False, 4)} | {pm(v['w_f1'], False, 4)} | {pm(v['macro_f1'], False, 4)} |")

    # ---- Tasks 2/3/5: open-set evaluation ----
    ev = load(p3 / "openset_eval_medium.json")
    if ev:
        b = "0.05"
        out.append("\n## Task 5 — leave-one-family-out, medium scale, global thresholds at a 5% validation false-rejection budget\n")
        out.append("| held out | test rows | unknown recall (95% CI, seed 1) | AUROC | AUPRC | false-unknown | benign FPR | known macro F1 | known-attack macro F1 |\n|---|---|---|---|---|---|---|---|---|")
        for f in FAMS:
            if f not in ev["summary"]:
                continue
            s = ev["summary"][f]["global"][b]
            seed0 = sorted(ev["results"][f]["seeds"], key=int)[0]
            r0 = ev["results"][f]["seeds"][seed0]
            ci = r0["strategies"]["global"][b]["recall_ci95"]
            out.append(f"| {f} | {r0['n_unknown']} | {pm(s['unknown_recall'])} ({ci[0]:.1%}–{ci[1]:.1%}) | "
                       f"{pm(s['auroc'], False, 3)} | {pm(s['auprc'], False, 3)} | {pm(s['false_unknown'], True, 2)} | "
                       f"{pm(s['benign_fpr'], True, 2)} | {pm(s['known_macro_f1'], False, 4)} | {pm(s['known_attack_macro_f1'], False, 4)} |")
        a = ev["family_average"]["global"][b]
        out.append(f"| **average** | | {pm(a['unknown_recall'])} | {pm(a['auroc'], False, 3)} | {pm(a['auprc'], False, 3)} | "
                   f"{pm(a['false_unknown'], True, 2)} | {pm(a['benign_fpr'], True, 2)} | {pm(a['known_macro_f1'], False, 4)} | "
                   f"{pm(a['known_attack_macro_f1'], False, 4)} |")

        out.append("\n## Task 2 — where held-out rows go (global, 5% budget, mean ± std over seeds)\n")
        out.append("| held out | (i) Unknown | (ii) a known family | (iii) Normal | held-out above tau_B | benign-support AUROC held-out vs known attacks | top-2 margin AUROC |\n|---|---|---|---|---|---|---|")
        for f in FAMS:
            if f not in ev["summary"]:
                continue
            s = ev["summary"][f]["global"][b]
            seed0 = sorted(ev["results"][f]["seeds"], key=int)[0]
            d = ev["results"][f]["seeds"][seed0]["distributions"]
            out.append(f"| {f} | {pm(s['unknown_recall'])} | {pm(s['held_to_known_family'])} | {pm(s['held_to_normal'])} | "
                       f"{d['held_out_above_tau_B']:.1%} | {d['benign'].get('auroc_held_vs_known_attack', float('nan')):.3f} | "
                       f"{d['top2_margin'].get('auroc_held_vs_known_attack', float('nan')):.3f} |")

        out.append("\n## Task 3 — threshold strategies, average across held-out families (mean ± std over seeds)\n")
        for bud in ("0.01", "0.05", "0.10"):
            out.append(f"\n**Validation false-rejection budget {float(bud):.0%}**\n")
            out.append("| strategy | unknown recall | held-out to Normal | false-unknown | benign FPR | known macro F1 | known-attack macro F1 | AUROC | AUPRC |\n|---|---|---|---|---|---|---|---|---|")
            for st in STRATS:
                key = "0.00" if st == "closed" else bud
                a = ev["family_average"][st][key]
                out.append(f"| {st} | {pm(a['unknown_recall'])} | {pm(a['held_to_normal'])} | {pm(a['false_unknown'], True, 2)} | "
                           f"{pm(a['benign_fpr'], True, 2)} | {pm(a['known_macro_f1'], False, 4)} | "
                           f"{pm(a['known_attack_macro_f1'], False, 4)} | {pm(a['auroc'], False, 3)} | {pm(a['auprc'], False, 3)} |")
        out.append("\n### Task 3 per family, unknown recall at a 5% budget\n")
        out.append("| held out | " + " | ".join(STRATS[1:]) + " |\n|---|" + "---|" * (len(STRATS) - 1))
        for f in FAMS:
            if f in ev["summary"]:
                out.append(f"| {f} | " + " | ".join(pm(ev["summary"][f][st]["0.05"]["unknown_recall"]) for st in STRATS[1:]) + " |")

    # ---- Task 4 ----
    for st in ("global", "margin", "classcond", "combined"):
        sf = load(p3 / f"scale_fusion_{st}.json")
        if not sf:
            continue
        out.append(f"\n## Task 4 — scale fusion ({st} thresholds, 5% budget, family average over seeds {sf['seeds']})\n")
        out.append("| rule | unknown recall | held-out to Normal | false-unknown | benign FPR | known macro F1 | AUROC | AUPRC |\n|---|---|---|---|---|---|---|---|")
        for rule, t in sf["family_average"].items():
            out.append(f"| {rule} | {pm(t['unknown_recall'])} | {pm(t['held_to_normal'])} | {pm(t['false_unknown'], True, 2)} | "
                       f"{pm(t['benign_fpr'], True, 2)} | {pm(t['known_macro_f1'], False, 4)} | {pm(t['auroc'], False, 3)} | {pm(t['auprc'], False, 3)} |")
        for cand, v in sf["verdict"].items():
            out.append(f"\n{cand} vs max: recall {v['recall_gain_vs_max'][0]:+.2%} (seed sd {v['recall_gain_vs_max'][1]:.2%}), "
                       f"known F1 {v['known_f1_change_vs_max'][0]:+.4f}, families improved {v['families_improved']} — "
                       f"{'measurable improvement' if v['measurable_improvement'] else 'no measurable improvement'}")
        if st == "global":
            out.append("\n### Complementarity per held-out family (first seed)\n")
            out.append("| held out | short | medium | long | union of scales | caught by all three | best single |\n|---|---|---|---|---|---|---|")
            seen = set()
            for k, e in sorted(sf["per_run"].items()):
                fam = k.rsplit("_s", 1)[0]
                if fam in seen:
                    continue
                seen.add(fam)
                c = e["complementarity"]; r = c["recall_per_scale"]
                out.append(f"| {fam} | {r['short']:.1%} | {r['medium']:.1%} | {r['long']:.1%} | {c['union_recall']:.1%} | "
                           f"{c['all_three_recall']:.1%} | {c['best_single_scale']} |")

    # ---- Task 6: baselines ----
    bl = load(p3 / "baselines_openset.json")
    if bl and ev:
        out.append("\n## Task 6 — open-set baselines vs TM (5% budget, family average)\n")
        out.append("| method | unknown recall | held-out to Normal | false-unknown | benign FPR | known macro F1 | AUROC | AUPRC |\n|---|---|---|---|---|---|---|---|")
        a = ev["family_average"]["global"]["0.05"]
        out.append(f"| TM (global thresholds) | {pm(a['unknown_recall'])} | {pm(a['held_to_normal'])} | {pm(a['false_unknown'], True, 2)} | "
                   f"{pm(a['benign_fpr'], True, 2)} | {pm(a['known_macro_f1'], False, 4)} | {pm(a['auroc'], False, 3)} | {pm(a['auprc'], False, 3)} |")
        methods = ["rf_msp", "xgb_msp", "iforest_rf", "mahalanobis"]
        for mth in methods:
            fams = [f for f in FAMS if f in bl]
            seeds = sorted(set.intersection(*[set(bl[f]["methods"][mth]["seeds"]) for f in fams]))
            def avg(metric):
                return [np.mean([bl[f]["methods"][mth]["seeds"][s]["0.05"][metric] for f in fams]) for s in seeds]
            row = {m: (float(np.mean(avg(m))), float(np.std(avg(m)))) for m in
                   ("unknown_recall", "held_to_normal", "false_unknown", "benign_fpr", "known_macro_f1", "auroc", "auprc")}
            out.append(f"| {mth} | {pm(row['unknown_recall'])} | {pm(row['held_to_normal'])} | {pm(row['false_unknown'], True, 2)} | "
                       f"{pm(row['benign_fpr'], True, 2)} | {pm(row['known_macro_f1'], False, 4)} | {pm(row['auroc'], False, 3)} | {pm(row['auprc'], False, 3)} |")
        out.append("\n### Unknown recall per family at 5% (mean over seeds)\n")
        out.append("| held out | TM | " + " | ".join(methods) + " |\n|---|---|" + "---|" * len(methods))
        for f in FAMS:
            if f in bl and f in ev["summary"]:
                cells = [f"{ev['summary'][f]['global']['0.05']['unknown_recall'][0]:.1%}"]
                for mth in methods:
                    sd = bl[f]["methods"][mth]["seeds"]
                    cells.append(f"{np.mean([sd[s]['0.05']['unknown_recall'] for s in sd]):.1%}")
                out.append(f"| {f} | " + " | ".join(cells) + " |")

    # ---- Task 6: explanations ----
    ex = load(p3 / "explain_openset.json")
    if ex:
        out.append("\n## Task 6 — explanations of the open-set decision (medium, seed 42)\n")
        out.append("Fidelity: agreement of the reduced-clause decision with the full decision.\n")
        groups = ["all_weighted", "normal", "known_attack", "unknown_caught", "unknown_to_family", "unknown_to_normal"]
        for k in ("1", "5", "25", "100"):
            out.append(f"\n**k = {k} clauses per bank**\n")
            out.append("| held out | " + " | ".join(groups) + " |\n|---|" + "---|" * len(groups))
            for f in FAMS:
                if f in ex:
                    cells = []
                    for g in groups:
                        v = ex[f]["fidelity"].get(g, {}).get(k)
                        cells.append("–" if v is None else f"{v:.1%}")
                    out.append(f"| {f} | " + " | ".join(cells) + " |")
        out.append("\n**Stability and complexity**\n")
        out.append("| held out | group | Jaccard (1 literal flipped) | decision change | cited clauses | literals per cited clause (median) |\n|---|---|---|---|---|---|")
        for f in FAMS:
            if f in ex:
                for g, s in ex[f]["stability"].items():
                    if s:
                        c = ex[f]["complexity"][g]
                        out.append(f"| {f} | {g} | {s['jaccard_mean']:.3f} | {s['decision_change_rate']:.1%} | "
                                   f"{c['cited_clauses_mean']:.1f} | {c['literals_per_cited_clause_median']:.0f} |")

    p3.mkdir(parents=True, exist_ok=True)
    (p3 / "tables.md").write_text("\n".join(out) + "\n")
    print(f"written {p3 / 'tables.md'} ({len(out)} lines)")

if __name__ == "__main__":
    main()
