"""
Generates the Phase 5 tables from the run outputs.

Reads (each optional; a missing input skips its tables):
  <v1 data>/phase5/task2_v1.json        Task 1: results before the corrections
  <v2 data>/phase5/task2.json           Tasks 1, 2 and 5 (UNSW-NB15)
  <v2 data>/scores_medium_seed*/        closed-set TM (Task 2)
  <v2 data>/phase4/encoding_study.json  closed-set trees (Task 2)
  <v2 data>/phase5/benchmark.json       Task 3
  <v2 data>/phase5/explain_compare.json Task 4
  <cic data>/phase5/task2.json          Task 6 (CICIDS2017)

Writes <v2 data>/phase5/tables.md.

Usage: 33_phase5_tables.py <v1 data> <v2 data> <cic data>
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
UNSW = ["analysis", "backdoor", "dos", "exploits", "fuzzers", "generic", "reconnaissance",
        "shellcode", "worms"]
CIC = ["bot", "bruteforce", "ddos", "dos", "portscan", "webattack"]
SMALL = 1000
BUDGETS = ["0.01", "0.05", "0.10"]
MAIN_SYSTEMS = ["TM selected", "TM selected, balanced objective", "TM two_stage smote global",
                "RF boolean selected", "XGB boolean selected",
                "RF continuous_medium selected", "XGB continuous_medium selected"]
NAMES = {"TM selected": "TM (selected on validation)",
         "TM selected, balanced objective": "TM, sensitivity: balanced validation objective (post hoc)",
         "TM two_stage smote global": "TM two-stage, SMOTE, global (Phase 3 headline; not selected)",
         "RF boolean selected": "RF, Boolean input", "XGB boolean selected": "XGBoost, Boolean input",
         "RF continuous_medium selected": "RF, continuous input",
         "XGB continuous_medium selected": "XGBoost, continuous input"}

def load(p):
    p = Path(p)
    return json.load(open(p)) if p.exists() else None

def pct(ms, d=1):
    if ms is None:
        return "–"
    m, s = ms if isinstance(ms, (list, tuple)) else (ms, None)
    return f"{100*m:.{d}f}%" + (f" ± {100*s:.{d}f}" if s is not None else "")

def num(ms, d=3):
    if ms is None:
        return "–"
    m, s = ms if isinstance(ms, (list, tuple)) else (ms, None)
    return f"{m:.{d}f}" + (f" ± {s:.{d}f}" if s is not None else "")

def table(head, rows):
    return ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + \
           ["| " + " | ".join(str(c) for c in r) + " |" for r in rows] + [""]

MAIN_COLS = ["unknown recall", "unknown precision", "known-attack macro F1", "known rejected",
             "Normal false alarms", "AUROC", "AUPRC"]

def main_row(fa):
    return [pct(fa["unknown_recall"]), pct(fa["unknown_precision"]), num(fa["known_attack_macro_f1"]),
            pct(fa["known_rejection"], 2), pct(fa["normal_false_alarm"], 2),
            num(fa.get("auroc")) if "auroc" in fa else "–", num(fa.get("auprc")) if "auprc" in fa else "–"]

def describe_choice(sel):
    ch = sel["chosen"]
    picks = {}
    for f, v in ch.items():
        picks.setdefault(" / ".join(v["config"]), []).append(f)
    return "; ".join(f"`{k}` for {len(v)} fold(s)" + ("" if len(v) == len(ch) else f" ({', '.join(v)})")
                     for k, v in sorted(picks.items(), key=lambda kv: -len(kv[1])))

def open_set_tables(md, t2, fams, title):
    s = t2["summary"]
    if "selection" in t2 and t2["selection"]:
        md += [f"#### {title}: configuration chosen on validation (nested over folds)", ""]
        for k, v in t2["selection"].items():
            md.append(f"- **{k}**: {describe_choice(v)}")
        md.append("")
        J = t2["selection"].get("TM", {}).get("J", {})
        rows = sorted(((k, np.mean(list(v.values()))) for k, v in J.items()), key=lambda kv: -kv[1])
        md += ["Validation objective J (open-set macro F1 over known classes + Unknown, pseudo-unknowns, "
               "5% budget), mean over all nine folds and seeds:", ""]
        md += table(["TM configuration (architecture | resampling | rule)", "mean J"],
                    [[k.replace("|", " / "), f"{v:.4f}"] for k, v in rows])
    for b in BUDGETS:
        md += [f"#### {title}: {float(b):.0%} validation budget — family average (mean ± std over seeds)", ""]
        rows = [[NAMES.get(n, n)] + main_row(s[n][b]["family_average"]) for n in MAIN_SYSTEMS
                if n in s and b in s[n]]
        md += table(["system"] + MAIN_COLS, rows)
        md += [f"Micro-averaged over the {len(fams)} folds (counts pooled across folds, then the rate; "
               "AUROC/AUPRC are not pooled because different folds' scores are not on one scale):", ""]
        rows = [[NAMES.get(n, n)] + main_row(s[n][b]["micro"])[:5] for n in MAIN_SYSTEMS
                if n in s and b in s[n]]
        md += table(["system"] + MAIN_COLS[:5], rows)
    b = "0.05"
    md += [f"#### {title}: per held-out family, 5% budget (mean ± std over seeds)", ""]
    rows = []
    for f in fams:
        for n in MAIN_SYSTEMS:
            if n not in s or b not in s[n] or f not in s[n][b]["per_family"]:
                continue
            e = s[n][b]["per_family"][f]
            ci = e["recall_ci95_pooled_seeds"]
            flag = " ⚠" if e["n_test_unknown"] < SMALL else ""
            rows.append([f + flag, e["n_test_unknown"], e["n_val_heldout"], NAMES.get(n, n),
                         pct(e["unknown_recall"]) + f" [{100*ci[0]:.0f}–{100*ci[1]:.0f}]",
                         pct(e["unknown_precision"]), num(e["known_attack_macro_f1"]),
                         pct(e["normal_false_alarm"], 2), num(e["auroc"])])
    md += table(["held out", "test unknowns", "val pseudo-unknowns", "system",
                 "unknown recall [95% CI]", "unknown precision", "known-attack F1", "Normal FA", "AUROC"], rows)
    md += [f"⚠ fewer than {SMALL} test unknowns: recall rests on few rows; read the interval, not the point.", ""]

def change_table(md, v1, v2):
    md += ["### Task 1 — before (v1) and after (v2) the window corrections, seeds 7/42/1234, 5% budget, family average", ""]
    rows = []
    for name, e1 in v1["summary"].items():
        if not name.startswith("TM ") or name not in v2["summary"] or "0.05" not in e1:
            continue
        a, b = e1["0.05"]["family_average"], v2["summary"][name]["0.05"]["family_average"]
        rows.append([name[3:], pct(a["unknown_recall"]), pct(b["unknown_recall"]),
                     f"{100*(b['unknown_recall'][0]-a['unknown_recall'][0]):+.1f}",
                     num(a["known_attack_macro_f1"]), num(b["known_attack_macro_f1"]),
                     num(a["auroc"]), num(b["auroc"])])
    md += table(["configuration", "unknown recall v1", "v2", "Δ (points)", "known-attack F1 v1", "v2",
                 "AUROC v1", "v2"], rows)
    rows = []
    for f in UNSW:
        n = "TM two_stage smote global"
        if n in v1["summary"] and n in v2["summary"]:
            a = v1["summary"][n]["0.05"]["per_family"].get(f)
            b = v2["summary"][n]["0.05"]["per_family"].get(f)
            if a and b:
                rows.append([f, pct(a["unknown_recall"]), pct(b["unknown_recall"]),
                             f"{100*(b['unknown_recall'][0]-a['unknown_recall'][0]):+.1f}"])
    md += ["Per family, the Phase 3 headline configuration (two-stage, SMOTE, global):", ""]
    md += table(["held out", "unknown recall v1", "v2", "Δ (points)"], rows)

def closed_tables(md, v2dir, v1dir):
    spec = importlib.util.spec_from_file_location("cs", HERE / "22_closed_summary.py")
    cs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cs)
    from sklearn.metrics import precision_recall_fscore_support
    meta = load(v2dir / "temporal_medium" / "meta.json")
    if meta is None:
        return
    names = meta["class_names"]
    counts = np.array(meta["fit_class_counts_real"], dtype=float)
    lp = np.log(counts / counts.sum())
    rows, per = [], {}
    for root, tag in ((v1dir, "v1"), (v2dir, "v2")):
        for st in ("smote", "none"):
            f1s, pcs = [], []
            for seed in (7, 42, 1234):
                d = root / f"scores_medium_seed{seed}{'' if st == 'smote' else '_' + st}"
                if not (d / "test_scores.npy").exists():
                    continue
                vs = np.load(d / "val_scores.npy").astype(float)
                ts = np.load(d / "test_scores.npy").astype(float)
                vs = vs + np.random.default_rng(0).uniform(0, 1e-3, vs.shape)
                ts = ts + np.random.default_rng(1).uniform(0, 1e-3, ts.shape)
                yv, yt = np.load(d / "y_val.npy"), np.load(d / "y_test.npy")
                p, _ = cs.calibrated(vs, yv, ts, lp)
                pr = precision_recall_fscore_support(yt, p, labels=range(len(names)), zero_division=0)
                f1s.append(float(pr[2].mean())); pcs.append(pr[2])
            if f1s:
                rows.append([f"TM {st} ({tag})", len(f1s), num((np.mean(f1s), np.std(f1s)), 4)])
                if tag == "v2":
                    per[f"TM {st}"] = np.mean(pcs, 0)
    enc = load(v2dir / "phase4" / "encoding_study.json")
    if enc and "trees" in enc:
        for rep, lab in (("boolean_q5", "Boolean"), ("continuous", "continuous")):
            for m in ("rf", "xgb"):
                e = enc["trees"].get(rep, {}).get(m)
                if e:
                    rows.append([f"{m.upper()} {lab} (v2)", 3,
                                 num(tuple(e["macro_f1"]), 4)])
                    if "per_class" in e:
                        per[f"{m.upper()} {lab}"] = np.array([e["per_class"][n]["f1"][0] for n in names])
    md += ["### Task 2 — closed set (10 classes, test), macro F1 (TM: prior-calibrated decision)", ""]
    md += table(["model", "seeds", "macro F1"], rows)
    if per:
        md += ["Per-class F1 (v2, mean over seeds):", ""]
        keys = list(per)
        md += table(["class"] + keys, [[n] + [f"{per[k][i]:.3f}" for k in keys] for i, n in enumerate(names)])

def bench_tables(md, b):
    hw = b["hardware"]
    md += ["### Task 3 — compute cost", "",
           f"Hardware: {hw['cpu']}, {hw['cores_logical']} cores ({hw['perf_cores']} performance + "
           f"{hw['eff_cores']} efficiency), {hw['memory_gb']:.0f} GB; {hw['os']}; numpy {hw['numpy']}, "
           f"scikit-learn {hw['sklearn']}, XGBoost {hw['xgboost']}. Fold: {b['fold']} held out. "
           "Energy not measured (needs root on macOS); CPU-seconds are reported instead.", ""]
    if "preprocess" in b:
        p = b["preprocess"]["1"]
        md += ["Preprocessing (1 thread), separate from model inference:", ""]
        md += table(["step", "rows", "µs per row (batch)", "µs per row (one at a time, median)", "peak memory MB"], [
            ["window features, 10 s (00b kernel)", p["windows_medium"]["rows"],
             f"{p['windows_medium']['us_per_row']:.2f}", "–", f"{p['windows_medium']['peak_mem_above_start_mb']:.0f}"],
            ["Boolean encoding (thresholds + one-hot)", p["boolean_encoding"]["rows"],
             f"{p['boolean_encoding']['us_per_row']:.2f}", f"{p['boolean_encoding']['single_row']['median_us']:.0f}",
             f"{p['boolean_encoding']['peak_mem_above_start_mb']:.0f}"],
            ["continuous encoding (trees)", p["continuous_encoding"]["rows"],
             f"{p['continuous_encoding']['us_per_row']:.2f}", f"{p['continuous_encoding']['single_row']['median_us']:.0f}",
             f"{p['continuous_encoding']['peak_mem_above_start_mb']:.0f}"]])
    def row(label, e, t, fast=True):
        inf = e.get("infer_fast", e.get("infer"))
        lat = e.get("latency_fast", e.get("latency"))
        return [label, t, f"{e['train']['wall_s']:.0f}", f"{e['train']['cpu_s']:.0f}",
                f"{e['train']['cores_used']:.1f}", f"{e['train']['peak_mem_above_start_mb']:.0f}",
                f"{inf['rows_per_s']:,.0f}", f"{inf['us_per_row']:.2f}", f"{inf['peak_mem_above_start_mb']:.0f}",
                f"{lat['median_us']:.0f} / {lat['p99_us']:.0f}", f"{e['size_bytes']/1024:.0f}"]
    head = ["model", "threads", "train s (wall)", "train CPU s", "cores used", "train peak MB",
            "batch rows/s", "µs/row (batch)", "inference peak MB", "one-row latency µs (median / p99)",
            "model size KB"]
    for level, tasks in (("Individual model, binary task (the normality bank's)",
                          [("bank_tm", "TM bank"), ("bank_rf", "RF"), ("bank_xgb", "XGBoost")]),
                         ("Individual model, multi-class task (Normal + known families)",
                          [("mc_tm", "TM multi-class"), ("mc_rf_boolean", "RF, Boolean"),
                           ("mc_xgb_boolean", "XGBoost, Boolean"), ("mc_rf_continuous", "RF, continuous"),
                           ("mc_xgb_continuous", "XGBoost, continuous")]),
                         ("Complete open-set pipeline (decision included)",
                          [("two_stage_tm", "TM two-stage (OpenSATM: all banks)"), ("mc_tm", "TM single-stage"),
                           ("mc_rf_boolean", "RF, Boolean"), ("mc_xgb_boolean", "XGBoost, Boolean"),
                           ("mc_rf_continuous", "RF, continuous"), ("mc_xgb_continuous", "XGBoost, continuous")])):
        rows = [row(lab, b[t][th], th) for t, lab in tasks if t in b for th in sorted(b[t], key=int)]
        md += [f"{level}:", ""] + table(head, rows)
    rows = []
    for t, lab in (("bank_tm", "TM bank"), ("mc_tm", "TM multi-class"), ("two_stage_tm", "TM two-stage")):
        if t in b:
            e = b[t]["1"] if "1" in b[t] else list(b[t].values())[0]
            if "infer_tmu" in e:
                rows.append([lab, f"{e['infer_tmu']['rows_per_s']:,.0f}", f"{e['latency_tmu']['median_us']:.0f}",
                             f"{e.get('infer_fast', {}).get('rows_per_s', 0):,.0f}",
                             f"{e.get('latency_fast', {}).get('median_us', 0):.0f}",
                             e.get("clauses_total", "–"), e.get("includes_total", "–")])
    if rows:
        md += ["TM inference through tmu itself against the matrix scorer (`tm_fast`, exact, verified against tmu):", ""]
        md += table(["model", "tmu rows/s", "tmu one-row µs", "tm_fast rows/s", "tm_fast one-row µs",
                     "clauses", "included literals"], rows)

def explain_tables(md, ex):
    md += ["### Task 4 — explanations under common criteria (5% budget, held-out folds, seed 42 as reference)", ""]
    models = [m for m in ("tm", "rf_boolean", "xgb_boolean", "rf_continuous", "xgb_continuous") if m in ex]
    label = {"tm": "TM (multi-class)", "rf_boolean": "RF, Boolean (TreeSHAP)", "xgb_boolean": "XGBoost, Boolean (TreeSHAP)",
             "rf_continuous": "RF, continuous (TreeSHAP)", "xgb_continuous": "XGBoost, continuous (TreeSHAP)"}
    for g, desc in (("known_correct", "why a known class was predicted (known rows, correct)"),
                    ("unknown_missed", "why a known class was predicted (held-out rows given a known label)"),
                    ("unknown_rejected", "why a row was rejected as Unknown (held-out rows caught)"),
                    ("known_rejected", "why a row was rejected as Unknown (known rows rejected)")):
        rows = []
        for m in models:
            vals = [r["groups"][g] for r in ex[m].values() if g in r["groups"] and "fidelity_at_k" in r["groups"][g]]
            if not vals:
                continue
            mean = lambda f: float(np.mean([f(v) for v in vals]))
            rows.append([label[m], int(sum(v["rows"] for v in vals)), f"{mean(lambda v: v['tie_share']):.2f}",
                         f"{mean(lambda v: v['completeness_err_median']):.3f}",
                         f"{mean(lambda v: v['fidelity_at_k']['1']):.2f} / {mean(lambda v: v['fidelity_at_k']['5']):.2f}",
                         f"{mean(lambda v: v['k_min_median']):.1f}", f"{mean(lambda v: v['k90_median']):.1f}",
                         f"{mean(lambda v: v['faithfulness_flip_top']['3']):.2f} vs {mean(lambda v: v['faithfulness_flip_random']['3']):.2f}",
                         f"{mean(lambda v: v['faithfulness_flip_top']['10']):.2f} vs {mean(lambda v: v['faithfulness_flip_random']['10']):.2f}",
                         f"{mean(lambda v: v['stability_seed_jaccard_top5']):.2f}",
                         f"{mean(lambda v: v.get('stability_perturb_jaccard_top5', np.nan)):.2f}",
                         f"{mean(lambda v: v.get('stability_perturb_decision_change', np.nan)):.2f}"])
        md += [f"**{desc}**", ""]
        md += table(["model", "rows", "tied share", "completeness error (median)", "fidelity@1 / @5",
                     "features to reproduce (median)", "features for 90% of |attribution|",
                     "faithfulness: flips, top-3 vs random-3", "top-10 vs random-10",
                     "seed stability (top-5 Jaccard)", "perturbation stability (top-5 Jaccard)",
                     "decision change under perturbation"], rows)
    if "tm" in ex:
        rows = []
        for g in ("known_correct", "unknown_missed", "unknown_rejected", "known_rejected"):
            vals = [r["groups"][g] for r in ex["tm"].values() if g in r["groups"] and "clause_fidelity_at_k" in r["groups"][g]]
            if vals:
                mean = lambda f: float(np.mean([f(v) for v in vals]))
                rows.append([g, f"{mean(lambda v: v['clause_fidelity_at_k']['1']):.2f}",
                             f"{mean(lambda v: v['clause_fidelity_at_k']['10']):.2f}",
                             f"{mean(lambda v: v['clauses_min_median']):.0f}",
                             f"{mean(lambda v: v['literals_per_cited_clause_mean']):.0f}",
                             f"{mean(lambda v: v['clauses_firing_top_class_median']):.0f}"])
        md += ["TM clause-level explanations (the model's native unit; unclipped clause votes):", ""]
        md += table(["rows", "clause fidelity@1", "@10", "clauses to reproduce (median)",
                     "literals per cited clause", "clauses firing for the top class"], rows)

def main():
    v1, v2, cic = (Path(a) for a in sys.argv[1:4])
    md = ["# Phase 5 tables (generated by 33_phase5_tables.py)", ""]
    t1, t2 = load(v1 / "phase5" / "task2_v1.json"), load(v2 / "phase5" / "task2.json")
    if t1 and t2:
        change_table(md, t1, t2)
    closed_tables(md, v2, v1)
    if t2:
        md += ["### Task 2 — open set, UNSW-NB15", ""]
        open_set_tables(md, t2, UNSW, "UNSW-NB15")
    b = load(v2 / "phase5" / "benchmark.json")
    if b:
        bench_tables(md, b)
    ex = load(v2 / "phase5" / "explain_compare.json")
    if ex:
        explain_tables(md, ex)
    tc = load(cic / "phase5" / "task2.json")
    if tc:
        md += ["### Task 6 — open set, CICIDS2017", ""]
        open_set_tables(md, tc, CIC, "CICIDS2017")
    out = v2 / "phase5" / "tables.md"
    out.write_text("\n".join(md))
    print(f"written {out} ({len(md)} lines)")

if __name__ == "__main__":
    main()
