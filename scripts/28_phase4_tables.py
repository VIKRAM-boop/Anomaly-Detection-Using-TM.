"""
Markdown tables for Phase 4 (resampling, representation, single-stage).

Reads the JSON outputs of 24, 26, 27 and 20 under data/phase4/ and
data/phase3/ and writes data/phase4/tables.md. Every number in
PHASE4_RESULTS.md is taken from here.

Usage: 28_phase4_tables.py
"""
import json
import os
from pathlib import Path

import numpy as np

FAMILIES = ["analysis", "backdoor", "dos", "exploits", "fuzzers", "generic",
            "reconnaissance", "shellcode", "worms"]
STRATS = ["smote", "ros", "none"]
RULES = ["global", "classcond", "margin"]
B = "0.05"

def find_dir(name):
    if name == "data":
        override = os.environ.get("TM_DATA_DIR")
        if override:
            return Path(override)
    here = Path(__file__).resolve().parent
    for c in (here.parent / name, here / name):
        if c.exists():
            return c
    return here / name

def load(p):
    return json.load(open(p)) if p.exists() else None

def pct(ms, d=1):
    if ms is None:
        return "–"
    m, s = ms if isinstance(ms, (list, tuple)) else (ms, None)
    if m is None or (isinstance(m, float) and np.isnan(m)):
        return "–"
    return f"{100*m:.{d}f}%" + (f" ± {100*s:.{d}f}" if s else "")

def num(ms, d=4):
    if ms is None:
        return "–"
    m, s = ms if isinstance(ms, (list, tuple)) else (ms, None)
    return f"{m:.{d}f}" + (f" ± {s:.{d}f}" if s else "")

def table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return out

OPEN_COLS = ["unknown recall", "known-attack macro F1", "Normal false alarms",
             "known rejected", "known attacks rejected", "unknown → Normal",
             "unknown → known family", "AUROC"]

def open_row(s):
    return [pct(s["unknown_recall"]), num(s["known_attack_macro_f1"]), pct(s["benign_fpr"], 2),
            pct(s["false_unknown"], 2), pct(s.get("known_attack_reject")), pct(s["held_to_normal"]),
            pct(s["held_to_known_family"]), num(s["auroc"], 3)]

def single_stage_files(p4):
    f = {}
    for src in ("banks", "mc"):
        for st in STRATS:
            j = load(p4 / f"single_stage_{src}_medium{'' if st == 'smote' else '_' + st}.json")
            if j:
                f[(src, st)] = j
    return f

def task3(md, ss):
    md += ["## Task 3 — single-stage vs two-stage (5% budget, family average, mean ± std over seeds)", ""]
    rows = []
    for (src, st), j in ss.items():
        if st != "smote":
            continue
        for arch in ("two_stage", "single"):
            if arch not in j["summary"]:
                continue
            name = {"two_stage": "two-stage (gate + family banks)",
                    "single": "single-stage, one-vs-rest banks" if src == "banks"
                    else "single-stage, multi-class TM"}[arch]
            for r in RULES:
                rows.append([name, r] + open_row(j["summary"][arch]["family_average"][r][B]))
    md += table(["architecture", "rule"] + OPEN_COLS, rows) + [""]
    for rule in ("global", "margin"):
        md += [f"### Task 3 per family, {rule} rule, 5%", ""]
        rows = []
        for f in FAMILIES:
            for (src, st), j in ss.items():
                if st != "smote":
                    continue
                for arch in ("two_stage", "single"):
                    if arch not in j["summary"] or f not in j["summary"][arch]["per_family"]:
                        continue
                    name = {"two_stage": "two-stage", "single": "single (banks)" if src == "banks"
                            else "single (multi-class)"}[arch]
                    rows.append([f, name] + open_row(j["summary"][arch]["per_family"][f][rule][B]))
        md += table(["held out", "architecture"] + OPEN_COLS, rows) + [""]
    return md

def task1(md, ss, validity, enc):
    md += ["## Task 1 — resampling", "", "### SMOTE output validity", ""]
    if validity:
        rows = []
        for name, r in validity.items():
            sv = r["synthetic_violation_rate"]
            copies = [pc["exact_copy_of_own_class"] for pc in r["per_class"].values()]
            rows.append([name, r["n_real"], r["n_synthetic"], pct(r["real_violation_rate"]["any"], 2),
                         pct(sv["any"], 2), pct(sv["onehot_proto"], 2), pct(sv["ladder"], 2),
                         pct(sv["relational"], 2),
                         f"{100*min(copies):.1f}–{100*max(copies):.1f}%" if copies else "–"])
        md += table(["dataset", "real rows", "synthetic rows", "real rows violating",
                     "synthetic violating (any)", "one-hot proto", "quantile ladder",
                     "relational", "synthetic rows identical to a real row of their class"],
                    rows) + [""]
        cm = validity.get("closed_medium")
        if cm:
            md += ["Closed set, per class:", ""]
            md += table(["class", "real", "synthetic", "violating", "copy of a real row", "new pattern",
                         "distinct synthetic patterns"],
                        [[c, pc["real_rows"], pc["synthetic_rows"], pct(pc["violation_rate"]["any"]),
                          pct(pc["exact_copy_of_own_class"]), pct(pc["new_pattern"]),
                          pc["distinct_synthetic_patterns"]] for c, pc in cm["per_class"].items()]) + [""]
    if enc and enc.get("tm"):
        md += ["### Closed-set TM by resampling strategy (test, mean ± std over seeds)", ""]
        rows = []
        for st in STRATS:
            t = enc["tm"].get(f"{st}_q5")
            if t:
                rows.append([st, t["seeds"], num(t["raw"]["macro_f1"]), num(t["calibrated"]["macro_f1"]),
                             num(t["calibrated"]["acc"])])
        md += table(["strategy", "seeds", "macro F1 (argmax)", "macro F1 (prior-calibrated)",
                     "accuracy (calibrated)"], rows) + [""]
        names = enc["class_names"]
        md += ["Per-class precision / recall / F1 (prior-calibrated):", ""]
        head = ["class"] + [f"{st} P / R / F1" for st in STRATS if f"{st}_q5" in enc["tm"]]
        rows = []
        for n in names:
            r = [n]
            for st in STRATS:
                t = enc["tm"].get(f"{st}_q5")
                if t:
                    pc = t["calibrated"]["per_class"][n]
                    r.append(f"{pc['precision'][0]:.3f} / {pc['recall'][0]:.3f} / {pc['f1'][0]:.3f}")
            rows.append(r)
        md += table(head, rows) + [""]
    md += ["### Open-set detection by resampling strategy (5% budget, family average)", ""]
    rows = []
    for st in STRATS:
        for src, arch, name in (("banks", "two_stage", "two-stage"),
                                ("banks", "single", "single (banks)"),
                                ("mc", "single", "single (multi-class)")):
            j = ss.get((src, st))
            if j and arch in j["summary"]:
                for r in RULES:
                    rows.append([st, name, r] + open_row(j["summary"][arch]["family_average"][r][B]))
    md += table(["strategy", "architecture", "rule"] + OPEN_COLS, rows) + [""]
    return md

def task2(md, enc, p3):
    md += ["## Task 2 — feature representation", ""]
    if enc and enc.get("trees"):
        tr, tm = enc["trees"], enc.get("tm", {})
        md += ["### Closed set: model × representation, identical real rows (test, mean ± std)", ""]
        rows = []
        t5 = tm.get("none_q5")
        if t5:
            rows.append(["Tsetlin Machine", "Boolean (5 thresholds)", t5["seeds"],
                         num(t5["calibrated"]["macro_f1"]), num(t5["calibrated"]["acc"])])
        for rep, lab in (("boolean_q5", "Boolean (5 thresholds)"), ("continuous", "continuous")):
            for m, ml in (("rf", "Random Forest"), ("xgb", "XGBoost")):
                if rep in tr:
                    rows.append([ml, lab, enc.get("tree_seeds", 3),
                                 num(tr[rep][m]["macro_f1"]), num(tr[rep][m]["acc"])])
        md += table(["model", "input", "seeds", "macro F1", "accuracy"], rows) + [""]
        names = enc["class_names"]
        md += ["Per-class F1:", ""]
        cols = [("TM bool", lambda n: t5["calibrated"]["per_class"][n]["f1"][0] if t5 else None)]
        for rep, lab in (("boolean_q5", "bool"), ("continuous", "cont")):
            for m in ("rf", "xgb"):
                if rep in tr:
                    cols.append((f"{m.upper()} {lab}",
                                 lambda n, rep=rep, m=m: tr[rep][m]["per_class"][n]["f1"][0]))
        md += table(["class"] + [c for c, _ in cols],
                    [[n] + [f"{f(n):.3f}" if f(n) is not None else "–" for _, f in cols]
                     for n in names]) + [""]
        md += ["### Quantile encodings: performance, dimensionality and cost", ""]
        rows = []
        info = enc.get("info", {})
        for q in (3, 5, 9, 19):
            t = tm.get(f"none_q{q}"); i = info.get(f"q{q}"); b = tr.get(f"boolean_q{q}")
            if not (t or i or b):
                continue
            rows.append([q, i["n_literals"] if i else "–", i["distinct_literal_columns"] if i else "–",
                         num(t["calibrated"]["macro_f1"]) if t else "–",
                         num(b["rf"]["macro_f1"]) if b else "–", num(b["xgb"]["macro_f1"]) if b else "–",
                         f"{t['train_seconds'][0]:.0f}" if t and t.get("train_seconds") else "–",
                         f"{t['score_seconds'][0]:.0f}" if t and t.get("score_seconds") else "–",
                         f"{b['rf']['train_seconds'][0]:.0f} / {b['xgb']['train_seconds'][0]:.0f}" if b else "–",
                         f"{sum(i['bytes_on_disk'].values())/1e6:.0f}" if i else "–",
                         pct(i["test_rows_with_pattern_shared_across_classes"]) if i else "–"])
        if "continuous" in tr:
            c = tr["continuous"]
            rows.append(["continuous", c["n_features"], "–", "–", num(c["rf"]["macro_f1"]),
                         num(c["xgb"]["macro_f1"]), "–", "–",
                         f"{c['rf']['train_seconds'][0]:.0f} / {c['xgb']['train_seconds'][0]:.0f}", "–", "–"])
        md += table(["thresholds", "literals", "distinct literal columns", "TM macro F1", "RF macro F1",
                     "XGB macro F1", "TM train s", "TM score s (val+test)", "RF / XGB fit s",
                     "data MB", "test rows sharing a pattern across classes"], rows) + [""]
        if info:
            md += ["### Can the Boolean encoding separate each family from Normal?",
                   "", "Share of each family's test rows whose literal vector is identical to some "
                   "Normal test row's:", ""]
            qs = [q for q in (3, 5, 9, 19) if f"q{q}" in info]
            fams = list(info[f"q{qs[0]}"]["families"])
            md += table(["family"] + [f"{q} thresholds" for q in qs],
                        [[f] + [pct(info[f"q{q}"]["families"][f]["identical_to_a_normal_row"])
                                for q in qs] for f in fams]) + [""]
    rows, per = [], {}
    for rep, fn in (("continuous (all scales)", "baselines_openset.json"),
                    ("continuous (medium only)", "baselines_openset_continuous_medium.json"),
                    ("Boolean (TM literals)", "baselines_openset_boolean.json")):
        j = load(p3 / fn)
        if not j:
            continue
        for m, ml in (("rf_msp", "RF"), ("xgb_msp", "XGB")):
            vals = {k: [] for k in ("unknown_recall", "known_attack_macro_f1", "benign_fpr",
                                    "held_to_normal", "auroc")}
            for f in FAMILIES:
                if f not in j:
                    continue
                seeds = j[f]["methods"][m]["seeds"]
                for k in vals:
                    vals[k].append(np.mean([seeds[s][B][k] for s in seeds]))
                per.setdefault(f, {})[f"{ml} {rep}"] = np.mean([seeds[s][B]["unknown_recall"] for s in seeds])
            if vals["unknown_recall"]:
                rows.append([ml, rep, len(vals["unknown_recall"]), pct(np.mean(vals["unknown_recall"])),
                             num(np.mean(vals["known_attack_macro_f1"])), pct(np.mean(vals["benign_fpr"]), 2),
                             pct(np.mean(vals["held_to_normal"])), num(np.mean(vals["auroc"]), 3)])
    if rows:
        md += ["### Open set: RF / XGBoost by representation (5% budget, family average)", ""]
        md += table(["model", "input", "families", "unknown recall", "known-attack macro F1",
                     "Normal false alarms", "unknown → Normal", "AUROC"], rows) + [""]
        cols = sorted({c for v in per.values() for c in v})
        md += ["Unknown recall per family:", ""]
        md += table(["held out"] + cols, [[f] + [pct(per[f].get(c)) for c in cols] for f in per]) + [""]
    return md

def main():
    data = find_dir("data")
    p3, p4 = data / "phase3", data / "phase4"
    ss = single_stage_files(p4)
    md = ["# Phase 4 tables (generated by 28_phase4_tables.py)", ""]
    md = task3(md, ss)
    md = task1(md, ss, load(p4 / "smote_validity.json"), load(p4 / "encoding_study.json"))
    md = task2(md, load(p4 / "encoding_study.json"), p3)
    (p4 / "tables.md").write_text("\n".join(md))
    print(f"written {p4 / 'tables.md'} ({len(md)} lines)")

if __name__ == "__main__":
    main()
