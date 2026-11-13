"""
Single-stage open-set decision, compared directly with the two-stage one.

Two-stage (existing, 18_openset_eval.py): the normality bank gates rows at
tau_B; only gated rows reach the family banks, where Unknown is decided.
A held-out attack that looks benign is admitted as Normal at the gate and
never reaches the novelty test.

Single-stage: no gate. Normal is one more known class. Every row gets one
support score per known class -- Normal and each known family -- and

  prediction = the most strongly supported known class, unless the
               evidence is insufficient, in which case Unknown

with three rejection rules:

  global     Unknown when the strongest class score is below one threshold
  classcond  Unknown when the predicted class's score is below that class's
             own threshold; per-class thresholds sit at a common quantile q
             of each class's validation scores, q chosen to meet the budget
  margin     Unknown when the best minus second-best class score is small

Score sources:

  banks   version 1. The existing one-vs-rest banks: the normality bank's
          margin is Normal's support, each family bank's margin that
          family's. All are binary TMs with the same configuration and the
          same [-2T, 2T] range, trained on balanced samples, so the scores
          are on a common scale. No retraining; the same runs (and the same
          tie-break jitter) as the two-stage results.
  mc      version 2. One multi-class TM over Normal + the known families
          (25_train_single_stage.py); class sums are the supports.

Both architectures are held to the same budget definition as before: the
share of ALL known validation rows (Normal and known attacks) predicted
Unknown. Validation never contains the held-out family, so it plays no part
in calibration. Two-stage numbers are recomputed here from the same bank
scores with 18's own Calibrated class, so both rows of every comparison
come from identical runs and identical metric code.

Reported per held-out family (mean +/- std over seeds) and averaged:
unknown recall, known-attack macro F1, Normal false-alarm rate (Normal rows
not predicted Normal, Unknown included), known-sample rejection rate (known
rows predicted Unknown), held-out -> Normal, held-out -> known family,
AUROC and AUPRC of the decision's novelty score.

The resampling strategy (smote, ros or none; Task 1) picks which runs are
read: banks_/single_<scale>_<family>[_<strategy>]_s<seed>/.

Writes data/phase4/single_stage_<source>_<scale>[_<strategy>].json.

Usage: 24_single_stage.py [banks|mc] [scale] [strategy]
"""
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             precision_recall_fscore_support, roc_auc_score)

_spec = importlib.util.spec_from_file_location(
    "ev", Path(__file__).resolve().parent / "18_openset_eval.py")
ev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ev)

BUDGETS = ev.BUDGETS
REPORT_BUDGET = ev.REPORT_BUDGET
STRATEGIES = ["closed", "global", "classcond", "margin"]
MIN_ROWS_CLASSCOND = ev.MIN_ROWS_CLASSCOND

class SingleStage:
    """Calibration on validation of the three rejection rules over a score
    matrix whose columns are the known classes (Normal included)."""
    def __init__(self, classes, vs, y_val, unk):
        self.classes = np.asarray(classes)
        self.unk = unk
        self.vs, self.vy = vs, y_val

    def parts(self, s):
        top = np.sort(s, axis=1)
        j = s.argmax(1)
        return top[:, -1], top[:, -1] - top[:, -2], self.classes[j], j

    def novelty(self, strat, s, thr):
        best, margin, _, j = self.parts(s)
        if strat == "global":
            return -best
        if strat == "margin":
            return -margin
        if strat == "classcond":
            return thr["tau"][j] - best
        return np.full(len(s), -np.inf)

    def predict(self, strat, s, thr):
        _, _, pred, _ = self.parts(s)
        pred = pred.astype(np.int64)
        nov = self.novelty(strat, s, thr)
        if strat != "closed":
            pred[nov > thr["theta"]] = self.unk
        return pred, nov

    def calibrate(self, strat, beta):
        if strat == "closed":
            return {"theta": np.inf}
        n = len(self.vs)
        if strat in ("global", "margin"):
            nov = self.novelty(strat, self.vs, None)
            k = int(round(beta * n))
            return {"theta": float(np.sort(nov)[::-1][k]) if k > 0 else float(nov.max())}
        # classcond: each class's threshold is quantile q of its own
        # validation rows' score for that class; q bisected to meet beta
        per = []
        for c_i, c in enumerate(self.classes):
            sel = self.vy == c
            per.append(self.vs[sel, c_i] if sel.sum() >= MIN_ROWS_CLASSCOND
                       else self.vs[:, c_i])
        def rate(q):
            tau = np.array([np.quantile(v, q) for v in per])
            return float((self.novelty("classcond", self.vs, {"tau": tau}) > 0).mean()), tau
        lo, hi = 0.0, 1.0
        for _ in range(40):
            mid = (lo + hi) / 2
            if rate(mid)[0] < beta:
                lo = mid
            else:
                hi = mid
        return {"theta": 0.0, "q": hi, "tau": rate(hi)[1]}

def metrics(y, pred, nov, vpred, ni, unk, fams):
    is_unk = y == unk
    known = ~is_unk
    known_classes = sorted(set(np.unique(y[known]).tolist()))
    p_unk = pred[is_unk]
    res = {
        "val_false_reject": float((vpred == unk).mean()),
        "unknown_recall": float((p_unk == unk).mean()),
        "held_to_known_family": float(np.isin(p_unk, fams).mean()),
        "held_to_normal": float((p_unk == ni).mean()),
        "false_unknown": float((pred[known] == unk).mean()),
        "benign_fpr": float((pred[y == ni] != ni).mean()),
        "normal_to_unknown": float((pred[y == ni] == unk).mean()),
        "known_attack_reject": float((pred[known & (y != ni)] == unk).mean()),
        "known_macro_f1": float(precision_recall_fscore_support(
            y[known], pred[known], labels=known_classes, average="macro", zero_division=0)[2]),
        "known_attack_macro_f1": float(precision_recall_fscore_support(
            y[known], pred[known], labels=fams, average="macro", zero_division=0)[2]),
        "auroc": float(roc_auc_score(is_unk, nov)) if np.isfinite(nov).all() else float("nan"),
        "auprc": float(average_precision_score(is_unk, nov)) if np.isfinite(nov).all() else float("nan"),
        "recall_ci95": ev.wilson(int((p_unk == unk).sum()), int(is_unk.sum())),
    }
    return res

def confusion(y, pred, unk):
    labels = sorted(set(np.unique(y[y != unk]).tolist())) + [unk]
    return {"labels": labels, "matrix": confusion_matrix(y, pred, labels=labels).tolist()}

def evaluate_single(classes, vs, ts, y_val, y_test, ni, unk, fams):
    cal = SingleStage(classes, vs, y_val, unk)
    out = {}
    for strat in STRATEGIES:
        out[strat] = {}
        for beta in (BUDGETS if strat != "closed" else [0.0]):
            thr = cal.calibrate(strat, beta)
            vpred, _ = cal.predict(strat, vs, thr)
            pred, nov = cal.predict(strat, ts, thr)
            res = metrics(y_test, pred, nov, vpred, ni, unk, fams)
            if "q" in thr:
                res["q"] = thr["q"]
            if strat == "closed" or abs(beta - REPORT_BUDGET) < 1e-9:
                res["confusion"] = confusion(y_test, pred, unk)
            out[strat][f"{beta:.2f}"] = res
    return out

def evaluate_two_stage(r):
    """18's two-stage strategies on the same run, through the same metrics."""
    cal = ev.Calibrated(r)
    out = {}
    for strat in STRATEGIES:
        out[strat] = {}
        for beta in (BUDGETS if strat != "closed" else [0.0]):
            thr = cal.calibrate(strat, beta)
            vpred, _ = cal.predict(strat, r["vb"], r["vf"], thr)
            pred, nov = cal.predict(strat, r["tb"], r["tf"], thr)
            res = metrics(r["y_test"], pred, nov, vpred, cal.ni, cal.unk, list(cal.fams))
            if strat == "closed" or abs(beta - REPORT_BUDGET) < 1e-9:
                res["confusion"] = confusion(r["y_test"], pred, cal.unk)
            out[strat][f"{beta:.2f}"] = res
    return out

# ---------------- score sources ----------------

def find_runs(data, prefix, scale, strategy, meta_name):
    """{family: {seed: dir}}, matching family names exactly so that one
    strategy's runs never pick up another's."""
    sfx = "" if strategy == "smote" else f"_{strategy}"
    runs = {}
    for fam in ev.FAMILY_ORDER:
        stem = f"{prefix}_{scale}_{fam.lower()}{sfx}_s"
        for d in sorted(data.glob(f"{stem}*")):
            m = re.fullmatch(re.escape(stem) + r"(\d+)", d.name)
            if m and (d / meta_name).exists():
                runs.setdefault(fam.lower(), {})[int(m.group(1))] = d
    return runs

def runs_banks(data, scale, strategy):
    return find_runs(data, "banks", scale, strategy, "banks_meta.json")

def load_banks(d, data):
    r = ev.load_run(d, data)      # same jittered scores the two-stage uses
    m = r["meta"]
    classes = [m["normal_idx"]] + list(r["families"])
    vs = np.column_stack([r["vb"], r["vf"]])
    ts = np.column_stack([r["tb"], r["tf"]])
    return r, classes, vs, ts

def runs_mc(data, scale, strategy):
    return find_runs(data, "single", scale, strategy, "meta.json")

def load_mc(d, data):
    meta = json.load(open(d / "meta.json"))
    rng = np.random.default_rng(0)
    vs = np.load(d / "val_scores.npy").astype(np.float64)
    ts = np.load(d / "test_scores.npy").astype(np.float64)
    vs = vs + rng.uniform(0, 1e-3, vs.shape)      # same tie-break as load_run
    ts = ts + rng.uniform(0, 1e-3, ts.shape)
    fam = meta["held_out_class"].lower()
    src = data / f"openset_{meta['scale']}_{fam}"
    r = {"meta": meta, "y_val": np.load(src / "y_val.npy").astype(np.int64),
         "y_test": np.load(src / "y_test.npy").astype(np.int64)}
    return r, meta["classes"], vs, ts

# ---------------- main ----------------

KEYS = ["unknown_recall", "known_attack_macro_f1", "benign_fpr", "false_unknown",
        "held_to_normal", "held_to_known_family", "known_attack_reject",
        "normal_to_unknown", "known_macro_f1", "auroc", "auprc"]

def summarise(results, arch):
    """mean/std over seeds per family, and the family average per seed."""
    summ, avg = {}, {}
    fams = list(results)
    seeds = sorted(set.intersection(*[set(results[f]) for f in fams]))
    for strat in STRATEGIES:
        for bkey in results[fams[0]][seeds[0]][arch][strat]:
            for f in fams:
                summ.setdefault(f, {}).setdefault(strat, {})[bkey] = {
                    k: ev.mean_std([results[f][s][arch][strat][bkey][k] for s in results[f]])
                    for k in KEYS}
            avg.setdefault(strat, {})[bkey] = {
                k: ev.mean_std([np.mean([results[f][s][arch][strat][bkey][k] for f in fams])
                                for s in seeds]) for k in KEYS}
    return summ, avg

def main():
    source = sys.argv[1] if len(sys.argv) > 1 else "banks"
    scale = sys.argv[2] if len(sys.argv) > 2 else "medium"
    strategy = sys.argv[3] if len(sys.argv) > 3 else "smote"
    data = ev.find_dir("data")
    finder, loader = {"banks": (runs_banks, load_banks), "mc": (runs_mc, load_mc)}[source]
    runs = finder(data, scale, strategy)
    if not runs:
        raise SystemExit(f"no {source} runs for scale {scale}")
    order = [x.lower() for x in ev.FAMILY_ORDER]
    results = {}
    for fam_l in sorted(runs, key=lambda f: order.index(f) if f in order else 99):
        results[fam_l] = {}
        for seed, d in sorted(runs[fam_l].items()):
            r, classes, vs, ts = loader(d, data)
            m = r["meta"]
            ni, unk = m["normal_idx"], m["unknown_label"]
            fams = [c for c in classes if c != ni]
            assert m["held_out_idx"] not in classes and not (r["y_val"] == m["held_out_idx"]).any(), \
                "held-out family present in the known classes or in validation"
            entry = {"single": evaluate_single(classes, vs, ts, r["y_val"], r["y_test"], ni, unk, fams)}
            if source == "banks":
                entry["two_stage"] = evaluate_two_stage(r)
            results[fam_l][seed] = entry
        print(f"evaluated {fam_l}: seeds {sorted(results[fam_l])}", flush=True)

    archs = ["single"] + (["two_stage"] if source == "banks" else [])
    summary = {a: summarise(results, a) for a in archs}
    out_dir = data / "phase4"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"single_stage_{source}_{scale}{'' if strategy == 'smote' else '_' + strategy}.json"
    with open(path, "w") as f:
        json.dump({"source": source, "scale": scale, "strategy": strategy, "budgets": BUDGETS,
                   "results": {f: {str(s): v for s, v in r.items()} for f, r in results.items()},
                   "summary": {a: {"per_family": summary[a][0], "family_average": summary[a][1]}
                               for a in archs}},
                  f, indent=2, default=float)

    b = f"{REPORT_BUDGET:.2f}"
    print(f"\n=== {source}, {scale}, {strategy}: family average at {REPORT_BUDGET:.0%} budget "
          f"(mean +/- std over seeds) ===")
    print(f"{'arch':<11}{'rule':<11}{'unk.recall':>15}{'atkF1':>8}{'NormalFA':>10}{'knownRej':>10}"
          f"{'->Normal':>10}{'->family':>10}{'AUROC':>8}")
    for a in archs:
        for strat in STRATEGIES:
            s = summary[a][1][strat]["0.00" if strat == "closed" else b]
            print(f"{a:<11}{strat:<11}{s['unknown_recall'][0]:>8.1%}±{s['unknown_recall'][1]:<5.1%}"
                  f"{s['known_attack_macro_f1'][0]:>8.4f}{s['benign_fpr'][0]:>10.2%}"
                  f"{s['false_unknown'][0]:>10.2%}{s['held_to_normal'][0]:>10.1%}"
                  f"{s['held_to_known_family'][0]:>10.1%}{s['auroc'][0]:>8.3f}")
    print(f"\nper family, global rule, {REPORT_BUDGET:.0%}: unknown recall / ->Normal")
    for f in results:
        row = f"  {f:<15}"
        for a in archs:
            s = summary[a][0][f]["global"][b]
            row += f"{a}: {s['unknown_recall'][0]:6.1%} / {s['held_to_normal'][0]:5.1%}   "
        print(row)
    print(f"\nwritten to {path}")

if __name__ == "__main__":
    main()
