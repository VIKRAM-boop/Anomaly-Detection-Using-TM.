"""
Checks whether SMOTE's synthetic rows are valid observations (Task 1).

Both pipelines -- 01c (closed set) and 09 (open set) -- encode first and
resample second: quantile, one-hot and relational literals are built from
the fit rows, then SMOTE interpolates between binary rows in float32 and
thresholds the result back to 0/1 at 0.5. So synthetic rows can break the
constraints every real row satisfies by construction:

  one-hot     proto_*, service_* and state_* each have exactly one active
              indicator
  ladder      a feature's quantile literals are nested: x > Q90 implies
              x > Q75 implies ... x > Q10
  relational  R_outbound_heavy and R_inbound_heavy cannot both hold;
              R_many_conn_same_dst requires ct_dst_ltm > Q75 and
              ct_srv_dst > Q75 (same thresholds as those literals), and
              R_slow_handshake requires synack > Q75 and tcprtt > Q75

Real rows are checked too, as a control (they must show zero violations).
Also reported: how many synthetic rows are exact copies of a real row of
their class (interpolation that changed nothing), how many are new
patterns, and how many coincide with a real row of a different class.

imblearn returns the original rows first and the synthetic rows after
them, so a dataset's synthetic rows are those past the capped real count.

Writes data/phase4/smote_validity.json.

Usage: 26_smote_validity.py
"""
import json
import os
import re
from pathlib import Path

import numpy as np

PER_CLASS_CAP = 40000
FAMILIES = ["analysis", "backdoor", "dos", "exploits", "fuzzers", "generic",
            "reconnaissance", "shellcode", "worms"]

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

def constraint_columns(names):
    idx = {n: i for i, n in enumerate(names)}
    groups = {g: [i for n, i in idx.items() if n.startswith(f"{g}_")]
              for g in ("proto", "service", "state")}
    ladders = {}
    for n, i in idx.items():
        m = re.match(r"(.+)>Q([\d.]+)$", n)
        if m:
            ladders.setdefault(m.group(1), []).append((float(m.group(2)), i))
    ladders = {f: [i for _, i in sorted(v)] for f, v in ladders.items() if len(v) > 1}
    rel = {
        "exclusive": (idx["R_outbound_heavy"], idx["R_inbound_heavy"]),
        "implies": [(idx["R_many_conn_same_dst"], [idx["ct_dst_ltm>Q75"], idx["ct_srv_dst>Q75"]]),
                    (idx["R_slow_handshake"], [idx["synack>Q75"], idx["tcprtt>Q75"]])],
    }
    return groups, ladders, rel

def violations(X, groups, ladders, rel):
    """Per-row boolean masks, one per constraint type."""
    out = {}
    for g, cols in groups.items():
        out[f"onehot_{g}"] = X[:, cols].sum(1) != 1
    lad = np.zeros(len(X), bool)
    for cols in ladders.values():
        L = X[:, cols]
        lad |= (np.diff(L.astype(np.int8), axis=1) > 0).any(1)   # higher level on, lower off
    out["ladder"] = lad
    a, b = rel["exclusive"]
    r = (X[:, a] & X[:, b]).astype(bool)
    for c, reqs in rel["implies"]:
        r |= X[:, c].astype(bool) & ~X[:, reqs].all(1).astype(bool)
    out["relational"] = r
    out["any"] = np.any(np.stack(list(out.values())), axis=0)
    return out

def row_keys(X):
    P = np.packbits(X, axis=1)
    return np.array([r.tobytes() for r in P], dtype=object)

def analyse(X, y, n_real, names, class_names):
    groups, ladders, rel = constraint_columns(names)
    real, syn = slice(0, n_real), slice(n_real, len(y))
    Xr, yr, Xs, ys = X[real], y[real], X[syn], y[syn]
    vr, vs = violations(Xr, groups, ladders, rel), violations(Xs, groups, ladders, rel)
    kr, ks = row_keys(Xr), row_keys(Xs)
    res = {"n_real": int(n_real), "n_synthetic": int(len(ys)),
           "real_violation_rate": {k: float(v.mean()) for k, v in vr.items()},
           "synthetic_violation_rate": {k: float(v.mean()) if len(v) else 0.0 for k, v in vs.items()},
           "per_class": {}}
    for c in np.unique(ys):
        sel = ys == c
        own = set(kr[yr == c].tolist())
        other = set(kr[yr != c].tolist())
        k = ks[sel]
        copy_own = np.array([x in own for x in k])
        in_other = np.array([x in other for x in k])
        res["per_class"][class_names[int(c)]] = {
            "real_rows": int((yr == c).sum()), "synthetic_rows": int(sel.sum()),
            "violation_rate": {kk: float(v[sel].mean()) for kk, v in vs.items()},
            "exact_copy_of_own_class": float(copy_own.mean()),
            "new_pattern": float((~copy_own).mean()),
            "matches_other_class_real_row": float((in_other & ~copy_own).mean()),
            "distinct_synthetic_patterns": int(len(set(k.tolist()))),
        }
    return res

def n_capped(counts):
    return int(sum(min(c, PER_CLASS_CAP) for c in counts if c > 0))

def main():
    data = find_dir("data")
    out = {}
    jobs = [("closed_medium", data / "temporal_medium")] + \
           [(f"open_{f}", data / f"openset_medium_{f}") for f in FAMILIES]
    for name, d in jobs:
        if not (d / "X_fit.npy").exists():
            continue
        meta = json.load(open(d / "meta.json"))
        class_names = meta.get("all_class_names") or meta["class_names"]
        X = np.load(d / "X_fit.npy").astype(np.uint8)
        y = np.load(d / "y_fit.npy").astype(np.int64)
        n_real = n_capped(meta["fit_class_counts_real"])
        # the real prefix must hold the capped real counts, class by class
        exp = [min(c, PER_CLASS_CAP) for c in meta["fit_class_counts_real"]]
        assert np.array_equal(np.bincount(y[:n_real], minlength=len(exp)), exp), name
        res = analyse(X, y, n_real, meta["feature_names"], class_names)
        out[name] = res
        sv, rv = res["synthetic_violation_rate"], res["real_violation_rate"]
        print(f"{name:20s} real {res['n_real']:>7} synth {res['n_synthetic']:>7} | "
              f"real rows violating: {rv['any']:.2%} | synthetic violating: any {sv['any']:.1%} "
              f"(proto {sv['onehot_proto']:.1%}, service {sv['onehot_service']:.1%}, "
              f"state {sv['onehot_state']:.1%}, ladder {sv['ladder']:.1%}, "
              f"relational {sv['relational']:.1%})", flush=True)
        if name == "closed_medium":
            for c, pc in res["per_class"].items():
                print(f"    {c:15s} real {pc['real_rows']:>6} synth {pc['synthetic_rows']:>6}  "
                      f"violating {pc['violation_rate']['any']:6.1%}  copy-of-real "
                      f"{pc['exact_copy_of_own_class']:6.1%}  new {pc['new_pattern']:6.1%}  "
                      f"equals-another-class {pc['matches_other_class_real_row']:5.1%}  "
                      f"distinct patterns {pc['distinct_synthetic_patterns']}", flush=True)
    p = data / "phase4" / "smote_validity.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(p, "w"), indent=2)
    print(f"written to {p}")

if __name__ == "__main__":
    main()
