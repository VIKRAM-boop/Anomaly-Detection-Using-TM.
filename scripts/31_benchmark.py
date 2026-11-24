"""
Phase 5, Task 3: computational cost of TM, Random Forest and XGBoost on the
same machine, data and thread counts.

Every measurement runs in a fresh child process (--task), so peak memory
and CPU time belong to that task alone. The parent records the hardware
and runs every task at the requested thread counts.

Levels compared, never mixed:

  individual model   one TM against one RF / one XGBoost solving the same
                     problem on the same rows:
                       bank   the normality bank's binary task (Normal vs
                              known attacks, the balanced sample 10 draws)
                       mc     the multi-class problem (Normal + known
                              families), the single-stage TM's task
  deployed pipeline  everything needed for one open-set decision:
                       TM two-stage  normality bank + every family bank +
                                     calibrated thresholds (OpenSATM)
                       TM single     one multi-class TM + threshold
                       RF / XGBoost  one multi-class model + threshold
  preprocessing      measured separately: window features (00b's kernel,
                     medium scale), Boolean encoding (quantile thresholds +
                     one-hot, 09's encoding) and the continuous encoding the
                     trees use

Per task: training wall time and CPU time; inference throughput on the
whole test partition (batch) and per-observation latency (one row per call,
median and 99th percentile over LATENCY_ROWS rows); peak resident memory
above the process's level just before the phase (sampled every 5 ms,
children included); CPU utilisation (CPU time / wall time, in cores); and
serialised model size (TM: the packed include masks and weights tm_fast
saves; RF: pickle; XGBoost: its UBJSON format).

Threads: N sets OMP/OpenBLAS/vecLib thread limits, RF n_jobs, XGBoost
nthread and the number of parallel bank-training processes. tmu itself
trains single-threaded. Energy is not measured: macOS exposes package
power only through powermetrics, which needs root.

Writes <data>/phase5/benchmark.json.

Usage: 31_benchmark.py [--fold dos] [--threads 1,8] [--only task,...]
"""
import argparse
import importlib.util
import json
import os
import pickle
import platform
import subprocess
import sys
import threading
import time
from pathlib import Path

LATENCY_ROWS = 1000
# the TM's training rows follow the resampling strategy selected in Task 2
TM_STRATEGY = os.environ.get("TM_STRATEGY", "smote")
HERE = Path(__file__).resolve().parent

def find_dir(name):
    if name == "data":
        override = os.environ.get("TM_DATA_DIR")
        if override:
            return Path(override)
    for c in (HERE.parent / name, HERE / name):
        if c.exists():
            return c
    return HERE / name

class Meter:
    """Peak RSS (process + children) above the starting level, CPU time."""
    def __init__(self):
        import psutil
        self.p = psutil.Process()
    def _rss(self):
        tot = self.p.memory_info().rss
        for c in self.p.children(recursive=True):
            try:
                tot += c.memory_info().rss
            except Exception:
                pass
        return tot
    def _cpu(self):
        t = self.p.cpu_times()
        import resource
        ch = resource.getrusage(resource.RUSAGE_CHILDREN)
        return t.user + t.system + ch.ru_utime + ch.ru_stime
    def __enter__(self):
        self.base, self.peak, self.stop = self._rss(), 0, False
        def loop():
            while not self.stop:
                self.peak = max(self.peak, self._rss())
                time.sleep(0.005)
        self.th = threading.Thread(target=loop, daemon=True)
        self.c0, self.t0 = self._cpu(), time.perf_counter()
        self.th.start()
        return self
    def __exit__(self, *a):
        self.wall = time.perf_counter() - self.t0
        self.cpu = self._cpu() - self.c0
        self.stop = True
        self.th.join()
        self.peak = max(self.peak, self._rss())
        self.out = {"wall_s": self.wall, "cpu_s": self.cpu, "cores_used": self.cpu / max(self.wall, 1e-9),
                    "peak_mem_above_start_mb": (self.peak - self.base) / 2**20,
                    "rss_at_start_mb": self.base / 2**20}

def latency(fn, X, n=LATENCY_ROWS):
    import numpy as np
    for i in range(20):
        fn(X[i:i + 1])
    t = []
    for i in range(n):
        a = time.perf_counter()
        fn(X[i:i + 1])
        t.append(time.perf_counter() - a)
    t = np.array(t) * 1e6
    return {"median_us": float(np.median(t)), "p99_us": float(np.percentile(t, 99))}

def throughput(fn, X, repeats=3):
    best = None
    for _ in range(repeats):
        with Meter() as m:
            fn(X)
        if best is None or m.out["wall_s"] < best["wall_s"]:
            best = m.out
    best["rows"] = int(len(X))
    best["rows_per_s"] = len(X) / best["wall_s"]
    best["us_per_row"] = best["wall_s"] / len(X) * 1e6
    return best

def load_module(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

# ---------------- data ----------------

def fold_data(fold, strategy):
    import numpy as np
    d = find_dir("data") / f"openset_medium_{fold}{'' if strategy == 'smote' else '_' + strategy}"
    meta = json.load(open(d / "meta.json"))
    return (np.load(d / "X_fit.npy"), np.load(d / "y_fit.npy").astype(np.int64),
            np.load(d / "X_test.npy"), meta)

def tree_params(fold, rep):
    p = find_dir("data") / "phase3" / f"baselines_openset_{rep}.json"
    sel = json.load(open(p))[fold]["selected"] if p.exists() else {}
    rf = sel.get("rf", {"n_estimators": 100, "max_depth": 30, "min_samples_leaf": 1,
                        "max_features": 0.3, "class_weight": "balanced"})
    xgb = sel.get("xgb", {"n_estimators": 300, "max_depth": 10, "learning_rate": 0.1,
                          "subsample": 0.9, "colsample_bytree": 0.8})
    return rf, xgb

def continuous_fold(fold):
    """The trees' continuous input for this fold, exactly as 20 builds it."""
    os.environ["BASELINE_REPR"] = "continuous_medium"
    b = load_module("bl", "20_openset_baselines.py")
    import pandas as pd
    data = find_dir("data")
    tr_all = pd.read_parquet(data / "temporal" / "train_windowed.parquet")
    te_all = pd.read_parquet(data / "temporal" / "test_windowed.parquet")
    excl = data / f"temporal_excl_{fold}"
    tr_f = pd.read_parquet(excl / "train_windowed.parquet") if excl.exists() else tr_all
    te_f = pd.read_parquet(excl / "test_windowed.parquet") if excl.exists() else te_all
    for f in (tr_all, te_all, tr_f, te_f):
        f["attack_cat"] = f["attack_cat"].replace({"Backdoors": "Backdoor"})
    name = {f.lower(): f for f in b.FAMILIES}[fold]
    Xf, yf, Xv, yv, Xt, yt, names, ni, _ = b.load_partitions(name, tr_f, te_f, tr_all)
    import numpy as np
    Xc, yc = b.cap(Xf, yf, np.random.default_rng(42))
    return Xc, yc, Xt

# ---------------- model wrappers ----------------

def fit_tm(X, y, cfg, seed=42):
    import numpy as np
    from tmu.tsetlin_machine import TMClassifier
    np.random.seed(seed)
    tm = TMClassifier(number_of_clauses=int(cfg["clauses"]), T=int(cfg["T"]), s=float(cfg["s"]),
                      weighted_clauses=False)
    Xu = np.ascontiguousarray(X, dtype=np.uint32)
    for _ in range(int(cfg["epochs"])):
        tm.fit(Xu, y.astype(np.uint32))
    return tm

def tm_size(model, tmp):
    import tm_fast
    tm_fast.save_model(model, tmp)
    return os.path.getsize(tmp)

def tree_models(threads):
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier
    return (lambda p: RandomForestClassifier(random_state=42, n_jobs=threads, **p),
            lambda p: XGBClassifier(random_state=42, n_jobs=threads, tree_method="hist",
                                    eval_metric="logloss", **p))

def size_of_tree(m, kind, tmp):
    if kind == "rf":
        return len(pickle.dumps(m, protocol=pickle.HIGHEST_PROTOCOL))
    m.get_booster().save_model(str(tmp) + ".ubj")
    return os.path.getsize(str(tmp) + ".ubj")

# ---------------- tasks (run in a child) ----------------

def task_preprocess(fold, threads, tmp):
    import numpy as np
    import pandas as pd
    w = load_module("w2", "00b_build_windows_v2.py")
    data = find_dir("data")
    df = pd.concat([pd.read_parquet(data / "temporal" / f"{p}_windowed.parquet",
                                    columns=["srcip", "dstip", "dsport", "proto", "state", "ts", "ltime",
                                             "dur", "sbytes", "dbytes", "spkts", "dpkts", "tcprtt"])
                    for p in ("train", "test")], ignore_index=True)
    w.build(df.iloc[:1000], ["medium"], True)                   # compile
    with Meter() as m:
        w.build(df, ["medium"], True)
    out = {"windows_medium": {**m.out, "rows": len(df), "us_per_row": m.out["wall_s"] / len(df) * 1e6}}
    # Boolean encoding: quantile thresholds (fitted on fit rows) + one-hot, on test rows
    te = pd.read_parquet(data / "temporal" / "test_windowed.parquet")
    meta = json.load(open(data / f"openset_medium_{fold}" / "meta.json"))
    num_cols = sorted({n.split(">Q")[0] for n in meta["feature_names"] if ">Q" in n})
    num_cols = [c for c in num_cols if c in te.columns]
    V = te[num_cols].to_numpy(np.float64)
    thr = np.nanquantile(V[:100000], meta["quantile_levels"], axis=0)  # thresholds' values do not affect cost
    cats = ["proto", "service", "state"]
    from sklearn.preprocessing import OneHotEncoder
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.uint8).fit(te[cats])
    def boolean(Vb, Cb):
        Z = np.nan_to_num(Vb, nan=-np.inf)
        lit = (Z[:, :, None] > thr.T[None]).reshape(len(Z), -1).astype(np.uint8)
        return np.concatenate([lit, enc.transform(Cb)], axis=1)
    C = te[cats]
    with Meter() as m:
        boolean(V, C)
    out["boolean_encoding"] = {**m.out, "rows": len(V), "us_per_row": m.out["wall_s"] / len(V) * 1e6,
                               "n_numeric": len(num_cols)}
    t = []
    for i in range(LATENCY_ROWS):
        a = time.perf_counter(); boolean(V[i:i + 1], C.iloc[i:i + 1]); t.append(time.perf_counter() - a)
    out["boolean_encoding"]["single_row"] = {"median_us": float(np.median(t) * 1e6),
                                             "p99_us": float(np.percentile(t, 99) * 1e6)}
    codes = {c: pd.Categorical(te[c]).categories for c in cats}
    def cont(Vb, Cb):
        X = [np.nan_to_num(Vb.astype(np.float32))]
        for c in cats:
            X.append(pd.Categorical(Cb[c], categories=codes[c]).codes[:, None].astype(np.float32))
        return np.concatenate(X, axis=1)
    with Meter() as m:
        cont(V, C)
    out["continuous_encoding"] = {**m.out, "rows": len(V), "us_per_row": m.out["wall_s"] / len(V) * 1e6}
    t = []
    for i in range(LATENCY_ROWS):
        a = time.perf_counter(); cont(V[i:i + 1], C.iloc[i:i + 1]); t.append(time.perf_counter() - a)
    out["continuous_encoding"]["single_row"] = {"median_us": float(np.median(t) * 1e6),
                                                "p99_us": float(np.percentile(t, 99) * 1e6)}
    return out

def task_bank(fold, threads, tmp, model):
    """One normality bank's binary problem, one model."""
    import numpy as np
    import banks, tm_fast
    Xf, yf, Xt, meta = fold_data(fold, TM_STRATEGY)
    ni = meta["all_class_names"].index("Normal")
    Xb, yb = banks.balanced_binary(Xf, yf, [ni], np.random.default_rng(42))
    del Xf
    cfg = json.load(open(find_dir("data") / "tuned" / f"open_{fold}.json"))["best"]
    out = {"rows": int(len(yb)), "features": int(Xb.shape[1])}
    if model == "tm":
        with Meter() as m:
            tm = fit_tm(Xb, yb, cfg)
        out["train"] = m.out
        mod = tm_fast.extract_tm(tm)
        Xt32 = np.ascontiguousarray(Xt[:20000], dtype=np.uint32)
        out["infer_tmu"] = throughput(lambda X: tm.predict(X), Xt32, repeats=1)
        out["latency_tmu"] = latency(lambda x: tm.predict(x), Xt32, n=200)
        out["infer_fast"] = throughput(lambda X: tm_fast.binary_margin(mod, X), Xt)
        out["latency_fast"] = latency(lambda x: tm_fast.binary_margin(mod, x), Xt)
        out["size_bytes"] = tm_size(mod, str(tmp) + ".npz")
        out["config"] = cfg
        out["clauses_total"] = int(sum(len(w) for w in mod["weights"]))
        out["includes_total"] = int(sum(m_.sum() for m_ in mod["include"]))
    else:
        rf_p, xgb_p = tree_params(fold, "boolean")
        mk_rf, mk_xgb = tree_models(threads)
        mdl = mk_rf(rf_p) if model == "rf" else mk_xgb(xgb_p)
        X32 = Xb.astype(np.float32)
        with Meter() as m:
            mdl.fit(X32, yb)
        out["train"] = m.out
        Xtf = Xt.astype(np.float32)
        out["infer"] = throughput(lambda X: mdl.predict_proba(X), Xtf)
        out["latency"] = latency(lambda x: mdl.predict_proba(x), Xtf)
        out["size_bytes"] = size_of_tree(mdl, model, tmp)
        out["params"] = rf_p if model == "rf" else xgb_p
    return out

def task_mc(fold, threads, tmp, model, rep="boolean"):
    """The multi-class problem (Normal + known families); with the rejection
    threshold this is also the whole single-stage pipeline."""
    import numpy as np
    import tm_fast
    out = {}
    if model == "tm":
        Xf, yf, Xt, meta = fold_data(fold, TM_STRATEGY)
        cfg = json.load(open(find_dir("data") / "tuned" / f"open_{fold}.json"))["best"]
        classes = sorted(set(np.unique(yf).tolist()))
        enc = {c: i for i, c in enumerate(classes)}
        y = np.array([enc[c] for c in yf])
        out["rows"], out["features"] = int(len(y)), int(Xf.shape[1])
        with Meter() as m:
            tm = fit_tm(Xf, y, cfg)
        out["train"] = m.out
        del Xf
        mod = tm_fast.extract_tm(tm)
        theta = 0.0
        decide = lambda X: np.where(tm_fast.class_sums(mod, X).max(1) < theta, 99, 0)
        Xt32 = np.ascontiguousarray(Xt[:20000], dtype=np.uint32)
        out["infer_tmu"] = throughput(lambda X: tm.predict(X), Xt32, repeats=1)
        out["latency_tmu"] = latency(lambda x: tm.predict(x), Xt32, n=200)
        out["infer_fast"] = throughput(decide, Xt)
        out["latency_fast"] = latency(decide, Xt)
        out["size_bytes"] = tm_size(mod, str(tmp) + ".npz")
        out["config"] = cfg
        out["clauses_total"] = int(sum(len(w) for w in mod["weights"]))
        out["includes_total"] = int(sum(m_.sum() for m_ in mod["include"]))
        return out
    if rep == "boolean":
        Xf, yf, Xt, meta = fold_data(fold, "none")
        Xf, Xt = Xf.astype(np.float32), Xt.astype(np.float32)
    else:
        Xf, yf, Xt = continuous_fold(fold)
    rf_p, xgb_p = tree_params(fold, rep)
    mk_rf, mk_xgb = tree_models(threads)
    classes = sorted(set(np.unique(yf).tolist()))
    enc = {c: i for i, c in enumerate(classes)}
    y = np.array([enc[c] for c in yf])
    out["rows"], out["features"] = int(len(y)), int(Xf.shape[1])
    if model == "rf":
        mdl = mk_rf(rf_p)
        with Meter() as m:
            mdl.fit(Xf, y)
    else:
        mdl = mk_xgb({**xgb_p})
        mdl.set_params(eval_metric="mlogloss")
        w = len(y) / (len(classes) * np.bincount(y))
        with Meter() as m:
            mdl.fit(Xf, y, sample_weight=w[y])
    out["train"] = m.out
    decide = lambda X: np.where(mdl.predict_proba(X).max(1) < 0.5, 99, 0)
    out["infer"] = throughput(decide, Xt)
    out["latency"] = latency(decide, Xt)
    out["size_bytes"] = size_of_tree(mdl, model, tmp)
    out["params"] = rf_p if model == "rf" else xgb_p
    return out

def task_two_stage(fold, threads, tmp):
    """The full OpenSATM pipeline: every bank trained (threads parallel
    processes), then normality gate + family banks + calibrated decision."""
    import numpy as np
    import banks, tm_fast
    Xf, yf, Xt, meta = fold_data(fold, TM_STRATEGY)
    ni = meta["all_class_names"].index("Normal")
    fams = [c for c in meta["known_classes"] if c != ni]
    cfg = json.load(open(find_dir("data") / "tuned" / f"open_{fold}.json"))["best"]
    with Meter() as m:
        models = banks.train_banks(Xf, yf, fams, ni, cfg, 42, n_workers=threads)
    out = {"banks": len(models), "train": m.out, "config": cfg}
    models = {k: v[int(cfg["epochs"])] for k, v in models.items()}
    del Xf
    names = ["normal"] + [f"fam{c}" for c in fams]
    tau_b, theta = 0.0, 0.0
    def decide(X):
        b = tm_fast.binary_margin(models["normal"], X)
        f = np.stack([tm_fast.binary_margin(models[n], X) for n in names[1:]], 1)
        return np.where(b >= tau_b, ni, np.where(f.max(1) < theta, 99, np.asarray(fams)[f.argmax(1)]))
    out["infer_fast"] = throughput(decide, Xt)
    out["latency_fast"] = latency(decide, Xt)
    out["size_bytes"] = int(sum(tm_size(models[n], f"{tmp}_{n}.npz") for n in names))
    out["clauses_total"] = int(sum(sum(len(w) for w in models[n]["weights"]) for n in names))
    out["includes_total"] = int(sum(sum(m_.sum() for m_ in models[n]["include"]) for n in names))
    return out

TASKS = {
    "preprocess": lambda f, t, p: task_preprocess(f, t, p),
    "bank_tm": lambda f, t, p: task_bank(f, t, p, "tm"),
    "bank_rf": lambda f, t, p: task_bank(f, t, p, "rf"),
    "bank_xgb": lambda f, t, p: task_bank(f, t, p, "xgb"),
    "mc_tm": lambda f, t, p: task_mc(f, t, p, "tm"),
    "mc_rf_boolean": lambda f, t, p: task_mc(f, t, p, "rf", "boolean"),
    "mc_xgb_boolean": lambda f, t, p: task_mc(f, t, p, "xgb", "boolean"),
    "mc_rf_continuous": lambda f, t, p: task_mc(f, t, p, "rf", "continuous_medium"),
    "mc_xgb_continuous": lambda f, t, p: task_mc(f, t, p, "xgb", "continuous_medium"),
    "two_stage_tm": lambda f, t, p: task_two_stage(f, t, p),
}
SINGLE_THREADED = {"bank_tm", "mc_tm"}     # tmu training has no thread option

def hardware():
    def sysctl(k):
        try:
            return subprocess.check_output(["sysctl", "-n", k], text=True).strip()
        except Exception:
            return None
    import numpy, sklearn, xgboost
    return {"machine": platform.machine(), "cpu": sysctl("machdep.cpu.brand_string"),
            "cores_logical": os.cpu_count(), "perf_cores": sysctl("hw.perflevel0.physicalcpu"),
            "eff_cores": sysctl("hw.perflevel1.physicalcpu"),
            "memory_gb": int(sysctl("hw.memsize") or 0) / 2**30, "os": platform.platform(),
            "python": platform.python_version(), "numpy": numpy.__version__,
            "sklearn": sklearn.__version__, "xgboost": xgboost.__version__}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", default="dos")
    ap.add_argument("--threads", default="1,8")
    ap.add_argument("--only", default=None)
    ap.add_argument("--task", default=None)
    a = ap.parse_args()
    sys.path.insert(0, str(HERE))
    out_dir = find_dir("data") / "phase5"
    out_dir.mkdir(parents=True, exist_ok=True)
    if a.task:                                    # child
        t = int(a.threads)
        res = TASKS[a.task](a.fold, t, out_dir / f"_bench_{a.task}_{t}")
        print("RESULT " + json.dumps(res, default=float))
        return
    path = out_dir / "benchmark.json"
    results = json.load(open(path)) if path.exists() else {}
    results["hardware"] = hardware()
    results["fold"] = a.fold
    tasks = a.only.split(",") if a.only else list(TASKS)
    for task in tasks:
        for t in [int(x) for x in a.threads.split(",")]:
            if task in SINGLE_THREADED and t != 1:
                continue
            if task == "preprocess" and t != 1:
                continue
            env = dict(os.environ, OMP_NUM_THREADS=str(t), OPENBLAS_NUM_THREADS=str(t),
                       VECLIB_MAXIMUM_THREADS=str(t), MKL_NUM_THREADS=str(t))
            t0 = time.perf_counter()
            p = subprocess.run([sys.executable, "-u", __file__, "--task", task, "--threads", str(t),
                                "--fold", a.fold], env=env, capture_output=True, text=True)
            line = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
            if p.returncode or not line:
                print(f"FAILED {task} threads={t}\n{p.stderr[-2000:]}", flush=True)
                continue
            results.setdefault(task, {})[str(t)] = json.loads(line[0][7:])
            json.dump(results, open(path, "w"), indent=1, default=float)
            print(f"{task} threads={t} done in {time.perf_counter()-t0:.0f}s", flush=True)
    print(f"written to {path}")

if __name__ == "__main__":
    main()
