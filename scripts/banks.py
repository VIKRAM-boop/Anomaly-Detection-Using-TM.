"""
Shared training and scoring for Step 5's clause banks.

A run trains one normality branch (Normal vs every known attack) and one
detector per known attack family (that family vs everything else), each a
binary TMClassifier on a balanced sample. Training is single-threaded inside
tmu, so banks are trained in parallel worker processes; each worker returns
only the extracted clause masks and weights (tm_fast), which are then scored
in bulk in the parent.

Used by 10_train_family_banks.py for the final runs and by 17_tune_tm.py for
hyperparameter selection, so both go through exactly the same code.
"""
import multiprocessing as mp

import numpy as np

import tm_fast

N_WORKERS = 5   # 9 banks finish in two rounds instead of three
DEFAULT_CONFIG = {"clauses": 2000, "T": 200, "s": 5.0, "epochs": 1}

def balanced_binary(X, y, positive_labels, rng, max_pos=None):
    """All rows of the positive labels (optionally subsampled to max_pos) plus
    an equal number drawn from the rest, so no bank is biased by how common
    its family happens to be."""
    pos = np.where(np.isin(y, positive_labels))[0]
    if max_pos is not None and len(pos) > max_pos:
        pos = rng.choice(pos, max_pos, replace=False)
    neg_pool = np.where(~np.isin(y, positive_labels))[0]
    neg = rng.choice(neg_pool, size=min(len(pos), len(neg_pool)), replace=False)
    idx = np.concatenate([pos, neg])
    rng.shuffle(idx)
    return X[idx], np.isin(y[idx], positive_labels).astype(np.uint32)

def _train_one(job):
    """Worker: train one binary bank; return its extracted model.

    If epochs_to_report is given, the model is extracted after each of those
    epoch counts, so one training run yields several epoch settings."""
    from tmu.tsetlin_machine import TMClassifier
    name, X, y, cfg, seed, verify_X, epochs_to_report = job
    np.random.seed(seed)                       # TMClassifier draws from the global RNG
    tm = TMClassifier(number_of_clauses=int(cfg["clauses"]), T=int(cfg["T"]),
                      s=float(cfg["s"]), weighted_clauses=False)
    X = np.ascontiguousarray(X, dtype=np.uint32)
    checkpoints = {}
    targets = sorted(set(epochs_to_report or [int(cfg["epochs"])]))
    done = 0
    for ep in targets:
        for _ in range(ep - done):
            tm.fit(X, y)
        done = ep
        checkpoints[ep] = tm_fast.extract_tm(tm)
    if verify_X is not None:
        tm_fast.verify_against_tmu(tm, verify_X, n=len(verify_X))
    return name, checkpoints

def train_banks(X_fit, y_fit, families, normal_idx, cfg, seed,
                max_pos=None, verify_X=None, epochs_to_report=None, n_workers=N_WORKERS):
    """Returns {bank_name: {epochs: model}} for the normality bank ("normal")
    and one bank per family ("fam<c>")."""
    rng = np.random.default_rng(seed)
    jobs = []
    Xb, yb = balanced_binary(X_fit, y_fit, [normal_idx], rng, max_pos)
    jobs.append(("normal", Xb, yb, cfg, seed, verify_X, epochs_to_report))
    for c in families:
        Xb, yb = balanced_binary(X_fit, y_fit, [c], rng, max_pos)
        jobs.append((f"fam{c}", Xb, yb, cfg, seed, None, epochs_to_report))
    # One fresh process per bank. tmu keeps random state inside its compiled
    # code that numpy's seed does not reset, so a reused worker carries state
    # from whichever bank it trained before -- and which worker gets which
    # bank depends on timing. Fresh processes make a run reproducible.
    ctx = mp.get_context("spawn")
    with ctx.Pool(min(n_workers, len(jobs)), maxtasksperchild=1) as pool:
        results = pool.map(_train_one, jobs, chunksize=1)
    return dict(results)

def score_banks(models, X):
    """{bank_name: margin array} -- positive minus negative class score."""
    return {name: tm_fast.binary_margin(m, X).astype(np.float32) for name, m in models.items()}

def closed_macro_f1(val_scores, y_val, families, normal_idx, missed_attack_budget=0.01):
    """Known-class macro F1 on validation for Step 5's decision with no
    unknown option: Normal when benign support clears tau_B (pinned so that at
    most `missed_attack_budget` of attack rows look benign), else the
    strongest family. Validation contains no withheld-family rows, so this is
    a legitimate selection objective."""
    from sklearn.metrics import precision_recall_fscore_support
    # margins are integers and saturate at +/-2T, so the tau_B quantile can
    # land on a value shared by most rows; a seeded jitter below the integer
    # gap breaks those ties without reordering distinct scores
    jit = np.random.default_rng(0)
    vb = val_scores["normal"] + jit.uniform(0, 1e-3, len(y_val))
    vf = np.stack([val_scores[f"fam{c}"] for c in families], axis=1)
    vf = vf + jit.uniform(0, 1e-3, vf.shape)
    atk = y_val != normal_idx
    tau_b = np.quantile(vb[atk], 1.0 - missed_attack_budget)
    pred = np.asarray(families)[vf.argmax(axis=1)]
    pred[vb >= tau_b] = normal_idx
    return float(precision_recall_fscore_support(y_val, pred, average="macro",
                                                  zero_division=0)[2])
