"""
Tree baselines (Random Forest, Decision Tree) reported with the same five
columns paper 2's Table II uses: Train Accuracy, Test Accuracy, Precision,
Recall, F1. These are the comparison models for the TM.

Metric convention: weighted, because that is demonstrably what their
baseline rows use -- in every one of their baseline rows recall equals
accuracy exactly, which is only true of weighted averaging. Macro is
printed alongside so the stricter figure is never hidden.

Features: continuous. Trees choose their own split points and are
measurably hurt by pre-binning (RF loses ~0.09 macro F1 on the binarized
matrix), so binarizing for them would understate them. The TM is scored
separately since it can only accept binary input.

Balancing: capped at 40k/class then SMOTE, matching the TM's pipeline.
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from imblearn.over_sampling import SMOTE

RANDOM_STATE = 42
PER_CLASS_CAP = 40000
DROP = ["srcip", "sport", "dstip", "dsport", "stime", "ltime", "ts", "label"]
CAT = ["proto", "service", "state"]
DATA = Path("/tmp/tm_work")

def main():
    tr = pd.read_parquet(DATA / "train_windowed.parquet")
    te = pd.read_parquet(DATA / "test_windowed.parquet")
    for f in (tr, te):
        f["attack_cat"] = f["attack_cat"].replace({"Backdoors": "Backdoor"})

    le = LabelEncoder().fit(tr["attack_cat"])
    y_full = le.transform(tr["attack_cat"])
    y_test = le.transform(te["attack_cat"])

    cols = [c for c in tr.columns if c not in DROP + ["attack_cat"]]
    Xtr, Xte = tr[cols].copy(), te[cols].copy()
    for c in CAT:
        cats = pd.Categorical(Xtr[c])
        Xtr[c] = cats.codes
        Xte[c] = pd.Categorical(Xte[c], categories=cats.categories).codes
    Xtr = np.nan_to_num(Xtr.to_numpy(np.float32), nan=0, posinf=0, neginf=0)
    Xte = np.nan_to_num(Xte.to_numpy(np.float32), nan=0, posinf=0, neginf=0)

    rng = np.random.default_rng(RANDOM_STATE)
    keep = []
    for c in np.unique(y_full):
        i = np.where(y_full == c)[0]
        keep.append(rng.choice(i, PER_CLASS_CAP, replace=False) if len(i) > PER_CLASS_CAP else i)
    keep = np.concatenate(keep)
    Xc, yc = Xtr[keep], y_full[keep]
    print(f"capped: {Xc.shape}, applying SMOTE...", flush=True)
    Xb, yb = SMOTE(random_state=RANDOM_STATE).fit_resample(Xc, yc)
    print(f"balanced: {Xb.shape}  test: {Xte.shape}\n", flush=True)

    models = [
        ("Random Forest", RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=RANDOM_STATE)),
        ("Decision Tree", DecisionTreeClassifier(random_state=RANDOM_STATE)),
    ]

    rows = []
    for name, m in models:
        t0 = time.perf_counter()
        m.fit(Xb, yb)
        tr_acc = accuracy_score(yb, m.predict(Xb))
        p = m.predict(Xte)
        te_acc = accuracy_score(y_test, p)
        wp, wr, wf, _ = precision_recall_fscore_support(y_test, p, average="weighted", zero_division=0)
        mp, mr, mf, _ = precision_recall_fscore_support(y_test, p, average="macro", zero_division=0)
        rows.append((name, tr_acc, te_acc, wp, wr, wf, mp, mr, mf))
        print(f"{name:<15s} train={tr_acc:.4f} test={te_acc:.4f} "
              f"| weighted P={wp:.4f} R={wr:.4f} F1={wf:.4f} "
              f"| macro P={mp:.4f} R={mr:.4f} F1={mf:.4f}  ({time.perf_counter()-t0:.0f}s)", flush=True)

    print("\n" + "=" * 78)
    print("WEIGHTED (paper 2's convention for baselines)")
    print(f"{'Model':<16}{'TrainAcc':>10}{'TestAcc':>10}{'Precision':>11}{'Recall':>9}{'F1':>9}")
    for n, tra, tea, wp, wr, wf, *_ in rows:
        print(f"{n:<16}{tra:>10.4f}{tea:>10.4f}{wp:>11.4f}{wr:>9.4f}{wf:>9.4f}")
    print("\nMACRO (stricter, shown for completeness)")
    print(f"{'Model':<16}{'TrainAcc':>10}{'TestAcc':>10}{'Precision':>11}{'Recall':>9}{'F1':>9}")
    for n, tra, tea, _, _, _, mp, mr, mf in rows:
        print(f"{n:<16}{tra:>10.4f}{tea:>10.4f}{mp:>11.4f}{mr:>9.4f}{mf:>9.4f}")

if __name__ == "__main__":
    main()
