# Paper 3 — Full Run Results

> **Status: partially superseded.** These numbers come from the pipeline as it
> stood before the Phase 1 corrections. The validation set used here was drawn
> from the post-SMOTE array and was therefore majority synthetic, so any figure
> below that depends on validation — the RF/DT hyperparameter choices, the
> calibration temperature, the ensemble selection — was selected against a
> contaminated surface. The *test* numbers remain valid, since the test set was
> always held out correctly.
>
> See `PHASE1_RESULTS.md` for the corrected pipeline.

All numbers below were computed locally from the saved model outputs in
`data/` (TM scores in `data/scores_*/`, baseline results in
`data/tuned_baselines.json`, Step 5 results in `data/openset_*fuzzers*`).
The Colab notebook runs the same scripts and will reproduce these numbers.

## Steps 1-4: Random Forest, Decision Tree, Tsetlin Machine

### Random Forest / Decision Tree (continuous features)

`07_tune_baselines.py` runs RF/DT on continuous features only — the
binarized-feature path was removed from the script, so those numbers no
longer apply to the current code.

| Model | Test Acc | Weighted Precision | Weighted Recall | Weighted F1 | Macro F1 |
|---|---|---|---|---|---|
| Random Forest | 0.9582 | 0.9751 | 0.9582 | 0.9649 | 0.5644 ± 0.0009 |
| Decision Tree | 0.9556 | 0.9705 | 0.9556 | 0.9618 | 0.5235 ± 0.0004 |

### Tsetlin Machine, per branch and seed

| Run | Test Acc | Weighted Precision | Weighted Recall | Weighted F1 | Macro F1 |
|---|---|---|---|---|---|
| short, seed 42 | 0.9449 | 0.9724 | 0.9449 | 0.9550 | 0.4401 |
| medium, seed 42 | 0.9432 | 0.9703 | 0.9432 | 0.9542 | 0.4295 |
| long, seed 42 | 0.9444 | 0.9704 | 0.9444 | 0.9542 | 0.4360 |
| medium, seed 7 | 0.9473 | 0.9702 | 0.9473 | 0.9564 | 0.4420 |
| medium, seed 1234 | 0.9470 | 0.9695 | 0.9470 | 0.9563 | 0.4424 |

Medium-scale macro F1 across 3 seeds: **0.4380 ± 0.0057** (mean ± std).
These are raw argmax predictions with no calibration applied.

### Multi-scale ensemble + calibration (`08_multiscale_ensemble.py`)

RF/DT receive short, medium and long window features together in one feature
vector. The single-scale TM branches above do not, so the comparable TM
configuration is the multi-scale ensemble, which combines all three branches
and applies prior calibration.

Selected on validation: combiner = `max`, τ = 30.

Validation macro F1 per branch: short 0.4991, medium 0.4913, long 0.5062.

Applied once to test:

| Configuration | Test Acc | Weighted F1 | Macro F1 |
|---|---|---|---|
| Medium branch alone (calibrated) | 0.9538 | 0.9597 | 0.4665 |
| **Multi-scale ensemble (max, τ=30)** | 0.9528 | 0.9591 | **0.4717** |

The ensemble gains ~0.005 macro F1 over the best single branch — real but
marginal. Most of the gain over the uncalibrated 0.4295 comes from
calibration (τ), not from combining scales.

### Headline comparison

TM entry uses the multi-scale ensemble, so all three models see all three
window scales.

| Model | Test Acc | Weighted F1 | Macro F1 |
|---|---|---|---|
| Random Forest | 0.9582 | 0.9649 | 0.5644 |
| Decision Tree | 0.9556 | 0.9618 | 0.5235 |
| Tsetlin Machine (multi-scale ensemble) | 0.9528 | 0.9591 | 0.4717 |

**Remaining asymmetry:** the TM still works from binarized
(quantile-threshold) literals while RF/DT use raw continuous values. The TM
also underfits — ~0.70 train accuracy vs. the trees' 0.87-0.94 — so the
macro F1 gap is a mix of representation and model capacity.

### Per-class breakdown, multi-scale ensemble

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Analysis | 0.06 | 0.14 | 0.09 | 678 |
| Backdoor | 0.10 | 0.03 | 0.05 | 702 |
| DoS | 0.33 | 0.74 | 0.46 | 5,161 |
| Exploits | 0.72 | 0.46 | 0.56 | 12,850 |
| Fuzzers | 0.38 | 0.79 | 0.51 | 6,711 |
| Generic | 1.00 | 0.97 | 0.99 | 68,460 |
| Normal | 1.00 | 0.98 | 0.99 | 408,930 |
| Reconnaissance | 0.69 | 0.77 | 0.73 | 4,064 |
| Shellcode | 0.16 | 0.65 | 0.26 | 440 |
| Worms | 0.05 | 0.27 | 0.08 | 52 |

Macro F1 is dragged down almost entirely by Analysis, Backdoor and Worms —
the rare, heavily-overlapping classes. Generic and Normal are effectively
solved (0.99), and Reconnaissance/Exploits/Fuzzers are workable.

---

## Step 5: Open-Set Recognition (Fuzzers held out)

Fuzzers was removed entirely from training (0 Fuzzers rows seen at any
point). Its 6,711 test rows were relabelled `UNKNOWN` and scored against
9 one-vs-rest clause banks (1 normality bank + 8 attack-family banks) chained
through the three-way decision rule: benign / known family / unknown.

### Corrected thresholding (tau_B pinned to a 1% missed-attack budget — the reported result)

| Held out | Rows | AUROC | Caught @1% | @5% | @10% | @20% | Known Acc | Closed-set Acc |
|---|---|---|---|---|---|---|---|---|
| Fuzzers | 6,711 | 0.845 | 15.4% | 28.6% | 39.5% | 51.6% | 0.9774 | 0.9565 |

- **AUROC 0.845** — how well the benign+family scores separate real known
  attacks from the novel class, independent of any specific threshold.
- **Caught @N%** — % of the 6,711 held-out Fuzzers rows correctly flagged
  UNKNOWN, when the system is tuned to tolerate at most N% false alarms on
  known traffic.
- **Known Acc** — accuracy on known classes when the reject option is active
  (at the 20% budget), vs. **Closed-set Acc** — the same model with no
  unknown option at all. The reject option *improves* known-class accuracy
  (0.9774 vs 0.9565): rows it declines to name are disproportionately rows
  it would have gotten wrong anyway.
- A closed-set model catches **0% of Fuzzers by construction** — it has no
  "unknown" label to assign, so it would confidently misfile all 6,711 rows.
- **18.9%** of Fuzzers rows were routed to Normal (never reached the
  unknown-detection step at all) — the main source of missed detections.
- At the 5% budget, Fuzzers' most common single outcome is **UNKNOWN (28.6%)**
  — ahead of any specific wrong-family label (next highest: Normal 18.9%,
  Exploits 16.1%).

---

## Where these numbers came from

```
scripts/07_tune_baselines.py          -> data/tuned_baselines.json
scripts/02c_train_with_scores.py      -> data/scores_<scale>_seed<seed>/
scripts/08_multiscale_ensemble.py     -> data/multiscale_ensemble_output.txt
scripts/09_openset_preprocess.py Fuzzers
scripts/10_train_family_banks.py Fuzzers
scripts/11_openset_decision.py Fuzzers -> data/banks_medium_fuzzers/summary.json
scripts/14_openset_redecide.py         -> data/openset_sweep_medium.json (corrected thresholds)
```
