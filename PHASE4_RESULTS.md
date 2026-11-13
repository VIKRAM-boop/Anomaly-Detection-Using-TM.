# Phase 4 — Resampling, Boolean encoding and a single-stage architecture

Produced 2026-11-12 to 2026-11-13 on the chronological pipeline (Phase 1) with
the Phase 3 tuned configurations. Every number in the tables at the end is
generated from the run outputs by `scripts/28_phase4_tables.py`; the
summaries below quote from those tables.

Protocol, unchanged from Phase 3:

- **Splits.** Closed set: chronological train / validation / test, validation
  being the last 10% of the training period. Open set: leave one family out;
  the held-out family is removed from fit **and** validation and appears only
  in test.
- **Fitting.** Quantile thresholds, resampling, calibration and
  hyperparameter selection use the fit or validation partition only.
- **Budget.** The rejection budget is the share of **all** known validation
  rows (Normal and known attacks) predicted Unknown. 5% is reported; 1% and
  10% are in the JSON outputs. Both architectures in Task 3 use this same
  definition.
- **Seeds.** TM results are mean ± std over 5 seeds (7, 42, 99, 1234, 2024).
  Tree baselines use 3 seeds (42, 7, 1234), because each model costs more
  than a TM run.

Task 3 was the stated priority, so it comes first.

## Task 3 — single-stage architecture

**Design (`24_single_stage.py`).** There is no benign gate. Normal is one
more known class, and every row gets one support score per known class.

- The prediction is the class with the strongest support.
- The row is predicted Unknown instead if a rejection rule fires:
  - **global**: the top score is below one threshold.
  - **class-conditional**: the top score is below the predicted class's own
    threshold. Each class's threshold sits at a common quantile q of that
    class's validation scores, with q bisected to meet the budget.
  - **margin**: the gap between the best and second-best scores is small.

Two score sources were run:

1. **One-vs-rest banks (version 1).** The existing Phase 3 binary banks:
   - the normality bank's margin is Normal's support, and each family bank's
     margin is that family's support;
   - all banks share one configuration and the [−2T, 2T] range, so the
     scores are on a common scale;
   - no retraining was needed. The two-stage numbers are recomputed in the
     same script, from the same runs, with 18's own `Calibrated` class. They
     reproduce the Phase 3 tables, so both rows of each comparison come from
     identical models and identical metric code.
2. **One multi-class TM (version 2, `25_train_single_stage.py`).** Trained
   over Normal plus the known families; its class sums are the supports.
   - Configuration: each held-out family uses its open-set tuned bank
     configuration (`tuned/open_<family>.json`), **not** the closed-set
     tuning. The closed-set search saw every family in validation, so using
     its result would leak the held-out family into model selection.

Ties at the top score are rare, between 0% and 2.3% of test rows. The
exception is the 52 Worms test rows, where 25% tie.

**Result (5% budget, family average).**

| | two-stage | single-stage (banks) | single-stage (multi-class) |
|---|---|---|---|
| unknown recall, global | 62.3% | 51.5% | 54.7% |
| unknown recall, margin | 70.1% | 64.2% | 62.5% |
| unknown → Normal (global) | 11.6% | 0.0% | 0.0% |
| unknown → known family (global) | 26.1% | 48.5% | 45.2% |
| Normal false-alarm rate (global) | 2.08% | 5.90% | 5.14% |
| known attacks rejected (global) | 24.7% | 15.3% | 15.9% |
| known-attack macro F1 (global) | 0.337 | 0.366 | 0.366 |
| AUROC, margin rule | 0.868 | 0.932 | 0.929 |

What this shows:

- **The leak to Normal disappears.** Without a gate, almost no held-out
  attack is called Normal: 0.0–0.1%, against 11.6% for two-stage. For
  Generic, the worst case, the leak falls from 51% to 0%.
- **But those rows go to a known attack family instead,** not to Unknown.
  Overall unknown recall is 7–11 points lower than two-stage under both the
  global and the margin rule. Single-stage does not beat two-stage on unknown
  recall at this budget.
- **The budget falls on different rows.** In two-stage, Normal rows mostly
  pass the gate and are never candidates for rejection. So the 5% budget
  lands on known attacks: about 25% of known-attack test rows are rejected as
  Unknown. In single-stage the budget is shared with Normal:
  - about 2% of Normal rows are rejected;
  - only 15–21% of known attacks are rejected;
  - known-attack F1 is higher, and Normal false alarms are higher
    (4.3–5.9% vs 2.1%).
- **Ranking is better.** With the margin rule, single-stage ranks unknowns
  better across all thresholds (AUROC 0.93 vs 0.87), even though its recall
  at the 5% operating point is lower.
- **Per family** (tables below):
  - Two-stage has higher unknown recall for most held-out families.
  - Single-stage wins for Fuzzers under the global rule (banks 55.4% vs
    47.3%). But with Fuzzers held out, 12–18% of Normal rows are flagged,
    with a seed std of about ±7 points. This is the least stable result in
    the table.
  - Generic under the margin rule is near 1.5% in every architecture.
    Held-out Generic rows are confidently assigned to a single known class.
- **Score source barely matters.** The banks and the multi-class TM give
  family averages within about 3 points of each other.
- **The class-conditional rule fails in single-stage:** 3% recall, AUROC
  0.11–0.14.
  - Held-out rows look like typical members of their nearest known family.
    Example, Analysis held out, seed 42: of 678 test rows, 201 are predicted
    Backdoor, 279 DoS and 174 Exploits. Each scores above those classes' own
    7.6% quantile, so none is rejected.
  - Normal is about three quarters of validation, so most of the
    class-conditional budget is spent rejecting Normal rows: Normal false
    alarms are 8%.
  - The rule is reported as it ran; it was not tuned further.

**Resampling × architecture.** Random oversampling gives the same picture as
SMOTE. Without oversampling:

- the global rule loses 11–18 points of unknown recall;
- the margin rule gains (single-stage 70–74%), but known-attack F1 falls to
  0.27–0.28;
- see the Task 1 open-set table.

**Cost.**

- One multi-class TM: about 270 s to train and 77 s to score, wall-clock with
  three models training concurrently.
- One run of the nine binary banks: about 125 s, wall-clock with all nine
  training concurrently.

## Task 1 — SMOTE

### Where SMOTE sits in the pipeline

The order is the same in the closed-set pipeline (`01c_preprocess_temporal.py`)
and the open-set pipeline (`09_openset_preprocess.py`):

1. Compute the continuous features.
2. Fit the quantile thresholds on the fit partition.
3. **Binarise.**
4. Cap majority classes at 40,000 rows (random undersampling).
5. **SMOTE on the Boolean matrix,** followed by rounding at 0.5.

Validation and test are never resampled.

### Do synthetic rows satisfy the feature constraints? (`26_smote_validity.py`)

Every synthetic row was checked against three constraints:

- the protocol, service and state one-hot groups hold exactly one 1;
- each feature's quantile ladder is monotone (a value above threshold k is
  also above every threshold below k);
- the relational features are consistent with each other.

**No synthetic row violates any constraint.** This holds on the closed set
and on all nine open-set fit sets, about 2.3 million synthetic rows in total.

The reason is stronger than "the constraints hold". **Every synthetic row is
an exact copy of a real row of its own class (100.0%).**

- SMOTE draws one λ per synthetic row and sets x_new = x + λ·(x_nn − x).
- On 0/1 data, every bit where x and its neighbour differ takes the same
  value λ, and every bit where they agree is unchanged.
- Rounding at 0.5 therefore returns all of x (λ < 0.5) or all of x_nn
  (λ ≥ 0.5).
- So SMOTE here creates no new patterns. It is random oversampling, with
  copies drawn in proportion to how often a row is someone's nearest
  neighbour. For example, Worms has 103 real fit rows and its 39,897
  synthetic rows contain exactly 103 distinct patterns.

### Alternatives under identical partitions

- **`ros`:** random duplication of minority rows up to the 40,000 cap.
- **`none`:** the capped real rows only, with no synthetic or duplicated rows.
  Majority classes are still undersampled to 40,000, so this is the
  balanced-sampling variant.
  - tmu's TMClassifier has no per-sample weight, so class weighting cannot be
    applied in TM training.
  - Its decision-time analogue is reported instead: the prior-calibrated
    column, an argmax over class sums + τ·log prior, with τ chosen on
    validation.
  - The tree baselines in Task 2 use real class weighting on these same rows.

**Identical partitions.** The partitions are the same across strategies, and
this was checked:

- Each variant's validation and test arrays were asserted byte-identical to
  the SMOTE build's, then symlinked to it.
- The real fit rows are identical too, because the same `rng(42)` drew the
  cap. The `none` fit set equals the real-row prefix of the SMOTE fit set.

**Closed set (test, 5 seeds):**

| strategy | macro F1, argmax | macro F1, prior-calibrated |
|---|---|---|
| SMOTE | 0.4415 ± 0.0113 | 0.4628 ± 0.0076 |
| random oversampling | 0.4413 ± 0.0086 | 0.4635 ± 0.0060 |
| none | 0.3910 ± 0.0017 | 0.3905 ± 0.0012 |

- SMOTE and random oversampling are indistinguishable, as the copy result
  predicts.
- Without oversampling, the TM never predicts Backdoor, Shellcode or Worms,
  and almost never predicts DoS (F1 0.000, 0.000, 0.000 and 0.005).
- Exploits and Fuzzers gain recall instead.
- The prior correction does not recover the missing classes.

**Open set (5% budget, family average).**

- Two-stage, global rule:
  - SMOTE: 62.3% unknown recall.
  - Random oversampling: 61.9%.
  - None: 49.7%.
- Margin rule without oversampling: 74.5% recall, the highest in the
  table, but known-attack F1 falls to 0.24 (0.29 with SMOTE). That recall is
  bought with worse decisions on the known attacks.

Per-class precision, recall and F1 for each strategy are in the tables.

## Task 2 — Boolean encoding

### Same rows, same features, five models (closed set)

All five models see the same real fit rows: the `none` set, the capped
training partition. They are evaluated on the same test rows.

- The **continuous** input is rebuilt by `27_encoding_study.py` from the
  same windows the TM uses (base + medium). The script asserts that the
  labels match row for row.
- The trees use class weighting: RF `class_weight="balanced"`, and XGBoost
  balanced sample weights. Both get the same small grid, selected on
  validation, for every representation.

| model | input | macro F1 |
|---|---|---|
| TM | Boolean, 5 thresholds | 0.3905 ± 0.0012 (0.4628 with SMOTE) |
| RF | Boolean, 5 thresholds | 0.4954 ± 0.0025 |
| XGBoost | Boolean, 5 thresholds | 0.4975 ± 0.0008 |
| RF | continuous | 0.5947 ± 0.0021 |
| XGBoost | continuous | 0.6094 ± 0.0007 |

Take the gap from the best TM (0.463) to continuous XGBoost (0.609) and split
it in two:

- About 0.035 is the model: RF and XGBoost against the TM on the **same**
  Boolean input.
- About 0.11 is the **encoding**: the same trees, Boolean against continuous.

Binarisation is therefore the larger cost. The loss is concentrated in the
rare families, comparing XGBoost on Boolean against continuous input:

| family | Boolean F1 | continuous F1 |
|---|---|---|
| Worms | 0.18 | 0.66 |
| Shellcode | 0.31 | 0.63 |
| Reconnaissance | 0.72 | 0.83 |
| Backdoor | 0.10 | 0.16 |

Generic and Normal are unaffected (0.99 under both).

### Number of thresholds

Thresholds are evenly spaced quantiles, fit on the training partition.
5 thresholds is the original ladder; 3, 9 and 19 were added.

| thresholds | Boolean features | TM | RF | XGBoost | TM train / score (s) | data size |
|---|---|---|---|---|---|---|
| 3 | 323 | 0.384 | 0.475 | 0.471 | 16 / 33 | 276 MB |
| 5 | 425 | 0.391 | 0.495 | 0.498 | 18 / 39 | 363 MB |
| 9 | 629 | 0.396 | 0.534 | 0.524 | 23 / 51 | 537 MB |
| 19 | 1139 | 0.407 | 0.550 | 0.555 | 37 / 85 | 972 MB |

- More thresholds help the trees steadily: 19 thresholds recover about half
  of the Boolean→continuous gap.
- The TM gains only about 0.02, with a std up to 0.01 at 9 and 19 thresholds.
- Cost grows roughly linearly with the number of features.

### Does the encoding separate Normal from each family?

Measured as the share of each family's test rows whose Boolean vector is
identical to some Normal test row's:

- **At 5 thresholds: at most 1.3%** (Fuzzers). It is 0.0% for Backdoor,
  DoS, Exploits, Generic and Worms.
- At 3 thresholds it rises to 7–17% for Analysis, Fuzzers, Reconnaissance,
  Shellcode and Worms.

So at 5 or more thresholds the encoding keeps enough information to tell
Normal apart from every family. What it loses is the detail **between attack
families**: 2.2% of test rows share their exact pattern with a row of another
class.

### Open set

`20_openset_baselines.py` was run with RF and XGBoost on two inputs:

- **Boolean:** the TM's own literals, with the same real fit rows as the
  `none` open-set TM, asserted label-identical.
- **Continuous, base + medium windows:** this removes a mismatch in the
  Phase 3 baselines, which used all three scales' windows while TM_medium
  uses only base + medium.

Both use the same per-family protocol and 5% budget as the TM.

These baselines are single-stage by construction: Normal is a class, and
novelty is 1 − max class probability, which is the global rule. The matching
TM row is therefore the single-stage, global one.

| model | input | unknown recall | known-attack F1 |
|---|---|---|---|
| RF | continuous, all scales (Phase 3) | 87.8% | 0.447 |
| XGBoost | continuous, all scales (Phase 3) | 86.8% | 0.459 |
| RF | continuous, base + medium | 89.7% | 0.438 |
| XGBoost | continuous, base + medium | 90.3% | 0.459 |
| RF | Boolean, TM literals | 83.6% | 0.333 |
| XGBoost | Boolean, TM literals | 84.5% | 0.347 |
| TM single-stage (multi-class), same real rows | Boolean | 43.5% | 0.320 |
| TM single-stage (multi-class), SMOTE | Boolean | 54.7% | 0.366 |
| TM two-stage, SMOTE (Phase 3 headline) | Boolean | 62.3% | 0.337 |

- **The Phase 3 mismatch did not favour the baselines.** With the same
  windows as TM_medium, they do slightly *better* (89.7–90.3% vs
  86.8–87.8%).
- **In the open set the encoding is the smaller cost.**
  - Going from continuous to Boolean costs the trees about 6 points of
    unknown recall.
  - On the same Boolean rows, under the same architecture and rule, the
    trees reach 84% and the TM 44%. With SMOTE and the gate, the TM's best
    is 62–70%.
  - This is the reverse of the closed-set split above, where the encoding
    was the larger part.
- **Per family, binarisation hurts most where held-out rows resemble a known
  class on fine-grained values.** RF unknown recall, continuous vs Boolean:
  - Generic: 99.3% → 58.7%.
  - Shellcode: 96.4% → 71.2%.
- **For Worms and Reconnaissance, the Boolean input is better.** RF: 85.3%
  vs 70.5% and 72.7% vs 64.8%.
- **Cost.** One family's tree baselines (grid selection plus 3 seeds) took
  about 5–6 minutes on continuous input and 11–16 minutes on the 425
  Boolean features.

## Scripts added or changed this phase

| script | change |
|---|---|
| `24_single_stage.py` | new: single-stage decision, three rejection rules, two-stage recomputed from the same runs |
| `25_train_single_stage.py` | new: multi-class TM over Normal + known families, per-family open-set config |
| `26_smote_validity.py` | new: constraint and copy checks on every synthetic row |
| `27_encoding_study.py` | new: TM / RF / XGBoost × Boolean (3, 5, 9, 19 thresholds) / continuous, plus the information and cost measures |
| `28_phase4_tables.py` | new: generates the tables below |
| `01c_preprocess_temporal.py`, `09_openset_preprocess.py` | `ros` and `none` strategies, a quantile-count argument, and identical-partition assertions |
| `02c_train_with_scores.py`, `10_train_family_banks.py` | strategy suffix, train and score timings |
| `18_openset_eval.py` | ignores variant folders when collecting runs |
| `20_openset_baselines.py` | `BASELINE_REPR` (continuous, continuous_medium, boolean) and `BASELINE_SEEDS` |

## Appendix — generated tables

### Task 3 — single-stage vs two-stage (5% budget, family average, mean ± std over seeds)

| architecture | rule | unknown recall | known-attack macro F1 | Normal false alarms | known rejected | known attacks rejected | unknown → Normal | unknown → known family | AUROC |
|---|---|---|---|---|---|---|---|---|---|
| two-stage (gate + family banks) | global | 62.3% ± 2.0 | 0.3370 ± 0.0071 | 2.08% ± 0.05 | 4.49% ± 0.05 | 24.7% ± 0.3 | 11.6% ± 2.1 | 26.1% ± 2.2 | 0.848 ± 0.020 |
| two-stage (gate + family banks) | classcond | 16.8% ± 2.3 | 0.3898 ± 0.0022 | 2.08% ± 0.05 | 4.26% ± 0.06 | 24.5% ± 0.4 | 11.6% ± 2.1 | 71.5% ± 0.8 | 0.770 ± 0.021 |
| two-stage (gate + family banks) | margin | 70.1% ± 0.7 | 0.2895 ± 0.0036 | 2.08% ± 0.05 | 4.92% ± 0.02 | 25.5% ± 0.1 | 11.6% ± 2.1 | 18.2% ± 2.4 | 0.868 ± 0.019 |
| single-stage, one-vs-rest banks | global | 51.5% ± 4.4 | 0.3655 ± 0.0022 | 5.90% ± 0.84 | 4.33% ± 0.11 | 15.3% ± 0.7 | 0.0% ± 0.0 | 48.5% ± 4.4 | 0.800 ± 0.026 |
| single-stage, one-vs-rest banks | classcond | 3.1% ± 0.8 | 0.4000 ± 0.0018 | 8.40% ± 0.45 | 4.44% ± 0.26 | 3.7% ± 0.4 | 0.1% ± 0.1 | 96.8% ± 0.7 | 0.137 ± 0.025 |
| single-stage, one-vs-rest banks | margin | 64.2% ± 1.9 | 0.3218 ± 0.0036 | 4.90% ± 0.80 | 4.90% ± 0.07 | 21.5% ± 0.8 | 0.1% ± 0.1 | 35.7% ± 1.9 | 0.932 ± 0.004 |
| single-stage, multi-class TM | global | 54.7% ± 2.2 | 0.3659 ± 0.0061 | 5.14% ± 0.72 | 4.41% ± 0.11 | 15.9% ± 1.0 | 0.0% ± 0.0 | 45.2% ± 2.1 | 0.816 ± 0.018 |
| single-stage, multi-class TM | classcond | 3.2% ± 0.7 | 0.4022 ± 0.0027 | 8.12% ± 0.75 | 4.64% ± 0.12 | 3.6% ± 0.2 | 0.1% ± 0.1 | 96.7% ± 0.7 | 0.112 ± 0.015 |
| single-stage, multi-class TM | margin | 62.5% ± 1.5 | 0.3276 ± 0.0051 | 4.28% ± 0.79 | 4.82% ± 0.10 | 21.1% ± 0.6 | 0.1% ± 0.1 | 37.4% ± 1.5 | 0.929 ± 0.007 |

#### Task 3 per family, global rule, 5%

| held out | architecture | unknown recall | known-attack macro F1 | Normal false alarms | known rejected | known attacks rejected | unknown → Normal | unknown → known family | AUROC |
|---|---|---|---|---|---|---|---|---|---|
| analysis | two-stage | 76.7% ± 3.2 | 0.3637 ± 0.0101 | 1.88% ± 0.14 | 4.59% ± 0.03 | 21.4% ± 0.2 | 10.4% ± 2.4 | 12.9% ± 4.1 | 0.866 ± 0.021 |
| analysis | single (banks) | 64.5% ± 8.2 | 0.3936 ± 0.0124 | 4.36% ± 0.43 | 4.50% ± 0.12 | 14.4% ± 1.7 | 0.0% | 35.5% ± 8.2 | 0.829 ± 0.030 |
| analysis | single (multi-class) | 66.1% ± 13.5 | 0.3921 ± 0.0141 | 4.49% ± 0.70 | 4.62% ± 0.39 | 14.7% ± 3.1 | 0.0% | 33.9% ± 13.5 | 0.850 ± 0.026 |
| backdoor | two-stage | 92.0% ± 1.5 | 0.3782 ± 0.0087 | 2.22% ± 0.21 | 4.57% ± 0.05 | 20.9% ± 0.5 | 1.1% ± 0.6 | 7.0% ± 1.3 | 0.965 ± 0.006 |
| backdoor | single (banks) | 77.7% ± 8.7 | 0.4046 ± 0.0148 | 4.37% ± 0.30 | 4.19% ± 0.21 | 13.3% ± 1.7 | 0.0% | 22.3% ± 8.7 | 0.923 ± 0.011 |
| backdoor | single (multi-class) | 80.7% ± 14.2 | 0.4018 ± 0.0173 | 4.07% ± 0.40 | 4.28% ± 0.40 | 14.6% ± 2.8 | 0.0% | 19.3% ± 14.2 | 0.927 ± 0.025 |
| dos | two-stage | 87.1% ± 1.3 | 0.3567 ± 0.0231 | 2.04% ± 0.09 | 4.67% ± 0.11 | 22.0% ± 0.5 | 2.0% ± 0.5 | 10.9% ± 1.0 | 0.956 ± 0.004 |
| dos | single (banks) | 73.3% ± 10.5 | 0.4004 ± 0.0094 | 5.21% ± 0.30 | 4.47% ± 0.29 | 11.1% ± 2.1 | 0.0% | 26.7% ± 10.5 | 0.898 ± 0.016 |
| dos | single (multi-class) | 81.1% ± 6.2 | 0.3971 ± 0.0044 | 4.50% ± 0.72 | 4.61% ± 0.26 | 14.7% ± 3.2 | 0.0% | 18.9% ± 6.2 | 0.906 ± 0.008 |
| exploits | two-stage | 81.5% ± 6.9 | 0.3290 ± 0.0080 | 2.16% ± 0.13 | 4.44% ± 0.09 | 22.4% ± 0.6 | 2.3% ± 2.1 | 16.3% ± 5.4 | 0.946 ± 0.025 |
| exploits | single (banks) | 68.8% ± 17.8 | 0.3664 ± 0.0229 | 5.08% ± 0.26 | 4.35% ± 0.53 | 10.9% ± 2.7 | 0.0% | 31.2% ± 17.8 | 0.872 ± 0.049 |
| exploits | single (multi-class) | 77.5% ± 6.7 | 0.3452 ± 0.0089 | 4.93% ± 0.41 | 4.73% ± 0.14 | 13.0% ± 1.1 | 0.0% | 22.5% ± 6.7 | 0.880 ± 0.031 |
| fuzzers | two-stage | 47.3% ± 5.3 | 0.3116 ± 0.0325 | 1.41% ± 0.15 | 4.23% ± 0.16 | 19.1% ± 0.9 | 31.6% ± 6.0 | 21.1% ± 3.5 | 0.696 ± 0.061 |
| fuzzers | single (banks) | 55.4% ± 11.2 | 0.3186 ± 0.0360 | 18.32% ± 7.10 | 4.78% ± 0.32 | 5.0% ± 2.0 | 0.1% ± 0.1 | 44.6% ± 11.3 | 0.800 ± 0.047 |
| fuzzers | single (multi-class) | 39.5% ± 18.5 | 0.3459 ± 0.0185 | 12.75% ± 7.53 | 4.49% ± 0.44 | 3.1% ± 2.5 | 0.3% ± 0.4 | 60.2% ± 18.3 | 0.774 ± 0.041 |
| generic | two-stage | 26.1% ± 17.1 | 0.2503 ± 0.0082 | 2.24% ± 0.03 | 4.48% ± 0.09 | 54.9% ± 1.4 | 51.1% ± 23.6 | 22.8% ± 17.4 | 0.467 ± 0.223 |
| generic | single (banks) | 14.4% ± 11.6 | 0.2933 ± 0.0072 | 3.77% ± 0.15 | 4.10% ± 0.14 | 40.2% ± 1.5 | 0.0% | 85.6% ± 11.6 | 0.766 ± 0.124 |
| generic | single (multi-class) | 24.7% ± 12.4 | 0.3002 ± 0.0162 | 3.94% ± 0.29 | 4.15% ± 0.26 | 38.2% ± 6.0 | 0.0% ± 0.0 | 75.3% ± 12.4 | 0.849 ± 0.132 |
| reconnaissance | two-stage | 40.9% ± 3.6 | 0.3060 ± 0.0030 | 2.18% ± 0.08 | 4.62% ± 0.22 | 22.1% ± 1.2 | 0.8% ± 0.6 | 58.3% ± 3.8 | 0.907 ± 0.006 |
| reconnaissance | single (banks) | 23.7% ± 6.4 | 0.3280 ± 0.0114 | 4.67% ± 0.42 | 4.30% ± 0.30 | 13.1% ± 2.4 | 0.0% | 76.3% ± 6.4 | 0.589 ± 0.040 |
| reconnaissance | single (multi-class) | 30.0% ± 8.9 | 0.3243 ± 0.0076 | 4.65% ± 0.23 | 4.52% ± 0.08 | 14.0% ± 1.1 | 0.0% | 70.0% ± 8.9 | 0.640 ± 0.037 |
| shellcode | two-stage | 72.4% ± 3.6 | 0.3590 ± 0.0142 | 2.60% ± 0.10 | 4.34% ± 0.21 | 18.8% ± 0.7 | 0.5% ± 0.3 | 27.2% ± 3.6 | 0.965 ± 0.005 |
| shellcode | single (banks) | 69.9% ± 2.8 | 0.3746 ± 0.0090 | 2.94% ± 0.03 | 4.27% ± 0.18 | 17.5% ± 0.5 | 0.0% | 30.1% ± 2.8 | 0.955 ± 0.011 |
| shellcode | single (multi-class) | 68.2% ± 6.5 | 0.3844 ± 0.0144 | 2.93% ± 0.03 | 4.13% ± 0.26 | 16.6% ± 1.2 | 0.0% | 31.8% ± 6.5 | 0.941 ± 0.012 |
| worms | two-stage | 36.5% ± 9.4 | 0.3789 ± 0.0034 | 2.01% ± 0.13 | 4.44% ± 0.06 | 20.5% ± 0.7 | 5.0% ± 0.9 | 58.5% ± 10.0 | 0.863 ± 0.009 |
| worms | single (banks) | 16.2% ± 2.3 | 0.4096 ± 0.0104 | 4.41% ± 0.22 | 4.01% ± 0.27 | 12.1% ± 1.6 | 0.0% | 83.8% ± 2.3 | 0.570 ± 0.039 |
| worms | single (multi-class) | 25.0% ± 7.5 | 0.4021 ± 0.0078 | 4.02% ± 0.23 | 4.16% ± 0.15 | 13.8% ± 1.3 | 0.0% | 75.0% ± 7.5 | 0.581 ± 0.081 |

#### Task 3 per family, margin rule, 5%

| held out | architecture | unknown recall | known-attack macro F1 | Normal false alarms | known rejected | known attacks rejected | unknown → Normal | unknown → known family | AUROC |
|---|---|---|---|---|---|---|---|---|---|
| analysis | two-stage | 88.0% ± 1.4 | 0.3233 ± 0.0082 | 1.88% ± 0.14 | 5.00% ± 0.07 | 22.4% ± 0.4 | 10.4% ± 2.4 | 1.6% ± 2.8 | 0.880 ± 0.022 |
| analysis | single (banks) | 88.7% ± 5.1 | 0.3454 ± 0.0048 | 3.33% ± 0.06 | 5.23% ± 0.06 | 20.3% ± 0.3 | 0.0% | 11.3% ± 5.1 | 0.976 ± 0.002 |
| analysis | single (multi-class) | 88.8% ± 2.9 | 0.3453 ± 0.0043 | 3.34% ± 0.19 | 5.20% ± 0.07 | 20.3% ± 0.6 | 0.0% | 11.2% ± 2.9 | 0.976 ± 0.001 |
| backdoor | two-stage | 92.8% ± 0.6 | 0.3522 ± 0.0058 | 2.22% ± 0.21 | 4.78% ± 0.06 | 20.9% ± 0.4 | 1.1% ± 0.6 | 6.1% ± 0.5 | 0.971 ± 0.005 |
| backdoor | single (banks) | 91.5% ± 0.6 | 0.3678 ± 0.0070 | 3.05% ± 0.07 | 4.70% ± 0.09 | 19.0% ± 0.3 | 0.0% | 8.5% ± 0.6 | 0.979 ± 0.000 |
| backdoor | single (multi-class) | 91.3% ± 0.7 | 0.3705 ± 0.0070 | 3.13% ± 0.08 | 4.69% ± 0.06 | 18.7% ± 0.4 | 0.0% | 8.7% ± 0.7 | 0.979 ± 0.001 |
| dos | two-stage | 93.3% ± 1.5 | 0.2753 ± 0.0108 | 2.04% ± 0.09 | 5.25% ± 0.06 | 21.7% ± 0.3 | 2.0% ± 0.5 | 4.7% ± 1.1 | 0.962 ± 0.005 |
| dos | single (banks) | 90.8% ± 0.5 | 0.3390 ± 0.0034 | 3.69% ± 0.07 | 5.23% ± 0.04 | 17.9% ± 0.2 | 0.0% ± 0.0 | 9.2% ± 0.5 | 0.972 ± 0.002 |
| dos | single (multi-class) | 90.6% ± 0.2 | 0.3457 ± 0.0094 | 3.74% ± 0.06 | 5.23% ± 0.06 | 17.7% ± 0.1 | 0.0% ± 0.0 | 9.4% ± 0.2 | 0.968 ± 0.001 |
| exploits | two-stage | 96.9% ± 1.9 | 0.1940 ± 0.0132 | 2.16% ± 0.13 | 5.23% ± 0.06 | 21.9% ± 0.5 | 2.3% ± 2.1 | 0.8% ± 0.3 | 0.959 ± 0.019 |
| exploits | single (banks) | 94.7% ± 1.5 | 0.2751 ± 0.0070 | 3.81% ± 0.05 | 5.16% ± 0.05 | 17.7% ± 0.2 | 0.0% | 5.3% ± 1.5 | 0.974 ± 0.001 |
| exploits | single (multi-class) | 93.2% ± 1.8 | 0.2811 ± 0.0086 | 3.83% ± 0.08 | 5.18% ± 0.04 | 17.6% ± 0.2 | 0.0% | 6.8% ± 1.8 | 0.974 ± 0.001 |
| fuzzers | two-stage | 51.7% ± 7.3 | 0.2323 ± 0.0215 | 1.41% ± 0.15 | 5.01% ± 0.19 | 22.4% ± 0.8 | 31.6% ± 6.0 | 16.6% ± 4.3 | 0.753 ± 0.062 |
| fuzzers | single (banks) | 31.9% ± 22.3 | 0.2891 ± 0.0214 | 17.59% ± 7.10 | 4.93% ± 0.35 | 9.7% ± 4.7 | 1.2% ± 0.9 | 66.9% ± 22.0 | 0.840 ± 0.047 |
| fuzzers | single (multi-class) | 17.3% ± 12.4 | 0.3120 ± 0.0205 | 11.91% ± 7.26 | 4.45% ± 0.47 | 7.8% ± 2.1 | 1.0% ± 1.1 | 81.6% ± 11.8 | 0.823 ± 0.019 |
| generic | two-stage | 1.6% ± 0.0 | 0.2529 ± 0.0064 | 2.24% ± 0.03 | 4.68% ± 0.13 | 57.1% ± 1.1 | 51.1% ± 23.6 | 47.4% ± 23.6 | 0.448 ± 0.215 |
| generic | single (banks) | 1.4% ± 0.0 | 0.2645 ± 0.0120 | 3.20% ± 0.14 | 4.61% ± 0.25 | 51.4% ± 2.4 | 0.0% ± 0.0 | 98.6% ± 0.0 | 0.846 ± 0.055 |
| generic | single (multi-class) | 1.3% ± 0.1 | 0.2699 ± 0.0135 | 3.27% ± 0.11 | 4.64% ± 0.27 | 50.2% ± 3.0 | 0.0% ± 0.0 | 98.7% ± 0.1 | 0.850 ± 0.050 |
| reconnaissance | two-stage | 65.1% ± 5.7 | 0.2914 ± 0.0052 | 2.18% ± 0.08 | 5.04% ± 0.04 | 21.6% ± 0.1 | 0.8% ± 0.6 | 34.1% ± 6.1 | 0.947 ± 0.004 |
| reconnaissance | single (banks) | 48.2% ± 7.8 | 0.3066 ± 0.0065 | 3.48% ± 0.21 | 4.95% ± 0.07 | 18.5% ± 0.5 | 0.0% | 51.8% ± 7.8 | 0.919 ± 0.009 |
| reconnaissance | single (multi-class) | 47.9% ± 5.9 | 0.3039 ± 0.0025 | 3.33% ± 0.18 | 4.91% ± 0.09 | 18.9% ± 0.4 | 0.0% | 52.1% ± 5.9 | 0.910 ± 0.024 |
| shellcode | two-stage | 59.9% ± 2.1 | 0.3458 ± 0.0054 | 2.60% ± 0.10 | 4.52% ± 0.04 | 20.0% ± 0.3 | 0.5% ± 0.3 | 39.7% ± 2.1 | 0.956 ± 0.003 |
| shellcode | single (banks) | 54.2% ± 2.7 | 0.3574 ± 0.0035 | 2.94% ± 0.02 | 4.54% ± 0.02 | 19.2% ± 0.2 | 0.0% | 45.8% ± 2.7 | 0.909 ± 0.011 |
| shellcode | single (multi-class) | 51.0% ± 6.2 | 0.3650 ± 0.0142 | 2.92% ± 0.02 | 4.40% ± 0.27 | 18.6% ± 1.3 | 0.0% | 49.0% ± 6.2 | 0.918 ± 0.016 |
| worms | two-stage | 81.9% ± 2.0 | 0.3383 ± 0.0070 | 2.01% ± 0.13 | 4.75% ± 0.04 | 21.3% ± 0.2 | 5.0% ± 0.9 | 13.1% ± 1.4 | 0.933 ± 0.009 |
| worms | single (banks) | 76.2% ± 1.5 | 0.3519 ± 0.0057 | 3.03% ± 0.06 | 4.70% ± 0.06 | 19.6% ± 0.2 | 0.0% | 23.8% ± 1.5 | 0.972 ± 0.002 |
| worms | single (multi-class) | 81.2% ± 2.2 | 0.3549 ± 0.0031 | 3.02% ± 0.05 | 4.66% ± 0.03 | 19.6% ± 0.1 | 0.0% | 18.8% ± 2.2 | 0.969 ± 0.002 |

### Task 1 — resampling

#### SMOTE output validity

| dataset | real rows | synthetic rows | real rows violating | synthetic violating (any) | one-hot proto | quantile ladder | relational | synthetic rows identical to a real row of their class |
|---|---|---|---|---|---|---|---|---|
| closed_medium | 142521 | 257479 | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 100.0–100.0% |
| open_analysis | 140856 | 219144 | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 100.0–100.0% |
| open_backdoor | 141199 | 218801 | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 100.0–100.0% |
| open_dos | 133838 | 226162 | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 100.0–100.0% |
| open_exploits | 116048 | 243952 | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 100.0–100.0% |
| open_fuzzers | 127156 | 232844 | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 100.0–100.0% |
| open_generic | 101210 | 258790 | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 100.0–100.0% |
| open_reconnaissance | 134379 | 225621 | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 100.0–100.0% |
| open_shellcode | 141605 | 218395 | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 100.0–100.0% |
| open_worms | 142418 | 217582 | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 100.0–100.0% |

Closed set, per class:

| class | real | synthetic | violating | copy of a real row | new pattern | distinct synthetic patterns |
|---|---|---|---|---|---|---|
| Analysis | 1659 | 38341 | 0.0% | 100.0% | 0.0% | 1428 |
| Backdoor | 1316 | 38684 | 0.0% | 100.0% | 0.0% | 1159 |
| DoS | 8627 | 31373 | 0.0% | 100.0% | 0.0% | 4903 |
| Exploits | 26438 | 13562 | 0.0% | 100.0% | 0.0% | 8564 |
| Fuzzers | 15349 | 24651 | 0.0% | 100.0% | 0.0% | 11073 |
| Reconnaissance | 8115 | 31885 | 0.0% | 100.0% | 0.0% | 6891 |
| Shellcode | 914 | 39086 | 0.0% | 100.0% | 0.0% | 894 |
| Worms | 103 | 39897 | 0.0% | 100.0% | 0.0% | 103 |

#### Closed-set TM by resampling strategy (test, mean ± std over seeds)

| strategy | seeds | macro F1 (argmax) | macro F1 (prior-calibrated) | accuracy (calibrated) |
|---|---|---|---|---|
| smote | 5 | 0.4415 ± 0.0113 | 0.4628 ± 0.0076 | 0.9546 ± 0.0004 |
| ros | 5 | 0.4413 ± 0.0086 | 0.4635 ± 0.0060 | 0.9534 ± 0.0017 |
| none | 5 | 0.3910 ± 0.0017 | 0.3905 ± 0.0012 | 0.9551 ± 0.0007 |

Per-class precision / recall / F1 (prior-calibrated):

| class | smote P / R / F1 | ros P / R / F1 | none P / R / F1 |
|---|---|---|---|
| Analysis | 0.081 / 0.109 / 0.093 | 0.077 / 0.147 / 0.098 | 0.122 / 0.051 / 0.069 |
| Backdoor | 0.064 / 0.020 / 0.030 | 0.053 / 0.052 / 0.045 | 0.000 / 0.000 / 0.000 |
| DoS | 0.337 / 0.337 / 0.320 | 0.340 / 0.395 / 0.358 | 0.324 / 0.003 / 0.005 |
| Exploits | 0.581 / 0.660 / 0.616 | 0.594 / 0.614 / 0.602 | 0.496 / 0.938 / 0.649 |
| Fuzzers | 0.402 / 0.787 / 0.532 | 0.402 / 0.789 / 0.532 | 0.373 / 0.837 / 0.515 |
| Generic | 0.998 / 0.975 / 0.986 | 0.999 / 0.975 / 0.987 | 0.999 / 0.975 / 0.987 |
| Normal | 0.997 / 0.976 / 0.987 | 0.998 / 0.975 / 0.987 | 1.000 / 0.974 / 0.987 |
| Reconnaissance | 0.651 / 0.765 / 0.703 | 0.636 / 0.770 / 0.696 | 0.792 / 0.618 / 0.693 |
| Shellcode | 0.165 / 0.638 / 0.263 | 0.157 / 0.687 / 0.255 | 0.000 / 0.000 / 0.000 |
| Worms | 0.071 / 0.185 / 0.099 | 0.049 / 0.254 / 0.076 | 0.000 / 0.000 / 0.000 |

#### Open-set detection by resampling strategy (5% budget, family average)

| strategy | architecture | rule | unknown recall | known-attack macro F1 | Normal false alarms | known rejected | known attacks rejected | unknown → Normal | unknown → known family | AUROC |
|---|---|---|---|---|---|---|---|---|---|---|
| smote | two-stage | global | 62.3% ± 2.0 | 0.3370 ± 0.0071 | 2.08% ± 0.05 | 4.49% ± 0.05 | 24.7% ± 0.3 | 11.6% ± 2.1 | 26.1% ± 2.2 | 0.848 ± 0.020 |
| smote | two-stage | classcond | 16.8% ± 2.3 | 0.3898 ± 0.0022 | 2.08% ± 0.05 | 4.26% ± 0.06 | 24.5% ± 0.4 | 11.6% ± 2.1 | 71.5% ± 0.8 | 0.770 ± 0.021 |
| smote | two-stage | margin | 70.1% ± 0.7 | 0.2895 ± 0.0036 | 2.08% ± 0.05 | 4.92% ± 0.02 | 25.5% ± 0.1 | 11.6% ± 2.1 | 18.2% ± 2.4 | 0.868 ± 0.019 |
| smote | single (banks) | global | 51.5% ± 4.4 | 0.3655 ± 0.0022 | 5.90% ± 0.84 | 4.33% ± 0.11 | 15.3% ± 0.7 | 0.0% ± 0.0 | 48.5% ± 4.4 | 0.800 ± 0.026 |
| smote | single (banks) | classcond | 3.1% ± 0.8 | 0.4000 ± 0.0018 | 8.40% ± 0.45 | 4.44% ± 0.26 | 3.7% ± 0.4 | 0.1% ± 0.1 | 96.8% ± 0.7 | 0.137 ± 0.025 |
| smote | single (banks) | margin | 64.2% ± 1.9 | 0.3218 ± 0.0036 | 4.90% ± 0.80 | 4.90% ± 0.07 | 21.5% ± 0.8 | 0.1% ± 0.1 | 35.7% ± 1.9 | 0.932 ± 0.004 |
| smote | single (multi-class) | global | 54.7% ± 2.2 | 0.3659 ± 0.0061 | 5.14% ± 0.72 | 4.41% ± 0.11 | 15.9% ± 1.0 | 0.0% ± 0.0 | 45.2% ± 2.1 | 0.816 ± 0.018 |
| smote | single (multi-class) | classcond | 3.2% ± 0.7 | 0.4022 ± 0.0027 | 8.12% ± 0.75 | 4.64% ± 0.12 | 3.6% ± 0.2 | 0.1% ± 0.1 | 96.7% ± 0.7 | 0.112 ± 0.015 |
| smote | single (multi-class) | margin | 62.5% ± 1.5 | 0.3276 ± 0.0051 | 4.28% ± 0.79 | 4.82% ± 0.10 | 21.1% ± 0.6 | 0.1% ± 0.1 | 37.4% ± 1.5 | 0.929 ± 0.007 |
| ros | two-stage | global | 61.9% ± 2.1 | 0.3426 ± 0.0063 | 2.10% ± 0.05 | 4.51% ± 0.12 | 24.8% ± 0.5 | 10.9% ± 2.8 | 27.2% ± 2.6 | 0.852 ± 0.026 |
| ros | two-stage | classcond | 17.3% ± 2.4 | 0.3894 ± 0.0015 | 2.10% ± 0.05 | 4.26% ± 0.09 | 24.4% ± 0.5 | 10.9% ± 2.8 | 71.8% ± 1.3 | 0.776 ± 0.027 |
| ros | two-stage | margin | 70.3% ± 1.2 | 0.2886 ± 0.0048 | 2.10% ± 0.05 | 4.93% ± 0.02 | 25.4% ± 0.1 | 10.9% ± 2.8 | 18.8% ± 3.7 | 0.872 ± 0.024 |
| ros | single (banks) | global | 51.1% ± 3.3 | 0.3678 ± 0.0036 | 5.81% ± 0.81 | 4.35% ± 0.07 | 15.2% ± 0.8 | 0.0% ± 0.0 | 48.9% ± 3.3 | 0.795 ± 0.016 |
| ros | single (banks) | classcond | 2.8% ± 0.6 | 0.3995 ± 0.0016 | 8.25% ± 0.59 | 4.39% ± 0.13 | 3.7% ± 0.2 | 0.1% ± 0.2 | 97.1% ± 0.6 | 0.139 ± 0.017 |
| ros | single (banks) | margin | 62.7% ± 2.1 | 0.3250 ± 0.0032 | 4.76% ± 0.78 | 4.82% ± 0.05 | 21.1% ± 0.5 | 0.1% ± 0.1 | 37.2% ± 2.0 | 0.931 ± 0.009 |
| ros | single (multi-class) | global | 52.3% ± 3.4 | 0.3710 ± 0.0062 | 4.50% ± 0.14 | 4.34% ± 0.06 | 15.5% ± 0.7 | 0.0% ± 0.0 | 47.7% ± 3.4 | 0.809 ± 0.013 |
| ros | single (multi-class) | classcond | 3.6% ± 0.7 | 0.4032 ± 0.0022 | 7.41% ± 0.08 | 4.56% ± 0.08 | 4.0% ± 0.1 | 0.0% ± 0.0 | 96.4% ± 0.7 | 0.113 ± 0.020 |
| ros | single (multi-class) | margin | 63.8% ± 2.8 | 0.3271 ± 0.0065 | 3.48% ± 0.12 | 4.87% ± 0.08 | 21.7% ± 0.9 | 0.0% ± 0.1 | 36.1% ± 2.8 | 0.935 ± 0.009 |
| none | two-stage | global | 49.7% ± 2.2 | 0.3492 ± 0.0026 | 2.05% ± 0.04 | 4.18% ± 0.10 | 23.2% ± 0.4 | 11.9% ± 2.9 | 38.4% ± 2.1 | 0.827 ± 0.028 |
| none | two-stage | classcond | 13.3% ± 2.2 | 0.3741 ± 0.0033 | 2.05% ± 0.04 | 4.09% ± 0.04 | 23.7% ± 0.2 | 11.9% ± 2.9 | 74.9% ± 1.9 | 0.755 ± 0.028 |
| none | two-stage | margin | 74.5% ± 1.0 | 0.2408 ± 0.0027 | 2.05% ± 0.04 | 4.88% ± 0.02 | 25.6% ± 0.2 | 11.9% ± 2.9 | 13.6% ± 3.3 | 0.861 ± 0.028 |
| none | single (banks) | global | 33.7% ± 4.0 | 0.3822 ± 0.0030 | 5.09% ± 0.15 | 3.93% ± 0.14 | 10.4% ± 0.9 | 0.0% ± 0.0 | 66.3% ± 4.0 | 0.699 ± 0.023 |
| none | single (banks) | classcond | 2.1% ± 0.9 | 0.3909 ± 0.0028 | 7.09% ± 0.22 | 4.32% ± 0.20 | 4.1% ± 0.2 | 0.0% ± 0.0 | 97.9% ± 0.9 | 0.096 ± 0.014 |
| none | single (banks) | margin | 73.8% ± 1.9 | 0.2667 ± 0.0033 | 3.13% ± 0.04 | 4.85% ± 0.06 | 23.5% ± 0.4 | 0.0% ± 0.0 | 26.2% ± 1.9 | 0.950 ± 0.016 |
| none | single (multi-class) | global | 43.5% ± 5.0 | 0.3203 ± 0.0023 | 4.67% ± 0.12 | 4.14% ± 0.14 | 12.3% ± 0.8 | 0.7% ± 0.2 | 55.7% ± 4.9 | 0.709 ± 0.028 |
| none | single (multi-class) | classcond | 4.8% ± 1.2 | 0.3435 ± 0.0021 | 7.36% ± 0.23 | 4.68% ± 0.18 | 4.2% ± 0.2 | 0.7% ± 0.2 | 94.5% ± 1.1 | 0.120 ± 0.008 |
| none | single (multi-class) | margin | 70.1% ± 1.7 | 0.2816 ± 0.0038 | 3.09% ± 0.02 | 4.88% ± 0.03 | 22.4% ± 0.1 | 0.3% ± 0.3 | 29.6% ± 1.4 | 0.960 ± 0.002 |

### Task 2 — feature representation

#### Closed set: model × representation, identical real rows (test, mean ± std)

| model | input | seeds | macro F1 | accuracy |
|---|---|---|---|---|
| Tsetlin Machine | Boolean (5 thresholds) | 5 | 0.3905 ± 0.0012 | 0.9551 ± 0.0007 |
| Random Forest | Boolean (5 thresholds) | 3 | 0.4954 ± 0.0025 | 0.9559 ± 0.0002 |
| XGBoost | Boolean (5 thresholds) | 3 | 0.4975 ± 0.0008 | 0.9551 ± 0.0002 |
| Random Forest | continuous | 3 | 0.5947 ± 0.0021 | 0.9595 ± 0.0001 |
| XGBoost | continuous | 3 | 0.6094 ± 0.0007 | 0.9612 ± 0.0000 |

Per-class F1:

| class | TM bool | RF bool | XGB bool | RF cont | XGB cont |
|---|---|---|---|---|---|
| Analysis | 0.069 | 0.101 | 0.101 | 0.105 | 0.095 |
| Backdoor | 0.000 | 0.096 | 0.103 | 0.158 | 0.163 |
| DoS | 0.005 | 0.473 | 0.419 | 0.491 | 0.485 |
| Exploits | 0.649 | 0.602 | 0.611 | 0.657 | 0.675 |
| Fuzzers | 0.515 | 0.549 | 0.557 | 0.568 | 0.570 |
| Generic | 0.987 | 0.987 | 0.988 | 0.988 | 0.990 |
| Normal | 0.987 | 0.988 | 0.988 | 0.988 | 0.988 |
| Reconnaissance | 0.693 | 0.729 | 0.719 | 0.833 | 0.834 |
| Shellcode | 0.000 | 0.287 | 0.310 | 0.513 | 0.629 |
| Worms | 0.000 | 0.142 | 0.178 | 0.647 | 0.663 |

#### Quantile encodings: performance, dimensionality and cost

| thresholds | literals | distinct literal columns | TM macro F1 | RF macro F1 | XGB macro F1 | TM train s | TM score s (val+test) | RF / XGB fit s | data MB | test rows sharing a pattern across classes |
|---|---|---|---|---|---|---|---|---|---|---|
| 3 | 323 | 283 | 0.3841 ± 0.0010 | 0.4751 ± 0.0005 | 0.4713 ± 0.0007 | 16 | 33 | 22 / 95 | 276 | 3.4% |
| 5 | 425 | 347 | 0.3905 ± 0.0012 | 0.4954 ± 0.0025 | 0.4975 ± 0.0008 | 18 | 39 | 24 / 176 | 363 | 2.2% |
| 9 | 629 | 468 | 0.3960 ± 0.0068 | 0.5344 ± 0.0024 | 0.5240 ± 0.0012 | 23 | 51 | 41 / 193 | 537 | 1.9% |
| 19 | 1139 | 728 | 0.4066 ± 0.0103 | 0.5504 ± 0.0004 | 0.5551 ± 0.0027 | 37 | 85 | 93 / 416 | 972 | 1.7% |
| continuous | 54 | – | – | 0.5947 ± 0.0021 | 0.6094 ± 0.0007 | – | – | 19 / 68 | – | – |

#### Can the Boolean encoding separate each family from Normal?

Share of each family's test rows whose literal vector is identical to some Normal test row's:

| family | 3 thresholds | 5 thresholds | 9 thresholds | 19 thresholds |
|---|---|---|---|---|
| Analysis | 7.5% | 0.4% | 0.0% | 0.0% |
| Backdoor | 1.3% | 0.0% | 0.0% | 0.0% |
| DoS | 1.1% | 0.0% | 0.0% | 0.0% |
| Exploits | 1.6% | 0.0% | 0.0% | 0.0% |
| Fuzzers | 14.7% | 1.3% | 0.8% | 0.2% |
| Generic | 0.1% | 0.0% | 0.0% | 0.0% |
| Reconnaissance | 13.4% | 0.1% | 0.0% | 0.0% |
| Shellcode | 17.0% | 0.7% | 0.0% | 0.0% |
| Worms | 7.7% | 0.0% | 0.0% | 0.0% |

#### Open set: RF / XGBoost by representation (5% budget, family average)

| model | input | families | unknown recall | known-attack macro F1 | Normal false alarms | unknown → Normal | AUROC |
|---|---|---|---|---|---|---|---|
| RF | continuous (all scales) | 9 | 87.8% | 0.4472 | 2.83% | 0.1% | 0.972 |
| XGB | continuous (all scales) | 9 | 86.8% | 0.4593 | 2.47% | 2.1% | 0.973 |
| RF | continuous (medium only) | 9 | 89.7% | 0.4383 | 2.85% | 0.2% | 0.973 |
| XGB | continuous (medium only) | 9 | 90.3% | 0.4591 | 2.49% | 2.2% | 0.974 |
| RF | Boolean (TM literals) | 9 | 83.6% | 0.3325 | 2.54% | 1.2% | 0.963 |
| XGB | Boolean (TM literals) | 9 | 84.5% | 0.3472 | 2.42% | 2.3% | 0.971 |

Unknown recall per family:

| held out | RF Boolean (TM literals) | RF continuous (all scales) | RF continuous (medium only) | XGB Boolean (TM literals) | XGB continuous (all scales) | XGB continuous (medium only) |
|---|---|---|---|---|---|---|
| analysis | 97.5% | 98.9% | 99.1% | 99.6% | 99.8% | 99.9% |
| backdoor | 97.5% | 99.1% | 99.4% | 95.9% | 97.9% | 98.8% |
| dos | 92.2% | 91.0% | 93.2% | 94.4% | 92.7% | 94.7% |
| exploits | 94.4% | 89.4% | 94.6% | 94.2% | 97.8% | 97.0% |
| fuzzers | 82.8% | 87.7% | 89.8% | 70.9% | 78.2% | 80.3% |
| generic | 58.7% | 98.7% | 99.3% | 80.1% | 87.0% | 96.2% |
| reconnaissance | 72.7% | 64.0% | 64.8% | 64.2% | 55.6% | 58.3% |
| shellcode | 71.2% | 95.0% | 96.4% | 72.7% | 96.5% | 97.5% |
| worms | 85.3% | 66.5% | 70.5% | 88.5% | 75.8% | 89.7% |
