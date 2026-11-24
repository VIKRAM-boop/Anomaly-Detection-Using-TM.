# Phase 5 — Audit, model selection, cost, interpretability and a second dataset

Produced 2026-11-23. Every number below comes from the run outputs. The
generated tables are in the appendix (`scripts/33_phase5_tables.py`).

Protocol for every result in this document, unless a section says otherwise:

- **Data.** UNSW-NB15 with the corrected temporal windows (Task 1).
- **Splits.** Chronological split; validation is the last 10% of the training
  period. Leave-one-family-out (LOFO) over all nine attack families: the
  held-out family is removed from fit and validation, and its test rows
  become Unknown.
- **Budgets.** 1%, 5% and 10% of known validation rows rejected as Unknown.
- **Seeds.** 7, 42 and 1234 for every model, TM and trees alike.

## Task 1 — Audit of the temporal features and the evaluation protocol

### 1.1 Window construction (`00_build_temporal_windows.py`)

`29_audit_temporal.py` recomputes every window independently on 200,000
randomly sampled flows and compares the result with the stored features.

| check | finding |
|---|---|
| Row alignment | **Correct.** The stored arrival rate equals the recomputed window count for 100% of sampled flows at all three scales. |
| Current flow | **Included.** Each flow is a member of its own window (100%), so its own duration, bytes and packets enter its window statistics. A flow's own per-flow features are already model inputs and are known when it is classified, so this is consistent with prediction at flow end. It is stated here as an assumption. |
| Prediction time | Per-flow features (`dur`, `sbytes`, …) exist only once a flow has ended, so the natural prediction time for flow *i* is its end, `ltime_i`. |
| **Flows still in progress** | **A leak.** The window averaged the *final* duration, bytes, packets, RTT and state of every earlier flow, including flows that had not ended when flow *i* ended. Share of flows with at least one such member: **29% (1 s), 40% (10 s), 44% (60 s)**. On average 3.6% of a 10 s window's members are unfinished (17% at 1 s). Affected features: `byte_rate`, `pkt_rate`, `mean_dur`, `std_dur`, `mean_tcprtt`, `failed_ratio`. The count features (arrival rate, unique destinations and ports, entropy, protocol count, burstiness) need only flow starts, so they are unaffected. |
| Same-second ordering | `stime` has whole-second resolution and 00 sorts with a non-stable sort, so the order of flows that start in the same second is arbitrary. 95% of flows share their (source IP, second) with another flow, and 73% have a same-second earlier member. Such members have started by the current flow's end, so this is not a leak. It does make the exact window contents arbitrary within a second. |
| Train/test boundary | **Correct.** Windows only look backward. No sampled fit or validation flow's window reaches a later partition. 0.05% (10 s) and 0.6% (60 s) of validation and test flows reach back into the previous partition; that is past data and legitimate. |
| **LOFO and windows** | **A leak.** Windows were built once over all flows, before 09 removes the held-out family. The held-out family's training-period flows therefore still shaped the windows of the known rows that remained. Share of known *attack* fit/validation rows whose 10 s window counts a held-out-family flow: Exploits 74%, Reconnaissance 53%, Generic 49%, Fuzzers 26%, DoS 26%, Analysis 13%, Backdoor 12%, Shellcode 8%, Worms 1%. UNSW-NB15's attacks come from a handful of source IPs, so families interleave in the same windows. |

**Correction (`00b_build_windows_v2.py`).**

- Window membership is unchanged.
- The six value features now use only members that ended by the current
  flow's end (`ltime_j ≤ ltime_i`).
- `--exclude F` rebuilds the windows with family F's training-period flows
  removed. 09 and 20 now read that family's rebuilt windows. F's
  test-period flows stay: in deployment, the new attack's traffic really is
  there.
- With the filter switched off (`--legacy`), the new implementation
  reproduces 00 exactly. The only differences are in 0.6% of rows, at the
  5th significant digit (pandas' rolling standard deviation drifts
  numerically).

### 1.2 Fitting restricted to the right partitions

Verified in code, and by assertions where noted.

| step | where fitted | status |
|---|---|---|
| Quantile thresholds, one-hot encoders, relational thresholds | fit rows only (01c, 09; 40 for CICIDS) | ✓ |
| SMOTE / random oversampling / capping | fit rows only; validation and test never resampled (asserted byte-identical across strategies in Phase 4) | ✓ |
| TM hyperparameters | `17_tune_tm.py` on validation, separately per held-out family with that family absent. The closed-set configuration saw every family and is used only in the closed set. | ✓ |
| Tree hyperparameters | validation known-class macro F1, family absent (20, 27, 41) | ✓ |
| Rejection thresholds (τ_B, τ_A, global, class-conditional, margin), prior calibration | known validation rows only (18, 22, 24) | ✓ |
| Configuration selection (Task 2) | validation data only, nested over folds (Task 2) | ✓ (new) |
| Test data | scored once per trained model and never used to choose anything | ✓ |
| Window features | backward-only, but see the two leaks above | **fixed** |

### 1.3 LOFO across all nine families, and the small families

- **Mechanics.**
  - The held-out family is removed before the fit/validation split, so the
    split time is recomputed without it. This is asserted equal between TM
    and baselines.
  - Class indices are fitted on the full label set. In 09 and 20 this is now
    train ∪ test, because the rebuilt training windows no longer contain the
    held-out family.
  - Raw "Backdoors" labels are merged into "Backdoor" in every script.
  - The validation rows hold the same class counts in v1 and v2. Only the
    order within a second differs (the non-stable sort).
- **Small families.**

  | family | fit rows | validation | test |
  |---|---|---|---|
  | Worms | 103 | 19 | 52 |
  | Shellcode | 914 | 157 | 440 |
  | Analysis | 1,659 | 340 | 678 |
  | Backdoor | 1,316 | 311 | 702 |

  - As the held-out family, Worms' unknown recall rests on 52 rows: a 95%
    interval is about ±13 points. Its pseudo-unknown validation set
    (Task 2) has 19 rows.
  - As a known class, Worms' 19 validation rows fall below the 20-row
    minimum for its own class-conditional threshold, so it uses the
    all-class fallback.
  - Every per-family table in this document gives the number of test
    unknowns and a Wilson interval. Families with fewer than 1,000 test
    rows are marked.
- **Dataset artifact (not a protocol leak).** The source TTL alone separates
  attack from Normal test flows with AUROC 0.985 (`ct_state_ttl` 0.988):
  `sttl` = 254 or 62 almost only occurs on attack traffic. This is a known
  property of how UNSW-NB15's attack traffic was generated. It makes the
  Normal/attack part of every model's task easier than real traffic would.
  It does not separate attack families from each other.

### 1.4 Do the corrections change the earlier results?

All TM runs were repeated on the corrected data (v2), with seeds 7, 42 and
1234 and the same tuned hyperparameters. They are compared below with the
same seeds on the old data (v1). Full table in the appendix.

**Closed set.** TM with SMOTE, prior-calibrated macro F1: 0.4602 (v1) vs
0.4643 (v2). No material change.

**Open set, 5% budget, family average.**

- 15 of the 18 TM configurations change by less than ±2.5 points of unknown
  recall. The largest changes are +5.1 (single-stage multi-class, margin),
  −3.6 (single-stage multi-class, global) and +3.3 (two-stage, global).
- The Phase 3 headline configuration (two-stage, SMOTE, global) goes from
  62.4% to **65.7%**. Per family:
  - Six families change by less than 2 points.
  - Fuzzers +5.5, Reconnaissance +4.7, Worms +9.0, Generic +10.2. Generic
    has a seed std of ±27 points, and Worms rests on 52 rows.
- AUROC and known-attack F1 are essentially unchanged.

**Conclusion.** Both leaks were real, and both are fixed. Neither
inflated the earlier TM results: the corrected numbers are the same or
slightly better. The rankings between configurations do not change. All
results below use the corrected data.

## Task 2 — Validation-only selection of the TM, and comparison with RF and XGBoost

### Selection protocol

**Candidates (TM): 18 configurations** (5 quantile thresholds, medium scale).

| choice | options |
|---|---|
| architecture | two-stage (gate + family banks); single-stage over the same banks; single-stage multi-class TM |
| rule | global; class-conditional; margin |
| resampling | SMOTE; none |

Encoding was not searched again. In Phase 4, more thresholds barely helped
the TM (+0.006 to +0.016 closed-set macro F1), at 1.5–2.7× the cost.

**Candidates (trees).** RF and XGBoost, each on Boolean input and on
continuous input, with the global or margin rule.

**Why known-class validation is not enough.** Validation never contains
the held-out family. On known validation rows alone, a rule that never
rejects anything looks best. The selection therefore needs some unknown
rows that are not test rows.

**Pseudo-unknowns (`30_task2_compare.py`).**

- Each fold *v* keeps its held-out family's own **validation-period** rows
  (09 → `X_val_heldout`). Fold *v* never used them, and they are
  validation-period data, not test data.
- Objective, fixed in advance: J = macro F1 over the known classes plus
  Unknown, on fold *v*'s validation rows plus its pseudo-unknowns, with
  thresholds calibrated on known validation rows at the 5% budget.
- **Nested choice.** For outer fold *u*, the configuration chosen is the
  one that maximises the mean J over the **other** folds *v ≠ u*. Fold *u*'s
  own held-out rows never influence its own configuration.
- The trees' rule is chosen by the same procedure.

### Result: selection on validation picks a configuration that rarely detects unknowns

The validation-only procedure chose **two-stage, SMOTE, class-conditional
for all nine folds**. On test, at a 5% budget, it detects **16.7%** of
unknown attacks (family average).

The cause is the objective. Unknown is one class out of ~10 in a macro F1,
so J mostly rewards known-class accuracy. The class-conditional rule
protects known-class accuracy because it rarely rejects. The pre-specified
result stands as the answer to the question asked.

**Sensitivity analysis (added after seeing the primary result, labelled as
such).** Changing the objective after seeing test results would be
test-informed selection, so this is reported only as a sensitivity check.

- Objective: J′ = ½ (pseudo-unknown recall + known-class macro F1).
- J′ picks margin or global rules, mixing architectures and resampling
  across folds.
- Result: 59.5% unknown recall, lower than the fixed two-stage + global
  configuration (65.7%), which neither objective selected. Validation-only
  selection did not recover the best test configuration under either
  objective.

### Comparison (UNSW-NB15, 5% budget; full 1/5/10% tables, micro averages and per-family results in the appendix)

| system | unknown recall | unknown precision | known-attack F1 | known rejected | Normal false alarms | AUROC | AUPRC |
|---|---|---|---|---|---|---|---|
| TM, selected on validation | 16.7% ± 2.3 | 7.1% | 0.390 | 4.33% | 2.11% | 0.754 | 0.063 |
| TM, sensitivity objective (post hoc) | 59.5% ± 1.6 | 8.3% | 0.284 | 4.80% | 2.61% | 0.844 | 0.089 |
| TM two-stage SMOTE global (not selected) | 65.7% ± 2.1 | 12.8% | 0.333 | 4.61% | 2.11% | 0.840 | 0.131 |
| RF, Boolean (same input as TM) | 85.8% ± 1.9 | 17.0% | 0.333 | 4.90% | 2.56% | 0.962 | 0.178 |
| XGBoost, Boolean | 78.3% ± 3.3 | 12.2% | 0.353 | 4.79% | 2.42% | 0.969 | 0.166 |
| RF, continuous | 89.7% ± 0.1 | 17.9% | 0.432 | 4.87% | 2.80% | 0.973 | 0.193 |
| XGBoost, continuous | 89.8% ± 2.3 | 17.3% | 0.459 | 4.82% | 2.47% | 0.974 | 0.187 |

**Closed set (10 classes), macro F1.**

| model | macro F1 |
|---|---|
| TM, SMOTE | 0.464 |
| RF, Boolean | 0.496 |
| XGBoost, Boolean | 0.501 |
| RF, continuous | 0.596 |
| XGBoost, continuous | 0.610 |

Per-class F1 is in the appendix. The TM is behind on the rare families:
Worms 0.09 vs 0.16–0.67, Shellcode 0.26 vs 0.28–0.63.

**Reading the open-set numbers.**

- **Unknown precision is low for every model** (7–18%). A 5% budget on
  ~500,000 known test rows produces about 24,000 false rejections, more
  than the unknown rows of most folds. Read every unknown-recall figure
  with this in mind.
- **Micro and family averages diverge.** Micro-pooled recall is dominated
  by Generic (68,460 of 99,118 unknowns). At 5%, the TM's two-stage global
  configuration drops from a 65.7% family average to 42.4% micro, because
  it misses most of Generic (28.8% ± 26.9). RF on continuous input stays at
  96.2% micro.
- **Small families.** Analysis, Backdoor, Shellcode and Worms each have
  fewer than 1,000 test unknowns. They are marked ⚠ in the per-family
  table, with Wilson intervals; Worms' interval spans about 25 points.
- **Budgets.** At 1% every model detects little (8–19%). At 10% the trees
  reach 98–100% recall, but known-attack F1 collapses to 0.11–0.21.
- **Verdict.** On UNSW-NB15, under identical partitions, input
  representation, budgets and metrics, RF and XGBoost dominate the TM on
  every open-set metric except Normal false alarms. On the same Boolean
  input, RF finds 85.8% of unknowns against the TM's best 65.7%.

## Task 3 — Computational cost

**Setup.**

- Hardware: Apple M1, 8 cores (4 performance + 4 efficiency), 8 GB.
- Software: numpy 1.26.4 (OpenBLAS), scikit-learn 1.4.2, XGBoost 3.4.1,
  tmu as installed.
- Each measurement runs in a fresh process (`31_benchmark.py`) on the DoS
  fold, at 1 thread and 8 threads.
- Preprocessing is timed separately from inference.
- Energy was **not measured**: macOS exposes package power only through
  `powermetrics`, which needs root. CPU-seconds are reported instead.

**Preprocessing.**

| step | batch cost per row | one row at a time |
|---|---|---|
| Window features, 10 s (shared by all models) | 0.83 µs | – |
| Boolean encoding (TM and Boolean trees) | 1.65 µs | 312 µs, mostly Python overhead |
| Continuous encoding (trees) | 0.22 µs | 116 µs |

**Results, at 1 thread unless stated.**

| | train wall s | train peak MB | batch rows/s | one-row latency (median) | model size |
|---|---|---|---|---|---|
| **Individual model, binary task** | | | | | |
| TM bank | 9 | 298 | 45,772 | 94 µs | **30 KB** |
| RF | 24 | 37 | 228,153 | 1,069 µs | 10.7 MB |
| XGBoost | 16 | 155 | 227,421 | 79 µs | 1.7 MB |
| **Individual model, multi-class** | | | | | |
| TM | 99 | 934 | 10,330 | 460 µs | **142 KB** |
| RF, Boolean | 42 | 22 | 170,512 | 1,071 µs | 223 MB |
| XGBoost, Boolean | 218 | 124 | 21,530 | 348 µs | 20.6 MB |
| RF, continuous | 41 | 144 | 300,006 | 1,088 µs | 76.9 MB |
| XGBoost, continuous | 34 | 99 | 42,797 | 298 µs | 7.8 MB |
| **Complete open-set pipeline** | | | | | |
| TM two-stage (OpenSATM, all 9 banks) | 143 | 899 | 5,084 | 927 µs | **293 KB** |
| TM single-stage | 99 | 934 | 10,330 | 460 µs | **142 KB** |
| RF / XGBoost pipelines | as the multi-class rows above (one model + threshold) | | | | |

At 8 threads:

- RF and XGBoost training is 2–4.5× faster, and their batch throughput is
  2.5–4× higher.
- The TM two-stage pipeline trains in 42 s (banks in parallel), but
  scores only 9,108 rows/s.
- RF's one-row latency at 8 threads rises to 16.7 ms, because thread
  start-up dominates single-row calls.

**What this shows.**

- **Size is the one clear TM advantage.** The TM pipeline is 27×
  (XGBoost, continuous) to 780× (RF, Boolean) smaller. The single-stage TM
  is 55–1,600× smaller than the multi-class trees.
- **Batch inference is slower.** The TM is 4–60× behind the trees in this
  implementation. tmu's own predict is slower still for batches (7,552
  multi-class rows/s).
- **One-row latency is similar.** The TM's 0.46–0.93 ms sits between
  XGBoost (0.30–0.35 ms) and RF (1.07–1.09 ms). These are Python-level
  calls; an embedded C implementation of any of these models would be much
  faster, and these figures do not show that.
- **Training memory is higher.** The TM needs ~900 MB at peak because tmu
  copies the inputs to uint32. The trees need 22–155 MB.
- **Training rows differ.** The TM trains on the SMOTE set (360,000 rows),
  the trees on the capped real rows with class weights (~134,000). The
  binary-bank rows use identical data for all three models.

## Task 4 — Interpretability, measured rather than assumed

`32_explain_compare.py` puts the TM and the trees through the same tests.

**Attributions.**

- **TM:** each firing clause's vote is split equally among its literals,
  and literals are summed per original feature. Clause-level explanations
  are also reported, since clauses are the TM's native unit.
- **RF:** TreeSHAP.
- **XGBoost:** its own TreeSHAP (`pred_contribs`), in log-odds space.

All models are compared at the level of the 58 original features, on the
DoS and Fuzzers folds, with 200 rows per decision group, seed 42 as the
reference and seeds 7 and 1234 for stability.

**Two explanation targets, kept separate.**

- **Known class:** why class *k* won. The target is S_k − S_runner-up.
- **Rejection:** why the best class's support fell short. The target is
  S_top − θ.

**Results (means over the two folds; full tables in the appendix).**

| | TM | RF / XGBoost (Boolean) | RF / XGBoost (continuous) |
|---|---|---|---|
| **Known class, correct rows** | | | |
| Completeness (attributions sum to the score) | **no**: median error 95%, because class sums saturate at ±T | exact | exact |
| Features carrying 90% of the attribution | 18.5 | 9.5 / 16.5 | 14 / 18 |
| Faithfulness: decisions flipped by resetting the top-3 vs 3 random features | 0.91 vs 0.07 | 1.00 vs 0.11 / 1.00 vs 0.08 | 1.00 vs 0.13 / 0.94 vs 0.07 |
| Stability across seeds (top-5 Jaccard) | **0.49** | 0.69 / 0.73 | 0.92 / 0.77 |
| Stability under one perturbed feature | 0.89 | 0.92 / 0.90 | 0.96 / 0.93 |
| Clauses needed (TM) | median 1 clause of 2 literals reproduces the decision; 164 clauses fire | – | – |
| **Rejected as Unknown, held-out rows** | | | |
| Faithfulness: top-3 vs random-3 | 0.12 vs 0.02 | 0.06 vs 0.01 / 0.20 vs 0.03 | 0.04 vs 0.00 / 0.12 vs 0.01 |
| Stability across seeds | 0.45 | 0.67 / 0.61 | 0.61 / 0.56 |
| **Rejected as Unknown, known rows** | | | |
| Decided by tie-break (no explanation possible) | **46%** | 0% | 0% |

**Findings.**

1. **Known-class explanations.**
   - The TM's are sparse in its native unit: a single 2-literal clause
     usually reproduces the decision.
   - Faithfulness is about the same as SHAP's.
   - They are **less stable across seeds**: top-5 overlap 0.49 against
     0.69–0.92 for the trees.
   - They are **not complete**. Class sums are clipped at ±T, and most
     confident predictions saturate, so the clause votes do not add up to
     the score the decision used.
2. **Misclassified unknowns** (held-out rows given a known label) are the
   hardest case for the TM: a median of 71 clauses of about 45 literals
   each is needed to reproduce the decision.
3. **Rejection explanations are a different problem, and they are weak for
   every model.**
   - Resetting the features that pulled the best class's support down
     rarely turns a rejection back into an acceptance (0.04–0.20 at
     top-3). A rejection is an absence of supporting evidence, and
     deleting features does not create evidence.
   - None of the explanation methods gives a faithful account of a
     rejection, and the TM's is no better than SHAP's.
   - For the TM, 46% of the known rows it rejects sit exactly on its
     integer threshold and are decided by the tie-break. Those rejections
     have no explanation at all. This is the saturation limitation of
     Phase 3, now seen from the explanation side.
4. **Verdict.** Being rule-based does not make the TM's explanations
   measurably better here. They are sparse in the native unit, but less
   stable across seeds and incomplete under saturation. For rejections,
   neither family of models explains well.

## Task 5 — Consolidated comparison: is there a useful trade-off?

UNSW-NB15 at a 5% budget unless stated. Cost columns are the complete
pipeline at 1 thread.

| | TM selected (two-stage) | TM two-stage, global (best TM, not selected) | RF Boolean | XGBoost Boolean | RF continuous | XGBoost continuous |
|---|---|---|---|---|---|---|
| Closed-set macro F1 | 0.464 | 0.464 | 0.496 | 0.501 | 0.596 | 0.610 |
| Unknown recall, family average / micro | 16.7% / 26.6% | 65.7% / 42.4% | 85.8% / 79.3% | 78.3% / 42.9% | 89.7% / 96.2% | 89.8% / 86.2% |
| Unknown precision | 7.1% | 12.8% | 17.0% | 12.2% | 17.9% | 17.3% |
| CICIDS2017 unknown recall (same configuration) | 5.7% | 5.7% | 91.4% | 65.1% | 47.1% | 49.2% |
| Explanation stability across seeds (known) | 0.49 | 0.49 | 0.69 | 0.73 | 0.92 | 0.77 |
| Explanation complete | no (saturation) | no | yes | yes | yes | yes |
| Model size | 293 KB | 293 KB | 223 MB | 20.6 MB | 76.9 MB | 7.8 MB |
| Batch throughput (rows/s) | 5,084 | 5,084 | 170,512 | 21,530 | 300,006 | 42,797 |
| One-row latency | 0.93 ms | 0.93 ms | 1.07 ms | 0.35 ms | 1.09 ms | 0.30 ms |
| Training peak memory | 899 MB | 899 MB | 22 MB | 124 MB | 144 MB | 99 MB |

**Conclusion.**

- **The TM's only measured advantage is model size**: 27–780× smaller
  pipelines.
- **There is no speed advantage.** Batch inference is slower, and one-row
  latency is in the same range as the trees'.
- **Training memory is higher.**
- **Explanations are not measurably better.**
- **Detection is far lower.** On UNSW-NB15 the best TM configuration
  detects 13–24 points fewer unknown attacks (family average) than RF and
  XGBoost; the configuration chosen on validation detects 62–73 points
  fewer.
  On CICIDS2017 the selected configuration fails (Task 6).
- A constrained-deployment contribution would need acceptable detection at
  the small size. On this evidence the detection gap is too large, so
  **this study finds no useful trade-off in the TM's favour.** That
  finding is reported as such.

## Task 6 — Generalisation: CICIDS2017

**Data and choices (`40_cicids_prepare.py`).**

- **Source:** the corrected CICIDS2017 of Engelen, Rimmer & Joosen (WTMC
  2021), which re-extracts the flows and relabels more than 20% of the
  original traces. 2,100,814 flows over 5 days.
- **Families defined for this dataset** (not mapped to UNSW-NB15's):

  | family | flows | test rows |
  |---|---|---|
  | BruteForce (FTP + SSH Patator) | 6,953 | 1,391 |
  | DoS (Hulk, GoldenEye, slowloris, Slowhttptest) | 171,779 | 34,356 |
  | DDoS | 95,123 | 19,025 |
  | PortScan | 159,151 | 31,831 |
  | Bot | 738 | 148 ⚠ |
  | WebAttack (brute force, XSS, SQL injection) | 190 | 38 ⚠ |
  | Normal | – | – |

- **Excluded as labelled rows:** Infiltration (32 flows) and Heartbleed
  (11), too few; and the "Attempted" flows, whose ground truth is neither
  benign nor a successful attack. They remain in other flows' windows as
  real traffic.
- **Per-class chronological split** (72% fit, 8% validation, 20% test, by
  time within each class). Each attack occupies its own hours of one day,
  so one global time cut would leave whole families in only one partition.
  Within every class, training precedes test. Across classes it does not:
  some training rows of one family are later than some test rows of
  another. That is a dataset-specific departure, stated here.
- **Same leakage-aware construction as UNSW-NB15:**
  - corrected windows (completed flows only), with the held-out family's
    training-period flows removed before the windows are built;
  - encoding fitted on fit rows;
  - pseudo-unknowns from the validation period.
- **Feature adaptations:**
  - CICIDS has no connection state or TCP RTT. A flow with no backward
    packets counts as failed, and the RTT window feature is dropped.
  - Identifiers (flow ID, IPs, ports, timestamp) are excluded, as on
    UNSW-NB15.
- **Configuration:**
  - TM hyperparameters are **not tuned on CICIDS**: the modal UNSW-NB15
    configuration (1000 clauses, T = 50, s = 5, 1 epoch).
  - Trees get 20's validation grid.

**The selected configuration, carried over unchanged (two-stage, SMOTE,
class-conditional).**

- Unknown recall at 5%: **5.7%** (micro 7.7%).
- The benign gate admits 62–100% of each held-out family as Normal. Normal
  false alarms are 0.00%: the gate's threshold, set at a 1% missed-attack
  budget on known attacks, is low enough that nearly everything passes,
  including the new attacks.
- The UNSW-NB15 choice does **not** generalise.

**Other TM configurations (not the selected one; for context).**

| configuration | unknown recall | unknown precision | known-attack F1 |
|---|---|---|---|
| Single-stage over the same banks, global (re-selected on CICIDS validation for 5 of 6 folds) | 76.1% (micro 81.0%) | 29.3% | 0.842 |
| Same, class-conditional | 88.1% | 31.3% | 0.863 |

| model | unknown recall | unknown precision | known-attack F1 |
|---|---|---|---|
| RF, Boolean | 91.4% | 30.0% | 0.783 |
| XGBoost, Boolean | 65.1% | 27.3% | 0.913 |
| RF, continuous | 47.1% | 43.4% | 0.872 |
| XGBoost, continuous | 49.2% | 27.0% | 0.935 |

**What generalises and what does not.**

1. **Selection does not transfer.** The UNSW-selected configuration fails
   on CICIDS. Architecture choice is dataset-dependent, and the two-stage
   gate in particular.
2. **The TM does better here.** Its single-stage form over the banks beats
   XGBoost and continuous-input RF on unknown recall in this dataset, though
   not RF on Boolean input. It also has lower known-attack F1 than
   XGBoost.
3. **Tree results are unstable across families.**
   - With DDoS held out, Boolean XGBoost catches 0.1% while Boolean RF
     catches 100%.
   - With Bot held out (148 rows), the continuous-input trees catch 0%.
   - WebAttack (38 test rows, 16 validation rows) and Bot are too small for
     any per-family claim.
4. **As on UNSW-NB15**, unknown precision is far below recall (27–43%), and
   family and micro averages differ substantially.

## Limitations

- **Seeds and folds.**
  - Three seeds per model.
  - Hyperparameters were tuned on the v1 (uncorrected) data and reused.
  - The explanations cover two folds.
- **Benchmark.** Python-level latencies on one laptop. No energy
  measurement. Training rows differ between the TM (SMOTE) and the trees
  (capped real rows with weights), except for the binary-bank comparison.
- **Saturation.** TM class sums saturate at ±T. Ties at the integer
  threshold decide some rejections and make the TM's attributions
  incomplete.
- **Dataset validity.** UNSW-NB15's TTL artifact makes the Normal/attack
  separation easier than real traffic would. On CICIDS2017, the per-class
  split departs from a single chronological cut.

## Scripts added or changed this phase

| script | change |
|---|---|
| `29_audit_temporal.py` | new: independent recomputation of the windows; alignment, current-flow, in-progress, same-second, boundary and LOFO checks |
| `00b_build_windows_v2.py` | new: corrected windows (completed flows only, `--exclude` per held-out family, `--legacy` reproduces 00) |
| `09_openset_preprocess.py` | reads per-family rebuilt windows; label encoder on train ∪ test; saves `X_val_heldout` (pseudo-unknowns) |
| `10_train_family_banks.py`, `25_train_single_stage.py` | score the pseudo-unknown rows |
| `20_openset_baselines.py` | per-family rebuilt windows; saves top-2 probabilities for validation, test and pseudo-unknowns |
| `30_task2_compare.py` | new: nested validation-only selection; all Task 2 metrics at 1/5/10%, micro and family averages, Wilson intervals |
| `31_benchmark.py` | new: compute benchmark |
| `32_explain_compare.py` | new: TM vs TreeSHAP under common criteria |
| `33_phase5_tables.py` | new: generates the appendix |
| `40_cicids_prepare.py`, `41_cicids_trees.py` | new: CICIDS2017 under the same protocol |

## Appendix — generated tables

### Task 1 — before (v1) and after (v2) the window corrections, seeds 7/42/1234, 5% budget, family average

| configuration | unknown recall v1 | v2 | Δ (points) | known-attack F1 v1 | v2 | AUROC v1 | v2 |
|---|---|---|---|---|---|---|---|
| two_stage smote global | 62.4% ± 1.2 | 65.7% ± 2.1 | +3.3 | 0.333 ± 0.007 | 0.333 ± 0.012 | 0.836 ± 0.018 | 0.840 ± 0.031 |
| two_stage smote classcond | 15.7% ± 1.7 | 16.7% ± 2.3 | +1.0 | 0.389 ± 0.003 | 0.390 ± 0.003 | 0.751 ± 0.017 | 0.754 ± 0.032 |
| two_stage smote margin | 70.4% ± 0.7 | 70.8% ± 0.8 | +0.4 | 0.290 ± 0.004 | 0.289 ± 0.005 | 0.856 ± 0.016 | 0.858 ± 0.029 |
| single_banks smote global | 50.6% ± 2.9 | 52.9% ± 4.1 | +2.4 | 0.364 ± 0.002 | 0.361 ± 0.015 | 0.807 ± 0.020 | 0.823 ± 0.002 |
| single_banks smote classcond | 2.9% ± 0.6 | 3.0% ± 0.6 | +0.0 | 0.399 ± 0.002 | 0.401 ± 0.002 | 0.120 ± 0.004 | 0.119 ± 0.012 |
| single_banks smote margin | 63.9% ± 2.4 | 65.8% ± 1.3 | +1.9 | 0.321 ± 0.004 | 0.321 ± 0.006 | 0.930 ± 0.002 | 0.938 ± 0.006 |
| single_mc smote global | 54.9% ± 1.3 | 51.3% ± 1.8 | -3.6 | 0.367 ± 0.003 | 0.372 ± 0.002 | 0.805 ± 0.016 | 0.800 ± 0.008 |
| single_mc smote classcond | 3.6% ± 0.4 | 4.0% ± 0.3 | +0.4 | 0.403 ± 0.002 | 0.402 ± 0.004 | 0.117 ± 0.012 | 0.116 ± 0.018 |
| single_mc smote margin | 62.1% ± 1.8 | 67.2% ± 3.4 | +5.1 | 0.329 ± 0.005 | 0.326 ± 0.004 | 0.927 ± 0.008 | 0.938 ± 0.011 |
| two_stage none global | 48.7% ± 2.4 | 50.7% ± 2.5 | +2.0 | 0.351 ± 0.002 | 0.349 ± 0.003 | 0.831 ± 0.034 | 0.816 ± 0.018 |
| two_stage none classcond | 12.8% ± 2.8 | 12.3% ± 0.9 | -0.5 | 0.373 ± 0.002 | 0.375 ± 0.001 | 0.754 ± 0.035 | 0.737 ± 0.014 |
| two_stage none margin | 73.9% ± 0.6 | 74.4% ± 0.6 | +0.5 | 0.240 ± 0.003 | 0.242 ± 0.003 | 0.864 ± 0.033 | 0.847 ± 0.015 |
| single_banks none global | 31.4% ± 3.5 | 29.7% ± 2.5 | -1.7 | 0.383 ± 0.003 | 0.386 ± 0.001 | 0.681 ± 0.004 | 0.701 ± 0.006 |
| single_banks none classcond | 1.9% ± 0.6 | 2.6% ± 0.2 | +0.7 | 0.389 ± 0.001 | 0.391 ± 0.001 | 0.088 ± 0.012 | 0.096 ± 0.018 |
| single_banks none margin | 72.3% ± 0.5 | 72.2% ± 1.1 | -0.1 | 0.266 ± 0.004 | 0.270 ± 0.002 | 0.945 ± 0.019 | 0.954 ± 0.007 |
| single_mc none global | 45.4% ± 5.7 | 47.3% ± 3.2 | +2.0 | 0.319 ± 0.002 | 0.324 ± 0.003 | 0.720 ± 0.032 | 0.739 ± 0.011 |
| single_mc none classcond | 5.6% ± 0.7 | 6.2% ± 1.5 | +0.5 | 0.344 ± 0.002 | 0.345 ± 0.003 | 0.119 ± 0.007 | 0.127 ± 0.021 |
| single_mc none margin | 70.3% ± 2.1 | 69.7% ± 1.6 | -0.6 | 0.280 ± 0.004 | 0.282 ± 0.002 | 0.960 ± 0.002 | 0.962 ± 0.001 |

Per family, the Phase 3 headline configuration (two-stage, SMOTE, global):

| held out | unknown recall v1 | v2 | Δ (points) |
|---|---|---|---|
| analysis | 75.3% ± 3.4 | 75.0% ± 4.9 | -0.2 |
| backdoor | 93.1% ± 0.5 | 93.0% ± 0.2 | -0.1 |
| dos | 88.0% ± 0.7 | 87.4% ± 0.9 | -0.5 |
| exploits | 82.6% ± 8.2 | 82.4% ± 2.6 | -0.2 |
| fuzzers | 51.6% ± 0.8 | 57.1% ± 3.5 | +5.5 |
| generic | 18.6% ± 6.7 | 28.8% ± 26.9 | +10.2 |
| reconnaissance | 41.8% ± 3.9 | 46.5% ± 7.4 | +4.7 |
| shellcode | 72.7% ± 3.9 | 74.2% ± 1.3 | +1.5 |
| worms | 37.8% ± 12.0 | 46.8% ± 4.0 | +9.0 |

### Task 2 — closed set (10 classes, test), macro F1 (TM: prior-calibrated decision)

| model | seeds | macro F1 |
|---|---|---|
| TM smote (v1) | 3 | 0.4602 ± 0.0071 |
| TM none (v1) | 3 | 0.3900 ± 0.0013 |
| TM smote (v2) | 3 | 0.4643 ± 0.0068 |
| TM none (v2) | 3 | 0.3867 ± 0.0019 |
| RF Boolean (v2) | 3 | 0.4963 ± 0.0006 |
| XGB Boolean (v2) | 3 | 0.5011 ± 0.0019 |
| RF continuous (v2) | 3 | 0.5959 ± 0.0009 |
| XGB continuous (v2) | 3 | 0.6095 ± 0.0010 |

Per-class F1 (v2, mean over seeds):

| class | TM smote | TM none | RF Boolean | XGB Boolean | RF continuous | XGB continuous |
|---|---|---|---|---|---|---|
| Analysis | 0.098 | 0.008 | 0.096 | 0.103 | 0.107 | 0.098 |
| Backdoor | 0.058 | 0.000 | 0.100 | 0.097 | 0.160 | 0.160 |
| DoS | 0.329 | 0.069 | 0.469 | 0.428 | 0.492 | 0.490 |
| Exploits | 0.599 | 0.628 | 0.602 | 0.611 | 0.656 | 0.671 |
| Fuzzers | 0.529 | 0.500 | 0.552 | 0.558 | 0.566 | 0.570 |
| Generic | 0.987 | 0.986 | 0.987 | 0.988 | 0.988 | 0.990 |
| Normal | 0.986 | 0.987 | 0.988 | 0.988 | 0.988 | 0.988 |
| Reconnaissance | 0.704 | 0.689 | 0.728 | 0.722 | 0.833 | 0.834 |
| Shellcode | 0.261 | 0.000 | 0.281 | 0.318 | 0.533 | 0.625 |
| Worms | 0.092 | 0.000 | 0.159 | 0.198 | 0.635 | 0.667 |

### Task 2 — open set, UNSW-NB15

#### UNSW-NB15: configuration chosen on validation (nested over folds)

- **TM**: `two_stage / smote / classcond` for 9 fold(s)
- **TM (balanced objective, sensitivity)**: `single_banks / none / margin` for 2 fold(s) (analysis, exploits); `single_mc / none / margin` for 2 fold(s) (dos, shellcode); `two_stage / none / margin` for 2 fold(s) (fuzzers, generic); `two_stage / smote / global` for 2 fold(s) (reconnaissance, worms); `two_stage / smote / margin` for 1 fold(s) (backdoor)
- **RF boolean**: `rf / boolean / global` for 8 fold(s) (analysis, backdoor, dos, exploits, fuzzers, reconnaissance, shellcode, worms); `rf / boolean / margin` for 1 fold(s) (generic)
- **RF continuous_medium**: `rf / continuous_medium / global` for 9 fold(s)
- **XGB boolean**: `xgb / boolean / margin` for 9 fold(s)
- **XGB continuous_medium**: `xgb / continuous_medium / margin` for 9 fold(s)

Validation objective J (open-set macro F1 over known classes + Unknown, pseudo-unknowns, 5% budget), mean over all nine folds and seeds:

| TM configuration (architecture | resampling | rule) | mean J |
|---|---|
| two_stage / smote / classcond | 0.3963 |
| single_mc / smote / classcond | 0.3941 |
| single_banks / smote / classcond | 0.3929 |
| single_banks / none / classcond | 0.3902 |
| two_stage / none / classcond | 0.3894 |
| single_banks / none / global | 0.3884 |
| single_mc / smote / global | 0.3857 |
| single_banks / smote / global | 0.3816 |
| two_stage / smote / global | 0.3661 |
| two_stage / none / global | 0.3650 |
| single_mc / smote / margin | 0.3588 |
| single_mc / none / global | 0.3581 |
| single_mc / none / classcond | 0.3555 |
| single_banks / smote / margin | 0.3536 |
| two_stage / smote / margin | 0.3321 |
| single_mc / none / margin | 0.3280 |
| single_banks / none / margin | 0.3189 |
| two_stage / none / margin | 0.2997 |

#### UNSW-NB15: 1% validation budget — family average (mean ± std over seeds)

| system | unknown recall | unknown precision | known-attack macro F1 | known rejected | Normal false alarms | AUROC | AUPRC |
|---|---|---|---|---|---|---|---|
| TM (selected on validation) | 8.1% ± 2.6 | 11.3% ± 2.1 | 0.411 ± 0.002 | 0.97% ± 0.02 | 2.11% ± 0.04 | 0.754 ± 0.032 | 0.063 ± 0.026 |
| TM, sensitivity: balanced validation objective (post hoc) | 10.5% ± 1.2 | 7.3% ± 0.6 | 0.383 ± 0.003 | 0.85% ± 0.04 | 2.40% ± 0.02 | 0.844 ± 0.013 | 0.089 ± 0.005 |
| TM two-stage, SMOTE, global (Phase 3 headline; not selected) | 12.8% ± 0.6 | 14.7% ± 0.3 | 0.411 ± 0.003 | 0.57% ± 0.04 | 2.11% ± 0.04 | 0.840 ± 0.031 | 0.131 ± 0.015 |
| RF, Boolean input | 8.3% ± 0.6 | 14.6% ± 1.1 | 0.458 ± 0.001 | 0.43% ± 0.01 | 2.21% ± 0.00 | 0.962 ± 0.000 | 0.178 ± 0.001 |
| XGBoost, Boolean input | 16.7% ± 0.6 | 11.1% ± 0.1 | 0.451 ± 0.000 | 0.94% ± 0.01 | 2.19% ± 0.00 | 0.969 ± 0.001 | 0.166 ± 0.002 |
| RF, continuous input | 7.7% ± 0.1 | 16.0% ± 0.9 | 0.561 ± 0.001 | 0.39% ± 0.00 | 2.24% ± 0.00 | 0.973 ± 0.000 | 0.193 ± 0.001 |
| XGBoost, continuous input | 18.5% ± 0.5 | 16.5% ± 1.8 | 0.565 ± 0.000 | 0.95% ± 0.04 | 2.17% ± 0.00 | 0.974 ± 0.001 | 0.187 ± 0.006 |

Micro-averaged over the 9 folds (counts pooled across folds, then the rate; AUROC/AUPRC are not pooled because different folds' scores are not on one scale):

| system | unknown recall | unknown precision | known-attack macro F1 | known rejected | Normal false alarms |
|---|---|---|---|---|---|
| TM (selected on validation) | 21.4% ± 18.5 | 27.9% ± 17.0 | 0.413 ± 0.001 | 0.97% ± 0.02 | 2.11% ± 0.04 |
| TM, sensitivity: balanced validation objective (post hoc) | 4.2% ± 0.6 | 9.8% ± 1.0 | 0.394 ± 0.002 | 0.85% ± 0.04 | 2.40% ± 0.02 |
| TM two-stage, SMOTE, global (Phase 3 headline; not selected) | 5.6% ± 0.4 | 17.9% ± 0.2 | 0.413 ± 0.003 | 0.56% ± 0.04 | 2.11% ± 0.04 |
| RF, Boolean input | 4.2% ± 0.5 | 17.9% ± 1.3 | 0.456 ± 0.001 | 0.42% ± 0.01 | 2.21% ± 0.00 |
| XGBoost, Boolean input | 6.3% ± 0.2 | 13.0% ± 0.2 | 0.451 ± 0.000 | 0.94% ± 0.01 | 2.19% ± 0.00 |
| RF, continuous input | 4.6% ± 0.4 | 20.7% ± 1.5 | 0.561 ± 0.000 | 0.39% ± 0.00 | 2.24% ± 0.00 |
| XGBoost, continuous input | 11.0% ± 1.9 | 20.1% ± 2.3 | 0.563 ± 0.000 | 0.96% ± 0.03 | 2.17% ± 0.00 |

#### UNSW-NB15: 5% validation budget — family average (mean ± std over seeds)

| system | unknown recall | unknown precision | known-attack macro F1 | known rejected | Normal false alarms | AUROC | AUPRC |
|---|---|---|---|---|---|---|---|
| TM (selected on validation) | 16.7% ± 2.3 | 7.1% ± 2.5 | 0.390 ± 0.003 | 4.33% ± 0.01 | 2.11% ± 0.04 | 0.754 ± 0.032 | 0.063 ± 0.026 |
| TM, sensitivity: balanced validation objective (post hoc) | 59.5% ± 1.6 | 8.3% ± 0.2 | 0.284 ± 0.004 | 4.80% ± 0.08 | 2.61% ± 0.05 | 0.844 ± 0.013 | 0.089 ± 0.005 |
| TM two-stage, SMOTE, global (Phase 3 headline; not selected) | 65.7% ± 2.1 | 12.8% ± 2.3 | 0.333 ± 0.012 | 4.61% ± 0.04 | 2.11% ± 0.04 | 0.840 ± 0.031 | 0.131 ± 0.015 |
| RF, Boolean input | 85.8% ± 1.9 | 17.0% ± 0.7 | 0.333 ± 0.000 | 4.90% ± 0.01 | 2.56% ± 0.00 | 0.962 ± 0.000 | 0.178 ± 0.001 |
| XGBoost, Boolean input | 78.3% ± 3.3 | 12.2% ± 3.1 | 0.353 ± 0.001 | 4.79% ± 0.01 | 2.42% ± 0.00 | 0.969 ± 0.001 | 0.166 ± 0.002 |
| RF, continuous input | 89.7% ± 0.1 | 17.9% ± 0.0 | 0.432 ± 0.001 | 4.87% ± 0.02 | 2.80% ± 0.01 | 0.973 ± 0.000 | 0.193 ± 0.001 |
| XGBoost, continuous input | 89.8% ± 2.3 | 17.3% ± 0.5 | 0.459 ± 0.001 | 4.82% ± 0.01 | 2.47% ± 0.00 | 0.974 ± 0.001 | 0.187 ± 0.006 |

Micro-averaged over the 9 folds (counts pooled across folds, then the rate; AUROC/AUPRC are not pooled because different folds' scores are not on one scale):

| system | unknown recall | unknown precision | known-attack macro F1 | known rejected | Normal false alarms |
|---|---|---|---|---|---|
| TM (selected on validation) | 26.6% ± 19.6 | 11.3% ± 7.4 | 0.394 ± 0.002 | 4.32% ± 0.01 | 2.11% ± 0.04 |
| TM, sensitivity: balanced validation objective (post hoc) | 23.5% ± 0.5 | 9.8% ± 0.1 | 0.301 ± 0.003 | 4.80% ± 0.08 | 2.61% ± 0.05 |
| TM two-stage, SMOTE, global (Phase 3 headline; not selected) | 42.4% ± 18.2 | 16.5% ± 5.7 | 0.342 ± 0.010 | 4.61% ± 0.04 | 2.11% ± 0.04 |
| RF, Boolean input | 79.3% ± 12.8 | 26.3% ± 3.2 | 0.341 ± 0.001 | 4.90% ± 0.01 | 2.56% ± 0.00 |
| XGBoost, Boolean input | 42.9% ± 20.9 | 16.0% ± 6.4 | 0.358 ± 0.000 | 4.80% ± 0.01 | 2.42% ± 0.00 |
| RF, continuous input | 96.2% ± 0.1 | 30.5% ± 0.1 | 0.437 ± 0.001 | 4.86% ± 0.02 | 2.80% ± 0.01 |
| XGBoost, continuous input | 86.2% ± 12.7 | 28.2% ± 3.1 | 0.463 ± 0.001 | 4.83% ± 0.01 | 2.47% ± 0.00 |

#### UNSW-NB15: 10% validation budget — family average (mean ± std over seeds)

| system | unknown recall | unknown precision | known-attack macro F1 | known rejected | Normal false alarms | AUROC | AUPRC |
|---|---|---|---|---|---|---|---|
| TM (selected on validation) | 34.1% ± 2.2 | 6.0% ± 2.2 | 0.333 ± 0.003 | 8.55% ± 0.03 | 2.11% ± 0.04 | 0.754 ± 0.032 | 0.063 ± 0.026 |
| TM, sensitivity: balanced validation objective (post hoc) | 78.6% ± 1.9 | 8.6% ± 1.3 | 0.166 ± 0.005 | 9.76% ± 0.03 | 4.21% ± 0.04 | 0.844 ± 0.013 | 0.089 ± 0.005 |
| TM two-stage, SMOTE, global (Phase 3 headline; not selected) | 73.2% ± 3.1 | 8.5% ± 2.1 | 0.245 ± 0.011 | 9.04% ± 0.02 | 2.11% ± 0.04 | 0.840 ± 0.031 | 0.131 ± 0.015 |
| RF, Boolean input | 98.0% ± 0.0 | 12.7% ± 0.0 | 0.161 ± 0.001 | 9.92% ± 0.09 | 5.20% ± 0.09 | 0.962 ± 0.000 | 0.178 ± 0.001 |
| XGBoost, Boolean input | 100.0% ± 0.0 | 12.6% ± 0.0 | 0.114 ± 0.000 | 10.21% ± 0.03 | 5.33% ± 0.03 | 0.969 ± 0.001 | 0.166 ± 0.002 |
| RF, continuous input | 99.6% ± 0.2 | 12.8% ± 0.0 | 0.213 ± 0.001 | 9.77% ± 0.02 | 5.31% ± 0.01 | 0.973 ± 0.000 | 0.193 ± 0.001 |
| XGBoost, continuous input | 100.0% ± 0.0 | 12.9% ± 0.1 | 0.121 ± 0.000 | 9.96% ± 0.06 | 4.93% ± 0.08 | 0.974 ± 0.001 | 0.187 ± 0.006 |

Micro-averaged over the 9 folds (counts pooled across folds, then the rate; AUROC/AUPRC are not pooled because different folds' scores are not on one scale):

| system | unknown recall | unknown precision | known-attack macro F1 | known rejected | Normal false alarms |
|---|---|---|---|---|---|
| TM (selected on validation) | 33.2% ± 20.6 | 7.7% ± 4.4 | 0.351 ± 0.003 | 8.55% ± 0.03 | 2.11% ± 0.04 |
| TM, sensitivity: balanced validation objective (post hoc) | 46.6% ± 9.2 | 9.5% ± 1.7 | 0.192 ± 0.003 | 9.77% ± 0.03 | 4.21% ± 0.04 |
| TM two-stage, SMOTE, global (Phase 3 headline; not selected) | 46.6% ± 20.3 | 10.1% ± 3.9 | 0.261 ± 0.011 | 9.04% ± 0.02 | 2.11% ± 0.04 |
| RF, Boolean input | 99.7% ± 0.0 | 18.2% ± 0.1 | 0.163 ± 0.001 | 9.93% ± 0.09 | 5.20% ± 0.09 |
| XGBoost, Boolean input | 100.0% ± 0.0 | 17.8% ± 0.0 | 0.115 ± 0.000 | 10.22% ± 0.03 | 5.33% ± 0.03 |
| RF, continuous input | 99.8% ± 0.1 | 18.5% ± 0.0 | 0.214 ± 0.001 | 9.77% ± 0.02 | 5.31% ± 0.01 |
| XGBoost, continuous input | 100.0% ± 0.0 | 18.2% ± 0.1 | 0.123 ± 0.001 | 9.97% ± 0.06 | 4.93% ± 0.08 |

#### UNSW-NB15: per held-out family, 5% budget (mean ± std over seeds)

| held out | test unknowns | val pseudo-unknowns | system | unknown recall [95% CI] | unknown precision | known-attack F1 | Normal FA | AUROC |
|---|---|---|---|---|---|---|---|---|
| analysis ⚠ | 678 | 340 | TM (selected on validation) | 2.3% ± 1.4 [1–4] | 0.1% ± 0.0 | 0.415 ± 0.007 | 1.89% ± 0.15 | 0.767 ± 0.010 |
| analysis ⚠ | 678 | 340 | TM, sensitivity: balanced validation objective (post hoc) | 82.1% ± 3.9 [79–85] | 2.2% ± 0.1 | 0.280 ± 0.008 | 3.03% ± 0.03 | 0.971 ± 0.001 |
| analysis ⚠ | 678 | 340 | TM two-stage, SMOTE, global (Phase 3 headline; not selected) | 75.0% ± 4.9 [72–78] | 2.1% ± 0.1 | 0.352 ± 0.022 | 1.89% ± 0.15 | 0.875 ± 0.010 |
| analysis ⚠ | 678 | 340 | RF, Boolean input | 97.2% ± 0.2 [96–98] | 2.6% ± 0.0 | 0.386 ± 0.003 | 2.52% ± 0.01 | 0.981 ± 0.001 |
| analysis ⚠ | 678 | 340 | XGBoost, Boolean input | 99.3% ± 0.2 [98–100] | 2.8% ± 0.0 | 0.410 ± 0.001 | 2.40% ± 0.00 | 0.982 ± 0.000 |
| analysis ⚠ | 678 | 340 | RF, continuous input | 99.1% ± 0.1 [98–100] | 2.7% ± 0.0 | 0.505 ± 0.003 | 2.82% ± 0.01 | 0.982 ± 0.000 |
| analysis ⚠ | 678 | 340 | XGBoost, continuous input | 99.9% ± 0.0 [99–100] | 2.8% ± 0.0 | 0.521 ± 0.002 | 2.40% ± 0.00 | 0.983 ± 0.000 |
| backdoor ⚠ | 702 | 311 | TM (selected on validation) | 4.4% ± 0.5 [3–6] | 0.1% ± 0.0 | 0.433 ± 0.007 | 2.16% ± 0.09 | 0.841 ± 0.002 |
| backdoor ⚠ | 702 | 311 | TM, sensitivity: balanced validation objective (post hoc) | 93.4% ± 0.5 [91–95] | 2.6% ± 0.0 | 0.350 ± 0.004 | 2.16% ± 0.09 | 0.974 ± 0.001 |
| backdoor ⚠ | 702 | 311 | TM two-stage, SMOTE, global (Phase 3 headline; not selected) | 93.0% ± 0.2 [91–95] | 2.7% ± 0.0 | 0.376 ± 0.013 | 2.16% ± 0.09 | 0.968 ± 0.002 |
| backdoor ⚠ | 702 | 311 | RF, Boolean input | 97.6% ± 0.1 [96–98] | 2.7% ± 0.0 | 0.386 ± 0.002 | 2.54% ± 0.01 | 0.983 ± 0.000 |
| backdoor ⚠ | 702 | 311 | XGBoost, Boolean input | 95.9% ± 0.2 [94–97] | 2.8% ± 0.0 | 0.410 ± 0.001 | 2.42% ± 0.01 | 0.981 ± 0.000 |
| backdoor ⚠ | 702 | 311 | RF, continuous input | 99.4% ± 0.1 [99–100] | 2.8% ± 0.0 | 0.500 ± 0.002 | 2.85% ± 0.01 | 0.985 ± 0.000 |
| backdoor ⚠ | 702 | 311 | XGBoost, continuous input | 98.8% ± 0.3 [98–99] | 2.8% ± 0.0 | 0.513 ± 0.003 | 2.57% ± 0.01 | 0.983 ± 0.000 |
| dos | 5161 | 2568 | TM (selected on validation) | 5.7% ± 0.8 [5–6] | 1.4% ± 0.2 | 0.416 ± 0.001 | 2.00% ± 0.14 | 0.834 ± 0.002 |
| dos | 5161 | 2568 | TM, sensitivity: balanced validation objective (post hoc) | 69.5% ± 8.9 [68–71] | 12.4% ± 1.2 | 0.294 ± 0.004 | 3.26% ± 0.21 | 0.957 ± 0.001 |
| dos | 5161 | 2568 | TM two-stage, SMOTE, global (Phase 3 headline; not selected) | 87.4% ± 0.9 [87–88] | 15.8% ± 0.4 | 0.370 ± 0.002 | 2.00% ± 0.14 | 0.958 ± 0.004 |
| dos | 5161 | 2568 | RF, Boolean input | 92.1% ± 0.3 [91–93] | 15.1% ± 0.0 | 0.314 ± 0.001 | 2.69% ± 0.01 | 0.969 ± 0.001 |
| dos | 5161 | 2568 | XGBoost, Boolean input | 93.7% ± 0.1 [93–94] | 15.3% ± 0.0 | 0.351 ± 0.003 | 2.69% ± 0.01 | 0.982 ± 0.000 |
| dos | 5161 | 2568 | RF, continuous input | 92.9% ± 0.1 [92–94] | 15.6% ± 0.1 | 0.449 ± 0.006 | 3.12% ± 0.02 | 0.977 ± 0.000 |
| dos | 5161 | 2568 | XGBoost, continuous input | 94.2% ± 0.1 [94–95] | 15.5% ± 0.0 | 0.481 ± 0.001 | 2.75% ± 0.02 | 0.984 ± 0.000 |
| exploits | 12850 | 5286 | TM (selected on validation) | 18.0% ± 1.3 [17–19] | 9.3% ± 1.0 | 0.418 ± 0.005 | 2.16% ± 0.16 | 0.820 ± 0.021 |
| exploits | 12850 | 5286 | TM, sensitivity: balanced validation objective (post hoc) | 91.0% ± 6.6 [91–92] | 31.3% ± 1.4 | 0.208 ± 0.030 | 3.99% ± 0.47 | 0.973 ± 0.003 |
| exploits | 12850 | 5286 | TM two-stage, SMOTE, global (Phase 3 headline; not selected) | 82.4% ± 2.6 [82–83] | 31.3% ± 1.9 | 0.306 ± 0.038 | 2.16% ± 0.16 | 0.932 ± 0.017 |
| exploits | 12850 | 5286 | RF, Boolean input | 95.3% ± 0.4 [95–96] | 31.8% ± 0.1 | 0.215 ± 0.002 | 3.06% ± 0.02 | 0.977 ± 0.000 |
| exploits | 12850 | 5286 | XGBoost, Boolean input | 93.7% ± 0.1 [93–94] | 31.5% ± 0.1 | 0.231 ± 0.001 | 2.85% ± 0.01 | 0.976 ± 0.000 |
| exploits | 12850 | 5286 | RF, continuous input | 92.5% ± 0.6 [92–93] | 33.6% ± 0.1 | 0.320 ± 0.003 | 2.92% ± 0.01 | 0.977 ± 0.000 |
| exploits | 12850 | 5286 | XGBoost, continuous input | 96.8% ± 0.2 [96–97] | 32.2% ± 0.0 | 0.398 ± 0.002 | 3.02% ± 0.00 | 0.978 ± 0.000 |
| fuzzers | 6711 | 2186 | TM (selected on validation) | 27.0% ± 3.1 [26–28] | 7.5% ± 0.7 | 0.351 ± 0.013 | 1.71% ± 0.16 | 0.725 ± 0.035 |
| fuzzers | 6711 | 2186 | TM, sensitivity: balanced validation objective (post hoc) | 51.5% ± 3.0 [50–53] | 12.3% ± 0.1 | 0.202 ± 0.012 | 1.63% ± 0.03 | 0.731 ± 0.011 |
| fuzzers | 6711 | 2186 | TM two-stage, SMOTE, global (Phase 3 headline; not selected) | 57.1% ± 3.5 [56–58] | 14.5% ± 0.9 | 0.304 ± 0.035 | 1.71% ± 0.16 | 0.773 ± 0.040 |
| fuzzers | 6711 | 2186 | RF, Boolean input | 81.7% ± 0.5 [81–83] | 18.5% ± 0.1 | 0.336 ± 0.001 | 2.10% ± 0.01 | 0.968 ± 0.000 |
| fuzzers | 6711 | 2186 | XGBoost, Boolean input | 68.3% ± 0.9 [67–69] | 16.1% ± 0.1 | 0.356 ± 0.001 | 1.74% ± 0.01 | 0.963 ± 0.000 |
| fuzzers | 6711 | 2186 | RF, continuous input | 90.1% ± 0.5 [89–91] | 19.5% ± 0.1 | 0.418 ± 0.003 | 2.34% ± 0.01 | 0.971 ± 0.000 |
| fuzzers | 6711 | 2186 | XGBoost, continuous input | 78.6% ± 0.2 [78–80] | 18.1% ± 0.0 | 0.465 ± 0.002 | 1.76% ± 0.01 | 0.967 ± 0.000 |
| generic | 68460 | 33324 | TM (selected on validation) | 30.1% ± 28.6 [30–30] | 39.2% ± 23.1 | 0.257 ± 0.002 | 2.22% ± 0.02 | 0.324 ± 0.293 |
| generic | 68460 | 33324 | TM, sensitivity: balanced validation objective (post hoc) | 1.8% ± 0.3 [2–2] | 5.4% ± 0.9 | 0.204 ± 0.010 | 2.21% ± 0.03 | 0.260 ± 0.119 |
| generic | 68460 | 33324 | TM two-stage, SMOTE, global (Phase 3 headline; not selected) | 28.8% ± 26.9 [28–29] | 39.3% ± 22.1 | 0.247 ± 0.004 | 2.22% ± 0.02 | 0.315 ± 0.285 |
| generic | 68460 | 33324 | RF, Boolean input | 75.4% ± 18.7 [75–76] | 69.8% ± 6.2 | 0.303 ± 0.003 | 2.68% ± 0.01 | 0.956 ± 0.002 |
| generic | 68460 | 33324 | XGBoost, Boolean input | 24.5% ± 30.3 [24–25] | 29.6% ± 28.5 | 0.298 ± 0.000 | 2.38% ± 0.01 | 0.950 ± 0.005 |
| generic | 68460 | 33324 | RF, continuous input | 99.4% ± 0.0 [99–99] | 75.6% ± 0.0 | 0.396 ± 0.003 | 2.78% ± 0.01 | 0.972 ± 0.001 |
| generic | 68460 | 33324 | XGBoost, continuous input | 85.7% ± 18.3 [85–86] | 73.7% ± 4.7 | 0.430 ± 0.002 | 2.43% ± 0.00 | 0.966 ± 0.007 |
| reconnaissance | 4064 | 1818 | TM (selected on validation) | 29.7% ± 3.4 [28–31] | 5.3% ± 0.6 | 0.378 ± 0.004 | 2.19% ± 0.10 | 0.860 ± 0.003 |
| reconnaissance | 4064 | 1818 | TM, sensitivity: balanced validation objective (post hoc) | 46.5% ± 7.4 [45–48] | 7.6% ± 1.0 | 0.311 ± 0.004 | 2.19% ± 0.10 | 0.916 ± 0.009 |
| reconnaissance | 4064 | 1818 | TM two-stage, SMOTE, global (Phase 3 headline; not selected) | 46.5% ± 7.4 [45–48] | 7.6% ± 1.0 | 0.311 ± 0.004 | 2.19% ± 0.10 | 0.916 ± 0.009 |
| reconnaissance | 4064 | 1818 | RF, Boolean input | 70.8% ± 1.0 [69–72] | 10.5% ± 0.1 | 0.310 ± 0.001 | 2.57% ± 0.02 | 0.957 ± 0.001 |
| reconnaissance | 4064 | 1818 | XGBoost, Boolean input | 64.0% ± 0.7 [62–65] | 9.7% ± 0.1 | 0.335 ± 0.001 | 2.48% ± 0.01 | 0.958 ± 0.000 |
| reconnaissance | 4064 | 1818 | RF, continuous input | 66.2% ± 0.7 [65–68] | 9.8% ± 0.1 | 0.396 ± 0.001 | 2.88% ± 0.00 | 0.957 ± 0.001 |
| reconnaissance | 4064 | 1818 | XGBoost, continuous input | 58.2% ± 1.2 [57–60] | 8.8% ± 0.2 | 0.416 ± 0.002 | 2.48% ± 0.01 | 0.957 ± 0.000 |
| shellcode ⚠ | 440 | 157 | TM (selected on validation) | 26.5% ± 0.5 [23–31] | 0.6% ± 0.0 | 0.405 ± 0.002 | 2.53% ± 0.01 | 0.852 ± 0.005 |
| shellcode ⚠ | 440 | 157 | TM, sensitivity: balanced validation objective (post hoc) | 52.7% ± 2.7 [48–57] | 1.0% ± 0.0 | 0.327 ± 0.005 | 2.84% ± 0.07 | 0.953 ± 0.003 |
| shellcode ⚠ | 440 | 157 | TM two-stage, SMOTE, global (Phase 3 headline; not selected) | 74.2% ± 1.3 [70–78] | 1.4% ± 0.0 | 0.356 ± 0.008 | 2.53% ± 0.01 | 0.967 ± 0.002 |
| shellcode ⚠ | 440 | 157 | RF, Boolean input | 69.9% ± 0.7 [66–74] | 1.3% ± 0.0 | 0.380 ± 0.003 | 2.45% ± 0.00 | 0.901 ± 0.001 |
| shellcode ⚠ | 440 | 157 | XGBoost, Boolean input | 73.0% ± 0.6 [69–77] | 1.4% ± 0.0 | 0.387 ± 0.000 | 2.39% ± 0.01 | 0.963 ± 0.000 |
| shellcode ⚠ | 440 | 157 | RF, continuous input | 96.1% ± 0.5 [94–98] | 1.8% ± 0.0 | 0.444 ± 0.002 | 2.77% ± 0.04 | 0.976 ± 0.000 |
| shellcode ⚠ | 440 | 157 | XGBoost, continuous input | 97.1% ± 0.1 [95–98] | 1.8% ± 0.0 | 0.449 ± 0.001 | 2.39% ± 0.01 | 0.978 ± 0.001 |
| worms ⚠ | 52 | 19 | TM (selected on validation) | 6.4% ± 1.8 [2–16] | 0.0% ± 0.0 | 0.436 ± 0.002 | 2.15% ± 0.03 | 0.764 ± 0.017 |
| worms ⚠ | 52 | 19 | TM, sensitivity: balanced validation objective (post hoc) | 46.8% ± 4.0 [33–59] | 0.1% ± 0.0 | 0.377 ± 0.004 | 2.15% ± 0.03 | 0.859 ± 0.015 |
| worms ⚠ | 52 | 19 | TM two-stage, SMOTE, global (Phase 3 headline; not selected) | 46.8% ± 4.0 [33–59] | 0.1% ± 0.0 | 0.377 ± 0.004 | 2.15% ± 0.03 | 0.859 ± 0.015 |
| worms ⚠ | 52 | 19 | RF, Boolean input | 92.3% ± 1.6 [82–97] | 0.2% ± 0.0 | 0.371 ± 0.001 | 2.46% ± 0.01 | 0.967 ± 0.001 |
| worms ⚠ | 52 | 19 | XGBoost, Boolean input | 92.3% ± 1.6 [82–97] | 0.2% ± 0.0 | 0.396 ± 0.000 | 2.41% ± 0.00 | 0.970 ± 0.000 |
| worms ⚠ | 52 | 19 | RF, continuous input | 71.2% ± 0.0 [58–82] | 0.2% ± 0.0 | 0.459 ± 0.002 | 2.76% ± 0.01 | 0.959 ± 0.001 |
| worms ⚠ | 52 | 19 | XGBoost, continuous input | 98.7% ± 1.8 [90–100] | 0.2% ± 0.0 | 0.456 ± 0.002 | 2.42% ± 0.01 | 0.966 ± 0.000 |

⚠ fewer than 1000 test unknowns: recall rests on few rows; read the interval, not the point.

### Task 3 — compute cost

Hardware: Apple M1, 8 cores (4 performance + 4 efficiency), 8 GB; macOS-26.3-arm64-arm-64bit; numpy 1.26.4, scikit-learn 1.4.2, XGBoost 3.4.1. Fold: dos held out. Energy not measured (needs root on macOS); CPU-seconds are reported instead.

Preprocessing (1 thread), separate from model inference:

| step | rows | µs per row (batch) | µs per row (one at a time, median) | peak memory MB |
|---|---|---|---|---|
| window features, 10 s (00b kernel) | 2540047 | 0.83 | – | 485 |
| Boolean encoding (thresholds + one-hot) | 508048 | 1.65 | 312 | 171 |
| continuous encoding (trees) | 508048 | 0.22 | 116 | 333 |

Individual model, binary task (the normality bank's):

| model | threads | train s (wall) | train CPU s | cores used | train peak MB | batch rows/s | µs/row (batch) | inference peak MB | one-row latency µs (median / p99) | model size KB |
|---|---|---|---|---|---|---|---|---|---|---|
| TM bank | 1 | 9 | 9 | 1.0 | 298 | 45,772 | 21.85 | 191 | 94 / 111 | 30 |
| RF | 1 | 24 | 31 | 1.3 | 37 | 228,153 | 4.38 | 1 | 1069 / 1147 | 10961 |
| RF | 8 | 12 | 87 | 7.1 | 106 | 559,741 | 1.79 | 12 | 16742 / 17160 | 10961 |
| XGBoost | 1 | 16 | 21 | 1.3 | 155 | 227,421 | 4.40 | 0 | 79 / 162 | 1705 |
| XGBoost | 8 | 7 | 52 | 7.5 | 151 | 915,300 | 1.09 | 5 | 103 / 363 | 1705 |

Individual model, multi-class task (Normal + known families):

| model | threads | train s (wall) | train CPU s | cores used | train peak MB | batch rows/s | µs/row (batch) | inference peak MB | one-row latency µs (median / p99) | model size KB |
|---|---|---|---|---|---|---|---|---|---|---|
| TM multi-class | 1 | 99 | 100 | 1.0 | 934 | 10,330 | 96.80 | 16 | 460 / 489 | 142 |
| RF, Boolean | 1 | 42 | 54 | 1.3 | 22 | 170,512 | 5.86 | 1 | 1071 / 1173 | 228058 |
| RF, Boolean | 8 | 19 | 135 | 7.2 | 249 | 351,650 | 2.84 | 24 | 16623 / 17207 | 228058 |
| XGBoost, Boolean | 1 | 218 | 287 | 1.3 | 124 | 21,530 | 46.45 | 182 | 348 / 552 | 21065 |
| XGBoost, Boolean | 8 | 91 | 691 | 7.6 | 52 | 74,939 | 13.34 | 736 | 506 / 1547 | 21065 |
| RF, continuous | 1 | 41 | 53 | 1.3 | 144 | 300,006 | 3.33 | 8 | 1088 / 1225 | 78754 |
| RF, continuous | 8 | 9 | 68 | 7.2 | 235 | 736,325 | 1.36 | 0 | 16658 / 17003 | 78754 |
| XGBoost, continuous | 1 | 34 | 44 | 1.3 | 99 | 42,797 | 23.37 | 0 | 298 / 627 | 7946 |
| XGBoost, continuous | 8 | 23 | 147 | 6.5 | 115 | 155,145 | 6.45 | 0 | 430 / 1733 | 7946 |

Complete open-set pipeline (decision included):

| model | threads | train s (wall) | train CPU s | cores used | train peak MB | batch rows/s | µs/row (batch) | inference peak MB | one-row latency µs (median / p99) | model size KB |
|---|---|---|---|---|---|---|---|---|---|---|
| TM two-stage (OpenSATM: all banks) | 1 | 143 | 184 | 1.3 | 899 | 5,084 | 196.68 | 608 | 927 / 1016 | 293 |
| TM two-stage (OpenSATM: all banks) | 8 | 42 | 253 | 6.0 | 1770 | 9,108 | 109.79 | 955 | 3088 / 11484 | 293 |
| TM single-stage | 1 | 99 | 100 | 1.0 | 934 | 10,330 | 96.80 | 16 | 460 / 489 | 142 |
| RF, Boolean | 1 | 42 | 54 | 1.3 | 22 | 170,512 | 5.86 | 1 | 1071 / 1173 | 228058 |
| RF, Boolean | 8 | 19 | 135 | 7.2 | 249 | 351,650 | 2.84 | 24 | 16623 / 17207 | 228058 |
| XGBoost, Boolean | 1 | 218 | 287 | 1.3 | 124 | 21,530 | 46.45 | 182 | 348 / 552 | 21065 |
| XGBoost, Boolean | 8 | 91 | 691 | 7.6 | 52 | 74,939 | 13.34 | 736 | 506 / 1547 | 21065 |
| RF, continuous | 1 | 41 | 53 | 1.3 | 144 | 300,006 | 3.33 | 8 | 1088 / 1225 | 78754 |
| RF, continuous | 8 | 9 | 68 | 7.2 | 235 | 736,325 | 1.36 | 0 | 16658 / 17003 | 78754 |
| XGBoost, continuous | 1 | 34 | 44 | 1.3 | 99 | 42,797 | 23.37 | 0 | 298 / 627 | 7946 |
| XGBoost, continuous | 8 | 23 | 147 | 6.5 | 115 | 155,145 | 6.45 | 0 | 430 / 1733 | 7946 |

TM inference through tmu itself against the matrix scorer (`tm_fast`, exact, verified against tmu):

| model | tmu rows/s | tmu one-row µs | tm_fast rows/s | tm_fast one-row µs | clauses | included literals |
|---|---|---|---|---|---|---|
| TM bank | 35,979 | 33 | 45,772 | 94 | 2000 | 29635 |
| TM multi-class | 7,552 | 128 | 10,330 | 460 | 9000 | 221803 |

### Task 4 — explanations under common criteria (5% budget, held-out folds, seed 42 as reference)

**why a known class was predicted (known rows, correct)**

| model | rows | tied share | completeness error (median) | fidelity@1 / @5 | features to reproduce (median) | features for 90% of |attribution| | faithfulness: flips, top-3 vs random-3 | top-10 vs random-10 | seed stability (top-5 Jaccard) | perturbation stability (top-5 Jaccard) | decision change under perturbation |
|---|---|---|---|---|---|---|---|---|---|---|---|
| TM (multi-class) | 400 | 0.00 | 0.950 | 1.00 / 1.00 | 1.0 | 18.5 | 0.91 vs 0.07 | 1.00 vs 0.22 | 0.49 | 0.89 | 0.06 |
| RF, Boolean (TreeSHAP) | 400 | 0.00 | 0.000 | 1.00 / 1.00 | 1.0 | 9.5 | 1.00 vs 0.11 | 1.00 vs 0.29 | 0.69 | 0.92 | 0.06 |
| XGBoost, Boolean (TreeSHAP) | 400 | 0.00 | 0.000 | 0.99 / 1.00 | 1.0 | 16.5 | 1.00 vs 0.08 | 1.00 vs 0.24 | 0.73 | 0.90 | 0.05 |
| RF, continuous (TreeSHAP) | 400 | 0.00 | 0.000 | 1.00 / 1.00 | 1.0 | 14.0 | 1.00 vs 0.13 | 1.00 vs 0.45 | 0.92 | 0.96 | 0.04 |
| XGBoost, continuous (TreeSHAP) | 400 | 0.00 | 0.000 | 0.99 / 1.00 | 1.0 | 18.0 | 0.94 vs 0.07 | 1.00 vs 0.21 | 0.77 | 0.93 | 0.01 |

**why a known class was predicted (held-out rows given a known label)**

| model | rows | tied share | completeness error (median) | fidelity@1 / @5 | features to reproduce (median) | features for 90% of |attribution| | faithfulness: flips, top-3 vs random-3 | top-10 vs random-10 | seed stability (top-5 Jaccard) | perturbation stability (top-5 Jaccard) | decision change under perturbation |
|---|---|---|---|---|---|---|---|---|---|---|---|
| TM (multi-class) | 400 | 0.02 | 0.173 | 0.95 / 0.97 | 1.0 | 23.5 | 0.87 vs 0.25 | 0.97 vs 0.47 | 0.54 | 0.92 | 0.10 |
| RF, Boolean (TreeSHAP) | 400 | 0.00 | 0.000 | 1.00 / 1.00 | 1.0 | 25.0 | 0.90 vs 0.40 | 1.00 vs 0.80 | 0.73 | 0.89 | 0.17 |
| XGBoost, Boolean (TreeSHAP) | 400 | 0.00 | 0.000 | 0.77 / 0.95 | 1.0 | 23.5 | 0.91 vs 0.24 | 0.99 vs 0.62 | 0.76 | 0.87 | 0.10 |
| RF, continuous (TreeSHAP) | 400 | 0.00 | 0.000 | 1.00 / 1.00 | 1.0 | 24.0 | 1.00 vs 0.45 | 1.00 vs 0.92 | 0.72 | 0.88 | 0.16 |
| XGBoost, continuous (TreeSHAP) | 302 | 0.00 | 0.000 | 0.78 / 0.94 | 2.0 | 23.5 | 0.99 vs 0.27 | 1.00 vs 0.75 | 0.70 | 0.87 | 0.09 |

**why a row was rejected as Unknown (held-out rows caught)**

| model | rows | tied share | completeness error (median) | fidelity@1 / @5 | features to reproduce (median) | features for 90% of |attribution| | faithfulness: flips, top-3 vs random-3 | top-10 vs random-10 | seed stability (top-5 Jaccard) | perturbation stability (top-5 Jaccard) | decision change under perturbation |
|---|---|---|---|---|---|---|---|---|---|---|---|
| TM (multi-class) | 400 | 0.02 | 0.000 | 0.96 / 0.98 | 1.0 | 26.5 | 0.12 vs 0.02 | 0.14 vs 0.04 | 0.45 | 0.89 | 0.01 |
| RF, Boolean (TreeSHAP) | 400 | 0.00 | 0.000 | 1.00 / 1.00 | 1.0 | 26.0 | 0.06 vs 0.01 | 0.07 vs 0.01 | 0.67 | 0.87 | 0.00 |
| XGBoost, Boolean (TreeSHAP) | 400 | 0.00 | 0.000 | 1.00 / 1.00 | 1.0 | 21.5 | 0.20 vs 0.03 | 0.24 vs 0.04 | 0.61 | 0.88 | 0.00 |
| RF, continuous (TreeSHAP) | 400 | 0.00 | 0.000 | 1.00 / 1.00 | 1.0 | 28.0 | 0.04 vs 0.00 | 0.03 vs 0.00 | 0.61 | 0.88 | 0.00 |
| XGBoost, continuous (TreeSHAP) | 400 | 0.00 | 0.000 | 1.00 / 1.00 | 1.0 | 20.5 | 0.12 vs 0.01 | 0.20 vs 0.04 | 0.56 | 0.85 | 0.01 |

**why a row was rejected as Unknown (known rows rejected)**

| model | rows | tied share | completeness error (median) | fidelity@1 / @5 | features to reproduce (median) | features for 90% of |attribution| | faithfulness: flips, top-3 vs random-3 | top-10 vs random-10 | seed stability (top-5 Jaccard) | perturbation stability (top-5 Jaccard) | decision change under perturbation |
|---|---|---|---|---|---|---|---|---|---|---|---|
| TM (multi-class) | 400 | 0.46 | 0.000 | 0.67 / 0.68 | 10.0 | 26.5 | 0.69 vs 0.17 | 0.71 vs 0.23 | 0.53 | 0.90 | 0.07 |
| RF, Boolean (TreeSHAP) | 400 | 0.00 | 0.000 | 1.00 / 1.00 | 1.0 | 25.5 | 0.12 vs 0.02 | 0.09 vs 0.01 | 0.70 | 0.90 | 0.01 |
| XGBoost, Boolean (TreeSHAP) | 400 | 0.00 | 0.000 | 0.99 / 0.99 | 1.0 | 21.5 | 0.23 vs 0.04 | 0.29 vs 0.08 | 0.65 | 0.85 | 0.01 |
| RF, continuous (TreeSHAP) | 400 | 0.00 | 0.000 | 1.00 / 1.00 | 1.0 | 27.0 | 0.06 vs 0.01 | 0.05 vs 0.01 | 0.62 | 0.90 | 0.00 |
| XGBoost, continuous (TreeSHAP) | 400 | 0.00 | 0.000 | 1.00 / 0.99 | 1.0 | 21.5 | 0.12 vs 0.01 | 0.21 vs 0.03 | 0.61 | 0.92 | 0.01 |

TM clause-level explanations (the model's native unit; unclipped clause votes):

| rows | clause fidelity@1 | @10 | clauses to reproduce (median) | literals per cited clause | clauses firing for the top class |
|---|---|---|---|---|---|
| known_correct | 1.00 | 0.85 | 1 | 2 | 164 |
| unknown_missed | 1.00 | 0.21 | 71 | 45 | 96 |
| unknown_rejected | 1.00 | 1.00 | 1 | 70 | 41 |
| known_rejected | 1.00 | 1.00 | 1 | 43 | 183 |

### Task 6 — open set, CICIDS2017

#### CICIDS2017: configuration chosen on validation (nested over folds)

- **TM**: `single_banks / smote / global` for 5 fold(s) (bot, ddos, dos, portscan, webattack); `single_banks / smote / classcond` for 1 fold(s) (bruteforce)
- **TM (balanced objective, sensitivity)**: `single_banks / smote / classcond` for 5 fold(s) (bot, bruteforce, dos, portscan, webattack); `single_banks / smote / global` for 1 fold(s) (ddos)
- **RF boolean**: `rf / boolean / global` for 5 fold(s) (bot, bruteforce, ddos, portscan, webattack); `rf / boolean / margin` for 1 fold(s) (dos)
- **RF continuous_medium**: `rf / continuous_medium / global` for 6 fold(s)
- **XGB boolean**: `xgb / boolean / global` for 6 fold(s)
- **XGB continuous_medium**: `xgb / continuous_medium / global` for 6 fold(s)

Validation objective J (open-set macro F1 over known classes + Unknown, pseudo-unknowns, 5% budget), mean over all nine folds and seeds:

| TM configuration (architecture | resampling | rule) | mean J |
|---|---|
| single_banks / smote / global | 0.8867 |
| single_banks / smote / margin | 0.8824 |
| single_banks / smote / classcond | 0.8780 |
| two_stage / smote / global | 0.7275 |
| two_stage / smote / margin | 0.7272 |
| two_stage / smote / classcond | 0.7260 |

#### CICIDS2017: 1% validation budget — family average (mean ± std over seeds)

| system | unknown recall | unknown precision | known-attack macro F1 | known rejected | Normal false alarms | AUROC | AUPRC |
|---|---|---|---|---|---|---|---|
| TM (selected on validation) | 38.3% ± 1.4 | 27.7% ± 1.9 | 0.918 ± 0.003 | 1.60% ± 0.05 | 0.75% ± 0.06 | 0.907 ± 0.003 | 0.309 ± 0.005 |
| TM, sensitivity: balanced validation objective (post hoc) | 26.7% ± 2.8 | 24.0% ± 1.8 | 0.895 ± 0.007 | 2.63% ± 0.27 | 0.68% ± 0.04 | 0.841 ± 0.003 | 0.260 ± 0.008 |
| TM two-stage, SMOTE, global (Phase 3 headline; not selected) | 5.4% ± 0.1 | 8.2% ± 2.1 | 0.759 ± 0.008 | 2.40% ± 0.33 | 0.00% ± 0.00 | 0.100 ± 0.005 | 0.047 ± 0.013 |
| RF, Boolean input | 48.7% ± 7.0 | 32.8% ± 4.9 | 0.873 ± 0.010 | 1.54% ± 0.03 | 0.53% ± 0.02 | 0.963 ± 0.007 | 0.368 ± 0.007 |
| XGBoost, Boolean input | 49.9% ± 1.9 | 29.8% ± 0.3 | 0.925 ± 0.002 | 1.30% ± 0.01 | 0.76% ± 0.00 | 0.966 ± 0.001 | 0.319 ± 0.005 |
| RF, continuous input | 17.1% ± 7.4 | 31.2% ± 4.7 | 0.888 ± 0.001 | 0.50% ± 0.02 | 0.57% ± 0.01 | 0.959 ± 0.005 | 0.434 ± 0.015 |
| XGBoost, continuous input | 18.7% ± 1.9 | 25.4% ± 0.5 | 0.929 ± 0.003 | 0.96% ± 0.04 | 0.65% ± 0.01 | 0.963 ± 0.002 | 0.319 ± 0.009 |

Micro-averaged over the 6 folds (counts pooled across folds, then the rate; AUROC/AUPRC are not pooled because different folds' scores are not on one scale):

| system | unknown recall | unknown precision | known-attack macro F1 | known rejected | Normal false alarms |
|---|---|---|---|---|---|
| TM (selected on validation) | 64.6% ± 9.0 | 58.7% ± 4.3 | 0.907 ± 0.015 | 1.61% ± 0.05 | 0.75% ± 0.06 |
| TM, sensitivity: balanced validation objective (post hoc) | 54.3% ± 4.8 | 42.3% ± 0.5 | 0.886 ± 0.013 | 2.66% ± 0.29 | 0.68% ± 0.04 |
| TM two-stage, SMOTE, global (Phase 3 headline; not selected) | 7.3% ± 0.1 | 9.9% ± 1.3 | 0.761 ± 0.007 | 2.42% ± 0.34 | 0.00% ± 0.00 |
| RF, Boolean input | 64.7% ± 8.7 | 59.8% ± 2.8 | 0.879 ± 0.009 | 1.54% ± 0.03 | 0.53% ± 0.02 |
| XGBoost, Boolean input | 53.5% ± 2.1 | 59.6% ± 1.2 | 0.925 ± 0.002 | 1.30% ± 0.01 | 0.76% ± 0.00 |
| RF, continuous input | 24.4% ± 8.4 | 62.5% ± 7.6 | 0.897 ± 0.001 | 0.48% ± 0.02 | 0.57% ± 0.01 |
| XGBoost, continuous input | 27.9% ± 0.5 | 51.1% ± 0.6 | 0.914 ± 0.007 | 0.96% ± 0.04 | 0.65% ± 0.01 |

#### CICIDS2017: 5% validation budget — family average (mean ± std over seeds)

| system | unknown recall | unknown precision | known-attack macro F1 | known rejected | Normal false alarms | AUROC | AUPRC |
|---|---|---|---|---|---|---|---|
| TM (selected on validation) | 76.1% ± 1.2 | 29.3% ± 1.7 | 0.842 ± 0.008 | 3.53% ± 0.64 | 2.04% ± 0.68 | 0.907 ± 0.003 | 0.309 ± 0.005 |
| TM, sensitivity: balanced validation objective (post hoc) | 75.3% ± 1.6 | 27.1% ± 0.6 | 0.858 ± 0.009 | 4.47% ± 0.31 | 1.90% ± 0.56 | 0.841 ± 0.003 | 0.260 ± 0.008 |
| TM two-stage, SMOTE, global (Phase 3 headline; not selected) | 5.7% ± 0.2 | 4.0% ± 0.7 | 0.642 ± 0.008 | 5.74% ± 0.22 | 0.00% ± 0.00 | 0.100 ± 0.005 | 0.047 ± 0.013 |
| RF, Boolean input | 91.4% ± 5.0 | 30.0% ± 1.6 | 0.783 ± 0.012 | 5.89% ± 0.48 | 1.99% ± 0.06 | 0.963 ± 0.007 | 0.368 ± 0.007 |
| XGBoost, Boolean input | 65.1% ± 2.7 | 27.3% ± 0.1 | 0.913 ± 0.002 | 2.32% ± 0.04 | 1.39% ± 0.05 | 0.966 ± 0.001 | 0.319 ± 0.005 |
| RF, continuous input | 47.1% ± 13.4 | 43.4% ± 5.9 | 0.872 ± 0.002 | 0.90% ± 0.11 | 0.82% ± 0.09 | 0.959 ± 0.005 | 0.434 ± 0.015 |
| XGBoost, continuous input | 49.2% ± 3.9 | 27.0% ± 0.4 | 0.935 ± 0.003 | 2.03% ± 0.02 | 1.34% ± 0.05 | 0.963 ± 0.002 | 0.319 ± 0.009 |

Micro-averaged over the 6 folds (counts pooled across folds, then the rate; AUROC/AUPRC are not pooled because different folds' scores are not on one scale):

| system | unknown recall | unknown precision | known-attack macro F1 | known rejected | Normal false alarms |
|---|---|---|---|---|---|
| TM (selected on validation) | 81.0% ± 1.9 | 45.4% ± 4.1 | 0.855 ± 0.010 | 3.54% ± 0.63 | 2.04% ± 0.68 |
| TM, sensitivity: balanced validation objective (post hoc) | 80.1% ± 2.2 | 39.0% ± 1.9 | 0.863 ± 0.013 | 4.50% ± 0.32 | 1.90% ± 0.56 |
| TM two-stage, SMOTE, global (Phase 3 headline; not selected) | 7.7% ± 0.3 | 4.6% ± 0.1 | 0.644 ± 0.007 | 5.76% ± 0.23 | 0.00% ± 0.00 |
| RF, Boolean input | 99.9% ± 0.0 | 37.8% ± 1.8 | 0.791 ± 0.013 | 5.91% ± 0.48 | 1.99% ± 0.06 |
| XGBoost, Boolean input | 67.1% ± 4.9 | 50.9% ± 1.4 | 0.913 ± 0.002 | 2.31% ± 0.03 | 1.39% ± 0.05 |
| RF, continuous input | 67.6% ± 21.1 | 72.2% ± 4.6 | 0.880 ± 0.003 | 0.88% ± 0.11 | 0.82% ± 0.09 |
| XGBoost, continuous input | 56.5% ± 1.8 | 50.1% ± 0.6 | 0.935 ± 0.004 | 2.01% ± 0.02 | 1.34% ± 0.05 |

#### CICIDS2017: 10% validation budget — family average (mean ± std over seeds)

| system | unknown recall | unknown precision | known-attack macro F1 | known rejected | Normal false alarms | AUROC | AUPRC |
|---|---|---|---|---|---|---|---|
| TM (selected on validation) | 78.5% ± 1.3 | 29.1% ± 2.0 | 0.837 ± 0.009 | 4.36% ± 1.17 | 2.78% ± 1.28 | 0.907 ± 0.003 | 0.309 ± 0.005 |
| TM, sensitivity: balanced validation objective (post hoc) | 78.0% ± 1.5 | 26.0% ± 0.9 | 0.838 ± 0.009 | 5.70% ± 0.73 | 2.56% ± 1.07 | 0.841 ± 0.003 | 0.260 ± 0.008 |
| TM two-stage, SMOTE, global (Phase 3 headline; not selected) | 6.0% ± 0.4 | 2.6% ± 0.3 | 0.446 ± 0.007 | 9.87% ± 0.09 | 0.00% ± 0.00 | 0.100 ± 0.005 | 0.047 ± 0.013 |
| RF, Boolean input | 92.5% ± 4.6 | 26.5% ± 1.5 | 0.771 ± 0.007 | 7.18% ± 0.45 | 3.01% ± 0.11 | 0.963 ± 0.007 | 0.368 ± 0.007 |
| XGBoost, Boolean input | 72.7% ± 0.6 | 27.1% ± 0.6 | 0.910 ± 0.001 | 2.68% ± 0.06 | 1.58% ± 0.05 | 0.966 ± 0.001 | 0.319 ± 0.005 |
| RF, continuous input | 65.1% ± 1.5 | 45.0% ± 1.3 | 0.832 ± 0.012 | 1.33% ± 0.13 | 1.04% ± 0.17 | 0.959 ± 0.005 | 0.434 ± 0.015 |
| XGBoost, continuous input | 63.2% ± 2.5 | 26.9% ± 0.8 | 0.928 ± 0.001 | 2.72% ± 0.05 | 1.58% ± 0.05 | 0.963 ± 0.002 | 0.319 ± 0.009 |

Micro-averaged over the 6 folds (counts pooled across folds, then the rate; AUROC/AUPRC are not pooled because different folds' scores are not on one scale):

| system | unknown recall | unknown precision | known-attack macro F1 | known rejected | Normal false alarms |
|---|---|---|---|---|---|
| TM (selected on validation) | 83.5% ± 2.7 | 41.5% ± 5.7 | 0.851 ± 0.010 | 4.37% ± 1.16 | 2.78% ± 1.28 |
| TM, sensitivity: balanced validation objective (post hoc) | 82.6% ± 3.0 | 34.3% ± 2.7 | 0.843 ± 0.013 | 5.72% ± 0.72 | 2.56% ± 1.07 |
| TM two-stage, SMOTE, global (Phase 3 headline; not selected) | 8.1% ± 0.6 | 2.9% ± 0.2 | 0.453 ± 0.006 | 9.89% ± 0.10 | 0.00% ± 0.00 |
| RF, Boolean input | 99.9% ± 0.0 | 33.4% ± 1.4 | 0.778 ± 0.008 | 7.16% ± 0.45 | 3.01% ± 0.11 |
| XGBoost, Boolean input | 75.6% ± 1.2 | 50.4% ± 0.8 | 0.910 ± 0.001 | 2.67% ± 0.06 | 1.58% ± 0.05 |
| RF, continuous input | 91.9% ± 4.3 | 71.7% ± 1.0 | 0.835 ± 0.016 | 1.31% ± 0.12 | 1.04% ± 0.17 |
| XGBoost, continuous input | 72.3% ± 5.6 | 48.7% ± 2.5 | 0.929 ± 0.001 | 2.72% ± 0.05 | 1.58% ± 0.05 |

#### CICIDS2017: per held-out family, 5% budget (mean ± std over seeds)

| held out | test unknowns | val pseudo-unknowns | system | unknown recall [95% CI] | unknown precision | known-attack F1 | Normal FA | AUROC |
|---|---|---|---|---|---|---|---|---|
| bot ⚠ | 148 | 59 | TM (selected on validation) | 58.1% ± 0.0 [50–66] | 0.6% ± 0.1 | 0.791 ± 0.014 | 1.98% ± 0.79 | 0.775 ± 0.009 |
| bot ⚠ | 148 | 59 | TM, sensitivity: balanced validation objective (post hoc) | 58.1% ± 0.0 [50–66] | 0.4% ± 0.1 | 0.842 ± 0.023 | 1.87% ± 0.71 | 0.620 ± 0.008 |
| bot ⚠ | 148 | 59 | TM two-stage, SMOTE, global (Phase 3 headline; not selected) | 0.0% ± 0.0 [0–3] | 0.0% ± 0.0 | 0.652 ± 0.003 | 0.00% ± 0.00 | 0.129 ± 0.028 |
| bot ⚠ | 148 | 59 | RF, Boolean input | 49.8% ± 30.8 [42–58] | 0.2% ± 0.1 | 0.725 ± 0.044 | 1.68% ± 0.28 | 0.858 ± 0.028 |
| bot ⚠ | 148 | 59 | XGBoost, Boolean input | 34.9% ± 5.4 [28–43] | 0.6% ± 0.1 | 0.905 ± 0.004 | 0.94% ± 0.05 | 0.932 ± 0.002 |
| bot ⚠ | 148 | 59 | RF, continuous input | 0.0% ± 0.0 [0–3] | 0.0% ± 0.0 | 0.932 ± 0.010 | 0.62% ± 0.01 | 0.833 ± 0.033 |
| bot ⚠ | 148 | 59 | XGBoost, continuous input | 0.0% ± 0.0 [0–3] | 0.0% ± 0.0 | 0.918 ± 0.002 | 0.91% ± 0.03 | 0.866 ± 0.006 |
| bruteforce | 1391 | 556 | TM (selected on validation) | 86.7% ± 11.9 [85–88] | 5.0% ± 0.2 | 0.817 ± 0.013 | 1.40% ± 0.23 | 0.789 ± 0.001 |
| bruteforce | 1391 | 556 | TM, sensitivity: balanced validation objective (post hoc) | 86.7% ± 11.9 [85–88] | 5.0% ± 0.2 | 0.817 ± 0.013 | 1.40% ± 0.23 | 0.789 ± 0.001 |
| bruteforce | 1391 | 556 | TM two-stage, SMOTE, global (Phase 3 headline; not selected) | 0.0% ± 0.0 [-0–0] | 0.0% ± 0.0 | 0.631 ± 0.016 | 0.00% ± 0.00 | 0.017 ± 0.001 |
| bruteforce | 1391 | 556 | RF, Boolean input | 99.9% ± 0.0 [99–100] | 6.4% ± 2.0 | 0.848 ± 0.027 | 1.49% ± 0.16 | 0.992 ± 0.001 |
| bruteforce | 1391 | 556 | XGBoost, Boolean input | 89.7% ± 0.8 [88–91] | 12.8% ± 0.2 | 0.907 ± 0.000 | 0.84% ± 0.02 | 0.989 ± 0.001 |
| bruteforce | 1391 | 556 | RF, continuous input | 73.0% ± 15.5 [71–75] | 36.7% ± 6.3 | 0.862 ± 0.009 | 0.70% ± 0.04 | 0.996 ± 0.001 |
| bruteforce | 1391 | 556 | XGBoost, continuous input | 92.0% ± 6.1 [90–93] | 13.4% ± 0.9 | 0.900 ± 0.011 | 0.88% ± 0.03 | 0.986 ± 0.003 |
| ddos | 19025 | 7610 | TM (selected on validation) | 16.1% ± 10.4 [16–17] | 19.5% ± 11.0 | 0.790 ± 0.007 | 1.78% ± 0.27 | 0.915 ± 0.027 |
| ddos | 19025 | 7610 | TM, sensitivity: balanced validation objective (post hoc) | 16.1% ± 10.4 [16–17] | 19.5% ± 11.0 | 0.790 ± 0.007 | 1.78% ± 0.27 | 0.915 ± 0.027 |
| ddos | 19025 | 7610 | TM two-stage, SMOTE, global (Phase 3 headline; not selected) | 33.3% ± 1.3 [33–34] | 22.0% ± 4.4 | 0.597 ± 0.034 | 0.00% ± 0.00 | 0.377 ± 0.039 |
| ddos | 19025 | 7610 | RF, Boolean input | 100.0% ± 0.0 [100–100] | 48.1% ± 11.7 | 0.725 ± 0.035 | 1.36% ± 0.18 | 0.980 ± 0.000 |
| ddos | 19025 | 7610 | XGBoost, Boolean input | 0.1% ± 0.0 [0–0] | 0.3% ± 0.0 | 0.907 ± 0.005 | 1.33% ± 0.03 | 0.902 ± 0.004 |
| ddos | 19025 | 7610 | RF, continuous input | 66.8% ± 46.7 [66–67] | 60.5% ± 38.2 | 0.871 ± 0.012 | 0.66% ± 0.04 | 0.995 ± 0.001 |
| ddos | 19025 | 7610 | XGBoost, continuous input | 0.3% ± 0.0 [0–0] | 0.9% ± 0.1 | 0.946 ± 0.007 | 0.99% ± 0.02 | 0.970 ± 0.005 |
| dos | 34356 | 13743 | TM (selected on validation) | 99.3% ± 0.5 [99–99] | 87.7% ± 3.0 | 0.851 ± 0.038 | 1.46% ± 0.38 | 0.999 ± 0.000 |
| dos | 34356 | 13743 | TM, sensitivity: balanced validation objective (post hoc) | 97.2% ± 1.1 [97–97] | 76.3% ± 3.4 | 0.931 ± 0.004 | 1.38% ± 0.32 | 0.985 ± 0.001 |
| dos | 34356 | 13743 | TM two-stage, SMOTE, global (Phase 3 headline; not selected) | 0.9% ± 0.3 [1–1] | 1.8% ± 0.8 | 0.582 ± 0.004 | 0.00% ± 0.00 | 0.034 ± 0.014 |
| dos | 34356 | 13743 | RF, Boolean input | 100.0% ± 0.0 [100–100] | 68.2% ± 10.4 | 0.744 ± 0.029 | 1.45% ± 0.29 | 1.000 ± 0.000 |
| dos | 34356 | 13743 | XGBoost, Boolean input | 83.0% ± 10.1 [83–83] | 89.1% ± 0.9 | 0.907 ± 0.003 | 0.96% ± 0.03 | 0.994 ± 0.001 |
| dos | 34356 | 13743 | RF, continuous input | 56.1% ± 15.0 [56–57] | 96.9% ± 1.3 | 0.810 ± 0.001 | 0.62% ± 0.01 | 0.996 ± 0.001 |
| dos | 34356 | 13743 | XGBoost, continuous input | 52.8% ± 3.0 [52–53] | 86.6% ± 0.4 | 0.967 ± 0.003 | 0.79% ± 0.01 | 0.994 ± 0.000 |
| portscan | 31831 | 12732 | TM (selected on validation) | 99.8% ± 0.1 [100–100] | 62.9% ± 11.3 | 0.827 ± 0.052 | 3.92% ± 2.88 | 0.977 ± 0.004 |
| portscan | 31831 | 12732 | TM, sensitivity: balanced validation objective (post hoc) | 99.8% ± 0.1 [100–100] | 61.4% ± 8.6 | 0.853 ± 0.009 | 3.39% ± 2.28 | 0.914 ± 0.018 |
| portscan | 31831 | 12732 | TM two-stage, SMOTE, global (Phase 3 headline; not selected) | 0.0% ± 0.0 [0–0] | 0.0% ± 0.0 | 0.596 ± 0.006 | 0.00% ± 0.00 | 0.011 ± 0.003 |
| portscan | 31831 | 12732 | RF, Boolean input | 100.0% ± 0.0 [100–100] | 57.0% ± 0.6 | 0.758 ± 0.001 | 4.58% ± 0.17 | 0.976 ± 0.002 |
| portscan | 31831 | 12732 | XGBoost, Boolean input | 88.9% ± 2.8 [89–89] | 60.5% ± 0.5 | 0.883 ± 0.002 | 3.26% ± 0.21 | 0.982 ± 0.000 |
| portscan | 31831 | 12732 | RF, continuous input | 80.7% ± 15.4 [80–81] | 65.4% ± 1.9 | 0.768 ± 0.001 | 1.71% ± 0.44 | 0.973 ± 0.001 |
| portscan | 31831 | 12732 | XGBoost, continuous input | 92.9% ± 5.0 [93–93] | 60.4% ± 1.1 | 0.888 ± 0.002 | 3.61% ± 0.36 | 0.968 ± 0.000 |
| webattack ⚠ | 38 | 16 | TM (selected on validation) | 96.5% ± 2.5 [87–100] | 0.3% ± 0.1 | 0.974 ± 0.002 | 1.69% ± 0.72 | 0.990 ± 0.004 |
| webattack ⚠ | 38 | 16 | TM, sensitivity: balanced validation objective (post hoc) | 93.9% ± 4.5 [83–99] | 0.2% ± 0.0 | 0.917 ± 0.035 | 1.59% ± 0.69 | 0.823 ± 0.041 |
| webattack ⚠ | 38 | 16 | TM two-stage, SMOTE, global (Phase 3 headline; not selected) | 0.0% ± 0.0 [0–9] | 0.0% ± 0.0 | 0.792 ± 0.030 | 0.00% ± 0.00 | 0.033 ± 0.005 |
| webattack ⚠ | 38 | 16 | RF, Boolean input | 99.1% ± 1.2 [91–100] | 0.1% ± 0.0 | 0.901 ± 0.004 | 1.37% ± 0.16 | 0.970 ± 0.016 |
| webattack ⚠ | 38 | 16 | XGBoost, Boolean input | 93.9% ± 1.2 [83–99] | 0.4% ± 0.0 | 0.968 ± 0.001 | 0.99% ± 0.03 | 0.995 ± 0.000 |
| webattack ⚠ | 38 | 16 | RF, continuous input | 6.1% ± 2.5 [1–17] | 0.7% ± 0.6 | 0.986 ± 0.002 | 0.63% ± 0.01 | 0.959 ± 0.003 |
| webattack ⚠ | 38 | 16 | XGBoost, continuous input | 57.0% ± 20.6 [42–72] | 0.6% ± 0.1 | 0.989 ± 0.000 | 0.86% ± 0.04 | 0.992 ± 0.001 |

⚠ fewer than 1000 test unknowns: recall rests on few rows; read the interval, not the point.
