# Phase 3 — Tuning, open-set analysis, fusion and baselines

Produced 2026-11-01 to 2026-11-02 on the corrected chronological pipeline
(Phase 1). Every number in the tables at the end is generated from the run
outputs by `scripts/23_phase3_tables.py`; the summaries below quote from
those tables.

Throughout, the held-out ("unknown") attack family is removed from **fit and
validation** before any encoder, threshold, calibration or hyperparameter is
chosen (`09_openset_preprocess.py`, asserted in `10` and `17`), and appears
only in test.

## Infrastructure changes this phase

- **Faster, verified scoring (`tm_fast.py`).** A trained clause bank is just a
  set of included literals, so a whole matrix of rows can be scored with one
  matrix product. Every run checks it against tmu's own prediction path,
  clause by clause; they agree exactly. Together with training the nine banks
  in parallel this cut a run from ~18 to ~5 minutes.
- **Reproducible training.** tmu keeps random state inside its compiled code
  that numpy's seed does not reset. When worker processes were reused, a bank's
  result depended on which bank that worker had trained before, so the same
  configuration and seed gave different scores (0.4176 vs 0.4124). Each model
  now trains in a fresh process, and repeated runs reproduce exactly.
- **Tie-breaking.** Bank margins are integers that saturate at ±2T, so a
  quantile threshold often lands on a value shared by many rows. Without
  handling this, strategies silently rejected fewer rows than their budget and
  the tuning objective scored small-T configurations as degenerate (macro F1
  0.0976 = "everything Normal"). A seeded jitter far below the integer gap now
  breaks ties without reordering distinct scores; every strategy now meets its
  validation budget exactly (1.00%, 5.00%, 10.00%).

## Task 1 — retuning on corrected validation, five seeds

Search (`17_tune_tm.py`): clauses ∈ {1000, 2000, 4000} × T/clauses ∈ {0.05,
0.10, 0.20}, then s ∈ {2.5, 5, 10}, then epochs ∈ {1, 3, 5}, scored on the full
validation partition. Within 0.005 macro F1 (about one Worms validation row's
worth) the cheaper configuration is kept. The open-set banks are tuned
**separately for each held-out family**, with that family absent from the data
the search sees; the objective is known-class macro F1 of the Step 5 decision,
since the held-out family cannot be used to score novelty.

Every search chose **1000 clauses**, almost always T = 50, with 1–5 epochs
depending on the family (table below). The closed-set TM chose 1000 clauses,
T = 50, s = 10, 1 epoch.

**Closed-set TM, five seeds:** calibrated macro F1 0.462–0.472 per scale
(short 0.4724 ± 0.0029), multi-scale ensemble 0.4692 ± 0.0048, accuracy
0.954–0.955. Retuning did **not** raise performance relative to Phase 1
(≈ 0.466–0.470); what it changed is model size (half the clauses for the same
result) and it provides error bars: seed-to-seed spread is ±0.003–0.010 macro
F1, so single-run differences smaller than that in earlier phases were noise.

One earlier assumption was overturned during tuning: with the old default
configuration, macro F1 rose from 0.428 (1 epoch) to 0.444 (5 epochs) even
though accuracy had plateaued, confirming the caveat that "one epoch is enough"
had only been checked on accuracy. Under the tuned closed-set configuration
extra epochs did not help (0.432 at 1 epoch, 0.420 at 3, 0.419 at 5); for the
banks it varied by family (Backdoor and Exploits selected 5, Reconnaissance 3,
the rest 1), so epochs are reported as selected per family.

## Task 2 — where unknown attacks go, and why

At the reference operating point (global thresholds, 5% validation
false-rejection, medium scale, five seeds), held-out rows split into (i)
Unknown, (ii) a known family, (iii) Normal as shown in the Task 2 table. Score
distributions for every family are in `figures/phase3/scores_<family>.png`.

The plots and two separability statistics explain the failures:

- **Benign support.** For most families the held-out rows look exactly as
  non-benign as known attacks (AUROC ≈ 0.5 between the two), so very few leak
  to Normal: DoS 2.0%, Backdoor 1.1%, Reconnaissance 0.8%, Shellcode 0.5%.
  **Generic** (AUROC 0.966) and **Fuzzers** (0.621) look *more* benign than
  known attacks; 51% of Generic and 32% of Fuzzers clear the benign gate and
  are called Normal, where no unknown-attack rule can see them.
- **Family support and margin.** For DoS, Analysis, Backdoor, Exploits and
  Worms the held-out rows reach high family scores but with a **small gap
  between the best and second-best family** (top-2 margin AUROC 0.08–0.11:
  their margins are smaller than known attacks'). The model is undecided
  between sibling families — the {DoS, Analysis, Backdoor} confusion found in
  Phase 2 — which is precisely what a margin-based rule can detect.
- **Generic is the opposite case.** Gated Generic rows receive a family score
  and a top-2 margin *as large as or larger than* genuine known attacks (margin
  AUROC 0.958). A known bank claims Generic confidently, so no confidence-based
  rule can flag it; it is also the most seed-sensitive result (recall 26% ±
  17%).

## Task 3 — four novelty-threshold strategies

All four are calibrated on validation only, share the benign gate where they
use one, and are held to the same budget: the share of **all known validation
rows** predicted Unknown. Note this denominator differs from Phase 2's (attack-
looking rows only), so a 5% budget here is much looser than Phase 2's 5%.

Family average, five seeds, 5% budget:

| strategy | unknown recall | held-out → Normal | benign FPR | known macro F1 |
|---|---|---|---|---|
| closed (no Unknown) | 0% | 11.6% | 2.08% | 0.4770 |
| global (existing) | 62.3% ± 2.0% | 11.6% | 2.08% | 0.4094 |
| class-conditional | 16.8% ± 2.3% | 11.6% | 2.08% | 0.4563 |
| confidence margin | 70.1% ± 0.7% | 11.6% | 2.08% | 0.3672 |
| combined scoring | 23.5% ± 0.8% | 0.0% | 7.01% | 0.4614 |

**Every gain in unknown detection costs known-class accuracy.** At 5%, rejecting
~4.5% of known test traffic removes roughly a quarter of known-attack rows from
their correct class; known macro F1 falls from 0.477 to 0.409 (global) or 0.367
(margin). At a 1% budget the cost nearly vanishes (known macro F1 ≈ 0.50 for
global) but recall drops to 0.2–23% per family.

**No strategy wins across families** (1% budget): the margin rule is best where
the held-out family is confused with siblings (Analysis 32%, Backdoor 23%, DoS
31%, Worms 27%); class-conditional thresholds are the only thing that catches
Generic (23%); combined scoring is best for Fuzzers (28%) and is the only
strategy that stops held-out attacks being labelled Normal — at the price of
3× more false alarms on benign traffic (7.0% vs 2.1%). The right rule depends
on how a family fails, which a single global threshold cannot express.

## Task 4 — temporal scales and fusion

Banks for all three scales (three seeds each) were combined by mean, max, the
Phase 2 margin-weighted rule, and two revised adaptive rules that weight each
bank's scales by their validation reliability (AUROC on known classes).

Family average, global thresholds, 5% budget:

| rule | unknown recall | held-out → Normal | AUROC |
|---|---|---|---|
| short | 61.1% ± 2.9% | 14.1% | 0.821 |
| medium | 62.4% ± 1.2% | 12.9% | 0.836 |
| **long** | **69.5% ± 3.2%** | **7.6%** | **0.889** |
| mean | 68.4% ± 4.4% | 14.2% | 0.830 |
| max | 58.5% ± 2.3% | 12.7% | 0.835 |
| adaptive (margin, Phase 2) | 67.1% ± 4.3% | 14.8% | 0.824 |
| adaptive (reliability, revised) | 68.5% ± 4.5% | 14.1% | 0.831 |
| select most reliable scale | 69.6% ± 4.1% | 8.2% | 0.884 |

- **Scales are complementary.** The union of rows caught by any scale exceeds
  the best single scale by 3–15 points (Fuzzers 53% → 66%, Reconnaissance 47%
  → 61%, Worms 39% → 55%), and the best scale differs by family (long for
  five, medium for two, short for two). **Generic is detectable only at the
  long scale** (76.5% vs 14–19%), with near-zero rank correlation between the
  scales' novelty scores (Spearman 0.08–0.23).
- **Max fusion under-performs.** With tuned banks, max is worse than any
  single scale under global thresholds; Phase 2's preference for max (one
  family, untuned, one seed) did not hold up.
- **The revised adaptive rules give no measurable improvement over the
  simplest strong option.** The first versions standardised scores over
  validation, which is ~78% Normal, and collapsed (3.6% and 9.0% recall); with
  that flaw removed, selecting each bank's most reliable scale recovers
  69.6% — but this is essentially the long-scale branch rediscovered (long
  alone: 69.5%), and it matches long under every strategy (69.7 vs 70.3,
  22.1 vs 22.8, 26.7 vs 25.7).

Following the task's rule, the adaptive mechanism is **not retained**; the
simpler approach is preferred, and on this evidence that is the **long-scale
branch alone (or mean fusion)**, not maximum-score fusion. The headroom shown
by the union of scales remains unexploited by any rule tested.

## Task 5 — all nine families, including Worms

Medium scale, global thresholds, 5% budget, five seeds (full table below):
average unknown recall **62.3% ± 2.0%**, AUROC 0.848, AUPRC 0.133,
false-unknown 4.49%, benign FPR 2.08%, known macro F1 0.409 (0.477 without the
Unknown option).

Per family, recall ranges from Backdoor 92% and DoS 87% down to Fuzzers 47%,
Reconnaissance 41%, Worms 37% and Generic 26%. AUPRC is low throughout because
held-out rows are a tiny share of test traffic (Worms 0.01%, AUPRC 0.001;
Generic 13%, AUPRC 0.26) — it should be read against that base rate.

**Small families must be read cautiously.** Worms has 52 test rows and 19
validation rows; its 95% interval on recall from a single seed spans roughly
15–38%, and its seed-to-seed spread is ±9 points. Analysis (678), Backdoor (702)
and Shellcode (440) carry intervals of several points as well. Generic's ±17
point spread across seeds comes from the benign gate admitting a large, seed-
dependent share of its 68,460 rows.

## Task 6 — open-set baselines

`20_openset_baselines.py` rebuilds every held-out family's partitions exactly
as the TM's are built (same ordering, same removal from fit and validation,
same validation cut-off — asserted against the TM dataset), calibrates each
method's novelty threshold on validation to the same false-rejection budgets,
and reports the same metrics. RF and XGBoost hyperparameters are re-selected
per held-out family on validation; stochastic methods use the same five seeds.

Family average, 5% budget:

| method | unknown recall | held-out → Normal | benign FPR | known macro F1 | AUROC |
|---|---|---|---|---|---|
| TM framework (global) | 62.3% ± 2.0% | 11.6% | 2.08% | 0.409 | 0.848 |
| RF, max-probability threshold | **87.8% ± 0.4%** | 0.1% | 2.83% | 0.507 | 0.972 |
| XGBoost, max-probability threshold | 86.8% ± 0.5% | 2.1% | 2.47% | **0.518** | **0.973** |
| Isolation Forest gate + RF | 21.4% ± 2.6% | 4.0% | 16.25% | 0.519 | 0.873 |
| Mahalanobis distance | 20.2% | 0.0% | 6.63% | 0.390 | 0.708 |

**The TM framework is outperformed by simple tree ensembles.** RF and XGBoost
with max-probability thresholding detect about 25 points more of the held-out
attacks, classify known classes better, and almost never send a held-out attack
to Normal, at the same validation false-rejection budget. They beat the TM on
every family, by most where the TM fails: Generic (98.7% vs 26.1%), Fuzzers
(87.7% vs 47.3%), Reconnaissance (64.0% vs 40.9%).

Two qualifications:

- **Representation.** The baselines use the continuous features natively; the
  TM must work from binarised literals. Earlier work in this project showed
  binarisation alone costs tree models several points of macro F1, so part of
  the gap is the encoding rather than the TM.
- **The two-stage design costs on its own.** Placing a Random Forest behind an
  Isolation Forest benign gate — the same structure as the TM framework —
  drops it from 88% to 21% recall and raises benign false positives to 16%.
  A separate benign-rejection stage is a liability here regardless of the
  model behind it; a single model that treats Normal as one more class and
  thresholds its confidence does far better.

The distance-based baseline (shared-covariance Mahalanobis on log-scaled
features) is weaker than the TM, as is the Isolation-Forest-gated RF, so the TM
is not the weakest method; but it is clearly behind the best standard
approaches. The case for the TM therefore rests on interpretability, examined
next.

### Explanations of the full open-set decision

`21_openset_explain.py` explains every three-way decision (Normal / known
family / Unknown) by the clauses that carried it: the normality bank's clauses
for or against "benign", and the best family bank's clauses for or against
that family, set against τ_B and τ_A. Clauses are ranked by how well they
separate their bank's class on validation (held-out family absent). The
analysis uses the medium scale, seed 42, global thresholds at a 5% budget, and
splits test rows into five groups: Normal, known attacks, held-out rows caught
as Unknown, and held-out rows missed (sent to a known family or to Normal).

A worked example, DoS held out, a DoS row caught as Unknown:

```
benign margin -100 vs tau_B -95            -> not benign
  against benign: sttl>Q25 AND sttl>Q50 AND medium_failed_ratio>Q10 AND NOT proto_icmp ...
best family: Exploits, margin 64 vs tau_A 100 -> no family claims it -> UNKNOWN
  against Exploits: sttl>Q90
  for Exploits:     NOT ct_srv_dst>Q10
```

**Fidelity.** Every bank is cut to its k highest-ranked clauses, the
thresholds are re-calibrated for the cut model, and we measure how often the
cut model makes the same decision as the full model.

- With **one clause per bank**, agreement over the whole test set is 82–93%
  (Generic lowest), and about 97% on Normal rows. A handful of clauses
  reproduces most decisions.
- Agreement is lower for known attacks (30–79% at k = 1, 71–86% at k = 100),
  and for held-out rows caught as Unknown (18–70% at k = 1, 28–95% at
  k = 100). The attack-side decisions depend on many clauses at once.
- **Missed unknowns are explained worst.** For held-out rows sent to a known
  family, agreement is mostly 0–35% at every k. These rows sit right at τ_A,
  so the slightest change to the model moves them. For held-out rows sent to
  Normal, results swing between about 0% and 84% as k changes. These groups
  are small: 2–116 rows, except Fuzzers (1,823) and Generic (31,426). They
  should not be read as trends.

**Stability.** For 150 rows per group, one literal is flipped (5 copies per
row). We measure how much the set of cited clauses changes (Jaccard overlap)
and how often the decision changes.

| group | Jaccard | decision change |
|---|---|---|
| Normal | 0.98–0.99 | 0–1.3% |
| known attack | 0.81–0.94 | 0.9–8.4% |
| held-out, caught as Unknown | 0.70–0.86 | 0–2% (Fuzzers 15.5%, Generic 45.6%) |
| held-out, sent to a known family | 0.66–0.94 | 13–78% |
| held-out, sent to Normal | 0.84–0.99 | 0–2.9% |

Explanations and decisions are stable for Normal, known attacks and most
caught unknowns. Held-out rows sent to a known family are fragile: one flipped
literal changes the decision for 13–78% of them. This matches the fidelity
result: misclassified unknowns are borderline cases, not confident errors.
For Generic, even the caught unknowns are fragile (45.6%), consistent with
Generic's weak and seed-dependent recall (26.1% ± 17.1%).

(An earlier run of this measure gave 74–100% decision changes. That came from
scoring the perturbed copies without the tie-break jitter that the
thresholds were set with, which flipped rows tied at the saturated margin.
It was fixed and checked: rescoring rows with no literal flipped reproduces
100% of decisions.)

**Complexity.** An explanation cites about 5 clauses for a Normal decision
and 10 for an attack or Unknown decision. The median cited clause has 1–5
literals for Normal and known attacks, but longer ones (9–37 literals) for
caught unknowns in Analysis, DoS, Exploits, Shellcode and Generic. The
explanations stay short in clause count, but some individual clauses are long
enough that a person would need to read them carefully.

**Summary for Task 6.** The TM's open-set decisions can be explained with a
few clauses. The explanations are faithful and stable for Normal traffic,
known attacks and caught unknowns, and weakest exactly where the system
fails: unknowns misrouted to a known family. That is useful in itself,
because it flags those decisions as borderline. Performance trails RF and
XGBoost by a wide margin, so the TM's case is interpretability, not
accuracy.

## Limitation: saturated margins and tie-breaking

With T = 50, a bank's margin is an integer in [−100, 100], and many rows sit
exactly at ±100. At a 5% budget the global τ_A falls on that saturated value,
100, in every T = 50 configuration. τ_B also falls on or near −100. Rows tied
at the threshold are split by the seeded tie-break jitter, not by the model.
Measured for seed 42:

- **Held-out rows called Unknown.** 0–7% of them come from the tie (Generic
  14%), so at most about 4 points of any family's unknown recall is
  tie-decided.
- **Known attacks wrongly called Unknown.** 9–49% of those decisions come
  from the tie.
- **Shellcode** (T = 100, τ_A = 152) has almost no ties (under 1%).

The jitter is needed to meet the budgets exactly, but these decisions carry
no information. A larger T, or weighted clauses that remove saturation, would
be the cleaner fix in future work.

## Appendix — generated tables

The tables below are produced by `scripts/23_phase3_tables.py` from the run
outputs, unedited. Score-distribution plots are in `figures/phase3/`.

### Task 1 — tuned hyperparameters

| model | clauses | T | s | epochs | val macro F1 |
|---|---|---|---|---|---|
| closed-set 10-class TM | 1000 | 50 | 10.0 | 1 | 0.4319 |
| banks, analysis held out | 1000 | 50 | 5.0 | 1 | 0.4874 |
| banks, backdoor held out | 1000 | 50 | 5.0 | 5 | 0.5083 |
| banks, dos held out | 1000 | 50 | 5.0 | 1 | 0.4897 |
| banks, exploits held out | 1000 | 50 | 5.0 | 5 | 0.4804 |
| banks, fuzzers held out | 1000 | 50 | 2.5 | 1 | 0.4299 |
| banks, generic held out | 1000 | 50 | 5.0 | 1 | 0.4058 |
| banks, reconnaissance held out | 1000 | 50 | 5.0 | 3 | 0.4354 |
| banks, shellcode held out | 1000 | 100 | 10.0 | 1 | 0.4639 |
| banks, worms held out | 1000 | 50 | 5.0 | 1 | 0.4928 |

### Closed-set TM over seeds [7, 42, 99, 1234, 2024]

| configuration | accuracy | weighted F1 | macro F1 |
|---|---|---|---|
| short_raw | 0.9447 ± 0.0009 | 0.9553 ± 0.0006 | 0.4420 ± 0.0069 |
| short_calibrated | 0.9542 ± 0.0013 | 0.9594 ± 0.0009 | 0.4724 ± 0.0029 |
| medium_raw | 0.9457 ± 0.0013 | 0.9560 ± 0.0008 | 0.4448 ± 0.0097 |
| medium_calibrated | 0.9546 ± 0.0004 | 0.9589 ± 0.0008 | 0.4628 ± 0.0076 |
| long_raw | 0.9453 ± 0.0010 | 0.9558 ± 0.0008 | 0.4411 ± 0.0053 |
| long_calibrated | 0.9545 ± 0.0009 | 0.9593 ± 0.0011 | 0.4623 ± 0.0091 |
| ensemble_max_calibrated | 0.9549 ± 0.0003 | 0.9598 ± 0.0007 | 0.4692 ± 0.0048 |

### Task 5 — leave-one-family-out, medium scale, global thresholds at a 5% validation false-rejection budget

| held out | test rows | unknown recall (95% CI, seed 1) | AUROC | AUPRC | false-unknown | benign FPR | known macro F1 | known-attack macro F1 |
|---|---|---|---|---|---|---|---|---|
| analysis | 678 | 76.7% ± 3.2% (75.7%–81.8%) | 0.866 ± 0.021 | 0.021 ± 0.001 | 4.59% ± 0.03% | 1.88% ± 0.14% | 0.4332 ± 0.0089 | 0.3637 ± 0.0101 |
| backdoor | 702 | 92.0% ± 1.5% (91.5%–95.2%) | 0.965 ± 0.006 | 0.036 ± 0.002 | 4.57% ± 0.05% | 2.22% ± 0.21% | 0.4460 ± 0.0077 | 0.3782 ± 0.0087 |
| dos | 5161 | 87.1% ± 1.3% (88.0%–89.7%) | 0.956 ± 0.004 | 0.240 ± 0.020 | 4.67% ± 0.11% | 2.04% ± 0.09% | 0.4269 ± 0.0205 | 0.3567 ± 0.0231 |
| exploits | 12850 | 81.5% ± 6.9% (85.8%–87.0%) | 0.946 ± 0.025 | 0.413 ± 0.045 | 4.44% ± 0.09% | 2.16% ± 0.13% | 0.4022 ± 0.0071 | 0.3290 ± 0.0080 |
| fuzzers | 6711 | 47.3% ± 5.3% (49.5%–51.8%) | 0.696 ± 0.061 | 0.138 ± 0.021 | 4.23% ± 0.16% | 1.41% ± 0.15% | 0.3871 ± 0.0289 | 0.3116 ± 0.0325 |
| generic | 68460 | 26.1% ± 17.1% (11.7%–12.2%) | 0.467 ± 0.223 | 0.258 ± 0.120 | 4.48% ± 0.09% | 2.24% ± 0.03% | 0.3323 ± 0.0073 | 0.2503 ± 0.0082 |
| reconnaissance | 4064 | 40.9% ± 3.6% (43.7%–46.7%) | 0.907 ± 0.006 | 0.060 ± 0.004 | 4.62% ± 0.22% | 2.18% ± 0.08% | 0.3818 ± 0.0027 | 0.3060 ± 0.0030 |
| shellcode | 440 | 72.4% ± 3.6% (69.3%–77.5%) | 0.965 ± 0.005 | 0.029 ± 0.004 | 4.34% ± 0.21% | 2.60% ± 0.10% | 0.4286 ± 0.0126 | 0.3590 ± 0.0142 |
| worms | 52 | 36.5% ± 9.4% (15.2%–38.2%) | 0.863 ± 0.009 | 0.001 ± 0.000 | 4.44% ± 0.06% | 2.01% ± 0.13% | 0.4467 ± 0.0030 | 0.3789 ± 0.0034 |
| **average** | | 62.3% ± 2.0% | 0.848 ± 0.020 | 0.133 ± 0.015 | 4.49% ± 0.05% | 2.08% ± 0.05% | 0.4094 ± 0.0063 | 0.3370 ± 0.0071 |

### Task 2 — where held-out rows go (global, 5% budget, mean ± std over seeds)

| held out | (i) Unknown | (ii) a known family | (iii) Normal | held-out above tau_B | benign-support AUROC held-out vs known attacks | top-2 margin AUROC |
|---|---|---|---|---|---|---|
| analysis | 76.7% ± 3.2% | 12.9% ± 4.1% | 10.4% ± 2.4% | 13.0% | 0.564 | 0.099 |
| backdoor | 92.0% ± 1.5% | 7.0% ± 1.3% | 1.1% ± 0.6% | 0.0% | 0.528 | 0.088 |
| dos | 87.1% ± 1.3% | 10.9% ± 1.0% | 2.0% ± 0.5% | 1.8% | 0.505 | 0.082 |
| exploits | 81.5% ± 6.9% | 16.3% ± 5.4% | 2.3% ± 2.1% | 0.5% | 0.507 | 0.095 |
| fuzzers | 47.3% ± 5.3% | 21.1% ± 3.5% | 31.6% ± 6.0% | 26.2% | 0.621 | 0.154 |
| generic | 26.1% ± 17.1% | 22.8% ± 17.4% | 51.1% ± 23.6% | 86.5% | 0.966 | 0.958 |
| reconnaissance | 40.9% ± 3.6% | 58.3% ± 3.8% | 0.8% ± 0.6% | 0.2% | 0.503 | 0.203 |
| shellcode | 72.4% ± 3.6% | 27.2% ± 3.6% | 0.5% ± 0.3% | 0.5% | 0.449 | 0.167 |
| worms | 36.5% ± 9.4% | 58.5% ± 10.0% | 5.0% ± 0.9% | 3.8% | 0.565 | 0.105 |

### Task 3 — threshold strategies, average across held-out families (mean ± std over seeds)


**Validation false-rejection budget 1%**

| strategy | unknown recall | held-out to Normal | false-unknown | benign FPR | known macro F1 | known-attack macro F1 | AUROC | AUPRC |
|---|---|---|---|---|---|---|---|---|
| closed | 0.0% ± 0.0% | 11.6% ± 2.1% | 0.00% ± 0.00% | 2.08% ± 0.05% | 0.4770 ± 0.0016 | 0.4130 ± 0.0018 | 0.848 ± 0.020 | 0.133 ± 0.015 |
| global | 12.1% ± 0.9% | 11.6% ± 2.1% | 0.50% ± 0.05% | 2.08% ± 0.05% | 0.4750 ± 0.0023 | 0.4108 ± 0.0026 | 0.848 ± 0.020 | 0.133 ± 0.015 |
| classcond | 6.9% ± 1.8% | 11.6% ± 2.1% | 0.90% ± 0.03% | 2.08% ± 0.05% | 0.4748 ± 0.0017 | 0.4106 ± 0.0019 | 0.765 ± 0.022 | 0.073 ± 0.019 |
| margin | 16.1% ± 1.0% | 11.6% ± 2.1% | 0.84% ± 0.03% | 2.08% ± 0.05% | 0.4698 ± 0.0027 | 0.4050 ± 0.0030 | 0.868 ± 0.019 | 0.117 ± 0.006 |
| combined | 8.5% ± 0.5% | 0.2% ± 0.1% | 0.65% ± 0.02% | 3.89% ± 0.75% | 0.4672 ± 0.0031 | 0.4031 ± 0.0031 | 0.751 ± 0.009 | 0.066 ± 0.003 |

**Validation false-rejection budget 5%**

| strategy | unknown recall | held-out to Normal | false-unknown | benign FPR | known macro F1 | known-attack macro F1 | AUROC | AUPRC |
|---|---|---|---|---|---|---|---|---|
| closed | 0.0% ± 0.0% | 11.6% ± 2.1% | 0.00% ± 0.00% | 2.08% ± 0.05% | 0.4770 ± 0.0016 | 0.4130 ± 0.0018 | 0.848 ± 0.020 | 0.133 ± 0.015 |
| global | 62.3% ± 2.0% | 11.6% ± 2.1% | 4.49% ± 0.05% | 2.08% ± 0.05% | 0.4094 ± 0.0063 | 0.3370 ± 0.0071 | 0.848 ± 0.020 | 0.133 ± 0.015 |
| classcond | 16.8% ± 2.3% | 11.6% ± 2.1% | 4.26% ± 0.06% | 2.08% ± 0.05% | 0.4563 ± 0.0020 | 0.3898 ± 0.0022 | 0.770 ± 0.021 | 0.076 ± 0.016 |
| margin | 70.1% ± 0.7% | 11.6% ± 2.1% | 4.92% ± 0.02% | 2.08% ± 0.05% | 0.3672 ± 0.0032 | 0.2895 ± 0.0036 | 0.868 ± 0.019 | 0.117 ± 0.006 |
| combined | 23.5% ± 0.8% | 0.0% ± 0.1% | 4.01% ± 0.19% | 7.01% ± 0.61% | 0.4614 ± 0.0057 | 0.3987 ± 0.0059 | 0.751 ± 0.009 | 0.066 ± 0.003 |

**Validation false-rejection budget 10%**

| strategy | unknown recall | held-out to Normal | false-unknown | benign FPR | known macro F1 | known-attack macro F1 | AUROC | AUPRC |
|---|---|---|---|---|---|---|---|---|
| closed | 0.0% ± 0.0% | 11.6% ± 2.1% | 0.00% ± 0.00% | 2.08% ± 0.05% | 0.4770 ± 0.0016 | 0.4130 ± 0.0018 | 0.848 ± 0.020 | 0.133 ± 0.015 |
| global | 72.1% ± 2.1% | 11.6% ± 2.1% | 8.95% ± 0.04% | 2.08% ± 0.05% | 0.3317 ± 0.0021 | 0.2496 ± 0.0023 | 0.848 ± 0.020 | 0.133 ± 0.015 |
| classcond | 34.7% ± 2.0% | 11.6% ± 2.1% | 8.55% ± 0.05% | 2.08% ± 0.05% | 0.4037 ± 0.0029 | 0.3306 ± 0.0033 | 0.788 ± 0.019 | 0.082 ± 0.015 |
| margin | 87.6% ± 2.3% | 11.6% ± 2.1% | 9.73% ± 0.13% | 2.08% ± 0.05% | 0.2060 ± 0.0035 | 0.1081 ± 0.0039 | 0.868 ± 0.019 | 0.117 ± 0.006 |
| combined | 41.4% ± 1.8% | 0.0% ± 0.0% | 8.10% ± 0.39% | 10.66% ± 0.56% | 0.4477 ± 0.0060 | 0.3857 ± 0.0065 | 0.751 ± 0.009 | 0.066 ± 0.003 |

### Task 3 per family, unknown recall at a 5% budget

| held out | global | classcond | margin | combined |
|---|---|---|---|---|
| analysis | 76.7% ± 3.2% | 2.7% ± 0.9% | 88.0% ± 1.4% | 23.8% ± 1.5% |
| backdoor | 92.0% ± 1.5% | 4.9% ± 1.0% | 92.8% ± 0.6% | 19.1% ± 1.7% |
| dos | 87.1% ± 1.3% | 5.5% ± 0.9% | 93.3% ± 1.5% | 36.4% ± 3.8% |
| exploits | 81.5% ± 6.9% | 16.4% ± 3.6% | 96.9% ± 1.9% | 24.8% ± 4.1% |
| fuzzers | 47.3% ± 5.3% | 21.3% ± 2.7% | 51.7% ± 7.3% | 61.1% ± 8.7% |
| generic | 26.1% ± 17.1% | 38.5% ± 19.2% | 1.6% ± 0.0% | 0.1% ± 0.0% |
| reconnaissance | 40.9% ± 3.6% | 29.0% ± 1.3% | 65.1% ± 5.7% | 6.5% ± 0.3% |
| shellcode | 72.4% ± 3.6% | 26.2% ± 2.0% | 59.9% ± 2.1% | 30.9% ± 2.0% |
| worms | 36.5% ± 9.4% | 6.9% ± 1.5% | 81.9% ± 2.0% | 8.8% ± 1.5% |

### Task 4 — scale fusion (global thresholds, 5% budget, family average over seeds [7, 42, 1234])

| rule | unknown recall | held-out to Normal | false-unknown | benign FPR | known macro F1 | AUROC | AUPRC |
|---|---|---|---|---|---|---|---|
| short | 61.1% ± 2.9% | 14.1% ± 1.5% | 4.68% ± 0.06% | 2.06% ± 0.02% | 0.4150 ± 0.0006 | 0.821 ± 0.015 | 0.116 ± 0.011 |
| medium | 62.4% ± 1.2% | 12.9% ± 1.8% | 4.53% ± 0.01% | 2.10% ± 0.05% | 0.4061 ± 0.0060 | 0.836 ± 0.018 | 0.125 ± 0.013 |
| long | 69.5% ± 3.2% | 7.6% ± 1.3% | 4.46% ± 0.04% | 2.08% ± 0.04% | 0.4037 ± 0.0056 | 0.889 ± 0.014 | 0.165 ± 0.011 |
| mean | 68.4% ± 4.4% | 14.2% ± 2.4% | 4.60% ± 0.04% | 2.00% ± 0.04% | 0.4033 ± 0.0042 | 0.830 ± 0.025 | 0.127 ± 0.018 |
| max | 58.5% ± 2.3% | 12.7% ± 2.1% | 4.49% ± 0.05% | 1.99% ± 0.04% | 0.4093 ± 0.0044 | 0.835 ± 0.019 | 0.126 ± 0.014 |
| adaptive_margin | 67.1% ± 4.3% | 14.8% ± 1.7% | 4.60% ± 0.04% | 2.00% ± 0.04% | 0.4025 ± 0.0039 | 0.824 ± 0.018 | 0.123 ± 0.014 |
| adaptive_rel | 3.6% ± 2.1% | 14.2% ± 2.4% | 4.30% ± 0.09% | 2.00% ± 0.04% | 0.4106 ± 0.0012 | 0.718 ± 0.023 | 0.052 ± 0.019 |
| select | 9.0% ± 1.4% | 8.2% ± 1.4% | 4.14% ± 0.03% | 2.03% ± 0.07% | 0.4091 ± 0.0006 | 0.776 ± 0.014 | 0.103 ± 0.012 |
| adaptive_rel_raw | 68.5% ± 4.5% | 14.1% ± 2.5% | 4.60% ± 0.04% | 2.00% ± 0.04% | 0.4034 ± 0.0041 | 0.831 ± 0.025 | 0.128 ± 0.018 |
| select_raw | 69.6% ± 4.1% | 8.2% ± 1.4% | 4.50% ± 0.02% | 2.03% ± 0.07% | 0.4120 ± 0.0035 | 0.884 ± 0.017 | 0.161 ± 0.014 |

adaptive_rel vs max: recall -54.86% (seed sd 3.32%), known F1 +0.0013, families improved 1/9 — no measurable improvement

select vs max: recall -49.46% (seed sd 3.53%), known F1 -0.0003, families improved 1/9 — no measurable improvement

adaptive_rel_raw vs max: recall +10.01% (seed sd 4.99%), known F1 -0.0059, families improved 8/9 — no measurable improvement

select_raw vs max: recall +11.12% (seed sd 5.81%), known F1 +0.0027, families improved 8/9 — no measurable improvement

### Complementarity per held-out family (first seed)

| held out | short | medium | long | union of scales | caught by all three | best single |
|---|---|---|---|---|---|---|
| analysis | 68.3% | 76.3% | 77.1% | 80.7% | 63.7% | long |
| backdoor | 92.2% | 92.5% | 91.9% | 95.6% | 88.5% | medium |
| dos | 81.7% | 87.7% | 83.8% | 89.2% | 78.3% | medium |
| exploits | 72.8% | 71.2% | 78.8% | 84.9% | 63.2% | long |
| fuzzers | 55.6% | 51.5% | 46.9% | 64.7% | 39.0% | short |
| generic | 3.9% | 16.0% | 77.7% | 80.1% | 2.6% | long |
| reconnaissance | 39.7% | 36.3% | 44.8% | 58.0% | 25.4% | long |
| shellcode | 76.6% | 67.5% | 77.0% | 79.8% | 66.8% | long |
| worms | 23.1% | 53.8% | 28.8% | 59.6% | 17.3% | medium |

### Task 4 — scale fusion (margin thresholds, 5% budget, family average over seeds [7, 42, 1234])

| rule | unknown recall | held-out to Normal | false-unknown | benign FPR | known macro F1 | AUROC | AUPRC |
|---|---|---|---|---|---|---|---|
| short | 68.4% ± 0.8% | 14.1% ± 1.5% | 4.85% ± 0.01% | 2.06% ± 0.02% | 0.3716 ± 0.0012 | 0.841 ± 0.014 | 0.105 ± 0.004 |
| medium | 70.4% ± 0.7% | 12.9% ± 1.8% | 4.92% ± 0.02% | 2.10% ± 0.05% | 0.3675 ± 0.0039 | 0.856 ± 0.016 | 0.113 ± 0.005 |
| long | 70.3% ± 1.0% | 7.6% ± 1.3% | 4.83% ± 0.03% | 2.08% ± 0.04% | 0.3693 ± 0.0025 | 0.903 ± 0.011 | 0.134 ± 0.007 |
| mean | 69.3% ± 0.8% | 14.2% ± 2.4% | 4.85% ± 0.02% | 2.00% ± 0.04% | 0.3699 ± 0.0024 | 0.846 ± 0.022 | 0.110 ± 0.008 |
| max | 69.0% ± 1.0% | 12.7% ± 2.1% | 4.88% ± 0.01% | 1.99% ± 0.04% | 0.3669 ± 0.0015 | 0.861 ± 0.019 | 0.108 ± 0.007 |
| adaptive_margin | 69.2% ± 0.7% | 14.8% ± 1.7% | 4.88% ± 0.04% | 2.00% ± 0.04% | 0.3658 ± 0.0037 | 0.840 ± 0.015 | 0.108 ± 0.005 |
| adaptive_rel | 40.0% ± 1.1% | 14.2% ± 2.4% | 4.48% ± 0.01% | 2.00% ± 0.04% | 0.3901 ± 0.0006 | 0.784 ± 0.024 | 0.073 ± 0.008 |
| select | 38.6% ± 1.6% | 8.2% ± 1.4% | 4.46% ± 0.04% | 2.03% ± 0.07% | 0.3926 ± 0.0013 | 0.837 ± 0.016 | 0.093 ± 0.011 |
| adaptive_rel_raw | 69.3% ± 0.8% | 14.1% ± 2.5% | 4.85% ± 0.02% | 2.00% ± 0.04% | 0.3699 ± 0.0024 | 0.846 ± 0.022 | 0.110 ± 0.008 |
| select_raw | 69.7% ± 0.8% | 8.2% ± 1.4% | 4.84% ± 0.02% | 2.03% ± 0.07% | 0.3693 ± 0.0005 | 0.900 ± 0.012 | 0.132 ± 0.008 |

adaptive_rel vs max: recall -28.95% (seed sd 1.32%), known F1 +0.0232, families improved 0/9 — no measurable improvement

select vs max: recall -30.39% (seed sd 2.21%), known F1 +0.0257, families improved 1/9 — no measurable improvement

adaptive_rel_raw vs max: recall +0.36% (seed sd 0.51%), known F1 +0.0030, families improved 6/9 — no measurable improvement

select_raw vs max: recall +0.79% (seed sd 0.33%), known F1 +0.0024, families improved 6/9 — measurable improvement

### Task 4 — scale fusion (classcond thresholds, 5% budget, family average over seeds [7, 42, 1234])

| rule | unknown recall | held-out to Normal | false-unknown | benign FPR | known macro F1 | AUROC | AUPRC |
|---|---|---|---|---|---|---|---|
| short | 16.1% ± 1.6% | 14.1% ± 1.5% | 4.53% ± 0.07% | 2.06% ± 0.02% | 0.4582 ± 0.0033 | 0.746 ± 0.016 | 0.057 ± 0.009 |
| medium | 15.7% ± 1.7% | 12.9% ± 1.8% | 4.28% ± 0.06% | 2.10% ± 0.05% | 0.4557 ± 0.0023 | 0.757 ± 0.018 | 0.067 ± 0.013 |
| long | 22.8% ± 2.2% | 7.6% ± 1.3% | 4.18% ± 0.09% | 2.08% ± 0.04% | 0.4572 ± 0.0028 | 0.809 ± 0.014 | 0.119 ± 0.013 |
| mean | 17.0% ± 2.7% | 14.2% ± 2.4% | 4.46% ± 0.07% | 2.00% ± 0.04% | 0.4625 ± 0.0032 | 0.746 ± 0.024 | 0.069 ± 0.019 |
| max | 16.4% ± 1.8% | 12.7% ± 2.1% | 4.28% ± 0.05% | 1.99% ± 0.04% | 0.4542 ± 0.0029 | 0.758 ± 0.021 | 0.070 ± 0.014 |
| adaptive_margin | 16.2% ± 2.1% | 14.8% ± 1.7% | 4.47% ± 0.08% | 2.00% ± 0.04% | 0.4615 ± 0.0035 | 0.739 ± 0.017 | 0.064 ± 0.013 |
| adaptive_rel | 10.0% ± 2.4% | 14.2% ± 2.4% | 4.43% ± 0.07% | 2.00% ± 0.04% | 0.4268 ± 0.0002 | 0.729 ± 0.024 | 0.048 ± 0.014 |
| select | 15.9% ± 1.3% | 8.2% ± 1.4% | 4.32% ± 0.03% | 2.03% ± 0.07% | 0.4252 ± 0.0009 | 0.787 ± 0.013 | 0.088 ± 0.010 |
| adaptive_rel_raw | 17.0% ± 2.8% | 14.1% ± 2.5% | 4.46% ± 0.07% | 2.00% ± 0.04% | 0.4625 ± 0.0032 | 0.746 ± 0.025 | 0.070 ± 0.019 |
| select_raw | 22.1% ± 2.2% | 8.2% ± 1.4% | 4.30% ± 0.02% | 2.03% ± 0.07% | 0.4588 ± 0.0032 | 0.803 ± 0.015 | 0.113 ± 0.013 |

adaptive_rel vs max: recall -6.33% (seed sd 1.59%), known F1 -0.0274, families improved 1/9 — no measurable improvement

select vs max: recall -0.49% (seed sd 2.04%), known F1 -0.0290, families improved 3/9 — no measurable improvement

adaptive_rel_raw vs max: recall +0.66% (seed sd 1.70%), known F1 +0.0083, families improved 4/9 — no measurable improvement

select_raw vs max: recall +5.68% (seed sd 2.38%), known F1 +0.0046, families improved 5/9 — measurable improvement

### Task 4 — scale fusion (combined thresholds, 5% budget, family average over seeds [7, 42, 1234])

| rule | unknown recall | held-out to Normal | false-unknown | benign FPR | known macro F1 | AUROC | AUPRC |
|---|---|---|---|---|---|---|---|
| short | 30.7% ± 1.7% | 0.1% ± 0.1% | 4.50% ± 0.04% | 6.94% ± 0.51% | 0.4642 ± 0.0052 | 0.741 ± 0.019 | 0.072 ± 0.003 |
| medium | 24.1% ± 0.6% | 0.1% ± 0.1% | 3.99% ± 0.07% | 7.20% ± 0.67% | 0.4592 ± 0.0064 | 0.754 ± 0.009 | 0.068 ± 0.002 |
| long | 25.7% ± 0.9% | 0.0% ± 0.0% | 3.48% ± 0.02% | 6.32% ± 0.78% | 0.4594 ± 0.0079 | 0.778 ± 0.014 | 0.082 ± 0.003 |
| mean | 25.0% ± 1.7% | 0.0% ± 0.0% | 3.73% ± 0.09% | 6.59% ± 0.47% | 0.4644 ± 0.0063 | 0.789 ± 0.013 | 0.078 ± 0.005 |
| max | 27.5% ± 1.8% | 0.0% ± 0.0% | 4.18% ± 0.11% | 6.96% ± 0.62% | 0.4589 ± 0.0059 | 0.730 ± 0.011 | 0.072 ± 0.001 |
| adaptive_margin | 26.1% ± 1.6% | 0.0% ± 0.0% | 3.86% ± 0.11% | 6.67% ± 0.55% | 0.4635 ± 0.0063 | 0.788 ± 0.014 | 0.079 ± 0.004 |
| adaptive_rel | 11.5% ± 0.2% | 0.0% ± 0.0% | 3.94% ± 0.15% | 7.12% ± 0.34% | 0.4175 ± 0.0022 | 0.236 ± 0.002 | 0.105 ± 0.003 |
| select | 11.3% ± 0.2% | 0.0% ± 0.0% | 4.04% ± 0.09% | 7.37% ± 0.46% | 0.4156 ± 0.0024 | 0.234 ± 0.002 | 0.109 ± 0.001 |
| adaptive_rel_raw | 25.0% ± 1.7% | 0.0% ± 0.0% | 3.73% ± 0.08% | 6.59% ± 0.47% | 0.4644 ± 0.0062 | 0.789 ± 0.013 | 0.079 ± 0.005 |
| select_raw | 26.7% ± 2.2% | 0.1% ± 0.1% | 3.92% ± 0.04% | 6.64% ± 0.43% | 0.4634 ± 0.0042 | 0.778 ± 0.015 | 0.080 ± 0.005 |

adaptive_rel vs max: recall -15.94% (seed sd 1.93%), known F1 -0.0414, families improved 1/9 — no measurable improvement

select vs max: recall -16.22% (seed sd 1.68%), known F1 -0.0433, families improved 1/9 — no measurable improvement

adaptive_rel_raw vs max: recall -2.46% (seed sd 0.38%), known F1 +0.0055, families improved 3/9 — no measurable improvement

select_raw vs max: recall -0.76% (seed sd 1.50%), known F1 +0.0045, families improved 5/9 — no measurable improvement

### Task 6 — open-set baselines vs TM (5% budget, family average)

| method | unknown recall | held-out to Normal | false-unknown | benign FPR | known macro F1 | AUROC | AUPRC |
|---|---|---|---|---|---|---|---|
| TM (global thresholds) | 62.3% ± 2.0% | 11.6% ± 2.1% | 4.49% ± 0.05% | 2.08% ± 0.05% | 0.4094 ± 0.0063 | 0.848 ± 0.020 | 0.133 ± 0.015 |
| rf_msp | 87.8% ± 0.4% | 0.1% ± 0.0% | 4.68% ± 0.03% | 2.83% ± 0.00% | 0.5070 ± 0.0019 | 0.972 ± 0.000 | 0.194 ± 0.003 |
| xgb_msp | 86.8% ± 0.5% | 2.1% ± 0.2% | 4.79% ± 0.00% | 2.47% ± 0.00% | 0.5180 ± 0.0005 | 0.973 ± 0.001 | 0.188 ± 0.007 |
| iforest_rf | 21.4% ± 2.6% | 4.0% ± 0.5% | 3.09% ± 0.08% | 16.25% ± 1.87% | 0.5194 ± 0.0052 | 0.873 ± 0.012 | 0.073 ± 0.007 |
| mahalanobis | 20.2% ± 0.0% | 0.0% ± 0.0% | 5.03% ± 0.00% | 6.63% ± 0.00% | 0.3895 ± 0.0000 | 0.708 ± 0.000 | 0.072 ± 0.000 |

### Unknown recall per family at 5% (mean over seeds)

| held out | TM | rf_msp | xgb_msp | iforest_rf | mahalanobis |
|---|---|---|---|---|---|
| analysis | 76.7% | 98.9% | 99.8% | 39.2% | 15.8% |
| backdoor | 92.0% | 99.1% | 97.9% | 46.3% | 16.2% |
| dos | 87.1% | 91.0% | 92.7% | 23.9% | 17.2% |
| exploits | 81.5% | 89.4% | 97.8% | 15.8% | 30.7% |
| fuzzers | 47.3% | 87.7% | 78.2% | 10.4% | 24.4% |
| generic | 26.1% | 98.7% | 87.0% | 10.4% | 16.8% |
| reconnaissance | 40.9% | 64.0% | 55.6% | 10.9% | 14.3% |
| shellcode | 72.4% | 95.0% | 96.5% | 30.8% | 11.8% |
| worms | 36.5% | 66.5% | 75.8% | 4.6% | 34.6% |

### Task 6 — explanations of the open-set decision (medium, seed 42)

Fidelity: agreement of the reduced-clause decision with the full decision.


**k = 1 clauses per bank**

| held out | all_weighted | normal | known_attack | unknown_caught | unknown_to_family | unknown_to_normal |
|---|---|---|---|---|---|---|
| analysis | 89.3% | 96.9% | 58.2% | 23.0% | 31.6% | 1.5% |
| backdoor | 92.9% | 97.2% | 75.4% | 59.2% | 7.9% | 0.0% |
| dos | 91.8% | 97.0% | 71.0% | 67.5% | 5.1% | 5.0% |
| exploits | 93.4% | 97.3% | 79.4% | 70.3% | 0.9% | 2.6% |
| fuzzers | 91.7% | 97.4% | 70.7% | 66.2% | 5.6% | 1.0% |
| generic | 82.0% | 97.0% | 30.5% | 47.1% | 6.3% | 1.4% |
| reconnaissance | 86.1% | 97.4% | 39.9% | 17.9% | 34.7% | 4.4% |
| shellcode | 90.6% | 97.5% | 62.4% | 44.4% | 10.0% | 0.0% |
| worms | 92.3% | 97.0% | 72.8% | 55.6% | 9.7% | 0.0% |

**k = 5 clauses per bank**

| held out | all_weighted | normal | known_attack | unknown_caught | unknown_to_family | unknown_to_normal |
|---|---|---|---|---|---|---|
| analysis | 92.1% | 97.1% | 71.5% | 46.6% | 18.0% | 3.0% |
| backdoor | 93.0% | 97.2% | 75.5% | 60.2% | 5.3% | 0.0% |
| dos | 91.9% | 97.0% | 71.5% | 66.8% | 5.6% | 6.2% |
| exploits | 93.7% | 97.4% | 80.3% | 71.5% | 0.5% | 66.4% |
| fuzzers | 92.2% | 97.9% | 70.8% | 57.4% | 46.5% | 0.7% |
| generic | 82.2% | 97.3% | 30.4% | 48.9% | 5.9% | 0.6% |
| reconnaissance | 93.0% | 97.3% | 77.2% | 56.4% | 18.5% | 0.0% |
| shellcode | 95.9% | 98.7% | 84.6% | 90.5% | 15.0% | 0.0% |
| worms | 92.4% | 97.1% | 73.0% | 61.1% | 3.2% | 0.0% |

**k = 25 clauses per bank**

| held out | all_weighted | normal | known_attack | unknown_caught | unknown_to_family | unknown_to_normal |
|---|---|---|---|---|---|---|
| analysis | 93.8% | 97.4% | 78.9% | 92.1% | 10.5% | 1.5% |
| backdoor | 93.0% | 97.3% | 75.2% | 68.2% | 2.6% | 77.8% |
| dos | 92.2% | 97.3% | 71.7% | 69.3% | 4.4% | 82.5% |
| exploits | 93.9% | 97.3% | 80.7% | 76.8% | 25.0% | 67.2% |
| fuzzers | 92.6% | 98.1% | 71.4% | 88.7% | 15.7% | 0.9% |
| generic | 82.4% | 97.0% | 32.4% | 55.9% | 6.2% | 0.6% |
| reconnaissance | 93.3% | 97.7% | 77.0% | 55.2% | 16.4% | 1.5% |
| shellcode | 96.2% | 98.7% | 86.2% | 94.1% | 12.0% | 0.0% |
| worms | 92.4% | 97.1% | 73.0% | 66.7% | 3.2% | 66.7% |

**k = 100 clauses per bank**

| held out | all_weighted | normal | known_attack | unknown_caught | unknown_to_family | unknown_to_normal |
|---|---|---|---|---|---|---|
| analysis | 93.4% | 97.2% | 77.6% | 75.2% | 34.6% | 43.9% |
| backdoor | 93.5% | 97.5% | 77.0% | 58.2% | 0.0% | 77.8% |
| dos | 92.8% | 97.4% | 74.4% | 79.9% | 1.7% | 83.8% |
| exploits | 95.1% | 97.9% | 82.5% | 95.0% | 35.6% | 77.6% |
| fuzzers | 92.9% | 98.3% | 72.1% | 80.7% | 40.9% | 5.1% |
| generic | 82.8% | 97.2% | 30.9% | 68.1% | 3.5% | 0.0% |
| reconnaissance | 93.5% | 97.3% | 79.6% | 84.6% | 2.7% | 1.5% |
| shellcode | 96.4% | 98.9% | 86.1% | 75.4% | 14.0% | 0.0% |
| worms | 93.0% | 97.1% | 76.1% | 27.8% | 0.0% | 66.7% |

**Stability and complexity**

| held out | group | Jaccard (1 literal flipped) | decision change | cited clauses | literals per cited clause (median) |
|---|---|---|---|---|---|
| analysis | normal | 0.984 | 1.1% | 5.2 | 2 |
| analysis | known_attack | 0.933 | 8.4% | 9.9 | 5 |
| analysis | unknown_caught | 0.821 | 1.3% | 10.0 | 14 |
| analysis | unknown_to_family | 0.886 | 52.3% | 10.0 | 4 |
| analysis | unknown_to_normal | 0.991 | 0.9% | 5.0 | 1 |
| backdoor | normal | 0.991 | 1.3% | 5.2 | 2 |
| backdoor | known_attack | 0.933 | 4.8% | 9.9 | 5 |
| backdoor | unknown_caught | 0.733 | 0.1% | 10.0 | 9 |
| backdoor | unknown_to_family | 0.812 | 28.4% | 10.0 | 6 |
| backdoor | unknown_to_normal | 0.970 | 0.0% | 5.0 | 1 |
| dos | normal | 0.984 | 0.5% | 5.1 | 2 |
| dos | known_attack | 0.939 | 6.3% | 10.0 | 4 |
| dos | unknown_caught | 0.834 | 1.1% | 10.0 | 13 |
| dos | unknown_to_family | 0.940 | 13.2% | 10.0 | 1 |
| dos | unknown_to_normal | 0.982 | 1.5% | 5.0 | 1 |
| exploits | normal | 0.989 | 1.2% | 5.1 | 1 |
| exploits | known_attack | 0.885 | 5.7% | 10.0 | 7 |
| exploits | unknown_caught | 0.791 | 1.2% | 10.0 | 24 |
| exploits | unknown_to_family | 0.803 | 62.5% | 10.0 | 39 |
| exploits | unknown_to_normal | 0.964 | 1.7% | 5.0 | 1 |
| fuzzers | normal | 0.989 | 0.5% | 5.1 | 2 |
| fuzzers | known_attack | 0.901 | 7.7% | 10.0 | 4 |
| fuzzers | unknown_caught | 0.824 | 15.5% | 10.0 | 2 |
| fuzzers | unknown_to_family | 0.839 | 29.6% | 10.0 | 4 |
| fuzzers | unknown_to_normal | 0.959 | 2.9% | 5.0 | 2 |
| generic | normal | 0.991 | 0.3% | 5.1 | 2 |
| generic | known_attack | 0.829 | 5.6% | 9.9 | 6 |
| generic | unknown_caught | 0.721 | 45.6% | 10.0 | 14 |
| generic | unknown_to_family | 0.655 | 78.1% | 10.0 | 14 |
| generic | unknown_to_normal | 0.944 | 2.4% | 5.0 | 4 |
| reconnaissance | normal | 0.989 | 0.0% | 5.1 | 2 |
| reconnaissance | known_attack | 0.934 | 3.5% | 9.9 | 3 |
| reconnaissance | unknown_caught | 0.807 | 1.9% | 10.0 | 9 |
| reconnaissance | unknown_to_family | 0.845 | 67.7% | 10.0 | 10 |
| reconnaissance | unknown_to_normal | 0.887 | 2.9% | 5.0 | 18 |
| shellcode | normal | 0.985 | 0.5% | 5.1 | 2 |
| shellcode | known_attack | 0.811 | 0.9% | 10.0 | 20 |
| shellcode | unknown_caught | 0.697 | 2.1% | 10.0 | 37 |
| shellcode | unknown_to_family | 0.787 | 14.8% | 10.0 | 37 |
| shellcode | unknown_to_normal | 0.843 | 0.0% | 5.0 | 28 |
| worms | normal | 0.992 | 0.3% | 5.1 | 2 |
| worms | known_attack | 0.933 | 5.9% | 10.0 | 3 |
| worms | unknown_caught | 0.862 | 0.0% | 10.0 | 3 |
| worms | unknown_to_family | 0.898 | 18.1% | 10.0 | 3 |
| worms | unknown_to_normal | 0.956 | 0.0% | 5.0 | 1 |
