# Phase 2 — Completing the Proposed Methodology

Produced on 2026-10-18. All runs use the Fuzzers hold-out, seed 42, one
training epoch, and the corrected chronological pipeline. Logs in
`data/phase2_logs/`.

## A leak was fixed here too

`09_openset_preprocess.py` had the same defect Phase 1 removed from the main
pipeline: it applied SMOTE first, and `10_train_family_banks.py` then carved
a validation split out of the already-resampled array. Step 5 results
reported before this point were selected against a majority-synthetic
validation surface.

Two things changed:

- The training period is split chronologically into fit and validation
  before any encoder is fitted, and resampling touches fit rows only.
- The held-out family is now removed from **both** fit and validation, not
  just training, so it is unseen during model selection as well.

Verified on the short-scale build: `fit_class_counts_real[Fuzzers] = 0`,
fit 1,813,002 rows before 2015-02-18 06:58:18, validation 201,462 rows after
it, and 6,711 Fuzzers rows present only in test, relabelled unknown.

## Item 1 — separate banks per temporal scale

`09` and `10` now take the scale as an argument, so Step 4's
`TM_short`/`TM_medium`/`TM_long` each get their own normality branch and
their own set of attack-family banks. Nine configurations were built: three
scales x three normality modes.

The family banks do not depend on the normality mode, so they are trained
once per scale and reused across modes. Without that, this run would have
trained 81 banks instead of 33.

## Item 3 — self-supervised pseudo-anomalies

Step 5 asks for the normality branch to be trained "primarily on benign
windows and self-supervised perturbed windows". Three modes were compared:

| mode | negatives for the normality branch |
|---|---|
| supervised | real attack rows (what the previous implementation used) |
| selfsup | perturbed copies of benign rows, no attack labels |
| combined | both |

Perturbations respect the literal encoding. A feature-level perturbation
re-draws a feature's quantile level, keeping its five monotone threshold
literals consistent; a temporal perturbation transplants another row's
window-literal block, severing the link between a flow and its traffic
context. Verified: 51 quantile groups, zero monotonicity violations, 16.4%
of literals changed per row, every row altered. Randomly flipping literals
would instead produce rows that cannot occur, such as `x > Q90` together
with `x <= Q10`.

Results, with tau_B pinned to a 1% missed-attack budget:

| scale | mode | caught @5% | caught @20% | novel->Normal | benign recall | known acc |
|---|---|---|---|---|---|---|
| long | supervised | **39.4%** | 76.2% | **2.2%** | 0.982 | 0.9817 |
| medium | supervised | 38.8% | 75.1% | **2.1%** | 0.982 | 0.9783 |
| short | supervised | 35.7% | 61.6% | 23.1% | 0.988 | 0.9811 |
| short | combined | 33.9% | 53.4% | 34.4% | 0.987 | 0.9822 |
| medium | combined | 27.9% | 57.8% | 26.7% | 0.983 | 0.9775 |
| long | combined | 26.2% | 55.5% | 29.7% | 0.985 | 0.9813 |
| short | selfsup | 0.0% | 0.0% | 13.5% | 0.835 | 0.8907 |
| medium | selfsup | 0.0% | 0.0% | 11.5% | 0.849 | 0.8988 |
| long | selfsup | 0.0% | 0.0% | 12.4% | 0.859 | 0.9085 |

**The self-supervised branch fails at every scale.** It catches nothing at
any budget, and the benign-recall column shows why: it keeps only 83.5-85.9%
of real benign traffic correctly classified, against 98.2-98.8% for the
supervised branch. Since benign is 80% of the test set, wrongly flagging
14-17% of it collapses known accuracy to about 0.90. It fails in both
directions at once -- floods false positives on benign traffic and detects
no novel attacks.

**The pseudo-anomalies are not merely useless, they are harmful.**
`combined` contains every real attack row that `supervised` contains, plus
pseudo-anomalies, and is worse on both metrics at every scale: caught drops
by 5-13 points and the benign leak rises from ~2% to 27-34%. The normality
branch's training accuracy falls from ~0.99 to ~0.82 once pseudo-anomalies
enter the negative set.

The mechanism is that a self-supervised branch learns "is this a distorted
copy of benign traffic?", which is a different question from "is this an
attack?". Real attacks are not random distortions of benign flows.

Caveat: this tests one perturbation scheme, not self-supervision in general.
A perturbation designed to mimic attack structure specifically could behave
differently. The defensible claim is that this formulation underperforms the
supervised branch, not that the spec's approach cannot work.

## Item 2 — fusion rules

Step 4 requires separate banks per scale but defines no rule for combining
them, so the confidence-adaptive rule below is an interpretation rather than
a transcription. Confidence for a scale on a row is the margin between its
best and second-best family score, floored at zero and normalised across
scales; a scale that cannot separate its top two candidates gets little say
in that row. Only 3 validation rows out of 201,462 had zero margin at every
scale, so the uniform fallback is essentially never used.

Supervised normality, fused across all three scales:

| rule | caught @5% | caught @20% | novel->Normal | known acc |
|---|---|---|---|---|
| **max** (already implemented) | **43.5%** | **77.5%** | **2.1%** | 0.9804 |
| mean | 37.5% | 66.9% | 14.6% | 0.9821 |
| adaptive (this interpretation) | 33.6% | 63.7% | 16.4% | 0.9812 |

The existing maximum-score rule wins, and the confidence-adaptive rule comes
last. Taking the per-class maximum means a family only needs one scale to
recognise it; averaging dilutes a strong signal from the scale that matters
with two uninformative ones. Margin-based confidence turns out to be a poor
proxy for correctness -- a scale can be confidently wrong.

Fusion does beat the best single scale, modestly: 43.5% caught against
39.4% for long alone, at the cost of the benign leak rising from 2.2% to
2.6%.

Under `selfsup`, all three rules catch 0.0% up to the 10% budget. Fusion
cannot rescue a misplaced benign gate, which makes the conclusion about
self-supervised normality robust to the fusion choice.

## Item 4 — why withheld Fuzzers were classified as Normal

Largely because the threshold was chosen on contaminated data.

| pipeline | Fuzzers -> Normal |
|---|---|
| Before (leaky validation, medium scale) | 18.9% |
| After (clean chronological validation, medium supervised) | **2.1%** |
| After, max-fused across scales | 2.6% |

The normality branch was not blind to Fuzzers. The benign threshold tau_B
was being selected against a validation set made mostly of SMOTE
interpolations, which are easier to separate than real traffic, so the
threshold landed in the wrong place. With a clean validation partition and
the held-out family excluded from validation as well as training, the leak
falls by roughly nine-fold and UNKNOWN becomes the dominant outcome for
held-out Fuzzers rows.

Note that the spec's proposed remedy for this symptom moves it the wrong
way: self-supervised normality leaks 11.5-13.5% and combined leaks 26.7-34.4%.

Leave-one-family-out across the remaining families is not yet run.

## A note on AUROC

AUROC is reported by these scripts but should not lead the results. It is
computed only over rows that clear each configuration's own benign gate, so
it cannot see a misplaced gate. The clearest case: `selfsup` fused by mean
scored **0.903**, the highest figure anywhere in this study, while catching
**0.0%** of held-out Fuzzers. Read alone it would identify the worst
configuration tested as the best.

Catch rate at a fixed false-unknown budget, together with the novel->Normal
leak, track what the system actually does and should be quoted instead.

## Known gaps

- Leave-one-attack-family-out covers Fuzzers only so far.
- Explanation metrics (item 5) are implemented in `16_clause_explain.py`
  with definitions given in its docstring; results pending.
- Single seed throughout, so differences under roughly 0.01 should not be
  read as real.
- The fusion rule and the perturbation scheme are both interpretations, as
  noted above.
