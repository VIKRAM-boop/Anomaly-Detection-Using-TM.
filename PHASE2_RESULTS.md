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

## Item 4 — leave-one-attack-family-out

Eight families were held out in turn, each excluded from fit and validation
and relabelled unknown in test. Medium scale, supervised normality. Worms is
omitted: 52 test rows is too few for a catch rate to mean anything.

All figures below come from `14_openset_redecide.py`, which pins tau_B to a
1% missed-attack budget. That matters -- see the threshold-sensitivity note
below -- and numbers from `11_openset_decision.py`, which maximises balanced
accuracy instead, are not comparable with these.

| held out | rows | AUROC | @1% | @5% | @10% | @20% | ->Normal | known acc |
|---|---|---|---|---|---|---|---|---|
| Fuzzers | 6,711 | 0.861 | 13.6% | **38.8%** | 52.0% | 75.1% | 2.1% | 0.9783 |
| Shellcode | 440 | 0.812 | 12.0% | 23.9% | 41.1% | 65.2% | 0.5% | 0.9746 |
| Exploits | 12,850 | 0.739 | 4.2% | 11.7% | 30.7% | 55.4% | 25.1% | 0.9823 |
| DoS | 5,161 | 0.825 | 2.2% | 7.0% | 38.0% | 77.2% | 2.2% | 0.9800 |
| Reconnaissance | 4,064 | 0.618 | 3.7% | 6.8% | 16.8% | **34.7%** | 0.2% | 0.9789 |
| Analysis | 678 | 0.806 | 0.1% | 5.8% | 25.2% | 71.2% | 0.4% | 0.9823 |
| Backdoor | 702 | 0.804 | 1.3% | 4.7% | 21.8% | 68.9% | 3.1% | 0.9757 |
| Generic | 68,460 | 0.872 | 0.1% | 0.6% | 2.1% | **16.2%** | **81.6%** | 0.9490 |

### Two mutually-confusable groups

Where held-out rows go at the 5% budget:

| held out | top destinations | share into its group |
|---|---|---|
| DoS | Analysis 49.4%, Backdoor 23.6% | 73.0% |
| Analysis | Backdoor 44.7%, DoS 28.9% | 73.6% |
| Backdoor | Analysis 50.1%, DoS 31.3% | 81.4% |
| Reconnaissance | Shellcode 60.8%, Worms 13.4% | 60.8% |
| Shellcode | Reconnaissance 27.7%, Backdoor 21.6% | 27.7% |

**{DoS, Analysis, Backdoor}** is a tight three-way cluster: each member sends
73-81% of its rows to the other two, and all three behave almost identically
(4.7-7.0% caught at the 5% budget, 69-77% at 20%). This reproduces the same
finding from the pre-correction sweep, so it is a property of the data rather
than an artifact of the leak.

**{Reconnaissance, Shellcode}** is a pair, but an asymmetric one.
Reconnaissance collapses into Shellcode far more readily than the reverse,
and Shellcode is four times more detectable as novel (23.9% vs 6.8%).

### Two failure modes, not one

The families fail for opposite reasons, and the `->Normal` column separates
them cleanly:

- **Family confusion.** Analysis, Backdoor, DoS, Reconnaissance and
  Shellcode leak 0.2-3.1% to Normal. Their benign gate works; they are lost
  to sibling families instead.
- **Benign leakage.** Generic (81.6%) and Exploits (25.1%) are waved through
  as normal traffic and never reach the unknown branch at all.

Fuzzers originally belonged to the second group at 18.9%, and the Phase 1
and Phase 2 leak corrections moved it to the first at 2.1%. Generic and
Exploits did not move comparably: Generic went 94.9% -> 81.6%, so its benign
leakage is real rather than a thresholding artifact.

### Generic is extraordinarily threshold-sensitive

Generic behaves completely differently under the two tau_B rules:

| rule | tau_B | ->Normal | caught @20% |
|---|---|---|---|
| balanced accuracy (`11`) | +104 | ~0% | 97.5% |
| 1% missed-attack budget (`14`) | -119 | 81.6% | 16.2% |

No other family comes close to this sensitivity. The cause is that Generic
looks benign *relative to the other attack families*: a threshold calibrated
so only 1% of known attacks appear benign lands at -119, and most Generic
rows score above it. Which threshold rule you pick decides whether Generic is
the best-detected family or the worst.

This is worth stating plainly because it is easy to mislead yourself here.
Two different rules, applied to identical bank scores, support opposite
conclusions about the same family.

### Why withheld Fuzzers were classified as Normal

| pipeline | Fuzzers -> Normal |
|---|---|
| Before (leaky validation) | 18.9% |
| After (clean chronological validation) | **2.1%** |

For Fuzzers specifically, the answer is that the benign threshold was being
chosen against a validation set composed mostly of SMOTE interpolations,
which are easier to separate than real traffic, so tau_B landed in the wrong
place. With a clean validation partition and the held-out family excluded
from validation as well as training, the leak falls roughly nine-fold and
UNKNOWN becomes the dominant outcome.

The spec's proposed remedy moves this symptom the wrong way: self-supervised
normality leaks 11.5-13.5% and combined leaks 26.7-34.4%.

## Item 5 — clause extraction and explanation quality

The methodology asks for explanations to be assessed but gives no measures,
so these definitions are used (and stated in `16_clause_explain.py`):

- **fidelity** — share of rows where a surrogate using only the k most
  discriminative clauses per class reproduces the full model's prediction.
  These banks run with `weighted_clauses` disabled, so every weight is 1 and
  a class sum is just a count of firing clauses. Ranking by weight is
  therefore meaningless; clauses are ranked by how much more often they fire
  on their own class than on others.
- **stability** — mean Jaccard overlap between a row's firing-clause set and
  that of a copy with one literal flipped, over five draws per row.
- **complexity** — literals per firing clause, and clauses firing per row.

Medium scale, smote strategy, 300 test rows, full-model accuracy 0.950:

| k | fidelity |
|---|---|
| 1 | **94.7%** |
| 3 | 96.0% |
| 10 | 95.7% |
| 25 | 97.0% |
| 100 | 97.0% |
| 500 | 96.7% |

| measure | value |
|---|---|
| stability (Jaccard, one literal flipped) | 0.832 |
| clauses firing per row | mean 289.9, median 281, max 498 |
| literals per firing clause | mean 10.5, median 1, max 339 |

**One discriminative clause per class reproduces the model's decision on
94.7% of rows**, and the curve is flat from there: going from one clause to
five hundred adds about two points. Around 290 clauses fire per row, but
they are heavily redundant rather than each carrying part of the decision.

Individual clauses are short -- the median firing clause contains a single
literal -- so explanations are readable in the ordinary sense. A worked
example, a correctly predicted Normal row with 307 firing clauses:

```
clause   5: NOT sttl>Q90 AND ct_srv_src>Q25
clause 704: NOT ct_state_ttl>Q50
clause 703: medium_dst_entropy>Q25
clause 700: NOT ct_state_ttl>Q50 AND ct_srv_src>Q50
clause 692: medium_dst_entropy>Q50
```

Stability of 0.832 means a single flipped literal leaves about 83% of the
firing set unchanged, so explanations do not reshuffle under small input
changes.

Caveat: fidelity is measured against the model's own output, not ground
truth. It says the explanation is faithful to the model, not that the model
is correct.

A first version of this measurement ranked clauses by weight and reported
fidelity of 1.0% at k=10. That was an artifact: with all weights equal to 1
every class's top-k sum is min(k, clauses fired), so all classes tie and
argmax always returns class 0. The 1.0% was simply how often class 0 was the
right answer.

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

- Single seed throughout, so differences under roughly 0.01 should not be
  read as real.
- The fusion rule and the perturbation scheme are both interpretations, as
  noted above.
