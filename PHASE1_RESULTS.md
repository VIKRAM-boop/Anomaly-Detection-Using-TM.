# Phase 1 — Corrected Pipeline Results

Everything below was produced by the corrected pipeline on 2026-10-18 and
measured from the saved run logs (`data/phase1_logs/`). All nine TM runs use
seed 42 and one training epoch.

## What was corrected

| Correction | What was wrong before | What it is now |
|---|---|---|
| Validation partition | Carved out of the **post-SMOTE** array, so it was majority synthetic | Chronological tail of the training period, 100% real, never resampled |
| Preprocessing fit | Quantiles, one-hot and relational thresholds fit on all training rows, including the rows that later became validation | Fit on the `fit` partition alone |
| Resampling scope | SMOTE applied before the fit/validation split | Applied to `fit` rows only; validation and test untouched |
| Model selection | Per-epoch monitoring reported test accuracy | Reports validation macro F1; test scored once, at the end |
| `is_ftp_login` encoding | Raw values 0/1/2/4. Training rows were silently binarised by SMOTE's `>= 0.5` threshold; test rows were not, so train and test used different encodings | Thresholded identically across fit, validation and test |
| Window leakage | (not a defect) | Audited: pandas time-rolling is strictly backward-looking over a pre-sorted index |

Partition boundaries, verified in the run log:

```
fit  1,828,780 rows  <  2015-02-18 06:58:36
val    203,219 rows  <  2015-02-18 08:22:37
test   508,048 rows
```

All ten classes appear in validation. Worms is the thinnest at 19 validation
rows, so its contribution to validation macro F1 is inherently noisy — each
row moves Worms recall by 5.3 points.

## The leak corrupted selection, not measurement

Mean test macro F1 across the three scales, before and after the fix:

| | mean test macro F1 |
|---|---|
| Old pipeline (leaky validation, 6 epochs) | 0.4352 |
| Corrected pipeline (clean validation, 1 epoch) | 0.4356 |

Test performance is unchanged. The test set was always held out correctly, so
previously reported test numbers stand. What the leak broke was the ability to
*select* anything: the old validation reported roughly 0.70 macro F1 against a
test reality of ~0.44, because its rows were mostly SMOTE interpolations of
training points and therefore trivially easy.

After the fix, validation tracks test closely:

| run | val macro F1 | test macro F1 | gap |
|---|---|---|---|
| short / smote | 0.4173 | 0.4295 | +0.012 |
| medium / smote | 0.4273 | 0.4382 | +0.011 |
| long / smote | 0.4253 | 0.4392 | +0.014 |
| medium / moderate | 0.4245 | 0.4243 | −0.000 |
| medium / undersample | 0.4153 | 0.4153 | 0.000 |

Validation is now usable for model selection, which it previously was not.

## Per-scale results, all three sampling strategies

| scale | strategy | val mF1 | test mF1 | test acc | train acc |
|---|---|---|---|---|---|
| short | smote | 0.4173 | 0.4295 | 0.9424 | 0.6849 |
| medium | smote | 0.4273 | 0.4382 | 0.9447 | 0.6862 |
| long | smote | 0.4253 | 0.4392 | 0.9452 | 0.6838 |
| short | moderate | 0.4267 | 0.4337 | 0.9521 | 0.7570 |
| medium | moderate | 0.4245 | 0.4243 | 0.9523 | 0.7572 |
| long | moderate | 0.4228 | 0.4204 | 0.9510 | 0.7581 |
| short | undersample | 0.4286 | 0.4324 | 0.9512 | 0.7610 |
| medium | undersample | 0.4153 | 0.4153 | 0.9504 | 0.7566 |
| long | undersample | 0.4180 | 0.4149 | 0.9504 | 0.7526 |

Window scale makes little difference within a strategy — the three scales span
0.010 under smote and 0.018 under undersample.

## Sampling strategy comparison

The three strategies, defined on the fit partition only:

| strategy | rule | fit rows | synthetic share |
|---|---|---|---|
| smote | every class raised to 40,000 | 400,000 | ~85% |
| moderate | minorities raised to at most 5x their real count, majorities capped at 40,000 | 259,960 | ~45% |
| undersample | majorities capped at 10,000, nothing synthesised | 60,734 | 0% |

What this does to the rarest class:

| class | real fit rows | smote | moderate | undersample |
|---|---|---|---|---|
| Worms | 103 | 40,000 (**388x**) | 515 (5x) | 103 (1x) |
| Shellcode | 914 | 40,000 (44x) | 4,570 (5x) | 914 (1x) |
| Normal | 1,652,486 | 40,000 (downsampled) | 40,000 | 10,000 |

Mean test macro F1 across the three scales:

| strategy | raw argmax | + calibration | + multi-scale ensemble |
|---|---|---|---|
| **smote** | **0.4356** | **0.4660** | **0.4699** |
| moderate | 0.4261 | 0.4234 | — |
| undersample | 0.4209 | 0.4161 | 0.4182 |

### Calibration is doing the work, not the resampling

Before calibration the three strategies span only 0.015, and `undersample`
lands within 0.005 of `smote` while using 15% of the rows and no synthetic
data at all. SMOTE's advantage appears only once prior calibration is applied:

| strategy | selected tau | val mF1 before | val mF1 after | test mF1 delta |
|---|---|---|---|---|
| smote | 30 | 0.4273 | 0.4660 | **+0.0278** |
| moderate | 2 | 0.4245 | 0.4256 | −0.0009 |
| undersample | 10 | 0.4153 | 0.4179 | +0.0007 |

The mechanism: SMOTE balances the fit set completely, so the model's implicit
class prior becomes roughly uniform — far from the real distribution, leaving
a large correction for tau to make. `moderate` and `undersample` preserve much
of the real imbalance, so the learned prior is already close to correct and
calibration has nothing to fix.

The practical consequence is that aggressive oversampling and prior correction
are a matched pair. Either alone is worth little; together they are worth
about +0.034 macro F1 over the next best configuration.

### No strategy fixes the rare classes

Per-class test F1 at the medium scale:

| class | support | smote (calibrated) | moderate (short, raw) |
|---|---|---|---|
| Normal | 408,930 | 0.99 | 0.99 |
| Generic | 68,460 | 0.99 | 0.99 |
| Reconnaissance | 4,064 | 0.67 | 0.67 |
| Exploits | 12,850 | 0.57 | 0.55 |
| Fuzzers | 6,711 | 0.51 | 0.51 |
| DoS | 5,161 | 0.45 | 0.48 |
| Shellcode | 440 | 0.25 | 0.08 |
| Analysis | 678 | 0.09 | 0.08 |
| Worms | 52 | 0.08 | 0.00 |
| Backdoor | 702 | 0.07 | 0.00 |

Under smote, Worms reaches F1 0.08 with precision 0.05 — the model predicts
Worms often and is wrong 95% of the time. Under moderate it reaches 0.00,
because the model stops predicting Worms at all. Both are failures, in
opposite directions, and neither is fixable by resampling: 103 real training
rows do not contain enough signal, and synthesis cannot add information that
was never there.

Worth separating from the macro F1 ranking: catching 21% of Worms at 5%
precision is something an analyst can triage; catching 0% is a blind spot. If
the operational goal is not missing a novel attack family, smote is preferable
for these classes even where macro F1 does not say so.

### Less synthetic data fits real data better

Train accuracy rises from 0.685 under smote to 0.757 under moderate and
undersample, and test accuracy rises with it (0.944 to 0.952). The synthetic
rows actively interfere with fitting the real distribution; smote only comes
out ahead once calibration compensates for the prior it distorts.

## Known gaps

- **TM hyperparameters were not re-tuned.** `clauses=2000, T=200, s=5.0` were
  selected by a sweep against the old, contaminated validation set. The
  validation set is fixed but those values were inherited. Re-selecting them on
  clean validation remains outstanding.
- **One epoch, so no early stopping.** The choice rests on a 15-epoch run in
  which train accuracy never moved off ~0.667 and test accuracy was within
  0.005 of its final value after a single epoch. That evidence is accuracy
  only; macro F1 per epoch was never measured, so "one epoch is enough for the
  minority classes" is inferred rather than verified.
- **Single seed.** All nine runs use seed 42. Earlier work measured TM seed
  variance at roughly ±0.006 macro F1, so differences below about 0.01 between
  configurations here should not be treated as real.
- **RF/DT baselines** are reported separately; they are a comparison point, not
  part of the resampling study, and run on real data with no synthesis.
