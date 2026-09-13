# External Validation Notes

The human audit in `docs/measurement_notes.md` establishes that the LLM
signals reproduce what a human coder reads in the same text. It says
nothing about whether they carry economic information. This note
records the separate exercise: whether the measures line up with
euro-area macro data outside the corpus.

`analysis/13_external_validation.py` produces everything below.

## Data

Three series from the ECB Data Portal, downloaded by
`src/download_macro.py` (no API key required):

| Series | Key | Frequency |
| --- | --- | --- |
| HICP annual rate | `ICP/M.U2.N.000000.4.ANR` | monthly |
| Deposit facility rate | `FM/B.U2.EUR.4F.KR.DFR.LEV` | change dates |
| Unemployment rate | `LFSI/M.I9.S.UNEHRT.TOTAL0.15_74.T` | monthly |

Two aggregation choices matter.

The deposit facility rate is published **only on the days it changes**.
The raw file has 67 observations for 1999-2025. Aggregating it directly
would weight quarters by how often the Governing Council happened to
move; it is forward-filled onto a daily calendar first. Quarterly
levels are taken end-of-quarter, because the rate is a policy decision
rather than a flow — what a speech can anticipate is the level the rate
reaches. A quarterly average is written alongside as a robustness
variant.

HICP and unemployment are averaged within the quarter. Unemployment
begins in 2000Q1, so it is unavailable for the first four quarters of
the speech sample.

## Inference

Both sides of every pair are persistent quarterly series, and the
policy-path variable is an overlapping two-quarter difference. An iid
correlation standard error of `1/sqrt(n-3)` would be far too small on
both counts — roughly 0.14 where the autocorrelation-robust error is
0.20 to 0.28.

`src/timeseries_stats.py` implements the alternative. A correlation is
estimated as the OLS slope of the standardised dependent variable on
the standardised regressor, which makes Newey-West standard errors
directly available. Bandwidth follows `floor(4 * (T/100)^(2/9))`,
floored at the moving-average order the overlapping differences imply.
A moving-block bootstrap gives a second, non-parametric interval, since
the HAC sandwich is asymptotic and the subsamples have roughly fifty
observations. scipy is not a project dependency, so the Student-t tail
probability is computed from the regularised incomplete beta function.

A difference in correlations across eras is tested directly, as the
interaction coefficient in a pooled regression on **within-era**
standardised data. Standardising within era rather than pooling is what
makes the interaction a difference in correlations rather than a
difference in slopes confounded by the change in either variable's
dispersion. That distinction is not cosmetic here: inflation was far
more volatile after 2021 than before. The script asserts that the
interaction equals the difference of the two separately estimated
correlations.

## Result

Attention-weighted inflation outlook, quarterly, break imposed at 2012:

| Target | 1999-2011 | 2012-2025 | Difference |
| --- | --- | --- | --- |
| HICP inflation | 0.305 (0.194) | 0.251 (0.229) | -0.054 (0.304) |
| Deposit rate, 2q change | 0.196 (0.198) | 0.480 (0.275) | +0.284 (0.342) |

Newey-West standard errors in parentheses; 52, 56 and 54 quarters.

**The rate-path half of the pattern is there. The realised-inflation
half is not.** Directional inflation talk became substantially more
correlated with the subsequent policy path, from 0.20 to 0.48. Its
correlation with realised inflation barely moved, 0.31 to 0.25, rather
than falling from 0.45 to 0.18 as an earlier ad-hoc calculation
suggested. No definitional variant recovered the ad-hoc figures: the
monthly frequency, the inflation gap from 2 per cent, leads and lags of
HICP, the unconditional attention-weighted tone `att_pi * out_pi`, and
every break year from 2009 to 2015 all leave the HICP correlation flat
at 0.24 to 0.32 in both eras.

**Nothing here is significant at conventional levels once the standard
errors are right.** The late-era rate-path correlation reaches p = 0.09;
the difference across eras, which is the actual claim, has p = 0.41.
Its 95 per cent interval runs from -0.39 to +0.96. With roughly fifty
quarters per era and both series heavily autocorrelated, a rotation of
this size simply cannot be distinguished from sampling variation. The
result is a suggestive pattern, not an established one, and should be
reported that way.

## Robustness

The rate-path rise is not an artefact of any single choice:

- **Break year.** Across 2009-2015 the early-era correlation falls
  monotonically from 0.23 to 0.17 and the late-era one rises from 0.43
  to 0.49. The pattern is not knife-edge on 2012.
- **Speech length.** Partialling mean log words and the truncated share
  out of the outlook measure within era leaves the correlations at
  0.18 and 0.49. The length problem that afflicts the attention series
  does not drive this one, which is expected: outlook is conditional on
  attention.
- **Rate definition.** Using the quarterly-average rate gives 0.35 and
  0.43 — a much weaker rotation, because the average partly reflects
  decisions already taken within the quarter.
- **Language.** English-only measures give 0.23 and 0.48.

## Full-sample concordance

Seven measure-target pairs are checked over 1999-2025 with a
pre-committed expected sign. Five carry it. Two do not:

- **Growth outlook against the unemployment level** is +0.26 where
  reasoning predicts negative. This is a level-versus-change artefact:
  against the *change* in unemployment over the following two quarters
  the same measure gives -0.35 (p = 0.02), the expected sign and the
  strongest relationship in the table. Optimistic growth talk
  accompanies high unemployment that is about to fall — which is what
  a recovery looks like.
- **Growth outlook against the rate path** is -0.07, indistinguishable
  from zero (p = 0.77). Growth talk carries no policy-path information;
  inflation talk does.

## Limitations

1. **The 2012 break is imposed, not estimated.** No structural-break
   test was run. The sensitivity table shows the pattern is not
   specific to 2012, but a proper break test would be needed to claim a
   date.
2. **These are correlations.** Nothing here identifies a direction of
   causation. Speeches discussing inflation shortly before a decision
   may be describing a stance already settled internally rather than
   anticipating one.
3. **The measures are generated regressors.** The misclassification
   correction in `src/measurement_error.py` applies to attention
   shares, not to the conditional outlook used here, and the sampling
   error in the LLM codes is not propagated into these standard errors.
   They are conditional on the measures being taken as data.
4. **All caveats in `docs/measurement_notes.md` still apply** — the
   validation was non-blind, single-coder, n = 30, and not seeded.
5. **The standardising moments are treated as known** in the HAC
   standard errors. The effect is second order relative to the
   serial-correlation adjustment, but it is an approximation.
