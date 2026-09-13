# Contemporaneous Salience

`docs/external_validation_notes.md` records what happens when the
measures are tested against things they need not track: the policy
surprise, the subsequent rate path. Those are forecasting tests, and a
valid attention measure can fail every one of them without being wrong
about anything.

This note records the opposite exercise. If speeches do not discuss
financial stability more when the euro area is under financial stress,
`att_fs` is not measuring what its name says. That is a test a measure
cannot fail and survive. Every pair below is contemporaneous; nothing
here is a prediction.

`analysis/25_salience_validation.py` produces everything, and
`src/download_salience_series.py` downloads the new series.

## The pre-registration

Five pairs and their signs were fixed before any of them was estimated,
and all five are reported below. Four alternative operationalisations
were also run, and all four are reported. **Nine correlations were
estimated in total and nine appear in this note.** No pair was dropped
and no further specification was run and discarded.

The set is recorded in machine-readable form in
`outputs/validation/salience_specification_count.csv`, and the
pre-registration itself is the `PAIRS` list at the top of the analysis
script.

## One premise did not hold

The brief described both growth pairs as already computed in
`analysis/13_external_validation.py`. Only one of them is.

`out_y` against the two-quarter unemployment change is in
`outputs/validation/external_concordance.csv`, and it is reproduced
here to the tenth decimal place; the script asserts that agreement and
fails if it breaks. `att_y` against that same change had never been
run. The stored table pairs `att_y` with the unemployment **level**,
which is a different quantity — and, as the earlier note itself
observes, the level is the variable on which the growth measures behave
oddly. So `att_y` against the change is a new estimate, not a
confirmation, and it is the weakest result in the table.

## Data

Two families were added, both from the ECB Data Portal, both fetched
one series at a time.

**CISS.** Both pre-committed keys resolve, but they are not the same
kind of series:

| Series | Key | Actually published | Coverage |
| --- | --- | --- | --- |
| CISS | `CISS/D.U2.Z0Z.4F.EC.SS_CI.IDX` | **weekly**, Fridays | to 2 May 2025 |
| New CISS | `CISS/D.U2.Z0Z.4F.EC.SS_CIN.IDX` | daily | to 31 Dec 2025 |

Both carry a daily frequency code. Only the second is daily: the first
gives about thirteen observations per quarter against the second's
sixty-five, and stops seven months earlier. The quarterly file carries
an observation count for each so this cannot be forgotten downstream.
`SS_CI` is kept as the primary because it is the series the
pre-registration named first; `SS_CIN` is its alternate.

**Uncertainty.** There is no implied-volatility series on the ECB Data
Portal to use. Searching the portal's web interface for "volatility"
returns only CISS subindices; "VSTOXX" and "implied volatility" return
nothing; the `FM` Datastream block carries EURO STOXX index *levels*
and no volatility counterpart. The proxy is therefore survey-based —
the cross-forecaster dispersion of Survey of Professional Forecasters
point forecasts, reported by the portal as a variance and used here as
its square root:

| Proxy | Key |
| --- | --- |
| Real GDP growth dispersion | `SPF/Q.U2.RGDP.POINT.P9M.Q.VAR` |
| Inflation dispersion | `SPF/M.U2.HICP.POINT.P12M.Q.VAR` |

Growth is primary and inflation the alternate. That order was set on
data quality alone, before any correlation was estimated: the growth
series is evenly spaced at exactly one quarter across the whole sample,
while the inflation series carries an extra February 2020 round that
puts two observations into 2020Q1.

The SPF horizon codes date the **target** window, not the round in
which forecasters answered. For a contemporaneous test the round is
what matters, so both series are shifted back by the offset their
horizon implies — two quarters for `P9M`, three for `P12M`. The offset
is checked against a fact outside the data: the largest real GDP
dispersion in the sample must fall in the April 2020 round, the one
conducted as Europe shut down. It does, at 2.96 against a sample median
of 0.36, and the script raises if it ever stops doing so.

## Results

Newey-West standard errors with the automatic bandwidth, four lags on
every pair; moving-block bootstrap with a block length of five. Holm's
step-down correction across the five pre-registered pairs.

| Pair | Sign | r | HAC se | p | Holm p | HAC 95% | Bootstrap 95% | n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `att_fs` vs CISS | + | **+0.246** | 0.145 | 0.092 | 0.277 | [-0.04, +0.53] | [-0.05, +0.45] | 106 |
| `att_unc` vs SPF growth dispersion | + | **+0.361** | 0.163 | 0.028 | 0.114 | [+0.04, +0.68] | [+0.19, +0.50] | 106 |
| `att_y` vs unemployment, 2q change | + | +0.060 | 0.102 | 0.555 | 0.637 | [-0.14, +0.26] | [-0.13, +0.29] | 102 |
| `out_y` vs unemployment, 2q change | − | **−0.349** | 0.145 | 0.018 | 0.091 | [-0.63, -0.06] | [-0.54, -0.12] | 102 |
| `att_pi` vs \|HICP − 2\| | + | +0.167 | 0.167 | 0.319 | 0.637 | [-0.16, +0.49] | [-0.17, +0.40] | 108 |

**All five carry the predicted sign.** That is the headline, and it is
worth more than any individual p-value: the signs were fixed in
advance, and five for five is not what a set of measures that tracked
nothing would produce.

**Nothing survives Holm at conventional levels.** Two pairs clear 0.05
on the raw p-value and neither clears it after correction. With about a
hundred quarters and both sides of every pair heavily autocorrelated,
this is the same power problem the earlier note ran into, and it should
be reported as such rather than dressed up.

Variants, outside the Holm family because they re-operationalise pairs
already in it:

| Variant | r | HAC se | p | HAC 95% | n |
| --- | --- | --- | --- | --- | --- |
| `att_fs` vs New CISS | +0.227 | 0.151 | 0.137 | [-0.07, +0.52] | 108 |
| `att_unc` vs SPF inflation dispersion | +0.349 | 0.166 | 0.037 | [+0.02, +0.67] | 105 |
| `att_pi` vs signed HICP − 2 | +0.200 | 0.158 | 0.208 | [-0.11, +0.51] | 108 |
| `att_y` vs unemployment level | +0.147 | 0.085 | 0.087 | [-0.02, +0.31] | 104 |

Every variant agrees in sign and closely in magnitude with the pair it
stands in for. The last reproduces the stored concordance figure
exactly, which is a second check that the panel has not moved.

## Reading the two noisy measures

Round 4 puts human–human kappa at 0.560 for financial-stability
attention and 0.280 for uncertainty attention
(`outputs/validation/round4_ceiling.csv`). Both correlations are
attenuated toward zero by an amount this exercise does not identify.

**Financial stability.** +0.246 is a real association in the right
direction, and against a reliability of 0.560 the attenuated estimate
is consistent with a considerably larger true correlation. But it is
the primary pair, it does not reach significance either raw or
corrected, and its HAC interval includes zero. This is genuine but
modest support for `att_fs`, not the strong validation the pair was
set up to look for.

**Uncertainty.** This is the surprise, and it cuts the opposite way to
what the brief anticipated. `att_unc` gives the *largest* correlation
in the pre-registered set, +0.361, and the one whose bootstrap interval
is furthest from zero — despite a reliability of 0.280, the worst of
any measure in the project. A measure that noisy producing the
strongest contemporaneous association is more impressive than the
number alone suggests, not less, because attenuation at that
reliability is severe. The ambiguity the brief warned about — that a
weak result could not be told apart from measurement error — does not
arise, because the result is not weak.

The asymmetry is worth stating plainly: the pair expected to validate
cleanly did so only modestly, and the pair expected to be uninformative
is the strongest in the table.

## What the weak pairs do and do not show

`att_y` against the unemployment change is +0.060, which is nothing.
Growth attention has a reliability of 0.662, so this is not a
measurement-error story; the measure is coded reliably and simply does
not co-move with the labour market. Growth is discussed in nearly every
speech regardless of conditions, which puts a ceiling on how much a
share-of-speeches attention measure can move.

`att_pi` against the absolute inflation gap is +0.167 with p = 0.32.
The signed gap gives +0.200, slightly *larger*. The pre-committed
reasoning was that the absolute deviation should do better, because it
sees the deflation-scare half of the sample; it does not. That is a
small negative result for the reasoning behind the pair, and it is
recorded here rather than buried, though neither estimate is precise
enough to distinguish the two.

## Limitations

1. **Nothing here is significant after correction.** Two pairs reach
   p < 0.05 raw; none does under Holm. The pattern of signs is the
   evidence, not any single interval.

2. **Length is not partialled out.** Attention rises mechanically with
   speech length, and mean speech length trends over the sample. The
   earlier note found this problem afflicts the attention series
   specifically. No length-adjusted variant was run here, because doing
   so after seeing the results would have reopened a specification set
   that was deliberately closed in advance. It is the first thing a
   follow-up should do.

3. **The 2% target is treated as symmetric throughout.** It was not:
   the pre-2021 formulation was "below, but close to, 2%", and the
   symmetric target dates only from the 2021 strategy review. The
   absolute-gap pair imposes one definition across a sample in which
   the objective changed.

4. **Forecaster disagreement is not uncertainty.** Dispersion across
   forecasters and the width of any one forecaster's own distribution
   are different quantities, and they come apart. Disagreement is what
   the Data Portal publishes, so it is what was used.

5. **The measures are generated regressors.** As in the earlier note,
   the sampling error in the LLM codes is not propagated into these
   standard errors; the kappas are reported alongside instead, and the
   attenuation they imply is not corrected for.

6. **CISS in its primary form is weekly and stops in May 2025**, so the
   primary pair rests on 106 of the sample's 108 quarters and on
   thirteen readings per quarter rather than sixty-five.
