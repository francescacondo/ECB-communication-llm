# Dictionary Benchmark Notes

What did the language model buy that counting words would not have
given for free? `analysis/16_dictionary_benchmark.py` answers this.

## Design

The comparison is only informative if the dictionary is a good-faith
baseline rather than a strawman, so three things are held fixed
(`src/dictionary_signals.py`):

1. **Same input.** The dictionary reads the truncated excerpts in
   `outputs/production/`, exactly the text the model was shown, not the
   full speeches.
2. **Same codebook.** Every term is drawn from the concepts
   `docs/coding_rules.md` lists as qualifying evidence. The dictionary
   is given the same instructions the prompt was given, in the only
   form a dictionary can accept them.
3. **Same base rates.** Attention thresholds and the outlook neutral
   band are calibrated so the dictionary reproduces the LLM's
   prevalence on the baseline sample. Any difference in agreement is
   then about *which* speeches are flagged, not how many.

The baseline also gets two standard refinements: direction is scored
only inside sentences that mention the topic, and a three-word negation
window flips the sign of negated directional terms.

Point 3 deserves emphasis — **the comparison is deliberately generous
to the dictionary.** Prevalence matching hands it the LLM's base rates,
which a standalone dictionary would not have. Calibration is exact to
within 0.008 except financial stability (0.044, from ties in the
density distribution).

## Level 1: against the hand codes

Both instruments scored on the same thirty hand-coded speeches, by
identical code in `src/agreement.py`.

| Variable | LLM | Dictionary | LLM κ | Dict κ |
| --- | --- | --- | --- | --- |
| Inflation attention | 0.900 | 0.900 | 0.791 | 0.791 |
| Inflation outlook | 0.933 | 0.833 | 0.773 | 0.479 |
| Growth attention | 0.933 | 0.633 | 0.865 | 0.240 |
| Growth outlook | 0.900 | 0.467 | 0.825 | 0.167 |
| Financial stability | 1.000 | 0.833 | 1.000 | 0.619 |
| Uncertainty | 1.000 | 0.700 | 1.000 | 0.389 |

Pooled cell agreement: **170/180 = 0.944 for the LLM, 131/180 = 0.728
for the dictionary.**

Three things in the detail matter more than the headline gap.

**Inflation attention is a tie, exactly.** Both instruments score 0.900
with κ = 0.791 and the same three false positives. Detecting whether a
speech substantively discusses inflation is a solved problem for either
method — the word "inflation" is a near-sufficient statistic. Paying
for an LLM to do this alone would be waste.

**Growth attention is where the dictionary collapses**, at specificity
0.357 against the LLM's 0.857. The vocabulary of real activity —
"growth", "demand", "investment", "output" — saturates ECB prose
regardless of whether activity is the subject, so a density threshold
cannot discriminate. This is a vocabulary-saturation failure, not a
composition failure, and it is the one case where a better-designed
dictionary might close much of the gap.

**Uncertainty is a different failure again.** The dictionary misses six
of nineteen true positives (sensitivity 0.684). The coding rules
explicitly warn that "risks remain broadly balanced" should not count
and that the word "risk" alone is insufficient. A word counter cannot
follow that instruction; the LLM does, scoring 1.000.

## Level 2: do the aggregates differ?

Speech-level disagreement can average out. It partly does:

| Measure | Correlation | Mean, LLM | Mean, dictionary |
| --- | --- | --- | --- |
| `att_pi` | 0.849 | 0.611 | 0.607 |
| `att_fs` | 0.891 | 0.694 | 0.658 |
| `att_unc` | 0.786 | 0.560 | 0.559 |
| `att_y` | 0.510 | 0.813 | 0.811 |
| `out_pi` | 0.482 | −0.076 | 0.199 |
| `out_y` | 0.390 | 0.147 | 0.426 |

Attention series are broadly interchangeable at quarterly frequency.
**Outlook series are not** — they correlate under 0.5, and the
dictionary carries a large systematic hawkish bias (`out_pi` mean +0.20
against −0.08). Words like "pressure" and "strong" are up-terms that
appear constantly in inflation discussion regardless of direction, so
the baseline reads the corpus as persistently more hawkish than either
the LLM or the human coder does.

## Level 3: the result that complicates the story

Repeating `analysis/13`'s correlations with both instruments:

| Measure | Target | LLM | Dictionary |
| --- | --- | --- | --- |
| `out_pi` | HICP inflation | 0.229 (p = 0.13) | **0.618 (p = 0.001)** |
| `out_pi` | Deposit rate, 2q change | 0.309 (p = 0.08) | 0.282 (p = 0.13) |
| `att_pi` | HICP inflation | 0.200 (p = 0.21) | 0.268 (p = 0.13) |

**The dictionary correlates far better with realised inflation than the
LLM does.** Taken at face value this looks like the benchmark winning
the only test that matters.

It is not, and the reason is the most useful thing in this note.

## What each instrument is actually measuring

`docs/coding_rules.md` instructs the coder to classify the
forward-looking trajectory and explicitly **"not merely whether the
current level of inflation is high or low."** Splitting the target
tests whether each instrument honoured that:

| Target | LLM | Dictionary |
| --- | --- | --- |
| HICP level | 0.229 (p = 0.13) | **0.618 (p = 0.001)** |
| HICP change, past quarter | **0.338 (p = 0.02)** | 0.268 (p = 0.06) |
| HICP change, next 2 quarters | −0.011 (p = 0.93) | −0.041 (p = 0.79) |

The dictionary's strongest relationship by a wide margin is with the
**level**. The LLM's strongest is with the **change**, and its level
correlation is less than half the dictionary's.

That is exactly the pattern the coding rules predict if the LLM
complied with the instruction and the dictionary could not. The
vocabulary describing a high level of inflation — "elevated", "strong",
"pressure" — is the same vocabulary describing a rising one, so a bag
of words cannot separate level from trajectory even in principle. The
LLM's *weaker* correlation with contemporaneous HICP is evidence it is
measuring the intended construct, not evidence it is a worse
instrument.

**The general lesson: a higher correlation with an external series is
not by itself evidence of a better measure.** It depends entirely on
what the measure was supposed to capture. Criterion validity without
construct validity is a trap, and this is a clean worked example of it.

One honest qualification: neither instrument predicts *future*
inflation (both ≈ 0, p > 0.79). "Forward-looking" here means the
recent trajectory the speaker is describing, not a forecast with
predictive content. Nothing in this project should be described as
predicting inflation.

## What the benchmark supports

- The LLM is clearly better against hand codes, at 0.944 against 0.728
  pooled, and the advantage is concentrated where the coding rules
  demand composition rather than vocabulary.
- For **attention** alone, a dictionary is a defensible cheap
  substitute at quarterly frequency, except for growth.
- For **outlook**, the two instruments measure different constructs.
  This is not a quality gap that a better word list would close.

## Limitations

1. **n = 30, non-blind, single coder.** Every caveat in
   `docs/measurement_notes.md` applies to both columns of the horse
   race equally. The gap is large enough that it is unlikely to be an
   artefact, but the individual κ values are imprecise — several
   dictionary intervals include zero.
2. **One dictionary.** These are hand-built lists, not Loughran-McDonald
   or Apel-Blix Grimaldi. A published dictionary might do better,
   particularly on growth attention. The claim is not that no
   dictionary can do this, only that a good-faith implementation of
   *this* codebook cannot.
3. **Prevalence matching flatters the baseline**, as described above.
4. **Thresholds are calibrated on the full baseline sample**, including
   the thirty audit speeches. With n = 30 out of 2,717 the resulting
   optimism is negligible, but it is not zero.
