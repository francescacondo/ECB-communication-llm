# Measurement Notes: Validating the LLM Signals

The six signals in `outputs/extracted_signals/ecb_signals_gpt5mini_v1.csv`
are generated measurements produced by `gpt-5-mini` under prompt
version 1. This note records what is known about their accuracy, and how
it came to be known.

Four validation rounds were run. Each was designed to remove a specific
weakness in the one before, and each changed the answer. The sequence is
kept in full rather than tidied into a final result, because the
sequence is what the project actually established.

---

## Summary of the current position

**The ceiling is 0.810.** Two coders working independently from the
revised codebook on identical full text agree on 243 of 300 cells,
interval [0.761, 0.853]. Every statement about accuracy below is
relative to that.

**The model does not miss topics.** Sensitivity is 1.000 in both rounds
that could measure it, including one where the coder read a full speech
the model had seen only two-thirds of. This is the only claim that has
survived all four rounds unchanged.

**The model currently sits below the ceiling**, at 0.737 and 0.743
against the two coders. It was extracted under version 1 of the rules
and has never seen the six decision rules the humans coded under.

**Uncertainty attention is not usable.** After four rounds and one
targeted rule, human–human kappa is 0.280 with an interval of
[−0.009, 0.544] that includes zero.

**No corrected attention level should be reported yet.** The correction
parameters in `outputs/validation/measurement_error_params.json` are
still round 1's, which are known to be optimistic, and no replacement
has been estimated against the revised rules.

---

## Round 1 — thirty speeches, non-blind, excerpt

`data/human/human_audit_30.csv`, scored by
`analysis/15_validation_stats.py`.

Cell agreement 170/180 = 0.944, Cohen's kappa 0.773 to 1.000.
Sensitivity 1.000 for every attention variable. Specificity 0.850
(inflation), 0.857 (growth), 1.000 (financial stability), 1.000
(uncertainty). All five outlook disagreements ran directional to
neutral, with no sign reversals.

Four weaknesses were recorded at the time: not blind, one coder, n = 30,
and no seeded sampling frame. Two more surfaced later.

The coder read **the same 2,000-word excerpt the model saw** — all
thirty rows match the production excerpt exactly, and twenty-five of the
thirty speeches were truncated. That design cannot observe a
truncation-induced miss, so the reported sensitivity of 1.000 was close
to vacuous. And one speech, `ecb_2682`, is a 69-word slide stub outside
the baseline sample contributing six trivially easy agreements;
excluding it moves agreement to 0.9425.

Round 1 is now a development set, not evidence about accuracy.

---

## Round 2 — sixty speeches, blind, full speech, seeded

Drawn by `src/select_validation_sample.py` with seed 20260826,
stratified on the number of attention variables the model codes one and
crossed with era, mildly oversampling the top because sensitivity was
1.000 and every observed error was therefore a false positive. Coded
blind against the full speech by coder A
(`data/human/round2_coder_a_full.csv`). Scored by
`analysis/17_validation_round2.py`.

### The estimator has to change

Selection depends on the model's codes, so sample proportions of
sensitivity and specificity are biased and **raw agreement in this
sample is not comparable with round 1's 0.944**. What selection does not
depend on is the human code given the model code, so the predictive
values are unbiased, and the rates follow by Bayes using the model's
marginal rate `q`, counted exactly over all 2,717 baseline speeches
rather than estimated:

    p     = PPV * q + (1 - NPV) * (1 - q)
    alpha = PPV * q / p
    beta  = NPV * (1 - q) / (1 - p)

### Results

| Variable | PPV | NPV | Sensitivity | Specificity |
| --- | --- | --- | --- | --- |
| Inflation attention | 0.737 | 1.000 | 1.000 | 0.718 [0.57, 0.83] |
| Growth attention | 0.714 | 1.000 | 1.000 | 0.446 [0.28, 0.58] |
| Financial stability | 0.696 | 1.000 | 1.000 | 0.569 [0.40, 0.69] |
| Uncertainty | 0.487 | 1.000 | 1.000 | 0.606 [0.50, 0.69] |

Two findings survived everything that followed.

**Sensitivity is 1.000 against the full speech.** No false negatives in
any of the sixty-eight cells the model coded zero, while forty-one of
sixty speeches were truncated with a median of 1,194 words withheld.

**The no-sign-reversals claim did not survive.** One inflation-outlook
sign reversal and three neutral-to-directional growth-outlook errors,
where round 1 found none of either. Outlook error remains predominantly
attenuating — six of seven and sixteen of nineteen disagreements run
directional to neutral — but the clean version of the claim is gone.

Read alone, the table above says specificity collapsed. That reading was
published in an earlier draft of this note and was wrong.

---

## Round 3 — an independent second coder

Coder B coded the excerpt for fifty-three of the same speeches from the
codebook, without consulting coder A
(`data/human/round3_coder_b_excerpt.csv`). Scored by
`analysis/18_coder_reliability.py`.

Twelve of those speeches were never truncated, so the excerpt is the
whole speech and all three coders read identical words.

| Pair | Agreement | 95% interval |
| --- | --- | --- |
| Human A vs Human B | 0.694 | [0.575, 0.798] |
| Model vs Human A | 0.889 | [0.793, 0.951] |
| Model vs Human B | 0.778 | [0.664, 0.867] |

Because the model and coder B both read the excerpt, model-versus-B is
also computable on all fifty-three speeches, where it pools to 0.739.

**Round 2 had been measuring the model against a moving target.** With a
second coder disagreeing on 30.6% of cells, specificity estimated
against one coder is a property of the instrument relative to that
coder, not of the instrument.

### Truncation, decomposed

On control speeches both coders read identical words, so disagreement
there is coder difference alone; on truncated speeches coder B
additionally saw less text. Averaged over variables, coder difference
accounts for 0.306 of cells and withheld text adds 0.089.

**Truncation is real and second-order**, dominated by coder
indeterminacy roughly three to one. The 2,000-word cap is not the main
thing degrading the measures.

### A corrupted codebook

Comparing `docs/coding_rules.md` with `prompts/extraction_prompt.txt`
showed the two had diverged in four places, one of them a contradiction.

1. **Section 5 carried a clause belonging to section 6.** The exclusion
   for technical, operational, legal, credit, settlement and
   financial-system risks is a rule about uncertainty. Sitting in the
   financial-stability section it told the coder not to code 1 merely
   because financial-system risks are discussed, contradicting that
   section's own inclusion list, which names systemic risk and financial
   fragility as qualifying evidence.
2. **Section 6 was left without it**, so nothing kept operational and
   credit risk out of uncertainty attention.
3. **Section 5 lacked an override the prompt carries.** The prompt
   excludes ordinary references to markets, rates and transmission
   *"unless systemic or stability-related concerns are substantively
   addressed"*. The codebook stated the exclusion flatly in the early
   draft and every version since. A coder following it literally codes 0 where
   markets are the vehicle even if systemic concerns are the subject;
   the model, holding the override, codes 1. This compounds the
   misfiled clause rather than offsetting it.
4. **The prompt's general rules contain one the codebook does not**:
   distinguish current conditions from the forward-looking outlook, with
   the example that weak current activity can coexist with a positive
   growth outlook. The codebook states the principle inside sections 2
   and 4 but not among its general principles and without the example.
   A difference of emphasis rather than a contradiction, and the only
   one **not** repaired — round 4 is already coded against version 2 and
   a further edit would open a fresh mismatch.

Everything else agreed: the remaining five variables' rules, both
logical constraints and the other general principles match
substantively, differing only in wording and in examples.

The repair therefore made two changes, not one: it moved the misfiled
clause back to section 6 and added the missing override to section 5.

**The model was working from the correct rules. The human coders were
not**, in all three rounds to that point. The base rates show the
consequence:

| Variable | Coder A | Coder B | Model |
| --- | --- | --- | --- |
| Financial stability | 0.509 | 0.868 | 0.755 |
| Uncertainty | 0.321 | 0.868 | 0.660 |

Coder A followed the misfiled exclusion and coded financial stability
low; coder B followed the inclusion list and coded it high; section 6
had lost its exclusion, so nothing kept operational and credit risk out
of uncertainty. The model sits between the two humans on both. Those two
variables account for 34 of round 2's 84 model–human disagreements and
52 of round 3's 119 human–human disagreements.

The file was repaired on 2026-08-27 and now carries a provenance note.

---

## From disagreement to revised rules

The 119 disagreements were worked through case by case
(`data/human/round3_adjudication_completed.csv`), following
`docs/adjudication_protocol.md`. The sheet was built by
`src/build_adjudication_sheet.py` and deliberately excluded the model's
codes, because the resolved codes were to become a reference standard
and a reference contaminated by the instrument cannot measure it.

**This was not a joint adjudication.** The protocol asked for a
two-person session and warned that solo completion would yield one
coder's opinion a second time. The two coders never met; the sheet was
completed by one of them. The artefact shows it: 119 rows carry only 16
distinct rule texts applied verbatim, and the resolution takes coder A's
code 100 times out of 119 and 20 of 22 on identical text. Read the rules
below as coder A articulating a standard they had been applying
implicitly, not as a negotiated consensus — and note that coder A is the
stricter of the two, so the rules encode the stricter thresholds.

Fifty-nine rows resolved to rules already in the repaired codebook. As
predicted, 45 of the 52 financial-stability and uncertainty rows were in
that group.

The remaining sixty reduced to **six distinct decision rules**:

| Rule | Rows | Placed in |
| --- | --- | --- |
| Substantive-attention threshold | 25 | general principles §8 |
| Attention does not require a forecast | 16 | general principles §9 |
| Outlook versus mechanism | 11 | general principles §10 |
| Uncertainty threshold | 5 | §6 |
| Conflicting inflation horizons | 2 | §2 |
| Payments and operations boundary | 1 | §5 |

The largest gap was not where round 3 pointed. Every inflation- and
growth-attention disagreement needed new rules, and they reduce to two:
what makes analysis substantive, and whether attention requires a
forecast. Those variables had the *best* human–human kappa, so nothing
in the reliability statistics identified them. The codebook had said
"a passing reference is not sufficient" and never said what is.

The resolution took coder A's code 100 times out of 119, and 20 of 22 on
identical text. On truncated rows that is expected, since coder B was
reading less text. On identical text it is not, and it cannot be
determined from the data whether coder A was more accurate or the
session deferred to them. The adjudicated reference is therefore closer
to coder A's codes than the label suggests.

Against that reference the model agrees on 0.917 of cells on identical
text and 0.802 across all fifty-three.

---

## Round 4 — the ceiling under revised rules

Fifty speeches nobody had coded before, drawn with seed 20260828
excluding all ninety previously coded, and with **flat stratification
weights**. The weighting changed deliberately: round 2 oversampled
speeches the model codes one because it was estimating specificity,
whereas round 4 estimates a property of the coding task, so the sample
should represent the corpus and must not be conditional on the model's
codes. Both coders worked blind from identical full text
(`data/human/round4_coder_1.csv`, `round4_coder_2.csv`). Scored by
`analysis/19_round4_ceiling.py`.

### The ceiling

**243/300 = 0.810, interval [0.761, 0.853]**, against 0.694 on twelve
speeches under the corrupted rules.

Restricted to speeches the model codes on three or more topics, matching
where round 3's stratified sample was concentrated, the figure is 0.817.
The improvement is therefore not an artefact of the change in sampling
weights.

### Did the six rules work?

| Variable | kappa round 3 | kappa round 4 | Change |
| --- | --- | --- | --- |
| Financial stability | −0.154 | 0.560 [0.32, 0.78] | +0.714 |
| Inflation outlook | 0.059 | 0.631 [0.41, 0.83] | +0.572 |
| Inflation attention | 0.636 | 0.800 [0.63, 0.96] | +0.164 |
| Growth attention | 0.500 | 0.662 [0.41, 0.87] | +0.162 |
| Uncertainty | 0.182 | 0.280 [−0.01, 0.54] | +0.098 |
| Growth outlook | 0.714 | 0.650 [0.44, 0.83] | −0.064 |

Five of six improved. Financial stability moving from worse-than-chance
to 0.560 confirms the corrupted-clause diagnosis. Inflation outlook at
0.631 says the medium-term horizon rule did real work. Growth outlook's
small fall is within noise.

**Uncertainty did not move materially and its interval includes zero.**
It has now failed four rounds and one targeted rule. The recommendation
is to stop patching the rule and reconsider whether the construct is
single: the codebook's own examples mix exceptional-uncertainty
language, geopolitical risk and forecast dispersion, which may not be
one thing.

### The model against the ceiling

| | Pooled agreement |
| --- | --- |
| Coder 1 vs coder 2 | 0.810 |
| Model vs coder 1 | 0.737 |
| Model vs coder 2 | 0.743 |

This reverses round 3's finding that the model sat inside the human
band. **The model did not get worse; the humans got better.** Its codes
were produced under version 1 and have never seen the six rules.

Per-variable, the model's weakest is uncertainty at kappa 0.247 and
0.191 — the same variable the humans cannot agree on, which is what one
would expect if the construct rather than the instrument is at fault.

The gap between 0.74 and 0.81 is the quantified case for re-extracting
under `prompts/extraction_prompt_v2.txt`. Whether it is worth the cost
is a separate question, but it is now a measurable claim.

---

## What this means for the published measures

The correction in `src/measurement_error.py` inverts one-sided
misclassification, and
`outputs/validation/measurement_error_params.json` still holds round 1's
parameters, which are optimistic. Substituting round 2's would give:

| Variable | Raw | Corrected (R1) | Corrected (R2) |
| --- | --- | --- | --- |
| Inflation attention | 0.599 | 0.528 | 0.442 |
| Growth attention | 0.813 | 0.782 | 0.581 |
| Financial stability | 0.714 | 0.714 | 0.496 |
| Uncertainty | 0.559 | 0.559 | 0.272 |

**This has deliberately not been done.** Round 3 showed round 2's
specificities are coder-specific, and round 4 changed the rules under
which any future reference will be produced. Re-estimating now would
replace one coder-dependent number with another. The correction should
be re-estimated only after a re-extraction under version 2, against a
reference built from the revised rules.

The external validation in `analysis/13_external_validation.py` uses the
conditional outlook, which the correction never touched, so its results
are mechanically unaffected. Its inflation-outlook measure now has a
human–human kappa of 0.631, which is respectable but not high, and that
qualifies rather than supports the analysis.

---

## The transferable finding

**Single-coder validation cannot distinguish model error from an
underdetermined codebook.** The same instrument looked near-perfect
against a non-blind coder, badly degraded against a blind one, and
unremarkable once a second coder established what agreement is
achievable at all. Nothing about the instrument changed across those
three readings.

An agreement rate of 0.767 is uninterpretable without knowing that two
humans achieve 0.694 on the same task, and both are uninterpretable
without knowing that one of the rules they were reading contradicted
itself. Most of the literature applying language models as annotators
reports the first number and neither of the others.

---

## Limitations

1. **The adjudicated reference leans towards coder A**, 100 of 119 and
   20 of 22 on identical text, and it cannot be established from the
   data whether that reflects accuracy or deference.
2. **Round 4's per-variable intervals remain wide.** Uncertainty's
   includes zero; growth attention's runs [0.41, 0.87].
3. **Rounds differ in more than one respect each**, so changes between
   them are only partly attributable. The truncation channel could be
   separated because round 3 happened to supply controls; blindness and
   coder strictness could not.
4. **The model is scored against rules it never received.** Round 4's
   humans coded under version 2; the extraction ran under version 1.
5. **Prompt sensitivity is measured once, on five speeches.**
   `openai_test5_gpt5mini_predecessor.csv` and its version-1 counterpart cover
   the same five speeches with the same model and reasoning effort, so
   the prompt is the only thing that differs: 28 of 30 cells agree.
   Exactly one substantive line separates the two prompt versions, and
   both differences are attributable. `uncertainty_attention` on
   `ecb_0066` moves 1 to 0; that speech is titled "Payments and the
   Eurosystem" and mentions payment 24 times, settlement 10 and
   clearing 3 against a single mention of uncertainty, so it is
   precisely the case the added clause excludes. `growth_attention` on
   `ecb_0175` moves 1 to 0 with no corresponding prompt change, leaving
   run-to-run nondeterminism or cross-clause spillover, which thirty
   cells cannot separate. The wording change did what it was written to
   do; how much the extraction moves when the instructions do not
   remains a one-cell-in-thirty estimate.
6. **The measures are generated regressors** downstream, and none of
   this sampling error is propagated into the standard errors in
   `analysis/13_external_validation.py`.

## Next steps

1. Re-extract under `prompts/extraction_prompt_v2.txt` and re-score
   against round 4's codes. That closes the version mismatch and tests
   whether the 0.74 to 0.81 gap is the rules or the model.
2. Redefine or retire `uncertainty_attention`.
3. Only then re-estimate the misclassification correction, and report
   every model–human figure alongside the human–human figure for the
   same variable.
