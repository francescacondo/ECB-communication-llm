# Adjudication Protocol

> **What actually happened.** This protocol asked for a two-person
> session. It did not take place: the two coders never met and the sheet
> was completed by one of them alone. The resulting rules are sound and
> round 4 shows they work, but they are one coder's articulation of
> their own standard rather than a negotiated one. Any future round
> should run the session this document describes.

The purpose of this session is **not** to settle 119 codes. It is to
discover the decision rules the two coders were using implicitly, and to
write them into `docs/coding_rules.md` so the next round has something
determinate to code against.

The resolved codes are a by-product. The rules are the output.

## Why this is necessary

Round 3 established that two coders working independently from the
current rules agree on only 0.694 of cells on identical text, and that
disagreement is concentrated:

| Variable | Human–human κ | Disagreements | Of which identical text |
| --- | --- | --- | --- |
| Financial stability | −0.154 | 21 | 5 |
| Inflation outlook | 0.059 | 10 | 4 |
| Uncertainty | 0.182 | 31 | 6 |
| Growth attention | 0.500 | 17 | 3 |
| Inflation attention | 0.636 | 23 | 2 |
| Growth outlook | 0.714 | 17 | 2 |

A κ of −0.154 means two careful readers do worse than chance. That is
not a hard task; it is an underdetermined one.

## Start here: a defect that has now been fixed

`docs/coding_rules.md` was corrupted, and it was corrupted in a way that
mechanically produces the disagreements this session was convened to
adjudicate. The repair was made on 2026-08-27 and is recorded in the
provenance note at the top of that file.

The exclusion clause for *technical, operational, legal, credit,
settlement and financial-system risks* belongs to section 6,
uncertainty. In the corrupted file it had been misfiled into section 5,
financial stability, where it directly contradicted section 5's own
inclusion list — that list instructs the coder to code `1` for systemic
risk and financial fragility, while the misfiled clause instructs the
coder not to code `1` merely because financial-system risks are
discussed. Section 6 was left without the clause entirely.

`prompts/extraction_prompt.txt`, which is the specification the
extraction was actually generated under, has always had it in the right
place. **The model was working from the correct rules. The human coders
were not.**

The consequences are visible in the base rates, and they run in exactly
the direction the corruption predicts:

| Variable | Coder A | Coder B | Model |
| --- | --- | --- | --- |
| Financial stability | 0.509 | 0.868 | 0.755 |
| Uncertainty | 0.321 | 0.868 | 0.660 |

Coder A followed the misfiled exclusion and coded financial stability
low. Coder B followed the inclusion list and coded it high. Section 6
had lost its exclusion, so nothing told coder B to keep operational and
credit risk out of uncertainty, and uncertainty was coded high too. The
model, holding the uncorrupted spec, sits between the two humans on both
variables.

Those two variables account for 34 of the 84 model–human disagreements
in round 2 and 52 of the 119 human–human disagreements in round 3 —
roughly 40% of everything this exercise set out to explain.

### What this means for the session

**Re-read the corrected sections 5 and 6 before starting, and expect a
large share of the financial-stability and uncertainty rows to
dissolve.** Where a disagreement is explained entirely by one coder
following the misfiled clause, the resolution is not a new rule; it is
the corrected rule, and `rule_is_new` should be `no`.

What remains after that subtraction is the genuine ambiguity, and it is
that residue the revision should be built on. Do not write new rules to
cover cases the repaired rules already decide.

The human–human κ of −0.154 on financial stability should therefore be
read as an artefact of a corrupted instruction rather than as evidence
that the construct is unmeasurable. Whether any indeterminacy survives
the repair is exactly what this session establishes.

## The sheet

`outputs/audit/adjudication_sheet.csv`, one row per disagreeing cell,
sorted so the rows that speak to the rules come first.

The `evidence` column distinguishes the two kinds of row. **Identical
text** means the speech was never truncated, so both coders read exactly
the same words and any disagreement is pure interpretation — 22 rows.
**B saw excerpt only** means coder B read the first 2,000 words while
coder A read the whole speech, so the disagreement may be about the
missing text rather than about the rules — 97 rows.

Work the 22 identical-text rows first. They are the evidence about the
codebook. The other 97 are worth resolving for the record, but a rule
should not be changed on their strength alone, because you cannot tell
whether the coders disagreed about meaning or simply read different
words.

Full speech text for every row is in
`outputs/audit/speech_texts_r2/`, named in the `text_file` column.

### Columns to fill

`resolved_code` — the code both coders agree on after discussion. If no
agreement is reached, leave it blank and say why in `rule_implication`;
an unresolvable case is itself a finding about the rules.

`rule_implication` — the general principle the resolution rests on,
written so that it decides future cases. "We agreed this is a 1 because
the speech treats bank resilience as a subject rather than mentioning it
in passing" is useful. "Agreed 1" is not.

`rule_is_new` — `yes` if the principle is not already stated in
`docs/coding_rules.md`. These are the entries that become the revision.

## Rules for the session

**Do not consult the model's codes.** They are deliberately absent from
the sheet. These adjudicated codes will become the reference against
which the model's sensitivity and specificity are re-estimated, and a
reference contaminated by the model's own output cannot measure it. The
model's codes are in `outputs/validation/validation_sample_r2_key.csv`
and should stay closed until the adjudication is finished.

**Adjudicate the rule, not the speech.** The question is never "what is
the right code here" in isolation; it is "what rule would decide this
case and the next hundred like it". If a resolution cannot be stated as
a rule, the case has not been adjudicated, only settled.

**Record disagreements that survive.** If both coders still disagree
after discussion, that is the most valuable row in the sheet. It means
the construct itself is contested, and the honest response may be to
narrow the variable's definition or to drop it.

**Watch for systematic threshold differences.** Coder B codes `1` in 87%
of speeches on three variables against coder A's 32 to 59%. That is not
a series of independent judgement calls; it is a different standard for
what "substantively discussed" means. If that is the finding, the rule
needed is a threshold rule — how much of a speech, or what kind of
treatment, counts as substantive — and one such rule will resolve many
rows at once.

## What each variable's ambiguity looks like

**Financial stability.** With the misfiled clause removed, the question
that remains is the boundary between system-wide risk to financial
intermediation and operational, settlement or payments risk. The
repaired section 5 excludes only ordinary references to markets, rates
and transmission, so a rule may still be needed for the payments and
operational cases such as `ecb_1732`.

**Uncertainty.** Section 6 now carries its exclusion clause again,
which should remove the cases where financial-system risk was read as
uncertainty attention. What the rules still lack is a positive
threshold: they say what does not count but not what does. Coder A reads acknowledged forecast
uncertainty as insufficient (`ecb_2421`); coder B reads it as
sufficient. A workable rule needs to say what distinguishes emphasis
from acknowledgement.

**Inflation outlook.** Two distinct problems appear in the notes. One is
the horizon: `ecb_2388` has headline inflation expected to rise in the
coming months while medium-term inflation stays below the aim, and the
coders split `+1` against `−1` — a genuine sign reversal driven by which
horizon counts. The other is whether a directional statement that is
conditional or hedged counts as directional at all. The rules should
name the horizon explicitly and say how to treat conditionality.

## After the session

1. Revise `docs/coding_rules.md` from the `rule_is_new` entries. Keep the
   revision under version control as a distinct commit so the before and
   after are legible.
2. Re-run `src/select_validation_sample.py` with a new seed to draw
   40–50 fresh speeches, excluding everything coded so far.
3. Have **both** coders code that sample blind, on identical text, so the
   next ceiling estimate does not rest on twelve speeches.
4. Only then re-estimate the misclassification correction, and report
   every model–human agreement alongside the human–human figure for the
   same variable.

The speeches coded in rounds 1 to 3 are now a development set. Accuracy
against the revised rules must be reported on the new sample, or the
revision will simply have been fitted to the cases that motivated it.
