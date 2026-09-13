# Coding Rules for ECB Speech Signal Extraction

> **Provenance.** This file is the human-facing copy of the
> specification in `prompts/extraction_prompt.txt`, which is what the
> extraction was actually generated under. The two must say the same
> thing. They did not between the initial commit and 2026-08-27: the
> exclusion clause for technical, operational, legal, credit,
> settlement and financial-system risks had been misfiled from
> section 6 into section 5, where it directly contradicted section 5's
> own inclusion list and left section 6 without it. Human coders in
> validation rounds 1 to 3 worked from the corrupted copy; the model
> did not. The divergence originated at the revision that produced
> version 1 of the prompt, which added the clause to the prompt's section 6
> and to the codebook's section 5; see `prompts/README.md`. Check
> this file against the prompt before any coding round.
>
> The 2026-08-27 repair made two changes: it moved the misfiled clause
> back to section 6, and it added to section 5 the override the prompt
> has always carried ("unless systemic or stability-related concerns are
> substantively addressed"), which this file had lacked since the early
> draft.
> One difference remains unrepaired: the prompt's general rules ask the
> coder to distinguish current conditions from the forward-looking
> outlook, with a worked example. That principle appears here inside
> sections 2 and 4 but not among the general principles. It is left
> alone because round 4 is already coded against version 2.
>
> **Version 2, 2026-08-27.** Six decision rules were added below,
> derived from the 119 disagreements between two independent coders
> (`outputs/audit/adjudication_sheet.csv`). Sixty of those
> disagreements reduced to those six rules; the remaining fifty-nine
> were already decided by the repaired text above.
>
> The rules were written by one of the two coders working alone, not by
> the two together — see the header of
> `docs/adjudication_protocol.md`. They are that coder's articulation of
> a standard they had been applying, and that coder is the stricter of
> the two.
>
> The extraction in `outputs/extracted_signals/` was generated under
> version 1, preserved byte-for-byte in
> `prompts/extraction_prompt.txt`. The revised rules are mirrored in
> `prompts/extraction_prompt_v2.txt` for any future extraction. **The
> current model codes therefore predate these rules and were produced
> without them**, which must be stated whenever the model is scored
> against human codes made under version 2.

## Purpose

The objective is to extract a small set of economically interpretable
communication measures from ECB speeches using an LLM.

The variables are generated measurements rather than ground truth.
Classifications must be based only on the supplied speech text.

The model must not use outside knowledge about the speaker, date,
macroeconomic conditions, or subsequent events.

---

## 1. Inflation attention

Variable:

`inflation_attention`

Allowed values:

- `0` = inflation or price developments are not substantively discussed
- `1` = inflation or price developments are substantively discussed

### Coding rule

Code `1` when inflation, prices, price pressures, price stability,
inflation expectations, wages as a source of inflation pressure,
or related inflation developments form a meaningful part of the discussion.

A passing reference is not sufficient.

### Examples

Code `1`:

- discussion of current inflation developments;
- discussion of inflation forecasts;
- discussion of upside or downside inflation risks;
- discussion of inflation persistence;
- discussion of price pressures or inflation expectations.

Code `0`:

- inflation is mentioned only briefly;
- price stability appears only as part of the ECB's mandate;
- the speech is primarily about payments, regulation, institutional issues,
  or another topic without substantive inflation discussion.

---

## 2. Inflation outlook

Variable:

`inflation_outlook`

Allowed values:

- `-1` = downside or disinflationary emphasis
- `0` = balanced, neutral, ambiguous, mixed, purely descriptive,
  or inflation not substantively discussed
- `1` = upside or inflationary emphasis

### Coding rule

This variable measures the dominant directional emphasis of the discussion
about inflation.

Classify the direction of the forward-looking inflation outlook or dominant
expected trajectory, not merely whether the current level of inflation is
high or low.

Code `-1` when the speech emphasizes developments such as:

- declining inflation;
- weak price pressures;
- subdued wage or cost pressures;
- inflation projected below target;
- downside risks to inflation;
- disinflationary forces.

Code `1` when the speech emphasizes developments such as:

- rising inflation;
- persistent inflation;
- strong price or wage pressures;
- inflation projected above target;
- upside inflation risks;
- inflationary forces.

Code `0` when:

- the discussion is balanced;
- both upside and downside risks are emphasized without a clear dominant direction;
- the text is descriptive rather than directional;
- the evidence is insufficient;
- inflation is not substantively discussed.

### Conflicting horizons

When near-term headline inflation and an explicitly stated medium-term
or policy-relevant inflation outlook point in opposite directions,
classify using the medium-term policy-relevant horizon. A transitory
near-term movement does not overturn an explicit medium-term direction.

### Logical constraint

If:

`inflation_attention = 0`

then:

`inflation_outlook = 0`

---

## 3. Growth attention

Variable:

`growth_attention`

Allowed values:

- `0` = real economic activity is not substantively discussed
- `1` = real economic activity is substantively discussed

### Coding rule

Code `1` when the speech meaningfully discusses topics such as:

- real GDP;
- economic growth;
- output;
- domestic demand;
- consumption;
- investment;
- employment when discussed as part of the activity outlook;
- recession or recovery;
- real economic activity.

A passing reference is not sufficient.

---

## 4. Growth outlook

Variable:

`growth_outlook`

Allowed values:

- `-1` = weakening or downside emphasis
- `0` = balanced, neutral, ambiguous, mixed, purely descriptive,
  or growth not substantively discussed
- `1` = strengthening or upside emphasis

### Coding rule

Classify the direction of the forward-looking growth outlook or dominant
expected trajectory, not merely whether the current level of economic
activity is strong or weak.

Code `-1` when the dominant emphasis concerns:

- weakening activity;
- slowing growth;
- recession;
- falling demand;
- deteriorating economic conditions;
- downside risks to growth.

Code `1` when the dominant emphasis concerns:

- strengthening activity;
- accelerating growth;
- recovery;
- increasing demand;
- improving economic conditions;
- upside risks to growth.

Code `0` when:

- the discussion is balanced;
- signals are mixed;
- the text is descriptive rather than directional;
- evidence is insufficient;
- growth is not substantively discussed.

### Logical constraint

If:

`growth_attention = 0`

then:

`growth_outlook = 0`

---

## 5. Financial stability attention

Variable:

`financial_stability_attention`

Allowed values:

- `0` = financial stability is not substantively discussed
- `1` = financial stability is substantively discussed

### Coding rule

Code `1` when the speech meaningfully discusses financial-system risks,
including topics such as:

- banking-system resilience;
- bank solvency or liquidity;
- systemic risk;
- financial fragility;
- financial-sector vulnerabilities;
- market stress;
- contagion;
- macroprudential risks;
- financial stability.

Do not code `1` merely because financial markets, interest rates, or
monetary-policy transmission are mentioned, unless systemic or
stability-related concerns are substantively addressed.

### Payments and operations boundary

Safety, security, settlement, payment-system continuity or operational
resilience does not by itself constitute financial-stability attention.
Code `1` only when the discussion is explicitly tied to system-wide
impairment, contagion, banking-system resilience, macroprudential risk,
or the capacity of financial intermediation to serve the economy.

---

## 6. Uncertainty attention

Variable:

`uncertainty_attention`

Allowed values:

- `0` = uncertainty or risk is not substantively emphasized
- `1` = uncertainty or risk is substantively emphasized

### Coding rule

Code `1` when uncertainty, unusually wide risks, unpredictability,
or alternative scenarios are an important part of the economic discussion.

Examples include:

- exceptional uncertainty;
- unusually high uncertainty;
- major geopolitical uncertainty;
- substantial uncertainty around forecasts;
- unusually wide ranges of possible outcomes;
- strong emphasis on risks surrounding the outlook.

Do not code `1` simply because the word "risk" appears.

Routine central-bank phrases such as "risks remain broadly balanced"
should not automatically imply strong uncertainty attention.

Do not code `1` merely because technical, operational, legal, credit,
settlement, or financial-system risks are discussed. Such risks count as
uncertainty attention only when the text substantively emphasizes
uncertainty, unpredictability, unusually wide possible outcomes, or
difficulty assessing the outlook.

### Threshold

Acknowledged, intrinsic, methodological or routine forecast uncertainty
is insufficient. Code `1` only when unpredictability, unusually wide
outcomes, exceptional uncertainty, or uncertainty surrounding the
economic outlook is itself emphasized, or materially shapes the
economic or policy argument being made.

---

## General coding principles

1. Use only the supplied text.
2. Do not use outside knowledge.
3. Code substantive discussion rather than keyword presence.
4. Prefer `0` when directional evidence is genuinely ambiguous.
5. Do not infer the speaker's view beyond what is explicitly supported.
6. For outlook variables, classify the dominant directional emphasis rather
   than counting positive and negative words.
7. When inflation or growth is not substantively discussed, the associated
   outlook variable must be `0`.

---

## Decision rules from the round-3 disagreements

These were derived from the disagreements between two independent
coders in round 3, and written by one of them. They resolve cases the rules above left open, and
they apply across variables.

### 8. Substantive-attention threshold

Code an attention variable `1` only when the construct is an object of
developed analysis: several connected claims, a focused passage, or a
material role in the argument. An isolated consequence, a mandate
reference, an illustrative example or a passing mention is `0`.

In a very short speech, proportional centrality can satisfy the
threshold. What matters is whether the construct is being analysed, not
how many words are spent on it in absolute terms.

### 9. Attention does not require a forecast

Attention measures whether a construct is substantively analysed, not
whether the speech offers a forecast about it. A developed conceptual,
historical or simulation-based treatment receives attention `1` even
when the associated outlook is `0`.

The independence runs in this direction only. The constraint in the
reverse direction still binds: outlook must be `0` when attention
is `0`.

### 10. Outlook versus mechanism

A hypothetical scenario, a causal channel, the intended effect of a
structural reform, a policy simulation or a backward-looking episode
does not by itself establish an outlook direction.

Assign `+1` or `-1` only when the speaker presents that direction as the
expected or baseline forward trajectory, or as the clearly dominant
forward risk assessment. Describing how something would work, or how it
did work, is not the same as saying it is expected to happen.