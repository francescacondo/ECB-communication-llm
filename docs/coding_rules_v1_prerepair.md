# Coding Rules for ECB Speech Signal Extraction

> **Superseded, and corrupted.** This is the codebook as it stood for
> validation rounds 1 to 3, reconstructed for the record. Its section 5
> carries an exclusion clause belonging to section 6, where it
> contradicts section 5's own inclusion list, and its section 5 lacks
> the systemic-concerns override the prompt has always carried. Do not
> code against this file. The current codebook is `coding_rules.md`,
> at version 2; the earlier draft is `coding_rules_early_draft.md`.

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

Do not code `1` merely because financial markets, interest rates,
or monetary-policy transmission are mentioned. Do not code `1` merely because technical, operational, legal, credit,
settlement, or financial-system risks are discussed. Such risks count as
uncertainty attention only when the text substantively emphasizes
uncertainty, unpredictability, unusually wide possible outcomes, or
difficulty assessing the outlook.


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

