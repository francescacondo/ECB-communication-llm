"""
Score the second validation round.

The second round differs from the first in three ways that all matter
for how it must be read:

    1. **Blind.** The coder did not see the model's codes. The first
       round did, so its 0.944 was an upper bound.

    2. **Full speech, not excerpt.** The first round coded the same
       2,000-word excerpt the model saw, which cannot observe a
       truncation-induced miss. This round codes the speech, so the
       error rates it estimates are those of the pipeline rather than
       of the model given its input.

    3. **Stratified on the model's codes**, oversampling speeches the
       model codes one, because sensitivity was estimated at one
       throughout and every observed error was therefore a false
       positive.

Point 3 changes the estimator, and using the first round's would give
the wrong answer. Selection depends on the model's code, so the sample
proportions of sensitivity and specificity are biased. What selection
does not depend on is the human code given the model code, so these
are estimable without bias:

    PPV = Pr(human = 1 | model = 1),
    NPV = Pr(human = 0 | model = 0).

The rates of interest follow by Bayes, using the model's marginal base
rate q = Pr(model = 1), which is not estimated but counted exactly over
all baseline speeches:

    p     = PPV * q + (1 - NPV) * (1 - q)
    alpha = PPV * q / p
    beta  = NPV * (1 - q) / (1 - p)

For the same reason, **raw agreement in this sample is not comparable
to the first round's 0.944**. The sample was built to concentrate on
the cells where errors live. The comparable quantities are alpha and
beta.

Run from the project root:

    python analysis/17_validation_round2.py
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from agreement import (  # noqa: E402
    bootstrap_kappa_ci,
    categories_for,
    clopper_pearson_ci,
    cohen_kappa,
)

CODED_FILE = PROJECT_ROOT / "data" / "human" / "round2_coder_a_full.csv"

KEY_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "validation"
    / "validation_sample_r2_key.csv"
)

PANEL_FILE = PROJECT_ROOT / "outputs" / "panel" / "speech_panel.csv"

# Truncation flags are derived, not primary, so they are joined from the
# round-2 coding sheet rather than duplicated into data/human/.
SHEET_FILE = (
    PROJECT_ROOT / "outputs" / "audit" / "coding_sheet_r2_full.csv"
)

ROUND1_FILE = PROJECT_ROOT / "outputs" / "validation" / "confusion_matrices.csv"

VALIDATION_DIR = PROJECT_ROOT / "outputs" / "validation"

# The excerpt the model read. Everything past this is text the
# full-speech coder saw and the model did not.
EXCERPT_WORD_CAP = 2000

ATTENTION_VARS = [
    "inflation_attention",
    "growth_attention",
    "financial_stability_attention",
    "uncertainty_attention",
]

OUTLOOK_PARENT = {
    "inflation_outlook": "inflation_attention",
    "growth_outlook": "growth_attention",
}

ALL_VARS = ATTENTION_VARS + list(OUTLOOK_PARENT)

LABELS = {
    "inflation_attention": "Inflation attention",
    "growth_attention": "Growth attention",
    "financial_stability_attention": "Financial stability",
    "uncertainty_attention": "Uncertainty",
    "inflation_outlook": "Inflation outlook",
    "growth_outlook": "Growth outlook",
}


# ============================================================
# Loading and checking
# ============================================================

def load() -> tuple[pd.DataFrame, dict[str, float]]:
    """Merge the hand codes with the model codes and the base rates."""

    coded = pd.read_csv(CODED_FILE)
    key = pd.read_csv(KEY_FILE)
    panel = pd.read_csv(PANEL_FILE)

    missing = coded[ALL_VARS].isna()

    if missing.any().any():
        incomplete = coded.loc[missing.any(axis=1), "speech_id"].tolist()

        raise ValueError(
            f"Uncoded cells remain for: {incomplete}. Every cell must "
            "be coded before the round can be scored."
        )

    for variable in ALL_VARS:
        allowed = set(categories_for(variable))
        found = set(coded[variable].unique())

        if not found <= allowed:
            raise ValueError(
                f"{variable} contains values outside {sorted(allowed)}: "
                f"{sorted(found - allowed)}"
            )

    for outlook, attention in OUTLOOK_PARENT.items():
        violating = coded[
            (coded[attention] == 0) & (coded[outlook] != 0)
        ]

        if not violating.empty:
            raise ValueError(
                f"{outlook} is non-zero where {attention} is zero for "
                f"{violating['speech_id'].tolist()}. The coding rules "
                "forbid this."
            )

    sheet = pd.read_csv(SHEET_FILE)[["speech_id", "was_truncated"]]

    merged = coded.merge(
        key[["speech_id"] + ALL_VARS],
        on="speech_id",
        how="left",
        suffixes=("_h", ""),
        validate="one_to_one",
    ).merge(
        sheet, on="speech_id", how="left", validate="one_to_one"
    ).merge(
        panel[["speech_id", "n_words"]],
        on="speech_id",
        how="left",
        validate="one_to_one",
    )

    if merged["n_words"].isna().any():
        raise ValueError(
            "Word counts missing; run analysis/10_panel.py first."
        )

    if merged["was_truncated"].isna().any():
        raise ValueError(
            "Truncation flags missing; regenerate the round-2 coding "
            "sheet with src/build_coding_sheet.py."
        )

    if merged[ALL_VARS].isna().any().any():
        raise ValueError("Coded speeches absent from the sealed key.")

    baseline = panel[panel["in_baseline"]]

    base_rates = {
        variable: float(baseline[variable].mean())
        for variable in ATTENTION_VARS
    }

    return merged, base_rates


# ============================================================
# Rates
# ============================================================

def recover_rates(
    ppv: float,
    npv: float,
    q: float,
) -> tuple[float, float, float]:
    """
    Sensitivity, specificity and prevalence from predictive values.

    q is the model's marginal rate of coding one, counted over the whole
    baseline sample rather than estimated from the audit, so it carries
    no sampling error.
    """

    p = ppv * q + (1.0 - npv) * (1.0 - q)

    if p <= 0.0 or p >= 1.0:
        return float("nan"), float("nan"), p

    alpha = ppv * q / p
    beta = npv * (1.0 - q) / (1.0 - p)

    return alpha, beta, p


def rate_table(
    data: pd.DataFrame,
    base_rates: dict[str, float],
) -> pd.DataFrame:
    """Predictive values, and the rates recovered from them."""

    rows = []

    for variable in ATTENTION_VARS:
        model = data[variable]
        human = data[f"{variable}_h"]

        flagged = model == 1
        unflagged = model == 0

        n_flagged = int(flagged.sum())
        n_unflagged = int(unflagged.sum())

        tp = int((flagged & (human == 1)).sum())
        tn = int((unflagged & (human == 0)).sum())

        ppv = tp / n_flagged if n_flagged else float("nan")
        npv = tn / n_unflagged if n_unflagged else float("nan")

        ppv_lo, ppv_hi = clopper_pearson_ci(tp, n_flagged)
        npv_lo, npv_hi = clopper_pearson_ci(tn, n_unflagged)

        q = base_rates[variable]

        alpha, beta, prevalence = recover_rates(ppv, npv, q)

        # The recovery is not monotone in the two predictive values
        # jointly, so bounds are taken over the corners of the
        # confidence box rather than endpoint by endpoint.
        corners = [
            recover_rates(a, b, q)
            for a in (ppv_lo, ppv_hi)
            for b in (npv_lo, npv_hi)
        ]

        alphas = [c[0] for c in corners if np.isfinite(c[0])]
        betas = [c[1] for c in corners if np.isfinite(c[1])]

        rows.append(
            {
                "variable": variable,
                "label": LABELS[variable],
                "n_model_1": n_flagged,
                "n_model_0": n_unflagged,
                "false_positives": n_flagged - tp,
                "false_negatives": n_unflagged - tn,
                "ppv": ppv,
                "ppv_lo": ppv_lo,
                "ppv_hi": ppv_hi,
                "npv": npv,
                "npv_lo": npv_lo,
                "npv_hi": npv_hi,
                "model_base_rate_q": q,
                "sensitivity": alpha,
                "sens_lo": min(alphas) if alphas else float("nan"),
                "sens_hi": max(alphas) if alphas else float("nan"),
                "specificity": beta,
                "spec_lo": min(betas) if betas else float("nan"),
                "spec_hi": max(betas) if betas else float("nan"),
                "implied_true_share": prevalence,
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# Agreement and error structure
# ============================================================

def agreement_table(data: pd.DataFrame) -> pd.DataFrame:
    """Agreement and kappa, with the sampling caveat attached."""

    rows = []

    for variable in ALL_VARS:
        model = data[variable].to_numpy()
        human = data[f"{variable}_h"].to_numpy()

        cats = categories_for(variable)

        observed, kappa = cohen_kappa(human, model, cats)
        lo, hi = bootstrap_kappa_ci(human, model, cats)

        rows.append(
            {
                "variable": variable,
                "label": LABELS[variable],
                "n": len(data),
                "n_disagree": int((model != human).sum()),
                "agreement": observed,
                "kappa": kappa,
                "kappa_lo": lo,
                "kappa_hi": hi,
            }
        )

    return pd.DataFrame(rows)


def outlook_errors(data: pd.DataFrame) -> pd.DataFrame:
    """
    Decompose outlook disagreements.

    The first round found every outlook error running from a directional
    model code to a neutral human code, with no sign reversals, which is
    what justified describing the error as attenuating rather than
    distorting. This checks whether that survives blind coding.
    """

    rows = []

    for outlook, attention in OUTLOOK_PARENT.items():
        model = data[outlook]
        human = data[f"{outlook}_h"]

        differ = model != human

        forced = differ & (data[attention] != data[f"{attention}_h"])

        rows.append(
            {
                "variable": outlook,
                "label": LABELS[outlook],
                "n_disagree": int(differ.sum()),
                "forced_by_attention": int(forced.sum()),
                "independent": int((differ & ~forced).sum()),
                "directional_to_neutral": int(
                    (differ & (model != 0) & (human == 0)).sum()
                ),
                "neutral_to_directional": int(
                    (differ & (model == 0) & (human != 0)).sum()
                ),
                "sign_reversals": int(
                    (differ & (model != 0) & (human != 0)).sum()
                ),
            }
        )

    return pd.DataFrame(rows)


def withheld_words(data: pd.DataFrame) -> dict[str, float]:
    """
    How much text the full-speech coder saw and the model did not.

    Round 2 is the only round in which a human read the whole speech
    while the model read the excerpt, so it is the only place the cost
    of truncation can be priced. The claim it supports is a negative
    one: not a single false negative arose even though the median
    truncated speech withheld well over a thousand words from the
    model. That is only worth stating if the amount withheld is stated
    with it, since "no false negatives" is unremarkable if the excerpt
    contained almost the whole speech.
    """

    truncated = data[data["was_truncated"]]

    omitted = truncated["n_words"] - EXCERPT_WORD_CAP

    return {
        "n_speeches": len(data),
        "n_truncated": len(truncated),
        "median_omitted": float(omitted.median()),
        "mean_omitted": float(omitted.mean()),
        "q1_omitted": float(omitted.quantile(0.25)),
        "q3_omitted": float(omitted.quantile(0.75)),
        "max_omitted": float(omitted.max()),
        "median_share_seen": float(
            (EXCERPT_WORD_CAP / truncated["n_words"]).median()
        ),
    }


def truncation_diagnostic(data: pd.DataFrame) -> pd.DataFrame:
    """
    Ask whether disagreements concentrate in truncated speeches.

    A false negative on a truncated speech may be a truncation artefact
    rather than a model error: the topic could lie beyond word 2,000.
    A false positive cannot be explained that way, since the model saw
    less text, not more.
    """

    rows = []

    for variable in ATTENTION_VARS:
        model = data[variable]
        human = data[f"{variable}_h"]

        truncated = data["was_truncated"]

        false_negative = (model == 0) & (human == 1)
        false_positive = (model == 1) & (human == 0)

        rows.append(
            {
                "variable": variable,
                "label": LABELS[variable],
                "fn_total": int(false_negative.sum()),
                "fn_truncated": int((false_negative & truncated).sum()),
                "fp_total": int(false_positive.sum()),
                "fp_truncated": int((false_positive & truncated).sum()),
                "share_truncated": float(truncated.mean()),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# Main
# ============================================================

def main() -> None:
    data, base_rates = load()

    print("\n=== ROUND 2: SAMPLE ===")
    print(f"Speeches: {len(data)}")
    print(
        f"Truncated at extraction: {int(data['was_truncated'].sum())} "
        f"({data['was_truncated'].mean():.0%})"
    )
    print(
        "Blind, full-speech, stratified on the model's codes. Raw "
        "agreement below is\nNOT comparable with round 1: the sample "
        "was built to concentrate on error cells."
    )

    agreement = agreement_table(data)

    print("\n=== AGREEMENT (this sample, not corpus-representative) ===")
    print(
        agreement[
            ["label", "n_disagree", "agreement", "kappa",
             "kappa_lo", "kappa_hi"]
        ]
        .round(3)
        .to_string(index=False)
    )

    total = len(data) * len(ALL_VARS)
    agree = total - int(agreement["n_disagree"].sum())

    print(f"\nPooled cells: {agree}/{total} = {agree / total:.3f}")

    rates = rate_table(data, base_rates)

    print("\n=== PREDICTIVE VALUES (unbiased under this design) ===")
    print(
        rates[
            ["label", "n_model_1", "n_model_0", "false_positives",
             "false_negatives", "ppv", "npv"]
        ]
        .round(3)
        .to_string(index=False)
    )

    print("\n=== RECOVERED RATES (Bayes, with known base rate q) ===")
    print(
        rates[
            ["label", "model_base_rate_q", "sensitivity", "sens_lo",
             "sens_hi", "specificity", "spec_lo", "spec_hi"]
        ]
        .round(3)
        .to_string(index=False)
    )

    outlook = outlook_errors(data)

    print("\n=== OUTLOOK ERROR STRUCTURE ===")
    print(outlook.drop(columns=["variable"]).to_string(index=False))

    reversals = int(outlook["sign_reversals"].sum())

    if reversals:
        print(
            f"\n{reversals} sign reversal(s). Round 1 found none, and "
            "that finding is quoted in\ndocs/measurement_notes.md as "
            "evidence that outlook error attenuates rather than\n"
            "distorts. It does not survive blind full-text coding."
        )
    else:
        print("\nNo sign reversals, as in round 1.")

    truncation = truncation_diagnostic(data)

    print("\n=== TRUNCATION DIAGNOSTIC ===")
    print(truncation.drop(columns=["variable"]).to_string(index=False))

    withheld = withheld_words(data)

    print(
        f"\n{withheld['n_truncated']} of {withheld['n_speeches']} "
        f"speeches exceeded the {EXCERPT_WORD_CAP:,}-word cap. Words "
        "withheld from the model,\namong those: median "
        f"{withheld['median_omitted']:,.0f}, quartiles "
        f"{withheld['q1_omitted']:,.0f} and {withheld['q3_omitted']:,.0f}, "
        f"max {withheld['max_omitted']:,.0f}."
    )
    print(
        "The model read a median of "
        f"{withheld['median_share_seen']:.0%} of a truncated speech."
    )

    fn_total = int(truncation["fn_total"].sum())
    fn_trunc = int(truncation["fn_truncated"].sum())

    print(
        f"\nFalse negatives: {fn_total}, of which {fn_trunc} on "
        "truncated speeches."
    )

    if fn_total:
        print(
            "Round 1 reported zero false negatives in 120 attention "
            "cells. It coded the\nexcerpt, so a topic beyond word 2,000 "
            "was invisible to both coder and model."
        )

    rates.to_csv(VALIDATION_DIR / "round2_error_rates.csv", index=False)
    agreement.to_csv(VALIDATION_DIR / "round2_agreement.csv", index=False)
    outlook.to_csv(VALIDATION_DIR / "round2_outlook_errors.csv", index=False)
    truncation.to_csv(
        VALIDATION_DIR / "round2_truncation_diagnostic.csv", index=False
    )

    print(f"\nSaved to: {VALIDATION_DIR}")


if __name__ == "__main__":
    main()
