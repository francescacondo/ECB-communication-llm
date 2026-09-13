"""
Benchmark the LLM extraction against a keyword dictionary.

Every reader of this project will ask what the language model bought
that counting words would not have given for free. This script answers
it on three levels, in increasing order of what is actually at stake:

    1. Against the human codes. Both instruments are scored on the same
       thirty hand-coded speeches, by the same code in src/agreement.py.
       This is the only comparison with a ground truth in it.

    2. Against each other, at the aggregate frequency the project
       actually uses. Two instruments can disagree at the speech level
       and still produce near-identical quarterly series, in which case
       the choice between them would not matter for any downstream
       result.

    3. Against the macro data, reusing analysis/13's design. If the
       dictionary series reproduce the external correlations, the LLM
       is an expensive way to reach the same conclusion.

The comparison is deliberately generous to the dictionary. It reads the
same truncated excerpts the model read, its vocabulary is drawn from
the same codebook the prompt was given, and its decision thresholds are
calibrated to the LLM's own base rates, which is information a
standalone dictionary would not have. See src/dictionary_signals.py.

The expectation being tested is not that the dictionary is bad. It is
that the gap is concentrated in the outlook variables, where coding
requires composition rather than vocabulary, and that attention, which
is closer to a topic-detection problem, is nearly a solved task for
either instrument.

Requires:

    python analysis/10_panel.py      (speech panel, baseline flags)
    python src/download_macro.py     (macro series, for level 3)

Run from the project root:

    python analysis/16_dictionary_benchmark.py
"""

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from agreement import (  # noqa: E402
    bootstrap_kappa_ci,
    categories_for,
    cohen_kappa,
    sensitivity_specificity,
)

from dictionary_signals import (  # noqa: E402
    code_attention,
    code_outlook,
    score_corpus,
    threshold_at_prevalence,
)

from timeseries_stats import (  # noqa: E402
    correlation_hac,
    newey_west_bandwidth,
    newey_west_ols,
    standardise,
    two_sided_t_pvalue,
)


EXCERPT_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "production"
    / "ecb_speeches_1999_2025_excerpts.csv"
)

PANEL_FILE = PROJECT_ROOT / "outputs" / "panel" / "speech_panel.csv"

HUMAN_FILE = PROJECT_ROOT / "data" / "human" / "human_audit_30.csv"

MACRO_FILE = (
    PROJECT_ROOT / "data" / "processed" / "macro_quarterly.csv"
)

VALIDATION_DIR = PROJECT_ROOT / "outputs" / "validation"
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"


# ============================================================
# Configuration
# ============================================================

# Project variable -> (dictionary topic, kind).
ATTENTION_MAP = {
    "inflation_attention": "inflation",
    "growth_attention": "growth",
    "financial_stability_attention": "financial_stability",
    "uncertainty_attention": "uncertainty",
}

OUTLOOK_MAP = {
    "inflation_outlook": ("inflation", "inflation_attention"),
    "growth_outlook": ("growth", "growth_attention"),
}

ALL_VARS = [
    "inflation_attention",
    "inflation_outlook",
    "growth_attention",
    "growth_outlook",
    "financial_stability_attention",
    "uncertainty_attention",
]

LABELS = {
    "inflation_attention": "Inflation attention",
    "inflation_outlook": "Inflation outlook",
    "growth_attention": "Growth attention",
    "growth_outlook": "Growth outlook",
    "financial_stability_attention": "Financial stability attention",
    "uncertainty_attention": "Uncertainty attention",
}

# Quarterly aggregates compared between instruments, and the macro
# series each is validated against. Mirrors analysis/13.
RATE_HORIZON = 2

EXTERNAL_PAIRS = [
    ("out_pi", "hicp_yoy"),
    ("out_pi", "d2_dfr"),
    ("att_pi", "hicp_yoy"),
]

# Targets separating the level of inflation from its trajectory. The
# coding rules instruct the coder to classify the forward-looking
# outlook and explicitly "not merely whether the current level of
# inflation is high or low". A tone measure built from word counts
# cannot honour that distinction, because the vocabulary describing a
# high level ("elevated", "pressure", "strong") is the same vocabulary
# describing a rising one. This is the test of whether the two
# instruments are measuring the same construct at all.
CONSTRUCT_TARGETS = {
    "hicp_yoy": "HICP level",
    "d_hicp": "HICP change, past quarter",
    "d2_hicp_fwd": f"HICP change, next {RATE_HORIZON}q",
}


# ============================================================
# Building the dictionary codes
# ============================================================

def load_inputs() -> pd.DataFrame:
    """
    Attach the dictionary scores to the speech panel.

    The excerpts are the text the model was actually shown. Using the
    full speeches here would give the dictionary more evidence than the
    LLM had and quietly invalidate the comparison, so the merge is
    validated and the excerpt is required to be present for every
    speech in the panel.
    """

    panel = pd.read_csv(PANEL_FILE, parse_dates=["date"])
    excerpts = pd.read_csv(EXCERPT_FILE)

    merged = panel.merge(
        excerpts[["speech_id", "excerpt", "excerpt_words"]],
        on="speech_id",
        how="left",
        validate="one_to_one",
    )

    missing = merged["excerpt"].isna()

    if missing.any():
        raise ValueError(
            "Speeches in the panel without an extraction excerpt: "
            f"{merged.loc[missing, 'speech_id'].tolist()[:10]}"
        )

    scores = score_corpus(merged["excerpt"])

    return pd.concat([merged.drop(columns=["excerpt"]), scores], axis=1)


def add_dictionary_codes(data: pd.DataFrame) -> pd.DataFrame:
    """
    Convert continuous dictionary scores into codes on the project's
    scale, calibrating every threshold to the LLM's base rate.

    Calibration uses the baseline sample only, since that is the sample
    every published measure is built from and the base rates differ
    once the slide stubs are dropped.
    """

    data = data.copy()

    baseline = data["in_baseline"]

    for variable, topic in ATTENTION_MAP.items():
        density = data[f"dens_{topic}"].to_numpy(float)

        target = float(data.loc[baseline, variable].mean())

        cutoff = threshold_at_prevalence(
            density[baseline.to_numpy()], target
        )

        data[f"dict_{variable}"] = (density > cutoff).astype(int)

        data.attrs.setdefault("thresholds", {})[variable] = cutoff

    for variable, (topic, parent) in OUTLOOK_MAP.items():
        neutral_rate = float(
            (data.loc[baseline, variable] == 0).mean()
        )

        data[f"dict_{variable}"] = code_outlook(
            data[f"tone_{topic}"].to_numpy(float),
            data[f"dict_{parent}"].to_numpy(int),
            neutral_rate,
        )

    return data


def base_rate_table(data: pd.DataFrame) -> pd.DataFrame:
    """
    Confirm the calibration worked, on the baseline sample.

    Prevalence matching is exact only up to ties in the density
    distribution, which are common because many speeches score zero.
    Reporting both rates makes any slippage visible instead of
    assumed.
    """

    baseline = data[data["in_baseline"]]

    rows = []

    for variable in ALL_VARS:
        if variable in ATTENTION_MAP:
            llm = float((baseline[variable] == 1).mean())
            dictionary = float(
                (baseline[f"dict_{variable}"] == 1).mean()
            )
            quantity = "share coded 1"
        else:
            llm = float((baseline[variable] == 0).mean())
            dictionary = float(
                (baseline[f"dict_{variable}"] == 0).mean()
            )
            quantity = "share coded 0"

        rows.append(
            {
                "variable": variable,
                "label": LABELS[variable],
                "quantity": quantity,
                "llm": llm,
                "dictionary": dictionary,
                "difference": dictionary - llm,
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# Level 1: against the human codes
# ============================================================

def load_human(data: pd.DataFrame) -> pd.DataFrame:
    """Attach the hand codes to the scored panel."""

    human = pd.read_csv(HUMAN_FILE)

    columns = ["speech_id"] + [f"{v}_h" for v in ALL_VARS]

    merged = human[columns].merge(
        data[
            ["speech_id"]
            + ALL_VARS
            + [f"dict_{v}" for v in ALL_VARS]
        ],
        on="speech_id",
        how="left",
        validate="one_to_one",
    )

    if merged.isna().any().any():
        raise ValueError(
            "Validation speeches missing from the scored panel."
        )

    return merged


def horse_race(human: pd.DataFrame) -> pd.DataFrame:
    """
    Score both instruments against the human codes, variable by
    variable, using identical code for each.
    """

    rows = []

    for variable in ALL_VARS:
        reference = human[f"{variable}_h"].to_numpy()

        cats = categories_for(variable)

        for instrument, column in [
            ("LLM", variable),
            ("Dictionary", f"dict_{variable}"),
        ]:
            candidate = human[column].to_numpy()

            observed, kappa = cohen_kappa(reference, candidate, cats)

            kappa_lo, kappa_hi = bootstrap_kappa_ci(
                reference, candidate, cats
            )

            row = {
                "variable": variable,
                "label": LABELS[variable],
                "instrument": instrument,
                "n": len(reference),
                "agreement": observed,
                "kappa": kappa,
                "kappa_lo": kappa_lo,
                "kappa_hi": kappa_hi,
            }

            if variable in ATTENTION_MAP:
                row.update(sensitivity_specificity(reference, candidate))

            rows.append(row)

    return pd.DataFrame(rows)


def error_overlap(human: pd.DataFrame) -> pd.DataFrame:
    """
    Do the two instruments fail on the same speeches?

    Counting errors tells you how often each instrument is wrong.
    Intersecting them tells you whether they are wrong in the same
    place, which is a different question and the more interesting one.
    Equal error counts on a variable look like equivalent behaviour and
    need not be: two instruments can each post three false positives
    and share none of them, in which case the tie in kappa is an
    accident of arithmetic rather than agreement about the text.

    Reported per attention variable, with the speech ids, so a claim
    about which speeches fool both can be checked rather than asserted.
    """

    rows = []

    for variable in ATTENTION_MAP:
        reference = human[f"{variable}_h"]

        llm = human[variable]
        dictionary = human[f"dict_{variable}"]

        for kind, llm_hit, dict_hit in [
            (
                "false positive",
                (reference == 0) & (llm == 1),
                (reference == 0) & (dictionary == 1),
            ),
            (
                "false negative",
                (reference == 1) & (llm == 0),
                (reference == 1) & (dictionary == 0),
            ),
        ]:
            llm_ids = set(human.loc[llm_hit, "speech_id"])
            dict_ids = set(human.loc[dict_hit, "speech_id"])

            shared = llm_ids & dict_ids

            rows.append(
                {
                    "variable": variable,
                    "label": LABELS[variable],
                    "kind": kind,
                    "n_llm": len(llm_ids),
                    "n_dictionary": len(dict_ids),
                    "n_shared": len(shared),
                    "llm_only": " ".join(sorted(llm_ids - dict_ids)),
                    "dictionary_only": " ".join(
                        sorted(dict_ids - llm_ids)
                    ),
                    "shared_ids": " ".join(sorted(shared)),
                }
            )

    return pd.DataFrame(rows)


def cell_agreement(human: pd.DataFrame) -> pd.DataFrame:
    """Overall cell-level agreement, pooling all six variables."""

    rows = []

    for instrument, prefix in [("LLM", ""), ("Dictionary", "dict_")]:
        matches = 0
        total = 0

        for variable in ALL_VARS:
            reference = human[f"{variable}_h"].to_numpy()
            candidate = human[f"{prefix}{variable}"].to_numpy()

            matches += int((reference == candidate).sum())
            total += len(reference)

        rows.append(
            {
                "instrument": instrument,
                "cells_agreeing": matches,
                "cells": total,
                "agreement": matches / total,
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# Level 2: aggregate series
# ============================================================

def quarterly_measures(
    data: pd.DataFrame,
    prefix: str = "",
) -> pd.DataFrame:
    """
    Aggregate one instrument's codes to quarterly attention shares and
    conditional outlooks, following analysis/12's definitions.
    """

    baseline = data[data["in_baseline"]]

    frames = {}

    keys = {
        "pi": ("inflation_attention", "inflation_outlook"),
        "y": ("growth_attention", "growth_outlook"),
        "fs": ("financial_stability_attention", None),
        "unc": ("uncertainty_attention", None),
    }

    for key, (att_var, out_var) in keys.items():
        attention = baseline[f"{prefix}{att_var}"]

        frames[f"att_{key}"] = attention.groupby(
            baseline["yq"]
        ).mean()

        if out_var is None:
            continue

        outlook = baseline[f"{prefix}{out_var}"]

        attending = baseline[attention == 1]

        frames[f"out_{key}"] = (
            attending[f"{prefix}{out_var}"]
            .groupby(attending["yq"])
            .mean()
        )

    return pd.DataFrame(frames).rename_axis("period").reset_index()


def series_comparison(
    llm: pd.DataFrame,
    dictionary: pd.DataFrame,
) -> pd.DataFrame:
    """
    Correlation between the two instruments' quarterly series.

    This is the question that decides whether the choice of instrument
    matters downstream. Speech-level disagreement can average out.
    """

    merged = llm.merge(
        dictionary, on="period", suffixes=("_llm", "_dict")
    )

    rows = []

    for column in ["att_pi", "att_y", "att_fs", "att_unc",
                   "out_pi", "out_y"]:
        pair = merged[[f"{column}_llm", f"{column}_dict"]].dropna()

        estimate = correlation_hac(
            pair[f"{column}_dict"].to_numpy(float),
            pair[f"{column}_llm"].to_numpy(float),
        )

        rows.append(
            {
                "measure": column,
                "n_quarters": estimate.n_obs,
                "correlation": estimate.correlation,
                "hac_se": estimate.standard_error,
                "p_value": estimate.p_value,
                "mean_llm": float(pair[f"{column}_llm"].mean()),
                "mean_dict": float(pair[f"{column}_dict"].mean()),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# Level 3: external validation
# ============================================================

def construct_difference(
    llm: pd.DataFrame,
    dictionary: pd.DataFrame,
) -> pd.DataFrame:
    """
    Test whether the dictionary tracks the inflation level more than the
    model does.

    Section 8.4 shows correlations of 0.618 and 0.229 against the HICP
    level and invites the reader to see a gap. A gap between two
    correlations measured on the same sample against the same target is
    a testable quantity, and leaving it untested would make the note's
    central construct claim rest on eyeballing.

    Two tests are reported. Williams's t is the standard test for
    dependent correlations sharing one variable, and assumes independent
    observations, which quarterly macro series violate. The second is
    autocorrelation-robust and is the one to quote: the difference in
    correlations is the mean of a difference of standardised
    cross-products,

        d_t = z_hicp,t * z_dict,t  -  z_hicp,t * z_llm,t,

    whose expectation is r_dict - r_llm, so a Newey-West standard error
    on that mean tests the difference with no independence assumption.
    """

    macro = pd.read_csv(MACRO_FILE)

    merged = (
        llm[["period", "out_pi"]]
        .merge(dictionary[["period", "out_pi"]], on="period",
               suffixes=("_llm", "_dict"))
        .merge(macro[["period", "hicp_yoy"]], on="period")
        .dropna()
    )

    n = len(merged)

    j = merged["hicp_yoy"].to_numpy(float)
    k = merged["out_pi_dict"].to_numpy(float)
    h = merged["out_pi_llm"].to_numpy(float)

    r_jk = float(np.corrcoef(j, k)[0, 1])
    r_jh = float(np.corrcoef(j, h)[0, 1])
    r_kh = float(np.corrcoef(k, h)[0, 1])

    determinant = (
        1 - r_jk**2 - r_jh**2 - r_kh**2 + 2 * r_jk * r_jh * r_kh
    )
    mean_r = (r_jk + r_jh) / 2

    t_williams = (r_jk - r_jh) * np.sqrt(
        (n - 1) * (1 + r_kh)
        / (
            2 * ((n - 1) / (n - 3)) * determinant
            + mean_r**2 * (1 - r_kh) ** 3
        )
    )

    products = (
        standardise(j) * standardise(k) - standardise(j) * standardise(h)
    ) * n / (n - 1)

    fit = newey_west_ols(
        products, np.ones((n, 1)), newey_west_bandwidth(n)
    )

    low, high = fit.confidence_interval(0)

    return pd.DataFrame(
        [
            {
                "n": n,
                "r_dictionary_level": r_jk,
                "r_llm_level": r_jh,
                "r_between_measures": r_kh,
                "difference": r_jk - r_jh,
                "williams_t": float(t_williams),
                "williams_p": two_sided_t_pvalue(float(t_williams), n - 3),
                "hac_se": float(fit.standard_errors[0]),
                "hac_t": float(fit.t_statistics[0]),
                "hac_p": float(fit.p_values[0]),
                "hac_ci_low": low,
                "hac_ci_high": high,
                "hac_lags": fit.lags,
            }
        ]
    )


def construct_comparison(
    llm: pd.DataFrame,
    dictionary: pd.DataFrame,
) -> pd.DataFrame:
    """
    Ask what each instrument's inflation outlook is actually tracking.

    A higher correlation with realised inflation is not by itself
    evidence of a better measure: it depends entirely on what the
    measure was supposed to capture. If the dictionary tracks the
    *level* of inflation while the LLM tracks its *trajectory*, then
    the dictionary's stronger contemporaneous correlation is evidence
    that it is measuring the wrong thing well, rather than the right
    thing better.
    """

    macro = pd.read_csv(MACRO_FILE)

    rows = []

    for instrument, measures in [
        ("LLM", llm),
        ("Dictionary", dictionary),
    ]:
        merged = measures.merge(macro, on="period", how="left")

        merged["d_hicp"] = merged["hicp_yoy"].diff()

        merged["d2_hicp_fwd"] = (
            merged["hicp_yoy"].shift(-RATE_HORIZON)
            - merged["hicp_yoy"]
        )

        for target, label in CONSTRUCT_TARGETS.items():
            pair = merged[["out_pi", target]].dropna()

            estimate = correlation_hac(
                pair["out_pi"].to_numpy(float),
                pair[target].to_numpy(float),
                min_lags=(
                    RATE_HORIZON - 1
                    if target == "d2_hicp_fwd"
                    else 0
                ),
            )

            rows.append(
                {
                    "instrument": instrument,
                    "target": target,
                    "target_label": label,
                    "n_quarters": estimate.n_obs,
                    "correlation": estimate.correlation,
                    "hac_se": estimate.standard_error,
                    "p_value": estimate.p_value,
                }
            )

    return pd.DataFrame(rows)


def external_comparison(
    llm: pd.DataFrame,
    dictionary: pd.DataFrame,
) -> pd.DataFrame:
    """
    Repeat analysis/13's headline correlations for both instruments.

    If the dictionary series reproduce the macro relationships, the
    language model is an expensive route to the same answer. If they do
    not, the difference between the instruments is economically
    meaningful and not merely a coding-agreement statistic.
    """

    macro = pd.read_csv(MACRO_FILE)

    rows = []

    for instrument, measures in [
        ("LLM", llm),
        ("Dictionary", dictionary),
    ]:
        merged = measures.merge(macro, on="period", how="left")

        merged["d2_dfr"] = (
            merged["dfr_eop"].shift(-RATE_HORIZON) - merged["dfr_eop"]
        )

        for measure, target in EXTERNAL_PAIRS:
            pair = merged[[measure, target]].dropna()

            estimate = correlation_hac(
                pair[measure].to_numpy(float),
                pair[target].to_numpy(float),
                min_lags=RATE_HORIZON - 1 if target == "d2_dfr" else 0,
            )

            rows.append(
                {
                    "instrument": instrument,
                    "measure": measure,
                    "target": target,
                    "n_quarters": estimate.n_obs,
                    "correlation": estimate.correlation,
                    "hac_se": estimate.standard_error,
                    "p_value": estimate.p_value,
                }
            )

    return pd.DataFrame(rows)


# ============================================================
# Output
# ============================================================

def write_horse_race_tex(race: pd.DataFrame, path: Path) -> None:
    """Agreement and kappa, both instruments, against the hand codes."""

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{LLM and keyword dictionary against hand-coded "
        r"speeches}",
        r"\label{tab:dictionary-benchmark}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"& \multicolumn{2}{c}{Agreement} & "
        r"\multicolumn{2}{c}{Cohen's $\kappa$} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
        r"Variable & LLM & Dictionary & LLM & Dictionary \\",
        r"\midrule",
    ]

    for variable in ALL_VARS:
        subset = race[race["variable"] == variable].set_index(
            "instrument"
        )

        llm = subset.loc["LLM"]
        dictionary = subset.loc["Dictionary"]

        def fmt(value: float) -> str:
            return "--" if not np.isfinite(value) else f"{value:.3f}"

        lines.append(
            f"{LABELS[variable]} "
            f"& {fmt(llm['agreement'])} "
            f"& {fmt(dictionary['agreement'])} "
            f"& {fmt(llm['kappa'])} "
            f"& {fmt(dictionary['kappa'])} \\\\"
        )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{minipage}{0.86\textwidth}",
        r"\vspace{0.6em}",
        r"\footnotesize Thirty speeches hand-coded against "
        r"\texttt{docs/coding\_rules.md}. Both instruments read the "
        r"same truncated excerpts and are scored by identical code. "
        r"The dictionary's decision thresholds are calibrated to "
        r"reproduce the LLM's base rates on the baseline sample, which "
        r"is information a standalone dictionary would not have; the "
        r"comparison is therefore generous to the baseline. $\kappa$ "
        r"is undefined where a coder uses a single category "
        r"throughout. The hand coding was not blind and n = 30, so "
        r"these are indicative rather than precise.",
        r"\end{minipage}",
        r"\end{table}",
    ]

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_benchmark(
    race: pd.DataFrame,
    llm: pd.DataFrame,
    dictionary: pd.DataFrame,
    path: Path,
) -> None:
    """Agreement by instrument, and the two inflation series."""

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    # --- Panel A: agreement by variable -------------------
    ax = axes[0]

    order = ALL_VARS

    positions = np.arange(len(order))
    width = 0.36

    for offset, instrument, colour in [
        (-width / 2, "LLM", "#4c72b0"),
        (width / 2, "Dictionary", "#c44e52"),
    ]:
        values = [
            float(
                race[
                    (race["variable"] == v)
                    & (race["instrument"] == instrument)
                ]["agreement"].iloc[0]
            )
            for v in order
        ]

        ax.bar(
            positions + offset,
            values,
            width,
            label=instrument,
            color=colour,
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(
        [LABELS[v].replace(" ", "\n", 1) for v in order],
        fontsize=8,
    )
    ax.set_ylim(0, 1.05)
    ax.axhline(1.0, color="0.75", linewidth=0.8, linestyle=":")
    ax.set_ylabel("Agreement with hand codes")
    ax.set_title("Against 30 hand-coded speeches", fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)

    # --- Panel B: quarterly inflation outlook -------------
    ax = axes[1]

    merged = llm.merge(
        dictionary, on="period", suffixes=("_llm", "_dict")
    )

    time = pd.PeriodIndex(merged["period"], freq="Q").to_timestamp()

    ax.axhline(0, color="0.7", linewidth=0.8)

    ax.plot(
        time,
        merged["out_pi_llm"],
        color="#4c72b0",
        linewidth=1.5,
        label="LLM",
    )

    ax.plot(
        time,
        merged["out_pi_dict"],
        color="#c44e52",
        linewidth=1.5,
        linestyle="--",
        label="Dictionary",
    )

    ax.set_ylabel("Conditional inflation outlook")
    ax.set_xlabel("Quarter")
    ax.set_title(
        "Quarterly inflation outlook, both instruments", fontsize=11
    )
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25, linewidth=0.6)

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def report(frame: pd.DataFrame, columns: list[str]) -> str:
    return frame[columns].round(3).to_string(index=False)


# ============================================================
# Main
# ============================================================

def main() -> None:
    for directory in (VALIDATION_DIR, TABLE_DIR, FIGURE_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    data = add_dictionary_codes(load_inputs())

    print("\n=== SCORED CORPUS ===")
    print(f"Speeches: {len(data):,}")
    print(f"Baseline: {int(data['in_baseline'].sum()):,}")
    print(
        "Dictionary read the extraction excerpts "
        f"(mean {data['excerpt_words'].mean():.0f} words)"
    )

    rates = base_rate_table(data)

    worst = rates.reindex(
        rates["difference"].abs().sort_values().index
    )

    print("\n=== BASE RATES AFTER CALIBRATION ===")
    others = rates[rates["variable"] != "financial_stability_attention"]
    print(
        "Prevalence matching is exact to within "
        f"{others['difference'].abs().max():.3f} for every variable "
        "except financial\nstability, where ties in the density "
        f"distribution leave a gap of "
        f"{abs(float(rates.loc[rates['variable'] == 'financial_stability_attention', 'difference'].iloc[0])):.3f}."
    )
    print(
        report(
            rates,
            ["label", "quantity", "llm", "dictionary", "difference"],
        )
    )

    # --------------------------------------------------
    # 1. Against the hand codes
    # --------------------------------------------------

    human = load_human(data)

    race = horse_race(human)

    print("\n=== AGAINST 30 HAND-CODED SPEECHES ===")
    print(
        report(
            race,
            [
                "label",
                "instrument",
                "agreement",
                "kappa",
                "kappa_lo",
                "kappa_hi",
            ],
        )
    )

    attention_race = race[race["variable"].isin(ATTENTION_MAP)]

    print("\n=== ATTENTION: ERROR STRUCTURE ===")
    print(
        report(
            attention_race,
            [
                "label",
                "instrument",
                "sensitivity",
                "specificity",
                "false_positive",
                "false_negative",
            ],
        )
    )

    overlap = error_overlap(human)

    print("\n=== DO THE TWO FAIL ON THE SAME SPEECHES? ===")
    print(
        report(
            overlap,
            ["label", "kind", "n_llm", "n_dictionary", "n_shared"],
        )
    )
    print(
        "Equal counts are not the same errors. Inflation attention is "
        "the case to watch:\nboth instruments post the same number of "
        "false positives and do not agree on which."
    )

    cells = cell_agreement(human)

    print("\n=== POOLED CELL AGREEMENT ===")
    print(report(cells, ["instrument", "cells_agreeing", "cells",
                         "agreement"]))

    # --------------------------------------------------
    # 2. Aggregate series
    # --------------------------------------------------

    llm_measures = quarterly_measures(data, prefix="")
    dict_measures = quarterly_measures(data, prefix="dict_")

    comparison = series_comparison(llm_measures, dict_measures)

    print("\n=== QUARTERLY SERIES: LLM AGAINST DICTIONARY ===")
    print(
        report(
            comparison,
            [
                "measure",
                "n_quarters",
                "correlation",
                "hac_se",
                "p_value",
                "mean_llm",
                "mean_dict",
            ],
        )
    )

    # --------------------------------------------------
    # 3. External validation
    # --------------------------------------------------

    external = external_comparison(llm_measures, dict_measures)

    print("\n=== EXTERNAL VALIDATION, BOTH INSTRUMENTS ===")
    print(
        report(
            external,
            [
                "measure",
                "target",
                "instrument",
                "n_quarters",
                "correlation",
                "hac_se",
                "p_value",
            ],
        )
    )

    construct = construct_comparison(llm_measures, dict_measures)

    difference = construct_difference(llm_measures, dict_measures)

    print("\n=== WHAT IS EACH OUTLOOK MEASURE TRACKING? ===")
    print(
        report(
            construct,
            [
                "target_label",
                "instrument",
                "n_quarters",
                "correlation",
                "hac_se",
                "p_value",
            ],
        )
    )

    # --------------------------------------------------
    # 4. Write outputs
    # --------------------------------------------------

    rates.to_csv(
        VALIDATION_DIR / "dictionary_base_rates.csv", index=False
    )

    race.to_csv(
        VALIDATION_DIR / "dictionary_horse_race.csv", index=False
    )

    cells.to_csv(
        VALIDATION_DIR / "dictionary_cell_agreement.csv", index=False
    )

    comparison.to_csv(
        VALIDATION_DIR / "dictionary_series_comparison.csv",
        index=False,
    )

    external.to_csv(
        VALIDATION_DIR / "dictionary_external_validation.csv",
        index=False,
    )

    construct.to_csv(
        VALIDATION_DIR / "dictionary_construct_check.csv", index=False
    )

    difference.to_csv(
        VALIDATION_DIR / "dictionary_construct_difference.csv",
        index=False,
    )

    row = difference.iloc[0]

    print("\n=== IS THE GAP REAL? DIFFERENCE OF DEPENDENT CORRELATIONS ===")
    print(
        f"  dictionary {row['r_dictionary_level']:.3f} vs model "
        f"{row['r_llm_level']:.3f}, measures correlated "
        f"{row['r_between_measures']:.3f}"
    )
    print(
        f"  difference {row['difference']:+.3f}   Williams t="
        f"{row['williams_t']:+.2f} p={row['williams_p']:.4f} (assumes iid)"
    )
    print(
        f"  autocorrelation-robust: HAC se {row['hac_se']:.3f}, "
        f"t={row['hac_t']:+.2f}, p={row['hac_p']:.4f}, "
        f"95% [{row['hac_ci_low']:+.3f}, {row['hac_ci_high']:+.3f}]"
    )

    dict_measures.to_csv(
        PROJECT_ROOT
        / "outputs"
        / "panel"
        / "measures_quarterly_dictionary.csv",
        index=False,
    )

    write_horse_race_tex(
        race, TABLE_DIR / "dictionary_benchmark.tex"
    )

    plot_benchmark(
        race,
        llm_measures,
        dict_measures,
        FIGURE_DIR / "dictionary_benchmark.png",
    )

    overlap.to_csv(
        VALIDATION_DIR / "dictionary_error_overlap.csv", index=False
    )

    print(f"\nSaved statistics to: {VALIDATION_DIR}")
    print(f"Saved table to: {TABLE_DIR / 'dictionary_benchmark.tex'}")
    print(
        f"Saved figure to: {FIGURE_DIR / 'dictionary_benchmark.png'}"
    )


if __name__ == "__main__":
    main()
