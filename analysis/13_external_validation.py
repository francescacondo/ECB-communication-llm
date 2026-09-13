"""
External validation of the ECB communication measures against
euro-area macro data.

The LLM signals are generated measurements. The human audit in
analysis/15_validation_stats.py establishes that they reproduce what a
human coder reads in the same text; it says nothing about whether they
carry economic information. This script asks the separate question:
do the measures line up with things outside the corpus that they ought
to line up with?

The headline exercise concerns the attention-weighted inflation
outlook, `out_pi`, and two external series:

    realised inflation  the HICP annual rate in the same quarter;
    the rate path       the change in the deposit facility rate over
                        the following two quarters.

The claim under test is that the relationship rotated: directional
inflation talk tracked realised inflation early in the sample and came
to track the policy path instead in the forward-guidance era. Both
correlations are reported by era, with an explicit test of the
difference rather than an eyeballed comparison of two numbers.

Inference is autocorrelation-robust throughout. Both sides of every
pair are persistent quarterly series, and the rate-path variable is an
overlapping two-quarter difference, so an iid correlation standard
error would be far too small. See src/timeseries_stats.py.

Requires:

    python src/download_macro.py   (quarterly macro panel)
    python analysis/12_measures.py (quarterly communication measures)

Run from the project root:

    python analysis/13_external_validation.py
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

from timeseries_stats import (  # noqa: E402
    bootstrap_block_length,
    correlation_hac,
    moving_block_bootstrap_ci,
    newey_west_bandwidth,
    newey_west_ols,
    standardise,
)


MEASURES_FILE = (
    PROJECT_ROOT / "outputs" / "panel" / "measures_quarterly.csv"
)

ENGLISH_MEASURES_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "panel"
    / "measures_quarterly_english_only.csv"
)

MACRO_FILE = (
    PROJECT_ROOT / "data" / "processed" / "macro_quarterly.csv"
)

PANEL_DIR = PROJECT_ROOT / "outputs" / "panel"
VALIDATION_DIR = PROJECT_ROOT / "outputs" / "validation"
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"

MERGED_FILE = PANEL_DIR / "communication_macro_quarterly.csv"


# ============================================================
# Configuration
# ============================================================

# Horizon for the policy-path variable, in quarters. Two quarters is
# the span over which a speech can plausibly be said to anticipate a
# decision without simply describing one already announced.
RATE_HORIZON = 2

# Era split. This is imposed, not estimated: 2012 is the first full
# year of the Draghi presidency and precedes the introduction of
# explicit forward guidance in July 2013. Because the date is a choice
# rather than a finding, break_sensitivity() reports the same
# correlations for every plausible alternative.
ERA_BREAK_YEAR = 2012

ERA_EARLY = f"1999-{ERA_BREAK_YEAR - 1}"
ERA_LATE = f"{ERA_BREAK_YEAR}-2025"

# Candidate break years for the sensitivity check.
CANDIDATE_BREAKS = [2009, 2010, 2011, 2012, 2013, 2014, 2015]

# Quarters with very few speeches would give unstable measures. The
# quarterly frequency was chosen precisely because none are sparse;
# this is asserted rather than assumed, so that a change to the
# underlying sample cannot pass unnoticed.
ASSERT_NO_SPARSE_QUARTERS = True

LABELS = {
    "att_pi": "Inflation attention",
    "out_pi": "Inflation outlook",
    "att_y": "Growth attention",
    "out_y": "Growth outlook",
    "att_fs": "Financial stability attention",
    "att_unc": "Uncertainty attention",
    "hicp_yoy": "HICP inflation",
    "d2_dfr": f"Deposit rate, {RATE_HORIZON}q change",
    "d2_dfr_avg": f"Deposit rate (avg), {RATE_HORIZON}q change",
    "d2_mro": f"MRO rate, {RATE_HORIZON}q change",
    "unemployment": "Unemployment rate",
    "d2_unemployment": f"Unemployment, {RATE_HORIZON}q change",
}

# Targets built as overlapping forward differences. Their residuals are
# MA(RATE_HORIZON - 1) by construction, which sets a floor on the HAC
# bandwidth and on the bootstrap block length.
OVERLAPPING_TARGETS = {
    "d2_dfr", "d2_dfr_avg", "d2_mro", "d2_unemployment",
}

# Full-sample concordance checks. Each pair carries the sign economic
# reasoning predicts, so that a sign flip is visible in the output
# instead of having to be worked out by the reader.
CONCORDANCE_PAIRS = [
    ("att_pi", "hicp_yoy", +1),
    ("out_pi", "hicp_yoy", +1),
    ("out_pi", "d2_dfr", +1),
    ("att_y", "unemployment", +1),
    ("out_y", "unemployment", -1),
    ("out_y", "d2_unemployment", -1),
    ("out_y", "d2_dfr", +1),
]

# The headline exercise: one communication measure, two targets.
HEADLINE_MEASURE = "out_pi"

HEADLINE_TARGETS = ["hicp_yoy", "d2_dfr"]

# Controls for the length robustness check. Attention rises
# mechanically with text length and mean speech length trends over the
# sample, so a correlation could in principle be an artefact of how
# long speeches were in a given period.
LENGTH_CONTROLS = ["mean_log_words", "share_truncated"]


# ============================================================
# Data
# ============================================================

def load_merged(
    measures_file: Path = MEASURES_FILE,
) -> pd.DataFrame:
    """
    Merge the quarterly communication measures with the macro panel and
    derive the forward-difference targets.

    The merge is validated one-to-one and required to cover every
    quarter of the speech sample: a macro series that silently stops
    short would otherwise shorten a subsample without saying so.
    """

    measures = pd.read_csv(measures_file)
    macro = pd.read_csv(MACRO_FILE)

    merged = measures.merge(
        macro,
        on="period",
        how="left",
        validate="one_to_one",
    )

    if len(merged) != len(measures):
        raise ValueError(
            "Merge changed the number of quarters: "
            f"{len(measures)} -> {len(merged)}"
        )

    quarters = pd.PeriodIndex(merged["period"], freq="Q")

    if not quarters.is_monotonic_increasing:
        raise ValueError("Quarters are not in ascending order.")

    expected = pd.period_range(
        quarters.min(), quarters.max(), freq="Q"
    )

    if len(quarters) != len(expected) or (quarters != expected).any():
        raise ValueError(
            "The quarterly index has gaps; the forward differences "
            "below assume consecutive quarters."
        )

    merged["quarter"] = quarters
    merged["year"] = quarters.year

    if ASSERT_NO_SPARSE_QUARTERS and merged["is_sparse"].any():
        sparse = merged.loc[merged["is_sparse"], "period"].tolist()

        raise ValueError(
            f"Sparse quarters present: {sparse}. The quarterly "
            "frequency was chosen because none exist; revisit the "
            "frequency or relax ASSERT_NO_SPARSE_QUARTERS knowingly."
        )

    # Every quarter must have a policy rate and an inflation reading,
    # since these carry the headline result.
    for column in ["hicp_yoy", "dfr_eop", "mro_eop"]:
        missing = merged.loc[merged[column].isna(), "period"].tolist()

        if missing:
            raise ValueError(
                f"{column} is missing for {len(missing)} quarters of "
                f"the speech sample, e.g. {missing[:5]}."
            )

    # --------------------------------------------------
    # Forward differences
    # --------------------------------------------------

    # Change over the NEXT RATE_HORIZON quarters, so the measure in
    # quarter t is compared with what happens after it. The last
    # RATE_HORIZON quarters are necessarily missing.
    merged["d2_dfr"] = (
        merged["dfr_eop"].shift(-RATE_HORIZON) - merged["dfr_eop"]
    )

    merged["d2_dfr_avg"] = (
        merged["dfr_avg"].shift(-RATE_HORIZON) - merged["dfr_avg"]
    )

    # The main refinancing rate, spliced across tender regimes in
    # src/download_macro.py. See policy_rate_robustness() for why it
    # matters which rate is used.
    merged["d2_mro"] = (
        merged["mro_eop"].shift(-RATE_HORIZON) - merged["mro_eop"]
    )

    merged["d2_unemployment"] = (
        merged["unemployment"].shift(-RATE_HORIZON)
        - merged["unemployment"]
    )

    merged["era"] = np.where(
        merged["year"] < ERA_BREAK_YEAR, ERA_EARLY, ERA_LATE
    )

    return merged


def era_slice(data: pd.DataFrame, era: str) -> pd.DataFrame:
    return data[data["era"] == era]


def paired(
    data: pd.DataFrame,
    measure: str,
    target: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Aligned, complete observations for one measure-target pair."""

    subset = data[[measure, target]].dropna()

    if len(subset) < 12:
        raise ValueError(
            f"Only {len(subset)} usable quarters for {measure} against "
            f"{target}; too few for autocorrelation-robust inference."
        )

    return (
        subset[measure].to_numpy(float),
        subset[target].to_numpy(float),
    )


def lag_floor(target: str) -> int:
    """
    Minimum HAC bandwidth implied by construction of the target.

    An overlapping h-quarter difference has MA(h-1) errors under the
    null, so a bandwidth below h-1 would discard autocorrelation known
    to be there.
    """

    return RATE_HORIZON - 1 if target in OVERLAPPING_TARGETS else 0


# ============================================================
# Correlation tables
# ============================================================

def correlation_row(
    data: pd.DataFrame,
    measure: str,
    target: str,
    sample: str,
    expected_sign: int | None = None,
) -> dict:
    """One correlation with HAC and block-bootstrap inference."""

    x, y = paired(data, measure, target)

    estimate = correlation_hac(x, y, min_lags=lag_floor(target))

    block = bootstrap_block_length(
        len(x), min_block=lag_floor(target) + 1
    )

    boot_low, boot_high = moving_block_bootstrap_ci(x, y, block)

    row = {
        "sample": sample,
        "measure": measure,
        "measure_label": LABELS[measure],
        "target": target,
        "target_label": LABELS[target],
        "n_quarters": estimate.n_obs,
        "correlation": estimate.correlation,
        "hac_se": estimate.standard_error,
        "t_stat": estimate.t_statistic,
        "p_value": estimate.p_value,
        "ci_low": estimate.ci_low,
        "ci_high": estimate.ci_high,
        "hac_lags": estimate.lags,
        "block_length": block,
        "boot_ci_low": boot_low,
        "boot_ci_high": boot_high,
    }

    if expected_sign is not None:
        row["expected_sign"] = expected_sign

        row["sign_as_expected"] = bool(
            np.sign(estimate.correlation) == np.sign(expected_sign)
        )

    return row


def concordance_table(data: pd.DataFrame) -> pd.DataFrame:
    """Full-sample correlations for every measure-target pair."""

    return pd.DataFrame(
        [
            correlation_row(
                data, measure, target, "1999-2025", expected_sign
            )
            for measure, target, expected_sign in CONCORDANCE_PAIRS
        ]
    )


def era_table(data: pd.DataFrame) -> pd.DataFrame:
    """The headline measure against each target, by era."""

    rows = []

    for target in HEADLINE_TARGETS:
        for era in (ERA_EARLY, ERA_LATE):
            rows.append(
                correlation_row(
                    era_slice(data, era),
                    HEADLINE_MEASURE,
                    target,
                    era,
                )
            )

    return pd.DataFrame(rows)


# ============================================================
# Testing the change across eras
# ============================================================

def era_difference_test(
    data: pd.DataFrame,
    measure: str,
    target: str,
) -> dict:
    """
    Test whether a correlation differs across the two eras.

    Within each era both variables are standardised, so the OLS slope
    of y on x in that era is exactly the era's correlation. Stacking
    the standardised data and interacting the regressor with a
    late-era dummy makes the interaction coefficient the difference in
    correlations, and a Newey-West standard error on it a test of the
    rotation.

    Standardising within era rather than pooling is what makes the
    interaction a difference in correlations rather than a difference
    in slopes confounded by the change in either variable's dispersion
    across eras. That distinction matters here: inflation was far more
    volatile after 2021 than before.
    """

    blocks = []

    for era, is_late in [(ERA_EARLY, 0.0), (ERA_LATE, 1.0)]:
        x, y = paired(era_slice(data, era), measure, target)

        blocks.append(
            pd.DataFrame(
                {
                    "y": standardise(y),
                    "x": standardise(x),
                    "late": is_late,
                }
            )
        )

    stacked = pd.concat(blocks, ignore_index=True)

    design = np.column_stack(
        [
            np.ones(len(stacked)),
            stacked["late"].to_numpy(),
            stacked["x"].to_numpy(),
            (stacked["late"] * stacked["x"]).to_numpy(),
        ]
    )

    lags = newey_west_bandwidth(
        len(stacked), min_lags=lag_floor(target)
    )

    fit = newey_west_ols(stacked["y"].to_numpy(), design, lags)

    low, high = fit.confidence_interval(3)

    # The interaction must equal the difference of the two separate
    # correlations; if it does not, the standardisation has gone wrong.
    early = correlation_hac(
        *paired(era_slice(data, ERA_EARLY), measure, target),
        min_lags=lag_floor(target),
    ).correlation

    late = correlation_hac(
        *paired(era_slice(data, ERA_LATE), measure, target),
        min_lags=lag_floor(target),
    ).correlation

    difference = float(fit.coefficients[3])

    if not np.isclose(difference, late - early, atol=1e-8):
        raise ValueError(
            "Interaction coefficient does not equal the difference in "
            f"correlations ({difference:.8f} vs "
            f"{late - early:.8f}); check the within-era "
            "standardisation."
        )

    return {
        "measure": measure,
        "measure_label": LABELS[measure],
        "target": target,
        "target_label": LABELS[target],
        "corr_early": early,
        "corr_late": late,
        "difference": difference,
        "hac_se": float(fit.standard_errors[3]),
        "t_stat": float(fit.t_statistics[3]),
        "p_value": float(fit.p_values[3]),
        "ci_low": low,
        "ci_high": high,
        "n_quarters": fit.n_obs,
        "hac_lags": fit.lags,
    }


def difference_table(data: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            era_difference_test(data, HEADLINE_MEASURE, target)
            for target in HEADLINE_TARGETS
        ]
    )


# ============================================================
# Robustness
# ============================================================

def break_sensitivity(data: pd.DataFrame) -> pd.DataFrame:
    """
    Recompute both correlations for every candidate break year.

    The 2012 split is imposed. If the pattern only appears at that one
    date it is a property of the split, not of the data.
    """

    rows = []

    for break_year in CANDIDATE_BREAKS:
        early = data[data["year"] < break_year]
        late = data[data["year"] >= break_year]

        row = {"break_year": break_year}

        for target in HEADLINE_TARGETS:
            for label, subset in [("early", early), ("late", late)]:
                x, y = paired(subset, HEADLINE_MEASURE, target)

                estimate = correlation_hac(
                    x, y, min_lags=lag_floor(target)
                )

                row[f"{target}_{label}"] = estimate.correlation
                row[f"{target}_{label}_se"] = estimate.standard_error
                row[f"{target}_{label}_n"] = estimate.n_obs

        rows.append(row)

    return pd.DataFrame(rows)


def policy_rate_robustness(data: pd.DataFrame) -> pd.DataFrame:
    """
    Repeat the headline policy-path result with the main refinancing
    rate in place of the deposit facility rate.

    Which rate represents the policy stance is not constant over this
    sample, and the change falls badly for the comparison being made.
    Under the pre-crisis corridor system the deposit rate was the floor
    and the main refinancing rate was the operative instrument, with
    overnight rates tracking the latter; only once excess liquidity had
    accumulated, from roughly 2014, did the deposit rate become the rate
    that governs the stance. Because the corridor was usually symmetric
    with fixed width, changes in the two largely coincide -- but the
    width itself moved repeatedly between 2008 and 2013.

    The deposit rate is therefore measured with some slippage precisely
    in the early era, which is the subsample carrying the "before" half
    of the rotation. If the rise in the correlation is an artefact of
    using the wrong rate early on, it should shrink or vanish when the
    main refinancing rate is used instead.
    """

    rows = []

    for target in ("d2_dfr", "d2_mro"):
        for era in (ERA_EARLY, ERA_LATE):
            row = correlation_row(
                era_slice(data, era), HEADLINE_MEASURE, target, era
            )

            row["policy_rate"] = (
                "Deposit facility"
                if target == "d2_dfr"
                else "Main refinancing"
            )

            rows.append(row)

    frame = pd.DataFrame(rows)

    return frame[
        [
            "policy_rate",
            "sample",
            "n_quarters",
            "correlation",
            "hac_se",
            "p_value",
            "ci_low",
            "ci_high",
            "hac_lags",
        ]
    ]


def length_adjusted_table(data: pd.DataFrame) -> pd.DataFrame:
    """
    Repeat the headline table with speech length partialled out.

    The data audit established that attention rises with text length
    and that mean speech length fell over the sample. The outlook
    measure is conditional on attention and so is less exposed to that
    than the attention shares are, but the check is cheap and the
    result should not depend on it.

    Length is removed from the communication measure within era, and
    the residual is correlated with the target. The reported quantity
    is therefore a partial correlation, not the raw one.
    """

    rows = []

    for target in HEADLINE_TARGETS:
        for era in (ERA_EARLY, ERA_LATE):
            subset = era_slice(data, era)

            columns = [HEADLINE_MEASURE, target] + LENGTH_CONTROLS

            complete = subset[columns].dropna()

            controls = np.column_stack(
                [np.ones(len(complete))]
                + [
                    complete[control].to_numpy(float)
                    for control in LENGTH_CONTROLS
                ]
            )

            measure = complete[HEADLINE_MEASURE].to_numpy(float)

            coefficients = np.linalg.lstsq(
                controls, measure, rcond=None
            )[0]

            residual = measure - controls @ coefficients

            estimate = correlation_hac(
                residual,
                complete[target].to_numpy(float),
                min_lags=lag_floor(target),
            )

            rows.append(
                {
                    "sample": era,
                    "target": target,
                    "target_label": LABELS[target],
                    "n_quarters": estimate.n_obs,
                    "partial_correlation": estimate.correlation,
                    "hac_se": estimate.standard_error,
                    "p_value": estimate.p_value,
                    "hac_lags": estimate.lags,
                }
            )

    return pd.DataFrame(rows)


def alternative_specifications(data: pd.DataFrame) -> pd.DataFrame:
    """
    The headline table under two changes of ingredient: the policy rate
    measured as a quarterly average rather than end of quarter, and the
    measures rebuilt from English-language speeches only.
    """

    rows = []

    for era in (ERA_EARLY, ERA_LATE):
        row = correlation_row(
            era_slice(data, era),
            HEADLINE_MEASURE,
            "d2_dfr_avg",
            era,
        )
        row["specification"] = "Quarterly-average policy rate"
        rows.append(row)

    if ENGLISH_MEASURES_FILE.exists():
        english = load_merged(ENGLISH_MEASURES_FILE)

        for target in HEADLINE_TARGETS:
            for era in (ERA_EARLY, ERA_LATE):
                row = correlation_row(
                    era_slice(english, era),
                    HEADLINE_MEASURE,
                    target,
                    era,
                )
                row["specification"] = "English-language speeches only"
                rows.append(row)

    frame = pd.DataFrame(rows)

    return frame[
        ["specification", "sample", "target_label", "n_quarters"]
        + ["correlation", "hac_se", "p_value", "hac_lags"]
    ]


# ============================================================
# Output formatting
# ============================================================

def stars(p_value: float) -> str:
    """Conventional significance markers."""

    if not np.isfinite(p_value):
        return ""

    if p_value < 0.01:
        return "***"

    if p_value < 0.05:
        return "**"

    if p_value < 0.10:
        return "*"

    return ""


def write_headline_tex(
    era: pd.DataFrame,
    differences: pd.DataFrame,
    path: Path,
) -> None:
    """Headline table: correlations by era, with the difference test."""

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Attention-weighted inflation outlook and euro-area "
        r"macro outcomes}",
        r"\label{tab:external-validation}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r" & "
        + f"{ERA_EARLY}"
        + r" & "
        + f"{ERA_LATE}"
        + r" & Difference \\",
        r"\midrule",
    ]

    for target in HEADLINE_TARGETS:
        cells = {
            row["sample"]: row
            for _, row in era[era["target"] == target].iterrows()
        }

        difference = differences[
            differences["target"] == target
        ].iloc[0]

        early = cells[ERA_EARLY]
        late = cells[ERA_LATE]

        lines.append(
            f"{LABELS[target]} "
            f"& {early['correlation']:.3f}{stars(early['p_value'])} "
            f"& {late['correlation']:.3f}{stars(late['p_value'])} "
            f"& {difference['difference']:+.3f}"
            f"{stars(difference['p_value'])} \\\\"
        )

        lines.append(
            f"& ({early['hac_se']:.3f}) "
            f"& ({late['hac_se']:.3f}) "
            f"& ({difference['hac_se']:.3f}) \\\\"
        )

        lines.append(
            f"\\quad Quarters & {early['n_quarters']} "
            f"& {late['n_quarters']} "
            f"& {difference['n_quarters']} \\\\"
        )

        lines.append(r"\addlinespace")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{minipage}{0.86\textwidth}",
        r"\vspace{0.6em}",
        r"\footnotesize Pearson correlations between the "
        r"attention-weighted inflation outlook and each macro series, "
        r"quarterly. The policy-path variable is the change in the "
        f"deposit facility rate over the following {RATE_HORIZON} "
        r"quarters. Newey--West standard errors in parentheses; the "
        r"bandwidth follows $\lfloor 4(T/100)^{2/9}\rfloor$, floored "
        r"at the moving-average order implied by the overlapping "
        r"differences. The difference column is the interaction "
        r"coefficient from a pooled regression on within-era "
        r"standardised data and equals the difference of the two "
        r"correlations by construction. The 2012 break is imposed, not "
        r"estimated. Confidence intervals are symmetric in the "
        r"standardised slope and are therefore not constrained to "
        r"$[-1,1]$; the moving-block bootstrap intervals reported in "
        r"\texttt{external\_validation\_by\_era.csv} respect the "
        r"bounds. $^{*}p<0.10$, $^{**}p<0.05$, $^{***}p<0.01$.",
        r"\end{minipage}",
        r"\end{table}",
    ]

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ============================================================
# Figures
# ============================================================

def plot_series(data: pd.DataFrame, path: Path) -> None:
    """The inflation outlook against each macro series over time."""

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    time = data["quarter"].dt.to_timestamp()

    break_date = pd.Timestamp(f"{ERA_BREAK_YEAR}-01-01")

    for ax, target in zip(axes, HEADLINE_TARGETS):
        ax.axhline(0, color="0.7", linewidth=0.8)

        ax.plot(
            time,
            data[HEADLINE_MEASURE],
            color="#4c72b0",
            linewidth=1.6,
            label=LABELS[HEADLINE_MEASURE],
        )

        ax.set_ylabel(LABELS[HEADLINE_MEASURE], color="#4c72b0")
        ax.tick_params(axis="y", labelcolor="#4c72b0")
        ax.set_ylim(-1.05, 1.05)

        twin = ax.twinx()

        twin.plot(
            time,
            data[target],
            color="#c44e52",
            linewidth=1.6,
            linestyle="--",
            label=LABELS[target],
        )

        twin.set_ylabel(LABELS[target], color="#c44e52")
        twin.tick_params(axis="y", labelcolor="#c44e52")

        ax.axvline(
            break_date, color="0.35", linewidth=1.0, linestyle=":"
        )

        ax.grid(alpha=0.2, linewidth=0.6)

        ax.set_title(
            f"{LABELS[HEADLINE_MEASURE]} and {LABELS[target]}",
            fontsize=11,
        )

    axes[1].set_xlabel("Quarter")

    fig.suptitle(
        "ECB inflation talk against realised inflation and the policy "
        f"path (break at {ERA_BREAK_YEAR})",
        fontsize=12,
    )

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_correlations(era: pd.DataFrame, path: Path) -> None:
    """Correlation estimates by era with both interval types."""

    fig, ax = plt.subplots(figsize=(8.5, 4.2))

    positions = []
    labels = []

    offset = {ERA_EARLY: -0.16, ERA_LATE: 0.16}
    colour = {ERA_EARLY: "#4c72b0", ERA_LATE: "#c44e52"}

    for index, target in enumerate(HEADLINE_TARGETS):
        positions.append(index)
        labels.append(LABELS[target])

        for era_name in (ERA_EARLY, ERA_LATE):
            row = era[
                (era["target"] == target)
                & (era["sample"] == era_name)
            ].iloc[0]

            x = index + offset[era_name]

            ax.plot(
                [x, x],
                [row["boot_ci_low"], row["boot_ci_high"]],
                color=colour[era_name],
                linewidth=1.0,
                alpha=0.45,
            )

            ax.plot(
                [x, x],
                [row["ci_low"], row["ci_high"]],
                color=colour[era_name],
                linewidth=2.6,
            )

            ax.plot(
                x,
                row["correlation"],
                "o",
                color=colour[era_name],
                markersize=7,
                label=(
                    era_name
                    if index == 0
                    else None
                ),
            )

    ax.axhline(0, color="0.4", linewidth=0.9)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_xlim(-0.5, len(positions) - 0.5)
    ax.set_ylabel("Correlation with inflation outlook")

    ax.set_title(
        "Thick bars: Newey--West 95% interval. Thin bars: "
        "moving-block bootstrap.",
        fontsize=9,
        color="0.35",
    )

    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ============================================================
# Main
# ============================================================

def report(frame: pd.DataFrame, columns: list[str]) -> str:
    return frame[columns].round(3).to_string(index=False)


def main() -> None:
    for directory in (PANEL_DIR, VALIDATION_DIR, TABLE_DIR, FIGURE_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    data = load_merged()

    print("\n=== MERGED QUARTERLY PANEL ===")
    print(f"Quarters: {len(data)}")
    print(f"Period: {data['period'].iloc[0]} to {data['period'].iloc[-1]}")
    print(
        f"Speeches: {int(data['n_speeches'].sum()):,} "
        f"(min {int(data['n_speeches'].min())} per quarter)"
    )
    print(
        f"Policy-path variable defined for "
        f"{int(data['d2_dfr'].notna().sum())} quarters "
        f"(last {RATE_HORIZON} necessarily missing)"
    )

    data.to_csv(MERGED_FILE, index=False)

    # --------------------------------------------------
    # 1. Full-sample concordance
    # --------------------------------------------------

    concordance = concordance_table(data)

    print("\n=== FULL-SAMPLE CONCORDANCE, 1999-2025 ===")
    print(
        report(
            concordance,
            [
                "measure_label",
                "target_label",
                "n_quarters",
                "correlation",
                "hac_se",
                "p_value",
                "expected_sign",
                "sign_as_expected",
            ],
        )
    )

    wrong_sign = concordance[~concordance["sign_as_expected"]]

    if not wrong_sign.empty:
        print(
            "\nNote: "
            f"{len(wrong_sign)} pair(s) carry the unexpected sign: "
            + ", ".join(
                f"{row['measure']}~{row['target']}"
                for _, row in wrong_sign.iterrows()
            )
        )

    # --------------------------------------------------
    # 2. Headline table by era
    # --------------------------------------------------

    era = era_table(data)

    print(f"\n=== HEADLINE: {LABELS[HEADLINE_MEASURE].upper()} BY ERA ===")
    print(
        report(
            era,
            [
                "target_label",
                "sample",
                "n_quarters",
                "correlation",
                "hac_se",
                "t_stat",
                "p_value",
                "ci_low",
                "ci_high",
                "hac_lags",
                "boot_ci_low",
                "boot_ci_high",
            ],
        )
    )

    differences = difference_table(data)

    print("\n=== TEST OF THE CHANGE ACROSS ERAS ===")
    print(
        report(
            differences,
            [
                "target_label",
                "corr_early",
                "corr_late",
                "difference",
                "hac_se",
                "t_stat",
                "p_value",
                "ci_low",
                "ci_high",
            ],
        )
    )

    # --------------------------------------------------
    # 3. Robustness
    # --------------------------------------------------

    breaks = break_sensitivity(data)

    print("\n=== SENSITIVITY TO THE BREAK YEAR ===")
    print(
        report(
            breaks,
            ["break_year"]
            + [
                f"{target}_{side}"
                for target in HEADLINE_TARGETS
                for side in ("early", "late")
            ],
        )
    )

    length = length_adjusted_table(data)

    print("\n=== LENGTH-ADJUSTED (PARTIAL) CORRELATIONS ===")
    print(
        report(
            length,
            [
                "target_label",
                "sample",
                "n_quarters",
                "partial_correlation",
                "hac_se",
                "p_value",
            ],
        )
    )

    rates = policy_rate_robustness(data)

    print("\n=== WHICH POLICY RATE? DFR AGAINST MRO ===")
    print(
        report(
            rates,
            ["policy_rate", "sample", "n_quarters", "correlation",
             "hac_se", "p_value"],
        )
    )

    mro_difference = era_difference_test(data, HEADLINE_MEASURE, "d2_mro")

    print(
        "\nEra difference with the MRO rate: "
        f"{mro_difference['difference']:+.3f} "
        f"(se {mro_difference['hac_se']:.3f}, "
        f"p = {mro_difference['p_value']:.3f})"
    )

    alternatives = alternative_specifications(data)

    print("\n=== ALTERNATIVE SPECIFICATIONS ===")
    print(
        report(
            alternatives,
            [
                "specification",
                "target_label",
                "sample",
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

    concordance.to_csv(
        VALIDATION_DIR / "external_concordance.csv", index=False
    )

    era.to_csv(
        VALIDATION_DIR / "external_validation_by_era.csv", index=False
    )

    differences.to_csv(
        VALIDATION_DIR / "external_era_difference_tests.csv",
        index=False,
    )

    breaks.to_csv(
        VALIDATION_DIR / "external_break_sensitivity.csv", index=False
    )

    length.to_csv(
        VALIDATION_DIR / "external_length_adjusted.csv", index=False
    )

    rates.to_csv(
        VALIDATION_DIR / "external_policy_rate_robustness.csv",
        index=False,
    )

    pd.DataFrame([mro_difference]).to_csv(
        VALIDATION_DIR / "external_mro_difference_test.csv", index=False
    )

    alternatives.to_csv(
        VALIDATION_DIR / "external_alternative_specs.csv", index=False
    )

    write_headline_tex(
        era, differences, TABLE_DIR / "external_validation.tex"
    )

    plot_series(data, FIGURE_DIR / "outlook_vs_macro.png")

    plot_correlations(
        era, FIGURE_DIR / "outlook_macro_correlations.png"
    )

    print(f"\nSaved merged panel to: {MERGED_FILE}")
    print(f"Saved statistics to: {VALIDATION_DIR}")
    print(f"Saved table to: {TABLE_DIR / 'external_validation.tex'}")
    print(f"Saved figures to: {FIGURE_DIR}")


if __name__ == "__main__":
    main()
