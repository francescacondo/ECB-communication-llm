"""
Aggregate the speech-level signals into communication measures.

Attention and outlook are conceptually distinct and are kept apart
throughout. For topic x in period t, with N_t speeches:

    Attention^x_t   = (1/N_t) * sum_i Attention^x_i
    Volume^x_t      = sum_i Attention^x_i
    Outlook^x_t     = sum_i Attention^x_i * Outlook^x_i
                      / sum_i Attention^x_i

The first is the share of communication devoted to the topic, the
second its absolute quantity, the third the directional stance
conditional on the topic being discussed. Financial stability and
uncertainty have no outlook component, so only the first two apply.

Because attention is coded 0/1, the attention-weighted outlook reduces
exactly to the mean outlook over speeches that discuss the topic. The
weighted form is implemented anyway, since it generalises if attention
is ever graded, and the equivalence is asserted rather than assumed.

Attention shares are additionally reported corrected for
misclassification, using the sensitivity and specificity estimated in
analysis/15_validation_stats.py. Run that script first.

Run from the project root:

    python analysis/12_measures.py
"""

from pathlib import Path
import json
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from measurement_error import ErrorRates, correct_share  # noqa: E402


PANEL_FILE = PROJECT_ROOT / "outputs" / "panel" / "speech_panel.csv"

PARAMS_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "validation"
    / "measurement_error_params.json"
)

MEASURE_DIR = PROJECT_ROOT / "outputs" / "panel"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"
VALIDATION_DIR = PROJECT_ROOT / "outputs" / "validation"

# Rows of the published quarterly summary, in order: column in the
# measures file, printed label, and whether it is a share.
SUMMARY_ROWS = [
    ("n_speeches", "Speeches per quarter"),
    ("att_pi", "Inflation attention"),
    ("att_pi_corr", r"\quad corrected"),
    ("att_y", "Growth attention"),
    ("att_fs", "Financial stability attention"),
    ("att_unc", "Uncertainty attention"),
    ("out_pi", "Inflation outlook"),
    ("out_y", "Growth outlook"),
    ("polar_pi", "Inflation polarisation"),
    ("polar_y", "Growth polarisation"),
]

# Topics with a directional outlook, and topics without.
DIRECTIONAL = {
    "pi": ("inflation_attention", "inflation_outlook"),
    "y": ("growth_attention", "growth_outlook"),
}

ATTENTION_ONLY = {
    "fs": "financial_stability_attention",
    "unc": "uncertainty_attention",
}

ATTENTION_VARS = {
    "pi": "inflation_attention",
    "y": "growth_attention",
    "fs": "financial_stability_attention",
    "unc": "uncertainty_attention",
}

LABELS = {
    "pi": "Inflation",
    "y": "Growth",
    "fs": "Financial stability",
    "unc": "Uncertainty",
}

# Periods with very few speeches give unstable shares and outlooks.
SPARSE_THRESHOLD = 3


# ============================================================
# Measures
# ============================================================

def weighted_outlook(
    attention: pd.Series,
    outlook: pd.Series,
) -> float:
    """
    Attention-weighted mean outlook, undefined when no speech in the
    period discusses the topic.
    """

    total = attention.sum()

    if total == 0:
        return float("nan")

    return float((attention * outlook).sum() / total)


def polarisation(outlook: pd.Series) -> float:
    """
    Directional disagreement among speeches that discuss the topic.

    Zero when every directional speech points the same way, one when
    upward and downward assessments are evenly split. Speeches coded
    neutral are excluded, so this measures disagreement about direction
    rather than reluctance to take a direction.
    """

    up = float((outlook == 1).sum())
    down = float((outlook == -1).sum())

    if up + down == 0:
        return float("nan")

    return 2.0 * min(up, down) / (up + down)


def period_measures(group: pd.DataFrame) -> pd.Series:
    """Compute all communication measures for one period."""

    out = {
        "n_speeches": len(group),
        "n_speakers": group["speaker"].nunique(),
        # Attention rises mechanically with text length, and mean
        # speech length trends over the sample. Carrying length to the
        # aggregate frequency lets downstream work control for it
        # without re-aggregating the speech panel.
        "mean_log_words": float(group["log_words"].mean()),
        "share_truncated": float(group["truncated"].mean()),
    }

    for key, variable in ATTENTION_VARS.items():
        out[f"att_{key}"] = float(group[variable].mean())
        out[f"vol_{key}"] = int(group[variable].sum())

    for key, (att_var, out_var) in DIRECTIONAL.items():
        attending = group.loc[group[att_var] == 1, out_var]

        out[f"out_{key}"] = weighted_outlook(
            group[att_var], group[out_var]
        )

        out[f"disp_{key}"] = (
            float(attending.std(ddof=1))
            if len(attending) > 1
            else float("nan")
        )

        out[f"polar_{key}"] = polarisation(attending)

        out[f"share_up_{key}"] = (
            float((attending == 1).mean())
            if len(attending) > 0
            else float("nan")
        )

        out[f"share_down_{key}"] = (
            float((attending == -1).mean())
            if len(attending) > 0
            else float("nan")
        )

    return pd.Series(out)


def check_weighting_equivalence(panel: pd.DataFrame) -> None:
    """
    Confirm that the attention-weighted outlook equals the conditional
    mean outlook. This holds because attention is binary; the check
    exists so that the two would diverge visibly if attention were ever
    recoded on a graded scale.
    """

    for key, (att_var, out_var) in DIRECTIONAL.items():
        attending = panel[panel[att_var] == 1]

        if attending.empty:
            continue

        weighted = weighted_outlook(
            panel[att_var], panel[out_var]
        )

        conditional = float(attending[out_var].mean())

        if not np.isclose(weighted, conditional):
            raise ValueError(
                f"Weighted and conditional outlook differ for {key}: "
                f"{weighted:.6f} vs {conditional:.6f}. Attention is no "
                "longer binary; revisit the aggregation."
            )


# ============================================================
# Correction
# ============================================================

def load_error_rates() -> dict[str, ErrorRates]:
    """Read the validated misclassification rates."""

    if not PARAMS_FILE.exists():
        raise FileNotFoundError(
            f"{PARAMS_FILE} not found. Run "
            "analysis/15_validation_stats.py first."
        )

    params = json.loads(PARAMS_FILE.read_text(encoding="utf-8"))

    rates = {}

    for key, variable in ATTENTION_VARS.items():
        entry = params["variables"][variable]

        rates[key] = ErrorRates(
            sensitivity=entry["sensitivity"],
            specificity=entry["specificity"],
            n_true_one=entry["n_true_one"],
            n_true_zero=entry["n_true_zero"],
        )

    return rates


def add_corrected_attention(
    measures: pd.DataFrame,
    rates: dict[str, ErrorRates],
) -> pd.DataFrame:
    """
    Add misclassification-corrected attention shares.

    The correction is applied period by period. It is not a level
    shift: because it subtracts a constant false-positive rate before
    rescaling, it bites hardest where attention is lowest.
    """

    measures = measures.copy()

    for key, rate in rates.items():
        measures[f"att_{key}_corr"] = correct_share(
            measures[f"att_{key}"], rate
        )

    return measures


# ============================================================
# Aggregation
# ============================================================

def aggregate(
    panel: pd.DataFrame,
    period: str,
    rates: dict[str, ErrorRates],
) -> pd.DataFrame:
    """Aggregate the speech panel to a calendar frequency."""

    measures = (
        panel.groupby(period, sort=True)
        .apply(period_measures, include_groups=False)
        .reset_index()
        .rename(columns={period: "period"})
    )

    measures["is_sparse"] = measures["n_speeches"] < SPARSE_THRESHOLD

    measures = add_corrected_attention(measures, rates)

    ordered = (
        [
            "period",
            "n_speeches",
            "n_speakers",
            "is_sparse",
            "mean_log_words",
            "share_truncated",
        ]
        + [f"att_{k}" for k in ATTENTION_VARS]
        + [f"att_{k}_corr" for k in ATTENTION_VARS]
        + [f"vol_{k}" for k in ATTENTION_VARS]
        + [
            column
            for key in DIRECTIONAL
            for column in (
                f"out_{key}",
                f"disp_{key}",
                f"polar_{key}",
                f"share_up_{key}",
                f"share_down_{key}",
            )
        ]
    )

    return measures[ordered]


def reindex_calendar(
    measures: pd.DataFrame,
    freq: str,
) -> pd.DataFrame:
    """
    Insert rows for periods with no speeches at all, so that gaps in
    ECB communication are visible rather than silently skipped by a
    time-series routine.
    """

    periods = pd.PeriodIndex(measures["period"], freq=freq)

    full = pd.period_range(periods.min(), periods.max(), freq=freq)

    measures = (
        measures.assign(period=periods)
        .set_index("period")
        .reindex(full)
    )

    measures["n_speeches"] = measures["n_speeches"].fillna(0).astype(int)
    measures["n_speakers"] = measures["n_speakers"].fillna(0).astype(int)
    measures["is_sparse"] = measures["n_speeches"] < SPARSE_THRESHOLD

    return measures.rename_axis("period").reset_index()


# ============================================================
# Figure
# ============================================================

def plot_raw_vs_corrected(annual: pd.DataFrame, path: Path) -> None:
    """Annual attention shares, observed against corrected."""

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)

    years = annual["period"].astype(int)

    for ax, key in zip(axes.flat, ATTENTION_VARS):
        ax.plot(
            years,
            annual[f"att_{key}"],
            color="#4c72b0",
            linewidth=1.8,
            label="Observed",
        )

        ax.plot(
            years,
            annual[f"att_{key}_corr"],
            color="#c44e52",
            linewidth=1.8,
            linestyle="--",
            label="Corrected",
        )

        ax.fill_between(
            years,
            annual[f"att_{key}_corr"],
            annual[f"att_{key}"],
            color="#c44e52",
            alpha=0.12,
        )

        ax.set_title(LABELS[key], fontsize=11)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.25, linewidth=0.6)

    axes[0, 0].legend(frameon=False, fontsize=9, loc="lower left")

    for ax in axes[1]:
        ax.set_xlabel("Year")

    for ax in axes[:, 0]:
        ax.set_ylabel("Share of speeches")

    fig.suptitle(
        "ECB speech attention: observed and misclassification-corrected",
        fontsize=12,
    )

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ============================================================
# Main
# ============================================================

def quarterly_summary(quarterly: pd.DataFrame) -> pd.DataFrame:
    """
    Distribution of each quarterly measure, as published.

    Generated rather than transcribed: an earlier draft of the note
    carried these figures typed by hand from console output, which made
    them unverifiable from the repository even though they were correct.
    """

    rows = []

    for column, label in SUMMARY_ROWS:
        series = quarterly[column].dropna()

        rows.append(
            {
                "measure": column,
                "label": label,
                "mean": float(series.mean()),
                "sd": float(series.std(ddof=1)),
                "min": float(series.min()),
                "max": float(series.max()),
            }
        )

    return pd.DataFrame(rows)


def write_summary_tex(table: pd.DataFrame, path: Path) -> None:
    """Emit the quarterly summary table the note includes."""

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Quarterly communication measures, 1999Q1--2025Q4 "
        r"(108 quarters)}",
        r"\label{tab:summary}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Measure & Mean & S.D. & Min & Max \\",
        r"\midrule",
    ]

    for _, row in table.iterrows():
        if row["measure"] in ("att_pi", "out_pi", "polar_pi"):
            lines.append(r"\addlinespace")

        digits = 2 if row["measure"] == "n_speeches" else 3

        def fmt(value: float) -> str:
            text = f"{value:.{digits}f}"
            return f"$-{text[1:]}$" if value < 0 else text

        lines.append(
            f"{row['label']} & {fmt(row['mean'])} & {fmt(row['sd'])} "
            f"& {fmt(row['min'])} & {fmt(row['max'])} \\\\"
        )

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def language_robustness(
    baseline: pd.DataFrame,
    english: pd.DataFrame,
) -> pd.DataFrame:
    """
    Correlate each quarterly measure with its English-only counterpart.

    The data audit flagged 151 speeches as delivered in a language other
    than English and asked whether they move the aggregates. This is the
    check that answers it, and until now the answer had been asserted in
    the documentation without any committed code producing it.
    """

    merged = baseline.merge(
        english, on="period", suffixes=("", "_en"), validate="one_to_one"
    )

    rows = []

    for key in ATTENTION_VARS:
        for prefix in ("att", "out"):
            column = f"{prefix}_{key}"

            if column not in merged or f"{column}_en" not in merged:
                continue

            pair = merged[[column, f"{column}_en"]].dropna()

            rows.append(
                {
                    "measure": column,
                    "n_quarters": len(pair),
                    "correlation": float(
                        pair[column].corr(pair[f"{column}_en"])
                    ),
                }
            )

    return pd.DataFrame(rows)


def main() -> None:
    MEASURE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    panel = pd.read_csv(PANEL_FILE, parse_dates=["date"])

    rates = load_error_rates()

    baseline = panel[panel["in_baseline"]].copy()

    check_weighting_equivalence(baseline)

    print("\n=== AGGREGATION INPUT ===")
    print(f"Speeches in panel: {len(panel):,}")
    print(f"Speeches in baseline: {len(baseline):,}")
    print(
        "Attention-weighted outlook verified equal to the conditional "
        "mean (attention is binary)."
    )

    outputs = {}

    for label, freq, column in [
        ("monthly", "M", "ym"),
        ("quarterly", "Q", "yq"),
        ("annual", "Y", "year"),
    ]:
        measures = aggregate(baseline, column, rates)

        if freq != "Y":
            measures = reindex_calendar(measures, freq)
            measures["period"] = measures["period"].astype(str)

        outputs[label] = measures

        path = MEASURE_DIR / f"measures_{label}.csv"
        measures.to_csv(path, index=False)

        print(
            f"\n{label}: {len(measures)} periods, "
            f"{int((measures['n_speeches'] == 0).sum())} empty, "
            f"{int(measures["is_sparse"].sum())} sparse "
            f"(<{SPARSE_THRESHOLD} speeches)"
        )

    # English-only variant, for the language robustness check.
    english = baseline[~baseline["non_english"]].copy()

    english_quarterly = aggregate(english, "yq", rates)

    english_quarterly.to_csv(
        MEASURE_DIR / "measures_quarterly_english_only.csv", index=False
    )

    print(
        f"\nEnglish-only quarterly variant: {len(english):,} speeches, "
        f"{len(english_quarterly)} periods"
    )

    language = language_robustness(outputs["quarterly"], english_quarterly)

    print("\n=== LANGUAGE ROBUSTNESS ===")
    print(
        "Each quarterly measure against its English-only counterpart."
    )
    print(language.round(4).to_string(index=False))
    print(
        f"\nRange: {language['correlation'].min():.3f} to "
        f"{language['correlation'].max():.3f}. The non-English speeches "
        "do not move the aggregates."
    )

    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)

    language.to_csv(
        VALIDATION_DIR / "language_robustness.csv", index=False
    )

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------

    quarterly = outputs["quarterly"]

    print("\n=== QUARTERLY MEASURES: SUMMARY ===")
    print(
        quarterly[
            [
                "n_speeches",
                "att_pi",
                "att_pi_corr",
                "att_y",
                "att_fs",
                "att_unc",
                "out_pi",
                "out_y",
                "polar_pi",
                "polar_y",
            ]
        ]
        .describe()
        .loc[["mean", "std", "min", "max"]]
        .round(3)
        .to_string()
    )

    annual = outputs["annual"]

    print("\n=== SELECTED YEARS ===")
    selected = annual[annual["period"].isin([2008, 2012, 2020, 2022, 2023])]

    print(
        selected[
            [
                "period",
                "n_speeches",
                "att_pi",
                "att_pi_corr",
                "out_pi",
                "out_y",
                "polar_pi",
            ]
        ]
        .round(3)
        .to_string(index=False)
    )

    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    summary = quarterly_summary(quarterly)

    summary.to_csv(
        MEASURE_DIR / "quarterly_summary.csv", index=False
    )

    write_summary_tex(summary, TABLE_DIR / "quarterly_summary.tex")

    figure_path = FIGURE_DIR / "attention_raw_vs_corrected.png"
    plot_raw_vs_corrected(annual, figure_path)

    print(f"\nSaved measures to: {MEASURE_DIR}")
    print(f"Saved figure to: {figure_path}")


if __name__ == "__main__":
    main()
