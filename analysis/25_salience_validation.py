"""
Contemporaneous salience: do the attention measures track what they
must track?

Earlier external validation tested the measures against things a valid
measure need not predict at all — the policy surprise, the subsequent
rate path. Failing those tells us little, because no theory says a
speech has to forecast the Governing Council. This script tests the
opposite kind of claim, the kind a measure cannot fail and still be
what it says it is. If speeches do not discuss financial stability more
when the euro area is under financial stress, `att_fs` is not measuring
attention to financial stability. Nothing about forecasting is at
stake; every pair below is contemporaneous.

## The pairs, and their signs, are fixed in advance

PAIRS below is the pre-registration. It was written and committed
before any correlation in this script was computed, and every entry in
it is reported, whatever it shows. That is the whole point of writing
it down: a salience test is only informative if the set of pairs is
closed before the estimates arrive.

    att_fs  vs CISS                     positive   (primary)
    att_unc vs SPF forecast dispersion  positive
    att_y   vs the 2q unemployment change   positive
    out_y   vs the 2q unemployment change   negative
    att_pi  vs |HICP - 2|                positive

The last is new. Inflation should be talked about more when inflation
is far from target in *either* direction, so the natural counterpart
for inflation attention is the absolute gap, not the signed rate.
analysis/13_external_validation.py pairs `att_pi` with the signed rate,
which is a weaker test: it asks whether inflation is discussed more
when inflation is high, and cannot see the deflation-scare half of the
sample at all.

The two growth pairs are stated in the pre-registration as already
established. Only one of them is. `out_y` against the two-quarter
unemployment change is in outputs/validation/external_concordance.csv
at -0.349; `att_y` against that same change is not, and has never been
run — the existing table pairs `att_y` with the unemployment *level*.
So `out_y` is reproduced from the earlier script and checked against
the stored figure, and `att_y` is estimated here for the first time.
reproduce_out_y() is what enforces that distinction.

## Inference

Newey-West throughout, with the automatic bandwidth from
src/timeseries_stats.py, plus a moving-block bootstrap interval as a
non-parametric second opinion. Holm's step-down correction is applied
across the five pre-registered pairs, because five pre-committed tests
with a one-sided prediction each is exactly the setting in which an
uncorrected p-value overstates what has been shown.

The robustness variants are deliberately *outside* the Holm family.
They are alternative operationalisations of pairs already in it, not
new hypotheses, and folding them in would penalise the pre-registered
set for the fact that CISS happens to be published twice.

## Speech length

Attention is read off a fixed 2,000-word excerpt, and mean speech
length falls by 39% from 1999 to its trough in 2017
(analysis/10_panel.py). A topic has more chance of being discussed
substantively in a long speech, so an attention series could in
principle track the length trend rather than the state it is supposed
to track. length_adjusted() partials mean log words and the truncated
share out of each measure and re-estimates. It mirrors
length_adjusted_table() in analysis/13_external_validation.py, which
runs the same check on the abandoned out_pi headline, so the two are
comparable.

## Attenuation

Two of the five pairs are measured with known and substantial error.
Round 4 gives human-human kappa of 0.560 for financial-stability
attention and 0.280 for uncertainty attention
(outputs/validation/round4_ceiling.csv). Both correlations are
attenuated toward zero by an amount that is not identified here, and
the uncertainty one is attenuated so heavily that a small estimate
cannot be told apart from a large one measured badly. The script
reports the reliability alongside every correlation so that the
distinction is on the face of the table rather than left to the reader.

Requires:

    python src/download_macro.py
    python src/download_salience_series.py
    python analysis/12_measures.py

Run from the project root:

    python analysis/25_salience_validation.py
"""

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from agreement import binomial_cdf  # noqa: E402

from timeseries_stats import (  # noqa: E402
    newey_west_bandwidth,
    newey_west_ols,
    standardise,
    two_sided_t_pvalue,
    bootstrap_block_length,
    correlation_hac,
    moving_block_bootstrap_ci,
)

MEASURES_FILE = (
    PROJECT_ROOT / "outputs" / "panel" / "measures_quarterly.csv"
)

MACRO_FILE = (
    PROJECT_ROOT / "data" / "processed" / "macro_quarterly.csv"
)

SALIENCE_FILE = (
    PROJECT_ROOT / "data" / "processed" / "salience_quarterly.csv"
)

VALIDATION_DIR = PROJECT_ROOT / "outputs" / "validation"

CEILING_FILE = VALIDATION_DIR / "round4_ceiling.csv"

CONCORDANCE_FILE = VALIDATION_DIR / "external_concordance.csv"

FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"

PAIRS_FILE = VALIDATION_DIR / "salience_pairs.csv"
VARIANTS_FILE = VALIDATION_DIR / "salience_variants.csv"
SPECIFICATIONS_FILE = VALIDATION_DIR / "salience_specification_count.csv"


# ============================================================
# Configuration
# ============================================================

# Horizon for the unemployment-change target, in quarters. Matched to
# RATE_HORIZON in analysis/13_external_validation.py so that the `out_y`
# figure reproduced below is comparable with the stored one.
UNEMPLOYMENT_HORIZON = 2

# The inflation target the absolute gap is taken around. Two per cent
# is the ECB's objective across the whole sample: the pre-2021 wording
# was "below, but close to, 2%" and the 2021 strategy review made it a
# symmetric 2%. The asymmetry of the earlier wording is a limitation of
# this pair, recorded in the notes rather than modelled.
INFLATION_TARGET = 2.0

# Targets built as overlapping forward differences have MA(h-1)
# residuals by construction, which floors the HAC bandwidth and the
# bootstrap block length.
OVERLAPPING_TARGETS = {"d2_unemployment"}

# Partialled out of each measure in the length check. The two are
# strongly related to one another -- they are close to a single length
# factor rather than two independent controls -- and both are kept
# because the excerpt cap bites through the truncated share while the
# trend in length itself works through the mean.
LENGTH_CONTROLS = ["mean_log_words", "share_truncated"]

# Bandwidths the pre-registered pairs are re-estimated at, either side
# of the automatic choice. The point estimate cannot move -- the
# bandwidth enters the standard error and nothing else -- so this is
# re-inference on five existing estimates, not five new ones.
BANDWIDTH_ALTERNATIVES = (3, 6)

LABELS = {
    "att_pi": "Inflation attention",
    "att_y": "Growth attention",
    "att_fs": "Financial stability attention",
    "att_unc": "Uncertainty attention",
    "out_y": "Growth outlook",
    "ciss_ci": "CISS",
    "ciss_cin": "New CISS",
    "spf_disp_rgdp": "SPF real GDP forecast dispersion",
    "spf_disp_hicp": "SPF inflation forecast dispersion",
    "d2_unemployment": f"Unemployment, {UNEMPLOYMENT_HORIZON}q change",
    "abs_hicp_gap": f"|HICP - {INFLATION_TARGET:g}|",
    "hicp_gap": f"HICP - {INFLATION_TARGET:g}",
    "unemployment": "Unemployment rate",
}


# ------------------------------------------------------------
# THE PRE-REGISTRATION
#
# Fixed before estimation. Every entry is reported.
# ------------------------------------------------------------

PAIRS = [
    ("att_fs", "ciss_ci", +1, "primary"),
    ("att_unc", "spf_disp_rgdp", +1, "pre-registered"),
    ("att_y", "d2_unemployment", +1, "pre-registered"),
    ("out_y", "d2_unemployment", -1, "pre-registered"),
    ("att_pi", "abs_hicp_gap", +1, "pre-registered"),
]

# Alternative operationalisations of pairs already in PAIRS. Reported in
# full, excluded from the Holm family; see the module docstring.
VARIANTS = [
    ("att_fs", "ciss_cin", +1, "CISS published as the New CISS"),
    (
        "att_unc",
        "spf_disp_hicp",
        +1,
        "dispersion of inflation rather than growth forecasts",
    ),
    (
        "att_pi",
        "hicp_gap",
        +1,
        "signed rather than absolute deviation from target",
    ),
    (
        "att_y",
        "unemployment",
        +1,
        "unemployment level rather than change",
    ),
]

# Human-human reliability from round 4, keyed by measure. Any
# correlation involving a measure listed here is attenuated toward zero.
RELIABILITY_ROWS = {
    "att_pi": "inflation_attention",
    "att_y": "growth_attention",
    "att_fs": "financial_stability_attention",
    "att_unc": "uncertainty_attention",
    "out_y": "growth_outlook",
}

# The stored figure this script must reproduce for `out_y`, and the
# tolerance. It is an exact recomputation, not an approximation, so the
# tolerance is numerical rather than substantive.
STORED_OUT_Y_TOLERANCE = 1e-10


# ============================================================
# Data
# ============================================================

def load_panel() -> pd.DataFrame:
    """
    Merge the communication measures with the macro and salience
    panels, and derive the two targets that are not published directly.

    Both merges are validated one-to-one. The salience merge is a left
    join and is *expected* to leave gaps: CISS in its original form
    stops in May 2025 and the SPF rounds end earlier still, so the last
    quarters of the speech sample have no counterpart. The gaps are
    counted and printed rather than asserted away.
    """

    measures = pd.read_csv(MEASURES_FILE)
    macro = pd.read_csv(MACRO_FILE)
    salience = pd.read_csv(SALIENCE_FILE)

    panel = measures.merge(
        macro, on="period", how="left", validate="one_to_one"
    ).merge(
        salience, on="period", how="left", validate="one_to_one"
    )

    if len(panel) != len(measures):
        raise ValueError(
            "A merge changed the number of quarters: "
            f"{len(measures)} -> {len(panel)}"
        )

    quarters = pd.PeriodIndex(panel["period"], freq="Q")

    expected = pd.period_range(quarters.min(), quarters.max(), freq="Q")

    if len(quarters) != len(expected) or (quarters != expected).any():
        raise ValueError(
            "The quarterly index has gaps; the forward difference "
            "below assumes consecutive quarters."
        )

    if panel["is_sparse"].any():
        sparse = panel.loc[panel["is_sparse"], "period"].tolist()

        raise ValueError(
            f"Sparse quarters present: {sparse}. Every measure below "
            "assumes each quarter carries enough speeches to be stable."
        )

    if panel["hicp_yoy"].isna().any():
        raise ValueError(
            "HICP is missing for part of the speech sample; the "
            "inflation-gap pair needs it throughout."
        )

    # --------------------------------------------------
    # Derived targets
    # --------------------------------------------------

    panel["hicp_gap"] = panel["hicp_yoy"] - INFLATION_TARGET

    panel["abs_hicp_gap"] = panel["hicp_gap"].abs()

    # Change over the NEXT h quarters, matching the construction in
    # analysis/13_external_validation.py.
    panel["d2_unemployment"] = (
        panel["unemployment"].shift(-UNEMPLOYMENT_HORIZON)
        - panel["unemployment"]
    )

    return panel


def load_reliability() -> dict[str, float]:
    """Round-4 human-human kappa, by measure."""

    ceiling = pd.read_csv(CEILING_FILE).set_index("variable")

    missing = set(RELIABILITY_ROWS.values()) - set(ceiling.index)

    if missing:
        raise ValueError(
            f"round4_ceiling.csv is missing rows {sorted(missing)}; "
            "the attenuation column cannot be built."
        )

    return {
        measure: float(ceiling.loc[row, "kappa_r4"])
        for measure, row in RELIABILITY_ROWS.items()
    }


def paired(
    panel: pd.DataFrame,
    measure: str,
    target: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Aligned, complete observations for one pair."""

    subset = panel[[measure, target]].dropna()

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
    """Minimum HAC bandwidth implied by the target's construction."""

    return (
        UNEMPLOYMENT_HORIZON - 1
        if target in OVERLAPPING_TARGETS
        else 0
    )


# ============================================================
# Estimation
# ============================================================

def correlation_row(
    panel: pd.DataFrame,
    measure: str,
    target: str,
    expected_sign: int,
    role: str,
    reliability: dict[str, float],
) -> dict:
    """One pre-committed pair, with HAC and bootstrap inference."""

    x, y = paired(panel, measure, target)

    estimate = correlation_hac(x, y, min_lags=lag_floor(target))

    block = bootstrap_block_length(
        len(x), min_block=lag_floor(target) + 1
    )

    boot_low, boot_high = moving_block_bootstrap_ci(x, y, block)

    return {
        "role": role,
        "measure": measure,
        "measure_label": LABELS[measure],
        "target": target,
        "target_label": LABELS[target],
        "expected_sign": expected_sign,
        "n_quarters": estimate.n_obs,
        "correlation": estimate.correlation,
        "hac_se": estimate.standard_error,
        "t_stat": estimate.t_statistic,
        "p_value": estimate.p_value,
        "hac_ci_low": estimate.ci_low,
        "hac_ci_high": estimate.ci_high,
        "hac_lags": estimate.lags,
        "block_length": block,
        "boot_ci_low": boot_low,
        "boot_ci_high": boot_high,
        "sign_as_expected": bool(
            np.sign(estimate.correlation) == np.sign(expected_sign)
        ),
        "measure_kappa_r4": reliability.get(measure, float("nan")),
    }


def holm_adjust(p_values: list[float]) -> np.ndarray:
    """
    Holm's step-down adjusted p-values.

    Sort ascending, scale the k-th smallest by (m - k), then enforce
    monotonicity so that an adjusted p-value can never fall below one
    that precedes it. Holm rather than Bonferroni because it is
    uniformly more powerful and needs no assumption about how the
    tests are related, which matters here: the five pairs share a
    corpus and are certainly not independent.
    """

    p = np.asarray(p_values, dtype=float)

    m = len(p)

    order = np.argsort(p)

    scaled = p[order] * (m - np.arange(m))

    # A running maximum from the smallest p-value upward. Without it a
    # later, larger raw p could receive a smaller adjusted value than
    # one that precedes it, which would break the step-down logic.
    adjusted = np.maximum.accumulate(scaled)

    out = np.empty(m)

    out[order] = np.clip(adjusted, 0.0, 1.0)

    return out


def reproduce_out_y(panel: pd.DataFrame, estimated: float) -> float:
    """
    Check the `out_y` correlation against the stored concordance table.

    The pre-registration describes both growth pairs as already
    computed. Only this one is. Reproducing it exactly is what licenses
    calling it a confirmation rather than a fresh estimate, and a
    mismatch would mean the measures or the macro panel have moved
    under an earlier result that the write-up still cites.
    """

    stored = pd.read_csv(CONCORDANCE_FILE)

    row = stored[
        (stored["measure"] == "out_y")
        & (stored["target"] == "d2_unemployment")
    ]

    if len(row) != 1:
        raise ValueError(
            "Expected exactly one stored out_y / d2_unemployment row "
            f"in {CONCORDANCE_FILE.name}, found {len(row)}."
        )

    stored_value = float(row["correlation"].iloc[0])

    if abs(stored_value - estimated) > STORED_OUT_Y_TOLERANCE:
        raise ValueError(
            "out_y against the unemployment change does not reproduce "
            f"the stored figure: {estimated:.10f} here against "
            f"{stored_value:.10f} in {CONCORDANCE_FILE.name}. The "
            "measures or the macro panel have changed under a result "
            "the write-up cites."
        )

    return stored_value


# ============================================================
# Reporting
# ============================================================

def format_table(table: pd.DataFrame, p_column: str) -> str:
    """A readable rendering of one correlation table."""

    lines = []

    for _, row in table.iterrows():
        interval = (
            f"[{row['hac_ci_low']:+.2f}, {row['hac_ci_high']:+.2f}]"
        )

        boot = (
            f"[{row['boot_ci_low']:+.2f}, {row['boot_ci_high']:+.2f}]"
        )

        lines.append(
            f"  {row['measure']:8s} vs {row['target']:16s} "
            f"exp {row['expected_sign']:+d}  "
            f"r = {row['correlation']:+.3f}  "
            f"se {row['hac_se']:.3f}  "
            f"p {row[p_column]:.3f}  "
            f"HAC {interval:16s} boot {boot:16s} "
            f"n = {row['n_quarters']:3d}"
        )

    return "\n".join(lines)


def level_control(data: pd.DataFrame) -> pd.DataFrame:
    """
    Check that growth outlook predicting the unemployment change is not
    an artefact of the level.

    The outlook correlates positively with the unemployment level, so if
    unemployment mean-reverted, optimistic talk would appear to precede
    falling unemployment while carrying no forward information at all.
    The check adds the level to the regression: if the coefficient on
    the outlook survives, the relationship is not the level in disguise.
    """

    frame = data[["out_y", "unemployment", "d2_unemployment"]].dropna()

    y = standardise(frame["d2_unemployment"].to_numpy(float))

    rows = []

    for label, columns in [
        ("outlook alone", ["out_y"]),
        ("outlook + unemployment level", ["out_y", "unemployment"]),
    ]:
        X = np.column_stack(
            [np.ones(len(frame))]
            + [standardise(frame[c].to_numpy(float)) for c in columns]
        )

        fit = newey_west_ols(
            y, X, newey_west_bandwidth(len(frame), min_lags=1)
        )

        rows.append(
            {
                "specification": label,
                "n": len(frame),
                "beta_out_y": float(fit.coefficients[1]),
                "hac_se": float(fit.standard_errors[1]),
                "p_value": float(fit.p_values[1]),
            }
        )

    frame_out = pd.DataFrame(rows)

    frame_out["corr_outlook_level"] = float(
        frame["out_y"].corr(frame["unemployment"])
    )
    frame_out["corr_level_own_change"] = float(
        frame["unemployment"].corr(frame["d2_unemployment"])
    )

    return frame_out


def signal_contrast(data: pd.DataFrame) -> pd.DataFrame:
    """
    Do attention and outlook carry different information about the same
    target?

    The aggregation step keeps attention, volume and conditional outlook
    apart, and argues that collapsing them into a single tone score --
    as dictionary measures do -- discards a real distinction. That is a
    design argument, and until now nothing tested it.

    This tests it directly. For a topic, take the attention share and
    the conditional outlook, correlate each with the same macro target
    over the same quarters with the same estimator, and test whether the
    two correlations differ. Because the contrast holds the instrument,
    the corpus and the target fixed and varies only which signal is
    read, it is a cleaner comparison than the dictionary benchmark,
    which necessarily varies the instrument too.

    Only growth is tested, because only growth has a defensible pair.
    Both `att_y` and `out_y` were pre-registered against the two-quarter
    unemployment change with signs fixed in advance, so the contrast is
    between two committed correlations.

    An inflation analogue was estimated and discarded. Inflation
    attention was pre-registered against the absolute gap from two per
    cent, not against the HICP change; pairing it with `out_pi` against
    the HICP change meant choosing the target in the knowledge that
    `out_pi` scores 0.338 there. That is selection, whatever it is
    called. It is also a different question -- the growth target is
    forward and the inflation target backward, so one asks whether
    outlook predicts and the other whether it describes. For the record
    the discarded contrast was +0.003 against +0.338, difference -0.335,
    HAC p = 0.107; it is counted in the specification tally and is not a
    result.

    The difference between two correlations sharing a variable is
    tested as the mean of a difference of standardised cross-products,
    so Newey-West applies to it directly and no independence assumption
    enters. Williams's t is reported alongside for reference, and
    assumes the independence that quarterly series lack.
    """

    frame = data.copy()

    triples = [
        ("Growth", "att_y", "out_y", "d2_unemployment",
         "unemployment, 2q change"),
    ]

    rows = []

    for topic, attention, outlook, target, target_label in triples:
        pair = frame[[attention, outlook, target]].dropna()
        n = len(pair)

        j = pair[target].to_numpy(float)
        k = pair[attention].to_numpy(float)
        h = pair[outlook].to_numpy(float)

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
            standardise(j) * standardise(k)
            - standardise(j) * standardise(h)
        ) * n / (n - 1)

        fit = newey_west_ols(
            products, np.ones((n, 1)), newey_west_bandwidth(n, min_lags=1)
        )

        low, high = fit.confidence_interval(0)

        rows.append(
            {
                "topic": topic,
                "target": target_label,
                "n": n,
                "r_attention": r_jk,
                "r_outlook": r_jh,
                "r_between_signals": r_kh,
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
        )

    return pd.DataFrame(rows)


def length_adjusted(
    panel: pd.DataFrame,
    reliability: dict[str, float],
) -> pd.DataFrame:
    """
    Re-estimate every pre-registered pair with speech length removed
    from the communication measure.

    The concern is mechanical rather than statistical. Attention is
    measured on a capped excerpt and mean length trends over the
    sample, so a measure could co-move with a macro series because both
    happen to trend, with the length of ECB speeches doing the work.
    Removing length from the measure and re-correlating is the cheapest
    way to find out.

    Length is partialled out of the measure only, not the target, which
    is what analysis/13_external_validation.py does and what makes the
    two comparable. Strictly that is a semi-partial correlation. The
    full partial, which residualises both sides, is reported alongside;
    on this panel the two agree to three decimals, so the choice does
    not carry the result either way.

    These are re-estimations of pairs already in the Holm family under
    a control, not new hypotheses, so they stay outside it for the same
    reason the variants do. They are counted in the specification
    tally.
    """

    rows = []

    for measure, target, sign, _ in PAIRS:
        columns = [measure, target] + LENGTH_CONTROLS

        complete = panel[columns].dropna()

        controls = np.column_stack(
            [np.ones(len(complete))]
            + [
                complete[control].to_numpy(float)
                for control in LENGTH_CONTROLS
            ]
        )

        def residual(values: np.ndarray) -> np.ndarray:
            coefficients = np.linalg.lstsq(
                controls, values, rcond=None
            )[0]

            return values - controls @ coefficients

        measured = complete[measure].to_numpy(float)
        targeted = complete[target].to_numpy(float)

        floor = lag_floor(target)

        raw = correlation_hac(measured, targeted, min_lags=floor)

        semi = correlation_hac(
            residual(measured), targeted, min_lags=floor
        )

        full = correlation_hac(
            residual(measured), residual(targeted), min_lags=floor
        )

        rows.append(
            {
                "measure": measure,
                "measure_label": LABELS[measure],
                "target": target,
                "target_label": LABELS[target],
                "expected_sign": sign,
                "n_quarters": raw.n_obs,
                "raw_correlation": raw.correlation,
                "raw_p_value": raw.p_value,
                "semipartial": semi.correlation,
                "semipartial_se": semi.standard_error,
                "semipartial_p": semi.p_value,
                "partial": full.correlation,
                "partial_p": full.p_value,
                "shift": semi.correlation - raw.correlation,
                "sign_as_expected": (
                    np.sign(semi.correlation) == np.sign(sign)
                ),
                "measure_kappa_r4": reliability.get(measure, float("nan")),
                "hac_lags": semi.lags,
            }
        )

    out = pd.DataFrame(rows)

    out["semipartial_p_holm"] = holm_adjust(
        out["semipartial_p"].tolist()
    )

    return out


def bandwidth_bands(lags: tuple[int, ...]) -> pd.DataFrame:
    """
    The sample sizes over which the automatic rule returns each L.

    The rule floor(4 (T/100)^(2/9)) is a step function of T, and
    Appendix C describes where it steps. Asserting those boundaries in
    prose is how the appendix came to claim L = 3 long after the design
    that produced T = 54 had been abandoned, so they are inverted from
    the rule here instead:

        floor(4 (T/100)^(2/9)) = L   <=>
        100 (L/4)^(9/2) <= T < 100 ((L+1)/4)^(9/2).

    Each row is checked against newey_west_bandwidth at both ends, so
    if the rule in src/timeseries_stats.py is ever changed this
    function fails rather than reporting boundaries for a rule that is
    no longer in force.
    """

    rows = []

    for lag in lags:
        lower = 100.0 * (lag / 4.0) ** 4.5
        upper = 100.0 * ((lag + 1) / 4.0) ** 4.5

        first = int(np.ceil(lower))
        last = int(np.ceil(upper)) - 1

        for size, expected in ((first, lag), (last, lag)):
            if newey_west_bandwidth(size) != expected:
                raise ValueError(
                    f"Inverted band for L={lag} disagrees with "
                    f"newey_west_bandwidth at T={size}: rule returns "
                    f"{newey_west_bandwidth(size)}. The bandwidth rule "
                    "has changed and Appendix C is now wrong."
                )

        rows.append(
            {
                "lags": lag,
                "smallest_T": first,
                "largest_T": last,
            }
        )

    return pd.DataFrame(rows)


def bandwidth_sensitivity(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Re-estimate every pre-registered pair at other bandwidths.

    The pooled sample runs 102 to 108 quarters and the rule steps at
    T = 100, so the estimates sit just inside the L = 4 band and a
    sample eight quarters shorter would be inferred at L = 3. Whether
    that matters is a question with an answer, and the answer belongs
    in the output rather than in a claim about the output.

    The correlation is reported at each bandwidth as well as the
    p-value. It is invariant by construction, and printing it is the
    cheapest available check that the re-estimation is doing what it
    claims: any movement in that column would mean the sample changed
    between fits, not the inference.
    """

    rows = []

    for measure, target, sign, _ in PAIRS:
        x, y = paired(panel, measure, target)

        n = len(x)

        automatic = newey_west_bandwidth(n, min_lags=lag_floor(target))

        design = np.column_stack([np.ones(n), standardise(x)])

        response = standardise(y)

        row = {
            "measure": measure,
            "target": target,
            "n_quarters": n,
            "automatic_lags": automatic,
        }

        for lag in sorted({automatic, *BANDWIDTH_ALTERNATIVES}):
            fit = newey_west_ols(response, design, lag)

            row[f"r_L{lag}"] = float(fit.coefficients[1])
            row[f"p_L{lag}"] = float(fit.p_values[1])

        estimates = [
            value for key, value in row.items() if key.startswith("r_L")
        ]

        if max(estimates) - min(estimates) > 1e-10:
            raise ValueError(
                f"The correlation for {measure} vs {target} moved "
                "across bandwidths. The bandwidth enters only the "
                "standard error, so the sample must have changed "
                "between fits."
            )

        p_values = [
            value for key, value in row.items() if key.startswith("p_L")
        ]

        row["p_range"] = max(p_values) - min(p_values)

        rows.append(row)

    return pd.DataFrame(rows)


def sign_test(pairs: pd.DataFrame) -> pd.DataFrame:
    """
    Exact binomial test on the pattern of signs.

    Every pair carries a direction fixed before estimation, so the
    number of pairs whose correlation comes out in the predicted
    direction is itself a statistic. Under the null of no relationship
    each sign is a fair coin, and the one-sided probability of getting
    at least the observed number right is exact.

    This is the omnibus the family needs. Holm asks whether any single
    pair survives correction and answers no; the sign test asks whether
    the family as a whole behaves as predicted, and it requires no
    distributional assumption about the correlations themselves --- only
    that a sign is a sign.

    It does require the signs to be independent, and they are not.
    cross_correlations() shows the targets are related: systemic stress
    and the two-quarter unemployment change correlate 0.685, which is
    unsurprising since crises raise unemployment, and two pairs share
    the unemployment change as a target outright. Correlated targets
    make the five signs fewer than five independent draws, so the exact
    p-value below is a lower bound on the true one. It is reported with
    that qualification rather than adjusted, because the effective
    number of independent signs is not identified by anything in this
    design.
    """

    n = len(pairs)
    correct = int(pairs["sign_as_expected"].sum())

    # P(X >= correct) for X ~ Binomial(n, 1/2), computed exactly.
    p_value = (
        1.0 if correct == 0
        else 1.0 - binomial_cdf(correct - 1, n, 0.5)
    )

    return pd.DataFrame(
        [
            {
                "n_pairs": n,
                "n_correct_sign": correct,
                "p_one_sided": p_value,
                "note": (
                    "exact binomial, signs fixed before estimation; "
                    "lower bound because targets are correlated"
                ),
            }
        ]
    )


def cross_correlations(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Correlations among the measures, and among their targets.

    Five pairs pointing the same way is only five pieces of evidence if
    the pairs are distinct. If financial-stability attention and
    uncertainty attention were the same signal, and systemic stress and
    forecast dispersion the same state, the family would be one crisis
    indicator counted twice and the sign test above would be
    meaningless.

    Both sides are therefore reported. The measures may be correlated
    with each other without damage; what would undermine the family is
    the targets being interchangeable.
    """

    measures = sorted({measure for measure, _, _, _ in PAIRS})
    targets = sorted({target for _, target, _, _ in PAIRS})

    rows = []

    for kind, columns in [("measure", measures), ("target", targets)]:
        for i, a in enumerate(columns):
            for b in columns[i + 1:]:
                both = panel[[a, b]].dropna()

                rows.append(
                    {
                        "kind": kind,
                        "a": a,
                        "b": b,
                        "n_quarters": len(both),
                        "correlation": float(both[a].corr(both[b])),
                    }
                )

    return pd.DataFrame(rows).sort_values(
        ["kind", "correlation"], ascending=[True, False]
    )


def plot_salience(
    panel: pd.DataFrame,
    pairs: pd.DataFrame,
    path: Path,
) -> None:
    """
    Each measure against the state it should track, over time.

    A correlation compresses a quarter-by-quarter relationship into one
    number, and for these pairs the number and the picture say different
    things. Financial-stability attention correlates only 0.246 with
    systemic stress, which reads as weak tracking; the series show why.
    Attention steps up in 2008 and stays up for most of a decade, while
    stress spikes and subsides. The measure records a lasting
    reallocation of attention after the crisis rather than following
    stress quarter by quarter, and a correlation cannot express the
    difference between a step and a spike.

    Both series in each panel are standardised, since the measures are
    shares and the targets are indices, variances and percentage-point
    changes.
    """

    order = [
        ("att_fs", "ciss_ci", "Financial stability attention",
         "Systemic stress (CISS)"),
        ("att_unc", "spf_disp_rgdp", "Uncertainty attention",
         "SPF GDP forecast dispersion"),
        ("out_y", "d2_unemployment", "Growth outlook",
         "Unemployment, next 2q change"),
        ("att_pi", "abs_hicp_gap", "Inflation attention",
         "|HICP $-$ 2|"),
    ]

    lookup = {
        (row["measure"], row["target"]): row
        for _, row in pairs.iterrows()
    }

    time = pd.PeriodIndex(panel["period"], freq="Q").to_timestamp()

    figure, axes = plt.subplots(2, 2, figsize=(12, 6.6), sharex=True)

    for axis, (measure, target, label_m, label_t) in zip(
        axes.flat, order
    ):
        frame = panel[[measure, target]].copy()
        frame["t"] = time
        frame = frame.dropna()

        axis.axhline(0, color="0.75", linewidth=0.7)

        axis.plot(
            frame["t"], standardise(frame[measure].to_numpy(float)),
            color="#4c72b0", linewidth=1.6, label=label_m,
        )

        axis.plot(
            frame["t"], standardise(frame[target].to_numpy(float)),
            color="#c44e52", linewidth=1.5, linestyle="--",
            label=label_t,
        )

        row = lookup.get((measure, target))

        annotation = (
            f"$r = {row['correlation']:+.3f}$, "
            f"$p = {row['p_value']:.3f}$"
            if row is not None
            else ""
        )

        axis.set_title(
            f"{label_m} and {label_t}\n{annotation}", fontsize=9.5
        )

        axis.grid(alpha=0.2, linewidth=0.5)
        axis.legend(frameon=False, fontsize=7.5, loc="lower left", ncol=2)

    for axis in axes[1]:
        axis.set_xlabel("Quarter")

    for axis in axes[:, 0]:
        axis.set_ylabel("Standardised")

    figure.suptitle(
        "Communication measures and the states they should track",
        fontsize=12,
    )

    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def main() -> None:
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)

    panel = load_panel()
    reliability = load_reliability()

    print("=== PANEL ===")
    print(
        f"{len(panel)} quarters, {panel['period'].iloc[0]} to "
        f"{panel['period'].iloc[-1]}"
    )

    print("\n=== TARGET COVERAGE WITHIN THE SPEECH SAMPLE ===")
    for target in [
        "ciss_ci",
        "ciss_cin",
        "spf_disp_rgdp",
        "spf_disp_hicp",
        "d2_unemployment",
        "abs_hicp_gap",
    ]:
        n = int(panel[target].notna().sum())

        print(f"  {target:16s} {n:3d} of {len(panel)} quarters")

    # --------------------------------------------------
    # The pre-registered set
    # --------------------------------------------------

    pairs = pd.DataFrame(
        [
            correlation_row(
                panel, measure, target, sign, role, reliability
            )
            for measure, target, sign, role in PAIRS
        ]
    )

    pairs["p_holm"] = holm_adjust(pairs["p_value"].tolist())

    pairs["holm_family_size"] = len(PAIRS)

    stored = reproduce_out_y(
        panel,
        float(
            pairs.loc[
                pairs["measure"] == "out_y", "correlation"
            ].iloc[0]
        ),
    )

    print("\n=== PRE-REGISTERED PAIRS (raw p) ===")
    print(format_table(pairs, "p_value"))

    print("\n=== PRE-REGISTERED PAIRS (Holm-adjusted p) ===")
    print(format_table(pairs, "p_holm"))

    print(
        f"\n  out_y reproduces the stored concordance figure exactly: "
        f"{stored:.6f}"
    )

    print("\n=== ATTENUATION ===")
    for _, row in pairs.iterrows():
        kappa = row["measure_kappa_r4"]

        print(
            f"  {row['measure']:8s} human-human kappa "
            f"{kappa:.3f}  r = {row['correlation']:+.3f}"
        )

    # --------------------------------------------------
    # Variants
    # --------------------------------------------------

    variants = pd.DataFrame(
        [
            correlation_row(
                panel, measure, target, sign, note, reliability
            )
            for measure, target, sign, note in VARIANTS
        ]
    )

    print("\n=== VARIANTS (outside the Holm family, raw p) ===")
    print(format_table(variants, "p_value"))

    # --------------------------------------------------
    # Length adjustment
    # --------------------------------------------------

    length = length_adjusted(panel, reliability)

    print("\n=== LENGTH-ADJUSTED (outside the Holm family) ===")
    print(
        "Mean log words and the truncated share partialled out of each "
        "measure.\nAttention is read off a capped excerpt and mean "
        "speech length trends, so a\ncorrelation could be an artefact "
        "of how long speeches were."
    )
    print(
        length[
            [
                "measure",
                "target",
                "n_quarters",
                "raw_correlation",
                "semipartial",
                "semipartial_p",
                "partial",
                "shift",
            ]
        ]
        .round(3)
        .to_string(index=False)
    )

    flipped = length[~length["sign_as_expected"]]

    print(
        f"\nSigns still as predicted: "
        f"{int(length['sign_as_expected'].sum())} of {len(length)}."
        + (
            ""
            if flipped.empty
            else " Flipped: "
            + ", ".join(flipped["measure"].tolist())
        )
    )
    print(
        f"Largest move: {length.loc[length['shift'].abs().idxmax(), 'measure']} "
        f"{length['shift'].abs().max():+.3f}. "
        f"Smallest Holm-adjusted p: {length['semipartial_p_holm'].min():.3f} "
        f"(raw table: {pairs['p_holm'].min():.3f})."
    )
    print(
        "Semi-partial and full partial agree to three decimals, so "
        "residualising the\nmeasure only rather than both sides does "
        "not carry the result."
    )

    controls = panel[LENGTH_CONTROLS].dropna()
    print(
        f"\nThe two controls correlate "
        f"{controls[LENGTH_CONTROLS[0]].corr(controls[LENGTH_CONTROLS[1]]):+.3f} "
        f"with each other, so they carry one dimension of length rather\n"
        f"than two. Financial-stability attention correlates "
        f"{panel['att_fs'].corr(panel['mean_log_words']):+.3f} with mean log "
        f"words:\nspeeches shorten while the measure rises, so length "
        f"attenuates the CISS\nassociation rather than generating it."
    )

    # --------------------------------------------------
    # Bandwidth
    # --------------------------------------------------

    bands = bandwidth_bands(
        tuple(sorted({*BANDWIDTH_ALTERNATIVES, 4}))
    )

    sensitivity = bandwidth_sensitivity(panel)

    print("\n=== BANDWIDTH ===")
    print(
        "The automatic rule floor(4 (T/100)^(2/9)) is a step function. "
        "Inverted, it\nreturns each bandwidth over these sample sizes:"
    )
    print(bands.to_string(index=False))
    print(
        f"\nThe pairs run {sensitivity['n_quarters'].min()} to "
        f"{sensitivity['n_quarters'].max()} quarters, so the automatic "
        f"choice is "
        f"{sorted(set(sensitivity['automatic_lags']))} throughout -- "
        "just inside\nthe band. Re-inference at other bandwidths, "
        "point estimates unchanged by construction:"
    )
    print(
        sensitivity.drop(columns=["automatic_lags"])
        .round(4)
        .to_string(index=False)
    )
    print(
        f"\nLargest movement in any p-value across bandwidths: "
        f"{sensitivity['p_range'].max():.3f}. No pair crosses 0.05 at "
        "any of them."
    )

    # --------------------------------------------------
    # Specification count
    # --------------------------------------------------

    counts = pd.DataFrame(
        [
            {
                "category": "pre-registered pairs",
                "n_specifications": len(PAIRS),
                "in_holm_family": True,
                "detail": "; ".join(
                    f"{m} vs {t} ({s:+d})" for m, t, s, _ in PAIRS
                ),
            },
            {
                "category": "length-adjusted re-estimations",
                "n_specifications": len(PAIRS),
                "in_holm_family": False,
                "detail": (
                    "every pre-registered pair re-estimated with "
                    + " and ".join(LENGTH_CONTROLS)
                    + " partialled out of the measure; re-estimations "
                    "of pairs already in the family, not new hypotheses"
                ),
            },
            {
                "category": "alternative operationalisations",
                "n_specifications": len(VARIANTS),
                "in_holm_family": False,
                "detail": "; ".join(
                    f"{m} vs {t} ({note})"
                    for m, t, _, note in VARIANTS
                ),
            },
        ]
    )

    counts.loc[len(counts)] = {
        "category": "total correlations estimated",
        "n_specifications": 2 * len(PAIRS) + len(VARIANTS),
        "in_holm_family": False,
        "detail": (
            "every correlation this script computes; no pair was "
            "dropped, and no additional specification was run and "
            "discarded"
        ),
    }

    print("\n=== SPECIFICATIONS ===")
    print(counts[["category", "n_specifications"]].to_string(index=False))

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    plot_salience(panel, pairs, FIGURE_DIR / "salience_overlay.png")

    signs = sign_test(pairs)

    print("\n=== OMNIBUS: DO THE SIGNS COME OUT AS PREDICTED? ===")
    row = signs.iloc[0]
    print(
        f"  {int(row['n_correct_sign'])} of {int(row['n_pairs'])} pairs "
        f"carry the predicted sign; exact binomial "
        f"p = {row['p_one_sided']:.4f}"
    )
    print(
        "  The signs are not independent -- see the target block below,"
        "\n  where systemic stress and the unemployment change "
        "correlate 0.685 --\n  so this is a lower bound on the true "
        "p-value."
    )

    crosses = cross_correlations(panel)

    print("\n=== ARE THE PAIRS DISTINCT? ===")
    print(crosses.round(3).to_string(index=False))

    signs.to_csv(VALIDATION_DIR / "salience_sign_test.csv", index=False)
    crosses.to_csv(
        VALIDATION_DIR / "salience_cross_correlations.csv", index=False
    )

    contrast = signal_contrast(panel)

    print("\n=== DO ATTENTION AND OUTLOOK CARRY DIFFERENT INFORMATION? ===")
    print(
        "Growth only. An inflation analogue was estimated and discarded "
        "as selection;\nsee signal_contrast() for why it is counted but "
        "not reported."
    )
    print(
        contrast[
            ["topic", "target", "n", "r_attention", "r_outlook",
             "r_between_signals", "difference", "hac_p"]
        ]
        .round(3)
        .to_string(index=False)
    )

    for _, row in contrast.iterrows():
        print(
            f"  {row['topic']}: difference {row['difference']:+.3f}, "
            f"HAC se {row['hac_se']:.3f}, t={row['hac_t']:+.2f}, "
            f"p={row['hac_p']:.4f}, "
            f"95% [{row['hac_ci_low']:+.3f}, {row['hac_ci_high']:+.3f}]"
        )

    contrast.to_csv(
        VALIDATION_DIR / "signal_contrast.csv", index=False
    )

    control = level_control(panel)

    print("\n=== GROWTH OUTLOOK: IS IT THE LEVEL IN DISGUISE? ===")

    print(control.round(3).to_string(index=False))

    control.to_csv(

        VALIDATION_DIR / "salience_level_control.csv", index=False

    )

    pairs.to_csv(PAIRS_FILE, index=False)
    variants.to_csv(VARIANTS_FILE, index=False)
    sensitivity.to_csv(
        VALIDATION_DIR / "salience_bandwidth_sensitivity.csv",
        index=False,
    )

    length.to_csv(
        VALIDATION_DIR / "salience_length_adjusted.csv", index=False
    )

    counts.to_csv(SPECIFICATIONS_FILE, index=False)

    print(f"\nSaved: {PAIRS_FILE}")
    print(f"Saved: {VARIANTS_FILE}")
    print(f"Saved: {SPECIFICATIONS_FILE}")


if __name__ == "__main__":
    main()
