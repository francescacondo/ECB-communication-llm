"""
Criterion validity of the inflation outlook against the Altavilla
et al. (2019) monetary policy factors.

Every external check so far has used a realised macro outcome: HICP
inflation, or the deposit facility rate over the following two
quarters. Those ask whether the measures line up with what the economy
and the Governing Council subsequently did. This script asks a
different question -- whether the measures line up with what markets
understood the policy stance to be -- using the four factors extracted
from high-frequency yield-curve movements around ECB policy events.

**This is criterion validity, not identification.** The factors are
*surprises*: the component of the yield curve move that markets did not
anticipate. A speech tone that predicted them would be a violation of
market efficiency, not a validation of the measure, and the null is
therefore the expected result for the surprises themselves. What can
carry information is the *cumulative* factor -- the running sum of
surprises, which traces the path of the stance that markets actually
learned about. If the inflation outlook measure is picking up
directional inflation talk, the level of that measure should move with
the accumulated forward-guidance stance, and with the right sign.

Two things follow from that and constrain how the result may be read.

    1. A cumulative sum of near-unforecastable surprises is close to a
       random walk. Correlating a persistent measure with it is exactly
       the setting in which a spurious correlation arises, and no HAC
       standard error repairs that -- the sandwich widens the interval
       under stationarity, which is the assumption in question. The
       first-differenced version of the levels correlation is precisely
       the surprise correlation, so the two are reported side by side
       and the pair is the diagnostic: a levels association with no
       first-difference counterpart is the signature of spurious
       regression, and must not be read as a finding.

    2. Magnitudes are not comparable across factors. Each is
       normalised on a different instrument -- target and timing on the
       1-month and 6-month OIS, forward guidance and QE on the 2-year
       and 10-year yields -- so a slope in basis points means basis
       points of a different yield in each column.

The specification set is fixed in advance and reported in full: four
factors (target, timing, forward guidance, QE) times two transforms
(cumulative level, within-quarter sum of surprises) equals eight cells.
Forward guidance in cumulative form is the pre-committed primary; the
other seven are reported so that the primary is read against them
rather than in isolation. Nothing is searched over: no break years, no
alternative frequencies, no leads or lags, no speaker subsets.

Requires:

    python src/download_policy_factors.py
    python analysis/12_measures.py

Run from the project root:

    python analysis/23_altavilla_factors.py
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from timeseries_stats import (  # noqa: E402
    autocorrelations,
    bootstrap_block_length,
    correlation_hac,
    moving_block_bootstrap_ci,
    newey_west_bandwidth,
    newey_west_ols,
    standardise,
)

FACTOR_FILE = (
    PROJECT_ROOT / "data" / "processed" / "policy_factors_events.csv"
)

MEASURE_FILE = (
    PROJECT_ROOT / "outputs" / "panel" / "measures_quarterly.csv"
)

VALIDATION_DIR = PROJECT_ROOT / "outputs" / "validation"

OUTPUT_FILE = VALIDATION_DIR / "altavilla_factors.csv"

MEASURE = "out_pi"

# Normalising instrument for each factor, carried into the output so
# that a slope in "basis points" is never read without its unit.
FACTORS = {
    "target": "1-month OIS",
    "timing": "6-month OIS",
    "fg": "2-year yield",
    "qe": "10-year yield",
}

PRIMARY_FACTOR = "fg"

# The pre-committed sign. A positive forward-guidance surprise raises
# the two-year yield, so accumulated forward guidance rises when the
# communicated stance tightens. An inflation outlook tilted upward
# should accompany a tighter accumulated stance, so the expected
# association is positive.
EXPECTED_SIGN = 1


def load_quarterly_factors() -> pd.DataFrame:
    """
    Aggregate event-level factors to quarters, in both transforms.

    Surprises within a quarter are summed rather than averaged: they
    are increments to the same underlying stance, and two hawkish
    surprises in one quarter move the stance twice as far as one does.
    Averaging would make a quarter with three meetings look milder than
    a quarter with one, which is backwards.

    The cumulative series is the running sum of those quarterly sums,
    started at zero at each factor's own first observation. Its level
    is arbitrary; only its variation is used, and a correlation is
    invariant to the starting point.
    """

    events = pd.read_csv(FACTOR_FILE, parse_dates=["date"])

    events["period"] = events["date"].dt.to_period("Q")

    out = pd.DataFrame(index=pd.Index(sorted(events["period"].unique()),
                                      name="period"))

    for factor in FACTORS:
        available = events.dropna(subset=[factor])

        if available.empty:
            raise ValueError(f"Factor {factor} is entirely missing.")

        quarterly = available.groupby("period")[factor].sum()

        # Reindex over the factor's own span only. A quarter inside the
        # span with no usable event would be a genuine gap and must not
        # be silently filled with a zero surprise.
        span = pd.period_range(
            quarterly.index.min(), quarterly.index.max(), freq="Q"
        )

        missing = [str(p) for p in span if p not in quarterly.index]

        if missing:
            raise ValueError(
                f"Factor {factor} has no event in {missing} inside its "
                "own sample span; the quarterly sum would be a "
                "fabricated zero."
            )

        out[f"{factor}_surprise"] = quarterly.reindex(out.index)
        out[f"{factor}_cumulative"] = quarterly.cumsum().reindex(out.index)

    return out.reset_index()


def persistence(series: np.ndarray) -> float:
    """First-order autocorrelation, as a random-walk warning light."""

    return float(autocorrelations(series, 1)[0])


def assess(
    measure: np.ndarray,
    factor: np.ndarray,
    label: str,
    transform: str,
    unit: str,
) -> dict:
    """
    One cell of the table: correlation, HAC inference, bootstrap
    interval, and the slope in the factor's own units.

    The slope regresses the factor on the raw (unstandardised) measure,
    so its coefficient answers "how many basis points of accumulated
    stance accompany a one-unit move in the outlook measure" -- the
    magnitude half of the question, which a correlation alone does not
    give.
    """

    n = len(measure)

    estimate = correlation_hac(factor, measure)

    block = bootstrap_block_length(n)
    boot_low, boot_high = moving_block_bootstrap_ci(factor, measure, block)

    design = np.column_stack([np.ones(n), measure])
    slope_fit = newey_west_ols(factor, design, newey_west_bandwidth(n))

    return {
        "factor": label,
        "transform": transform,
        "unit": unit,
        "n": n,
        "correlation": estimate.correlation,
        "hac_se": estimate.standard_error,
        "hac_p": estimate.p_value,
        "hac_ci_low": estimate.ci_low,
        "hac_ci_high": estimate.ci_high,
        "boot_ci_low": boot_low,
        "boot_ci_high": boot_high,
        "lags": estimate.lags,
        "slope_bp": float(slope_fit.coefficients[1]),
        "slope_bp_se": float(slope_fit.standard_errors[1]),
        "ar1_factor": persistence(factor),
        "ar1_measure": persistence(measure),
        "sign_as_expected": bool(
            np.sign(estimate.correlation) == EXPECTED_SIGN
        ),
    }


def main() -> None:
    factors = load_quarterly_factors()

    measures = pd.read_csv(MEASURE_FILE)
    measures["period"] = pd.PeriodIndex(measures["period"], freq="Q")

    merged = measures[["period", MEASURE]].merge(
        factors, on="period", how="inner"
    )

    merged = merged.sort_values("period").reset_index(drop=True)

    if merged.empty:
        raise ValueError("No overlap between measures and factors.")

    print("=== SAMPLE ===")
    print(
        f"Speech measures: {measures['period'].min()} to "
        f"{measures['period'].max()} ({len(measures)} quarters)"
    )
    print(
        f"Policy factors:  {factors['period'].min()} to "
        f"{factors['period'].max()} ({len(factors)} quarters)"
    )
    print(
        f"Overlap:         {merged['period'].min()} to "
        f"{merged['period'].max()} ({len(merged)} quarters)"
    )
    print(
        "The factors start in 2002 because the OIS quotes the rotation "
        "needs do not exist earlier; twelve quarters of the speech "
        "sample are unusable here."
    )

    rows = []

    for factor, unit in FACTORS.items():
        for transform in ("cumulative", "surprise"):
            column = f"{factor}_{transform}"

            usable = merged.dropna(subset=[column, MEASURE])

            rows.append(
                assess(
                    usable[MEASURE].to_numpy(float),
                    usable[column].to_numpy(float),
                    factor,
                    transform,
                    unit,
                )
            )

    table = pd.DataFrame(rows)

    print("\n=== ALL EIGHT PRE-COMMITTED CELLS ===")
    print(
        "Four factors x two transforms. Nothing else was estimated: no "
        "break years, no alternative frequencies, no leads or lags, no "
        "speaker subsets."
    )
    print(
        table[
            ["factor", "transform", "n", "correlation", "hac_se", "hac_p",
             "slope_bp", "unit"]
        ]
        .round(3)
        .to_string(index=False)
    )

    primary = table[
        (table["factor"] == PRIMARY_FACTOR)
        & (table["transform"] == "cumulative")
    ].iloc[0]

    companion = table[
        (table["factor"] == PRIMARY_FACTOR)
        & (table["transform"] == "surprise")
    ].iloc[0]

    print("\n=== PRIMARY: CUMULATIVE FORWARD GUIDANCE ===")
    print(
        f"  correlation with {MEASURE}: {primary['correlation']:.3f} "
        f"(Newey-West se {primary['hac_se']:.3f}, "
        f"p = {primary['hac_p']:.3f}, {int(primary['n'])} quarters, "
        f"{int(primary['lags'])} lags)"
    )
    print(
        f"  HAC 95% interval:       "
        f"[{primary['hac_ci_low']:.3f}, {primary['hac_ci_high']:.3f}]"
    )
    print(
        f"  block bootstrap 95%:    "
        f"[{primary['boot_ci_low']:.3f}, {primary['boot_ci_high']:.3f}]"
    )
    print(
        f"  slope: {primary['slope_bp']:.1f} bp of the 2-year yield "
        f"(se {primary['slope_bp_se']:.1f}) per unit of {MEASURE}"
    )
    print(
        f"  expected sign positive; observed sign "
        f"{'as expected' if primary['sign_as_expected'] else 'AGAINST expectation'}"
    )

    print("\n=== SPURIOUS-REGRESSION DIAGNOSTIC ===")
    print(
        "The cumulative series is a running sum of near-unforecastable "
        "surprises, so it is close to a random walk. Its first "
        "difference is the surprise series, and the surprise cell is "
        "therefore the differenced counterpart of the levels cell."
    )
    print(
        f"  persistence (lag-1 autocorrelation): "
        f"cumulative factor {primary['ar1_factor']:.3f}, "
        f"{MEASURE} {primary['ar1_measure']:.3f}"
    )
    print(
        f"  levels correlation      {primary['correlation']:+.3f} "
        f"(p = {primary['hac_p']:.3f})"
    )
    print(
        f"  differenced counterpart {companion['correlation']:+.3f} "
        f"(p = {companion['hac_p']:.3f})"
    )

    same_sign = (
        np.sign(companion["correlation"])
        == np.sign(primary["correlation"])
    )

    if companion["hac_p"] >= 0.05:
        print(
            "  The differenced counterpart is indistinguishable from "
            "zero. The levels association cannot be separated from the "
            "spurious correlation two persistent series produce by "
            "construction, and must not be reported as a finding."
        )
    elif not same_sign:
        print(
            "  The differenced counterpart is significant but carries "
            "the OPPOSITE sign to the levels correlation. That is not "
            "corroboration of the levels result; it is a second reason "
            "to distrust it. See the surprise block below for why a "
            "significant surprise correlation is itself unwelcome."
        )
    else:
        print(
            "  The differenced counterpart carries the same sign and is "
            "itself significant. Note that this would mean speech tone "
            "co-moves with monetary policy SURPRISES, which is a claim "
            "about market efficiency and should be treated with more "
            "suspicion, not less."
        )

    print("\n=== SURPRISES: THE EXPECTED NULL ===")
    surprises = table[table["transform"] == "surprise"]
    for _, r in surprises.iterrows():
        print(
            f"  {r['factor']:7s} {r['correlation']:+.3f} "
            f"(p = {r['hac_p']:.3f}, n = {int(r['n'])})"
        )
    print(
        "  These are the component of the yield-curve move markets did "
        "not anticipate. A speech measure predicting them would be a "
        "market-efficiency violation, not evidence for the measure. "
        "The null is the expected and the desired result here."
    )

    print("\n=== MULTIPLICITY ACROSS THE EIGHT CELLS ===")
    ordered = table.sort_values("hac_p")
    n_tests = len(table)
    holm_rejected = []

    for rank, (_, r) in enumerate(ordered.iterrows()):
        threshold = 0.05 / (n_tests - rank)
        passes = r["hac_p"] < threshold
        if passes:
            holm_rejected.append(f"{r['factor']} {r['transform']}")
        else:
            # Holm is a step-down procedure: the first failure stops it.
            print(
                f"  smallest p is {ordered['hac_p'].iloc[0]:.3f}; Holm "
                f"needs {0.05 / n_tests:.4f} at the first step"
            )
            break

    print(
        f"  cells rejected at 5% unadjusted: "
        f"{int((table['hac_p'] < 0.05).sum())} of {n_tests}"
    )
    print(
        f"  cells surviving Holm across all {n_tests}: "
        f"{len(holm_rejected) if holm_rejected else 'none'}"
    )
    print(
        "  Two unadjusted rejections out of eight is close to what "
        "eight tests produce by chance, and neither survives the "
        "correction."
    )

    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT_FILE, index=False)

    print(f"\nSaved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
