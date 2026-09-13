"""
Event-level test of the inflation outlook against the Altavilla et al.
(2019) factors.

analysis/23_altavilla_factors.py aligned the measures and the factors at
quarterly frequency. That alignment wastes the design the data actually
supports and has a defect: a quarter contains speeches delivered *after*
the policy events in it, so a contemporaneous quarterly correlation
cannot distinguish speeches anticipating a surprise from speeches
reacting to one.

The inter-meeting window fixes both. For each policy event k, the
window is every speech strictly after event k-1 and strictly before
event k. The outlook is aggregated over that window and related to
event k's factor. The window closes before the event, so nothing on the
right-hand side can be contaminated by the event itself, and the number
of observations rises from 96 quarters to roughly 230 windows.

**What a positive result would and would not mean.** The factors are
surprises: the component of the yield-curve move markets did not
anticipate. Every speech in the window is public information, available
to markets well before the event. So a window measure that predicted
the surprise would not be evidence that the measure is well
constructed -- it would be a claim that markets systematically fail to
price public ECB speech content. That is a strong claim about market
efficiency, and the burden on it is heavier than the burden on a
validation exercise, not lighter. The null remains the expected result.
What the design buys is that a null is now *informative*: it can no
longer be blamed on reverse causality.

Speeches are excluded on the event date itself. A speech delivered on a
meeting day may fall after the press conference, which would reopen
exactly the contamination the window design exists to close.

The first event has no predecessor and is dropped rather than given a
three-year window back to the start of the speech sample.

The specification set is fixed in advance and reported in full: four
factors times two transforms equals eight cells, mirroring the
quarterly table so the two are directly comparable. Window-length and
window-density diagnostics are printed but are *not* re-estimated
under alternative screens; no minimum-speech threshold is searched
over.

Requires:

    python src/download_policy_factors.py
    python analysis/10_panel.py
    python analysis/12_measures.py

Run from the project root:

    python analysis/24_event_window_factors.py
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from timeseries_stats import (  # noqa: E402
    bootstrap_block_length,
    correlation_hac,
    moving_block_bootstrap_ci,
    newey_west_bandwidth,
    newey_west_ols,
)

FACTOR_FILE = (
    PROJECT_ROOT / "data" / "processed" / "policy_factors_events.csv"
)

PANEL_FILE = PROJECT_ROOT / "outputs" / "panel" / "speech_panel.csv"

QUARTERLY_FILE = (
    PROJECT_ROOT / "outputs" / "panel" / "measures_quarterly.csv"
)

VALIDATION_DIR = PROJECT_ROOT / "outputs" / "validation"

OUTPUT_FILE = VALIDATION_DIR / "event_window_factors.csv"

WINDOW_FILE = VALIDATION_DIR / "event_window_measures.csv"

ATTENTION_VAR = "inflation_attention"
OUTLOOK_VAR = "inflation_outlook"

FACTORS = {
    "target": "1-month OIS",
    "timing": "6-month OIS",
    "fg": "2-year yield",
    "qe": "10-year yield",
}

PRIMARY_FACTOR = "fg"

# A positive forward-guidance surprise raises the two-year yield. An
# inflation outlook tilted upward during the window should, if it
# anticipates anything, precede a hawkish surprise.
EXPECTED_SIGN = 1


def weighted_outlook(attention: pd.Series, outlook: pd.Series) -> float:
    """
    Attention-weighted mean outlook, undefined when no speech in the
    window discusses inflation.

    Duplicated from analysis/12_measures.py rather than imported,
    because that file is a script and not on the import path. The
    duplication is verified against the committed quarterly series in
    check_reproduces_quarterly() rather than trusted.
    """

    total = attention.sum()

    if total == 0:
        return float("nan")

    return float((attention * outlook).sum() / total)


def check_reproduces_quarterly(panel: pd.DataFrame) -> None:
    """
    Confirm the local aggregator reproduces the committed quarterly
    out_pi exactly. If it does not, the window measure is a different
    object from the one every other result in the project uses, and no
    comparison between this script and the quarterly table would be
    meaningful.
    """

    quarterly = pd.read_csv(QUARTERLY_FILE)

    baseline = panel[panel["in_baseline"]]

    rebuilt = (
        baseline.groupby("yq")
        .apply(
            lambda g: weighted_outlook(g[ATTENTION_VAR], g[OUTLOOK_VAR]),
            include_groups=False,
        )
        .rename("rebuilt")
        .reset_index()
    )

    merged = quarterly[["period", "out_pi"]].merge(
        rebuilt, left_on="period", right_on="yq", how="inner"
    )

    if len(merged) != len(quarterly):
        raise ValueError(
            f"Rebuilt {len(merged)} quarters against {len(quarterly)} "
            "in the committed file; the grouping key differs."
        )

    gap = (merged["out_pi"] - merged["rebuilt"]).abs().max()

    if not np.isclose(gap, 0.0, atol=1e-12):
        raise ValueError(
            f"Local aggregator departs from the committed out_pi by "
            f"{gap:.2e}. The window measure would not be the same "
            "object as the quarterly one."
        )

    print(
        f"  aggregator reproduces committed out_pi across "
        f"{len(merged)} quarters (max |gap| {gap:.1e})"
    )


def build_windows(panel: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """
    Assign each speech to the inter-meeting window that ends at the
    next policy event, and aggregate.

    searchsorted with side="right" maps a speech to the first event
    strictly after it, which is the window definition. The side matters
    and is not a detail: with side="left" a speech dated exactly on an
    event day lands in the window *ending* at that event, and a speech
    delivered after the press conference would then sit on the
    right-hand side of a regression explaining that conference's own
    surprise. side="right" pushes it into the next window instead.
    Thirty baseline speeches fall on event dates, so the choice moves
    the estimates.
    """

    event_dates = events["date"].to_numpy("datetime64[ns]")

    speech_dates = panel["date"].to_numpy("datetime64[ns]")

    position = np.searchsorted(event_dates, speech_dates, side="right")

    # Fail loudly if the side ever stops excluding same-day speeches.
    on_event = np.isin(speech_dates, event_dates)

    if on_event.any():
        landed = position[on_event]
        matched = np.searchsorted(event_dates, speech_dates[on_event])

        if (landed <= matched).any():
            raise ValueError(
                "A speech dated on an event day was assigned to the "
                "window ending at that event. The window would include "
                "post-conference speech and the design is broken."
            )

    # Speeches after the last event have nowhere to go.
    inside = position < len(event_dates)

    assigned = panel.loc[inside].copy()
    assigned["event_index"] = position[inside]

    # The first event has no predecessor: its window would reach back
    # to the start of the speech sample.
    assigned = assigned[assigned["event_index"] > 0]

    grouped = assigned.groupby("event_index")

    windows = pd.DataFrame(
        {
            "n_speeches": grouped.size(),
            "n_attending": grouped[ATTENTION_VAR].sum(),
            "out_pi": grouped.apply(
                lambda g: weighted_outlook(
                    g[ATTENTION_VAR], g[OUTLOOK_VAR]
                ),
                include_groups=False,
            ),
        }
    )

    windows = windows.reindex(range(1, len(event_dates)))

    windows["n_speeches"] = windows["n_speeches"].fillna(0).astype(int)
    windows["n_attending"] = windows["n_attending"].fillna(0).astype(int)

    windows["event_date"] = event_dates[1:]
    windows["window_start"] = event_dates[:-1]
    windows["window_days"] = (
        pd.Series(windows["event_date"]).values
        - pd.Series(windows["window_start"]).values
    ) / np.timedelta64(1, "D")

    for factor in FACTORS:
        windows[factor] = events[factor].to_numpy()[1:]
        windows[f"{factor}_cumulative"] = (
            events[factor].cumsum().to_numpy()[1:]
        )

    return windows.reset_index(names="event_index")


def assess(
    measure: np.ndarray,
    factor: np.ndarray,
    label: str,
    transform: str,
    unit: str,
) -> dict:
    """One cell: correlation, HAC inference, bootstrap, and slope."""

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
        "sign_as_expected": bool(
            np.sign(estimate.correlation) == EXPECTED_SIGN
        ),
    }


def main() -> None:
    panel = pd.read_csv(PANEL_FILE, parse_dates=["date"])
    panel = panel.sort_values("date").reset_index(drop=True)

    print("=== AGGREGATOR CHECK ===")
    check_reproduces_quarterly(panel)

    panel = panel[panel["in_baseline"]].copy()

    events = pd.read_csv(FACTOR_FILE, parse_dates=["date"])
    events = events.sort_values("date").reset_index(drop=True)

    windows = build_windows(panel, events)

    print("\n=== WINDOW CONSTRUCTION ===")
    print(
        f"  {len(events)} policy events, "
        f"{len(windows)} inter-meeting windows "
        "(the first event has no predecessor and is dropped)"
    )
    print(
        f"  window length in days: median "
        f"{windows['window_days'].median():.0f}, "
        f"range {windows['window_days'].min():.0f} to "
        f"{windows['window_days'].max():.0f}"
    )
    print(
        f"  speeches per window: median "
        f"{windows['n_speeches'].median():.0f}, "
        f"range {windows['n_speeches'].min():.0f} to "
        f"{windows['n_speeches'].max():.0f}"
    )
    print(
        f"  windows with no speech at all: "
        f"{int((windows['n_speeches'] == 0).sum())}"
    )
    print(
        f"  windows with no speech discussing inflation "
        f"(out_pi undefined): "
        f"{int(windows['out_pi'].isna().sum())}"
    )

    thin = int((windows["n_attending"].between(1, 2)).sum())
    print(
        f"  windows resting on one or two attending speeches: {thin} "
        "-- these make the measure noisy, which attenuates every "
        "correlation below toward zero. No minimum-speech screen is "
        "applied; doing so would be a specification search."
    )

    rows = []

    for factor, unit in FACTORS.items():
        for transform in ("cumulative", "surprise"):
            column = (
                f"{factor}_cumulative" if transform == "cumulative"
                else factor
            )

            usable = windows.dropna(subset=[column, "out_pi"])

            rows.append(
                assess(
                    usable["out_pi"].to_numpy(float),
                    usable[column].to_numpy(float),
                    factor,
                    transform,
                    unit,
                )
            )

    table = pd.DataFrame(rows)

    print("\n=== ALL EIGHT PRE-COMMITTED CELLS, EVENT LEVEL ===")
    print(
        "Four factors x two transforms, mirroring the quarterly table. "
        "No minimum-speech screen, no break years, no leads or lags, "
        "no speaker subsets."
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
        & (table["transform"] == "surprise")
    ].iloc[0]

    print("\n=== PRIMARY: FORWARD GUIDANCE SURPRISE ON THE PRIOR WINDOW ===")
    print(
        f"  correlation with window out_pi: "
        f"{primary['correlation']:.3f} "
        f"(Newey-West se {primary['hac_se']:.3f}, "
        f"p = {primary['hac_p']:.3f}, {int(primary['n'])} windows, "
        f"{int(primary['lags'])} lags)"
    )
    print(
        f"  HAC 95% interval:    "
        f"[{primary['hac_ci_low']:.3f}, {primary['hac_ci_high']:.3f}]"
    )
    print(
        f"  block bootstrap 95%: "
        f"[{primary['boot_ci_low']:.3f}, {primary['boot_ci_high']:.3f}]"
    )
    print(
        f"  slope: {primary['slope_bp']:.1f} bp of the 2-year yield "
        f"(se {primary['slope_bp_se']:.1f}) per unit of window out_pi"
    )

    print("\n=== MULTIPLICITY ACROSS THE EIGHT CELLS ===")
    n_tests = len(table)
    smallest = table["hac_p"].min()
    print(
        f"  cells rejected at 5% unadjusted: "
        f"{int((table['hac_p'] < 0.05).sum())} of {n_tests}"
    )
    print(
        f"  smallest p is {smallest:.3f}; Holm needs "
        f"{0.05 / n_tests:.4f} at the first step -- "
        f"{'survives' if smallest < 0.05 / n_tests else 'does not survive'}"
    )

    print("\n=== WHAT THE DIRECTIONAL DESIGN SETTLES ===")
    print(
        "  The window closes before the event, so no speech on the "
        "right-hand side can be a reaction to the factor on the left. "
        "Reverse causality is closed by construction here, unlike in "
        "the quarterly alignment."
    )
    print(
        "  A null therefore means the measure carries no information "
        "about the coming surprise -- which is what an efficient market "
        "implies, since every speech in the window was public. A "
        "rejection would be a claim about market efficiency and would "
        "need to clear a far higher bar than this exercise sets."
    )

    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT_FILE, index=False)
    windows.to_csv(WINDOW_FILE, index=False)

    print(f"\nSaved to: {OUTPUT_FILE}")
    print(f"          {WINDOW_FILE}")


if __name__ == "__main__":
    main()
