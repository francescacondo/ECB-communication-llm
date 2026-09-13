"""
Download the contemporaneous-salience counterparts from the ECB Data
Portal.

`src/download_macro.py` pulls the series the outlook measures are
supposed to *anticipate* — realised inflation, the policy rate, the
unemployment rate. This script pulls the series the *attention*
measures must track if they measure attention at all. A speech corpus
that talks about financial stress when there is financial stress, and
about uncertainty when forecasters disagree, is doing the minimum a
valid salience measure has to do. Nothing here is a prediction
exercise; every pair is contemporaneous by construction.

Two families are downloaded.

**CISS.** The Composite Indicator of Systemic Stress, in both published
variants. Their metadata both declare daily frequency, but only one is:

    SS_CI    the original CISS. Published *weekly*, on Fridays,
             1,374 observations for 1999-2025, and it stops on
             2 May 2025.
    SS_CIN   the New CISS. Genuinely daily, roughly 261 observations a
             year, complete through December 2025.

That discrepancy is recorded rather than smoothed over: the quarterly
file carries an observation count for each, so a quarter built from
thirteen weekly readings is never mistaken for one built from sixty-five
daily ones. SS_CI is kept as the primary because it is the series the
pre-registered pair named first; SS_CIN is its alternate.

**SPF forecast dispersion.** The Data Portal carries no euro-area
implied-volatility index — there is no VSTOXX series on it, and a
search of the portal for "volatility" returns only CISS subindices.
The available uncertainty proxy is therefore survey-based: the
cross-forecaster variance of Survey of Professional Forecasters point
forecasts, published as the VAR source code in the SPF dataflow.

Real GDP growth is the primary proxy and HICP inflation the alternate.
That order is set by data quality alone, before any correlation was
estimated: the real GDP series is evenly spaced at exactly one quarter
throughout, while the HICP series carries an extra February 2020 round
that puts two observations in 2020Q1.

The horizon codes date the *target* period, not the survey round, and
the two differ by a fixed number of quarters that the code below states
and then asserts. The alignment is checked against a fact outside the
data: the largest real GDP disagreement in the sample must fall in the
April 2020 round, which is the one conducted as Europe shut down.

Run from the project root:

    python src/download_salience_series.py
"""

from pathlib import Path
import io

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "salience"

OUTPUT_FILE = (
    PROJECT_ROOT / "data" / "processed" / "salience_quarterly.csv"
)

API_BASE = "https://data-api.ecb.europa.eu/service/data"

START_PERIOD = "1998-01-01"
END_PERIOD = "2025-12-31"

USER_AGENT = "ecb-communication-llm research project"

REQUEST_TIMEOUT = 90


# ============================================================
# Series definitions
# ============================================================

# The Data Portal blocks wildcard queries, so every series is named in
# full and fetched on its own request.

CISS_SERIES = {
    "ciss_ci": ("CISS", "D.U2.Z0Z.4F.EC.SS_CI.IDX"),
    "ciss_cin": ("CISS", "D.U2.Z0Z.4F.EC.SS_CIN.IDX"),
}

# name -> (flow, key, quarters to subtract from the target period to
#          recover the survey round)
#
# P9M  target ends 9 months after the round opens  -> 2 quarters back
# P12M target ends 12 months after the round opens -> 3 quarters back
SPF_SERIES = {
    "spf_disp_rgdp": ("SPF", "Q.U2.RGDP.POINT.P9M.Q.VAR", 2),
    "spf_disp_hicp": ("SPF", "M.U2.HICP.POINT.P12M.Q.VAR", 3),
}

# The first SPF round was conducted in January 1999. Recovering the
# round from the target period must reproduce that.
SPF_FIRST_ROUND = pd.Period("1999Q1", freq="Q")

# The April 2020 round is the external check on the alignment: it is
# the round in which forecasters were asked about growth during the
# first European lockdowns, and it must carry the sample's largest
# real GDP disagreement by a wide margin.
SPF_ALIGNMENT_CHECK_ROUND = pd.Period("2020Q2", freq="Q")

# CISS is a normalised index on the unit interval by construction. A
# value outside it means the key resolved to something else.
CISS_BOUNDS = (0.0, 1.0)

# Forecast dispersion is a variance and cannot be negative. The upper
# bound is loose; it exists to catch a key that silently returned a
# forecast level instead of its variance.
DISPERSION_BOUNDS = (0.0, 25.0)


# ============================================================
# Download
# ============================================================

def fetch_series(flow: str, key: str) -> pd.DataFrame:
    """Retrieve one named series from the Data Portal."""

    response = requests.get(
        f"{API_BASE}/{flow}/{key}",
        params={
            "format": "csvdata",
            "startPeriod": START_PERIOD,
            "endPeriod": END_PERIOD,
            "detail": "dataonly",
        },
        headers={"User-Agent": USER_AGENT},
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()

    data = pd.read_csv(io.StringIO(response.text))

    missing = {"TIME_PERIOD", "OBS_VALUE"} - set(data.columns)

    if missing:
        raise ValueError(
            f"{flow}/{key} returned columns {list(data.columns)}; "
            f"expected TIME_PERIOD and OBS_VALUE. Missing: {missing}"
        )

    if data.empty:
        raise ValueError(
            f"{flow}/{key} returned no observations. Check the key "
            "against the Data Portal web interface."
        )

    return data[["TIME_PERIOD", "OBS_VALUE"]].copy()


def download_all() -> dict[str, pd.DataFrame]:
    """Fetch every configured series and cache the raw response."""

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    raw = {}

    definitions = {
        **{n: (f, k) for n, (f, k) in CISS_SERIES.items()},
        **{n: (f, k) for n, (f, k, _) in SPF_SERIES.items()},
    }

    for name, (flow, key) in definitions.items():
        print(f"Downloading {name} ({flow}/{key}) ...")

        data = fetch_series(flow, key)

        data.to_csv(RAW_DIR / f"{name}.csv", index=False)

        print(
            f"  {len(data):,} observations, "
            f"{data['TIME_PERIOD'].iloc[0]} to "
            f"{data['TIME_PERIOD'].iloc[-1]}"
        )

        raw[name] = data

    return raw


# ============================================================
# Quarterly aggregation
# ============================================================

def ciss_to_quarterly(data: pd.DataFrame, name: str) -> pd.DataFrame:
    """
    Average a CISS series within the quarter.

    The observation count is carried alongside the mean. SS_CI is
    published weekly despite its daily frequency code, so the two
    variants give quarters built from very different numbers of
    readings; the count makes that visible in the output file instead
    of leaving it to be rediscovered.
    """

    dates = pd.to_datetime(data["TIME_PERIOD"])

    series = pd.Series(
        data["OBS_VALUE"].astype(float).to_numpy(), index=dates
    ).sort_index()

    if series.index.has_duplicates:
        raise ValueError(f"{name} has duplicate observation dates.")

    low, high = CISS_BOUNDS

    outside = series[(series < low) | (series > high)]

    if not outside.empty:
        raise ValueError(
            f"{name} has {len(outside)} values outside [{low}, {high}]; "
            "CISS is normalised to the unit interval, so the key is "
            "probably not what it is labelled."
        )

    quarters = series.index.to_period("Q")

    frame = pd.DataFrame(
        {
            name: series.groupby(quarters).mean(),
            f"{name}_n_obs": series.groupby(quarters).size(),
        }
    )

    return frame.rename_axis("quarter")


def spf_to_quarterly(
    data: pd.DataFrame,
    name: str,
    quarters_back: int,
) -> pd.DataFrame:
    """
    Date an SPF dispersion series by its survey round.

    The published period is the end of the forecast target window, not
    the date the forecasters answered. For a contemporaneous test the
    round is what matters: the question is whether speeches given while
    forecasters disagreed also talked about uncertainty, not whether
    they talked about it a year before the disagreement resolved.

    The HICP series carries an extra February 2020 round, so two
    observations can land in one quarter; those are averaged. The real
    GDP series has no such case and the assertion below would catch it
    if a future vintage introduced one.
    """

    targets = pd.PeriodIndex(data["TIME_PERIOD"], freq="Q")

    rounds = targets - quarters_back

    series = pd.Series(
        data["OBS_VALUE"].astype(float).to_numpy(), index=rounds
    ).sort_index()

    low, high = DISPERSION_BOUNDS

    outside = series[(series < low) | (series > high)]

    if not outside.empty:
        raise ValueError(
            f"{name} has {len(outside)} values outside [{low}, {high}], "
            f"e.g. {outside.iloc[0]}. The key may return a forecast "
            "level rather than its variance."
        )

    if series.index.min() != SPF_FIRST_ROUND:
        raise ValueError(
            f"{name} implies a first survey round of "
            f"{series.index.min()}, but the SPF began in "
            f"{SPF_FIRST_ROUND}. The horizon offset of "
            f"{quarters_back} quarters is wrong."
        )

    n_duplicated = int(series.index.duplicated().sum())

    frame = pd.DataFrame(
        {
            # Reported as a standard deviation. The published quantity
            # is a variance, which is in squared percentage points and
            # far more skewed; the square root keeps the proxy in the
            # same units as the forecasts themselves.
            name: series.groupby(level=0).mean() ** 0.5,
            f"{name}_n_rounds": series.groupby(level=0).size(),
        }
    )

    print(
        f"  {name}: rounds {frame.index.min()} to {frame.index.max()}, "
        f"{len(frame)} quarters, {n_duplicated} duplicated round(s) "
        "averaged"
    )

    return frame.rename_axis("quarter")


def check_spf_alignment(quarterly: pd.DataFrame) -> None:
    """
    Verify the survey-round dating against the April 2020 round.

    An off-by-one in the horizon offset would shift every SPF
    observation relative to the speech measures and quietly weaken or
    invent a correlation. The COVID round is the cheapest external
    check: it must be the maximum, and by a margin no plausible
    ordinary quarter would produce.
    """

    series = quarterly["spf_disp_rgdp"].dropna()

    peak = series.idxmax()

    if peak != SPF_ALIGNMENT_CHECK_ROUND:
        raise ValueError(
            "Largest real GDP forecast dispersion falls in "
            f"{peak}, not the April 2020 round "
            f"({SPF_ALIGNMENT_CHECK_ROUND}). The horizon offset that "
            "recovers the survey round is wrong."
        )

    runner_up = series.drop(peak).max()
    typical = float(series.median())

    # The COVID cluster runs for several rounds, so the peak need not
    # stand far above the round that follows it. What it must do is
    # stand far above an ordinary quarter, and the median is the right
    # comparison for that.
    if series[peak] < 2.0 * typical:
        raise ValueError(
            f"The {SPF_ALIGNMENT_CHECK_ROUND} dispersion "
            f"({series[peak]:.3f}) is under twice the sample median "
            f"({typical:.3f}). The maximum landing on the right round "
            "is then weak evidence, and the alignment is unverified."
        )

    print(
        f"  Alignment check passed: {SPF_ALIGNMENT_CHECK_ROUND} "
        f"dispersion {series[peak]:.3f}, next largest {runner_up:.3f}, "
        f"median {typical:.3f}"
    )


def build_quarterly(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Assemble the quarterly salience panel."""

    pieces = [
        ciss_to_quarterly(raw[name], name) for name in CISS_SERIES
    ]

    pieces += [
        spf_to_quarterly(raw[name], name, back)
        for name, (_, _, back) in SPF_SERIES.items()
    ]

    panel = pd.concat(pieces, axis=1).sort_index()

    check_spf_alignment(panel)

    panel.index = panel.index.astype(str)

    return panel.rename_axis("period").reset_index()


def main() -> None:
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    raw = download_all()

    print("\n=== QUARTERLY AGGREGATION ===")

    panel = build_quarterly(raw)

    print("\n=== COVERAGE ===")
    for column in ["ciss_ci", "ciss_cin", "spf_disp_rgdp", "spf_disp_hicp"]:
        available = panel.loc[panel[column].notna(), "period"]

        print(
            f"{column:16s} {len(available):3d} quarters, "
            f"{available.iloc[0]} to {available.iloc[-1]}"
        )

    print("\n=== OBSERVATIONS PER QUARTER ===")
    print(
        panel[["ciss_ci_n_obs", "ciss_cin_n_obs"]]
        .describe()
        .loc[["mean", "min", "max"]]
        .round(1)
        .to_string()
    )

    print("\n=== SUMMARY ===")
    print(
        panel[["ciss_ci", "ciss_cin", "spf_disp_rgdp", "spf_disp_hicp"]]
        .describe()
        .loc[["mean", "std", "min", "max"]]
        .round(3)
        .to_string()
    )

    panel.to_csv(OUTPUT_FILE, index=False)

    print(f"\nSaved quarterly salience panel to: {OUTPUT_FILE}")
    print(f"Cached raw responses in: {RAW_DIR}")


if __name__ == "__main__":
    main()
