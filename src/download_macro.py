"""
Download euro-area macro series from the ECB Data Portal.

Three series are pulled, each chosen because it is the natural external
counterpart to one of the communication measures:

    HICP annual rate     realised inflation, against inflation talk;
    deposit facility rate the policy stance, against the outlook
                          measures that are supposed to anticipate it;
    unemployment rate     real activity, against growth talk.

The Data Portal's SDMX REST endpoint needs no key. Raw responses are
cached under data/raw/macro/ so that the quarterly build is
reproducible without re-hitting the API, and so that a revision to the
published series is visible as a change to a stored file rather than as
an unexplained change in a downstream table.

Quarterly aggregation differs by series and the difference matters:

    * HICP and unemployment are averaged within the quarter. They are
      flow-like monthly readings and a quarterly average is the
      conventional summary.

    * The deposit facility rate is taken at the end of the quarter. It
      is a policy decision, not a flow: what a speech in quarter t can
      anticipate is the level the rate reaches, not its average path.
      A quarterly average is also written, for robustness.

The rate series is published only on the days it changes, so it must be
forward-filled onto a daily calendar before any of that is meaningful.
Aggregating the raw file directly would silently weight quarters by how
often the Governing Council happened to move.

Run from the project root:

    python src/download_macro.py
"""

from pathlib import Path
import io

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_MACRO_DIR = PROJECT_ROOT / "data" / "raw" / "macro"

OUTPUT_FILE = (
    PROJECT_ROOT / "data" / "processed" / "macro_quarterly.csv"
)

API_BASE = "https://data-api.ecb.europa.eu/service/data"

# Start one year before the speech sample so that year-on-year and
# lagged transformations are defined from 1999Q1.
START_PERIOD = "1998-01-01"

# The speech sample ends in 2025; pulling beyond it is harmless and
# keeps the end of the panel from depending on the download date.
END_PERIOD = "2025-12-31"

USER_AGENT = "ecb-communication-llm research project"

REQUEST_TIMEOUT = 90


# ============================================================
# Series definitions
# ============================================================

# name -> (flow reference, series key, published frequency)
SERIES = {
    "hicp_yoy": ("ICP", "M.U2.N.000000.4.ANR", "M"),
    "dfr": ("FM", "B.U2.EUR.4F.KR.DFR.LEV", "B"),
    "unemployment": ("LFSI", "M.I9.S.UNEHRT.TOTAL0.15_74.T", "M"),
    # The main refinancing rate is not one series. Under fixed-rate
    # tenders the operative rate is the fixed rate; under the variable
    # rate tenders that ran from June 2000 to October 2008 it is the
    # minimum bid rate. The two are spliced in splice_mro_rate().
    "mro_fixed": ("FM", "B.U2.EUR.4F.KR.MRR_FR.LEV", "B"),
    "mro_minbid": ("FM", "B.U2.EUR.4F.KR.MRR_MBR.LEV", "B"),
}

# Plausibility bounds. These are wide enough to admit anything the euro
# area has actually done and narrow enough to catch a series key that
# silently resolves to an index level or a different unit.
BOUNDS = {
    "hicp_yoy": (-2.0, 12.0),
    "dfr": (-2.0, 10.0),
    "unemployment": (5.0, 15.0),
    "mro": (0.0, 10.0),
}

# The variable-rate tender regime, during which the minimum bid rate
# rather than a fixed rate was the operative policy rate. Both bounds
# are the dates the published series themselves change over, so the
# splice introduces no judgement of its own.
MBR_REGIME_START = "2000-06-28"
MBR_REGIME_END = "2008-10-15"


# ============================================================
# Download
# ============================================================

def fetch_series(flow: str, key: str) -> pd.DataFrame:
    """Retrieve one series from the Data Portal as tidy observations."""

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
            f"{flow}/{key} returned no observations. Check the series "
            "key against the Data Portal."
        )

    return data[["TIME_PERIOD", "OBS_VALUE"]].copy()


def download_all() -> dict[str, pd.DataFrame]:
    """Fetch every configured series and cache the raw response."""

    RAW_MACRO_DIR.mkdir(parents=True, exist_ok=True)

    raw = {}

    for name, (flow, key, _) in SERIES.items():
        print(f"Downloading {name} ({flow}/{key}) ...")

        data = fetch_series(flow, key)

        path = RAW_MACRO_DIR / f"{name}.csv"
        data.to_csv(path, index=False)

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

def monthly_to_quarterly(
    data: pd.DataFrame,
    name: str,
) -> pd.DataFrame:
    """
    Average a monthly series within the quarter.

    Quarters with fewer than three monthly readings are kept but
    flagged, so that a partial quarter at the end of the sample is
    visible rather than silently averaged over one month.
    """

    months = pd.PeriodIndex(data["TIME_PERIOD"], freq="M")

    series = pd.Series(
        data["OBS_VALUE"].astype(float).to_numpy(),
        index=months,
    ).sort_index()

    if series.index.has_duplicates:
        raise ValueError(f"{name} has duplicate monthly observations.")

    quarters = series.index.asfreq("Q")

    frame = pd.DataFrame(
        {
            name: series.groupby(quarters).mean(),
            f"{name}_n_months": series.groupby(quarters).size(),
        }
    )

    return frame.rename_axis("quarter")


def daily_rate_to_quarterly(
    data: pd.DataFrame,
    name: str,
) -> pd.DataFrame:
    """
    Turn a change-date policy-rate series into quarterly levels.

    The published file lists the rate only on days it changes. Between
    changes the rate is constant, so the series is reindexed onto a
    daily calendar and forward-filled before aggregation. Both the
    end-of-quarter level and the within-quarter average are returned.
    """

    dates = pd.to_datetime(data["TIME_PERIOD"])

    series = pd.Series(
        data["OBS_VALUE"].astype(float).to_numpy(),
        index=dates,
    ).sort_index()

    if series.index.has_duplicates:
        raise ValueError(f"{name} has duplicate daily observations.")

    calendar = pd.date_range(
        series.index.min(), pd.Timestamp(END_PERIOD), freq="D"
    )

    daily = series.reindex(calendar).ffill()

    if daily.isna().any():
        raise ValueError(
            f"{name} has unfilled days at the start of the calendar."
        )

    quarters = daily.index.to_period("Q")

    frame = pd.DataFrame(
        {
            f"{name}_eop": daily.groupby(quarters).last(),
            f"{name}_avg": daily.groupby(quarters).mean(),
        }
    )

    return frame.rename_axis("quarter")


def splice_mro_rate(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Reconstruct one continuous main refinancing rate.

    The deposit facility rate used elsewhere in this project is the
    right measure of the policy stance only once excess liquidity has
    accumulated, from roughly 2014. Under the pre-crisis corridor system
    the deposit rate was the floor and the main refinancing rate was the
    operative instrument, with overnight rates tracking the latter. The
    corridor was usually symmetric with fixed width, so changes in the
    two largely coincide, but the width itself moved repeatedly between
    2008 and 2013.

    The MRO rate is published as two non-overlapping series because the
    tender procedure changed. The fixed rate applies to January 1999 to
    June 2000 and again from October 2008; the minimum bid rate applies
    in between. Splicing at the published changeover dates gives one
    series with no judgement beyond the regime dates themselves.
    """

    fixed = raw["mro_fixed"].copy()
    minbid = raw["mro_minbid"].copy()

    for frame in (fixed, minbid):
        frame["date"] = pd.to_datetime(frame["TIME_PERIOD"])

    start = pd.Timestamp(MBR_REGIME_START)
    end = pd.Timestamp(MBR_REGIME_END)

    # Fixed rate outside the variable-tender window, minimum bid rate
    # inside it. The changeover date itself belongs to the fixed-rate
    # regime, which is when full allotment resumed.
    outside = fixed[(fixed["date"] < start) | (fixed["date"] >= end)]
    inside = minbid[(minbid["date"] >= start) & (minbid["date"] < end)]

    spliced = (
        pd.concat([outside, inside])
        .sort_values("date")
        .drop_duplicates(subset="date", keep="first")
        .reset_index(drop=True)
    )

    if spliced["date"].min() > pd.Timestamp("1999-01-04"):
        raise ValueError(
            "Spliced MRO series does not reach the start of the sample."
        )

    # A long gap between observations is normal here: the series lists
    # change dates only, and the MRO rate sat at zero from March 2016 to
    # September 2019. What must be checked instead is that the splice
    # itself is seamless -- that each regime actually contributed
    # observations, and that the changeovers are bracketed on both
    # sides rather than leaving a regime unrepresented.
    if inside.empty:
        raise ValueError(
            "The variable-tender window contributed no minimum bid "
            "rate observations; the regime dates are wrong."
        )

    for boundary in (start, end):
        before = (spliced["date"] < boundary).any()
        after = (spliced["date"] >= boundary).any()

        if not (before and after):
            raise ValueError(
                f"No observations bracket the regime change at "
                f"{boundary.date()}."
            )

    if spliced["date"].duplicated().any():
        raise ValueError("Spliced MRO series has duplicate dates.")

    return spliced[["TIME_PERIOD", "OBS_VALUE"]]


def build_quarterly(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Assemble the quarterly macro panel."""

    pieces = [
        monthly_to_quarterly(raw["hicp_yoy"], "hicp_yoy"),
        monthly_to_quarterly(raw["unemployment"], "unemployment"),
        daily_rate_to_quarterly(raw["dfr"], "dfr"),
        daily_rate_to_quarterly(splice_mro_rate(raw), "mro"),
    ]

    macro = pd.concat(pieces, axis=1).sort_index()

    macro.index = macro.index.astype(str)

    return macro.rename_axis("period").reset_index()


def check_plausible(macro: pd.DataFrame) -> None:
    """
    Reject values outside historically possible ranges.

    A mistyped series key usually still returns a valid CSV, just of
    the wrong quantity. This is the cheapest way to notice.
    """

    checks = {
        "hicp_yoy": "hicp_yoy",
        "unemployment": "unemployment",
        "dfr_eop": "dfr",
        "dfr_avg": "dfr",
        "mro_eop": "mro",
        "mro_avg": "mro",
    }

    for column, bound_key in checks.items():
        low, high = BOUNDS[bound_key]

        values = macro[column].dropna()

        outside = values[(values < low) | (values > high)]

        if not outside.empty:
            raise ValueError(
                f"{column} has {len(outside)} values outside "
                f"[{low}, {high}], e.g. {outside.iloc[0]}. The series "
                "key may not be what it is labelled."
            )


def main() -> None:
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    raw = download_all()

    macro = build_quarterly(raw)

    check_plausible(macro)

    print("\n=== QUARTERLY MACRO PANEL ===")
    print(f"Quarters: {len(macro)}")
    print(f"Period: {macro['period'].iloc[0]} to {macro['period'].iloc[-1]}")

    print("\n=== COVERAGE ===")
    for column in ["hicp_yoy", "unemployment", "dfr_eop", "mro_eop"]:
        available = macro.loc[macro[column].notna(), "period"]

        print(
            f"{column:14s} {len(available):3d} quarters, "
            f"{available.iloc[0]} to {available.iloc[-1]}"
        )

    print("\n=== SUMMARY ===")
    print(
        macro[["hicp_yoy", "unemployment", "dfr_eop", "mro_eop"]]
        .describe()
        .loc[["mean", "std", "min", "max"]]
        .round(2)
        .to_string()
    )

    macro.to_csv(OUTPUT_FILE, index=False)

    print(f"\nSaved quarterly macro panel to: {OUTPUT_FILE}")
    print(f"Cached raw responses in: {RAW_MACRO_DIR}")


if __name__ == "__main__":
    main()
