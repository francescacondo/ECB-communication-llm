"""
Download the euro-area monetary policy surprise factors.

The validation targets used so far -- realised HICP inflation, the
deposit facility rate path -- are all *realised* macro outcomes. They
say whether the communication measures line up with what the economy
and the Governing Council subsequently did. They cannot say whether the
measures line up with what markets understood the policy stance to be.
The Altavilla, Brugnolini, Gurkaynak, Motto and Ragusa (2019) factors
are the standard series for that, so they are pulled here.

Two sources, and the distinction matters:

    Dataset_EA-MPD.xlsx   the event-study database itself, from the ECB.
                          It contains only the *raw* window changes in
                          OIS rates, bond yields, equities and exchange
                          rates. It does NOT contain the rotated
                          factors, despite being the file usually cited
                          for them.

    gragusa.org/factors/  the rotated factors, published by one of the
                          authors and updated per event. Target comes
                          from the press release window; timing,
                          forward guidance and QE come from the press
                          conference window.

The workbook is downloaded even though nothing downstream reads it,
because it is the primary source and its Notes sheet is what documents
the event windows. Storing it makes the provenance of the factors
checkable without a second trip to the web.

Normalisation, as stated by the authors: target and timing have unit
effect on the 1-month and 6-month OIS surprises respectively; forward
guidance and QE have unit effect on the 2-year and 10-year yields. The
factors are therefore in basis points of the yield they are normalised
on, and are NOT comparable to one another in magnitude.

The factor files are vintage-stamped by the date of the last event they
cover. The vintage is pinned in FACTOR_VINTAGE rather than resolved to
"latest", so that re-running this script does not silently extend the
sample and move every downstream number.

Run from the project root:

    python src/download_policy_factors.py
"""

from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_FACTOR_DIR = PROJECT_ROOT / "data" / "raw" / "eampd"

OUTPUT_FILE = (
    PROJECT_ROOT / "data" / "processed" / "policy_factors_events.csv"
)

EAMPD_URL = "https://www.ecb.europa.eu/pub/pdf/annex/Dataset_EA-MPD.xlsx"

FACTOR_BASE = "https://gragusa.org/factors/data"

# Pinned vintage; see the docstring.
FACTOR_VINTAGE = "2025-10-30"

USER_AGENT = "ecb-communication-llm research project"

REQUEST_TIMEOUT = 90

# The speech corpus starts in 1999 but the factors start in 2002,
# because the OIS quotes needed for the rotation do not exist earlier.
# This is an expected loss of twelve quarters, not a merge failure, and
# is asserted rather than discovered downstream.
FACTORS_FIRST_YEAR = 2002


def fetch(url: str, destination: Path) -> Path:
    """Download to `destination`, reusing a cached copy if present."""

    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists():
        print(f"  cached  {destination.name}")
        return destination

    response = requests.get(
        url,
        headers={"User-Agent": USER_AGENT},
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()

    destination.write_bytes(response.content)

    print(f"  fetched {destination.name} ({len(response.content):,} bytes)")

    return destination


def load_factors() -> pd.DataFrame:
    """
    Merge the press release and press conference factor files.

    Target is kept from the press release window because that is where
    the authors identify it: the press release carries the rate
    decision, the conference carries the explanation. Calling target a
    press-conference factor would misattribute it.
    """

    conference = pd.read_csv(
        RAW_FACTOR_DIR
        / f"press_conference_factors_{FACTOR_VINTAGE}.csv",
        parse_dates=["date"],
    )

    release = pd.read_csv(
        RAW_FACTOR_DIR / f"press_release_factors_{FACTOR_VINTAGE}.csv",
        parse_dates=["date"],
    )

    expected_conference = {"date", "timing", "fg", "qe"}
    expected_release = {"date", "target"}

    if set(conference.columns) != expected_conference:
        raise ValueError(
            "Press conference file has columns "
            f"{sorted(conference.columns)}, expected "
            f"{sorted(expected_conference)}. The publisher changed the "
            "format; do not guess at the mapping."
        )

    if set(release.columns) != expected_release:
        raise ValueError(
            "Press release file has columns "
            f"{sorted(release.columns)}, expected "
            f"{sorted(expected_release)}."
        )

    merged = release.merge(conference, on="date", how="outer")

    merged = merged.sort_values("date").reset_index(drop=True)

    if merged["date"].duplicated().any():
        raise ValueError(
            "Duplicate event dates after the merge; the two windows "
            "disagree about the event calendar."
        )

    if merged["date"].dt.year.min() != FACTORS_FIRST_YEAR:
        raise ValueError(
            f"Factors begin in {merged['date'].dt.year.min()}, expected "
            f"{FACTORS_FIRST_YEAR}. The vintage or the source changed."
        )

    # QE is missing before the asset purchase programmes, by
    # construction rather than by accident: there is no third factor to
    # identify when the long end is not being moved by policy. Left as
    # missing so that any downstream use has to confront the shorter
    # sample rather than treat pre-2014 QE as zero.
    if merged["qe"].notna().any():
        first_qe = merged.loc[merged["qe"].notna(), "date"].min()
        print(f"  QE factor first available {first_qe.date()}")

    return merged


def main() -> None:
    print("Downloading the event-study database and the factors...")

    fetch(EAMPD_URL, RAW_FACTOR_DIR / "Dataset_EA-MPD.xlsx")

    for window in ("press_release", "press_conference"):
        name = f"{window}_factors_{FACTOR_VINTAGE}.csv"
        fetch(f"{FACTOR_BASE}/{name}", RAW_FACTOR_DIR / name)

    factors = load_factors()

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    factors.to_csv(OUTPUT_FILE, index=False)

    print(f"\n{len(factors)} policy events, "
          f"{factors['date'].min().date()} to "
          f"{factors['date'].max().date()}")
    print(factors[["target", "timing", "fg", "qe"]].notna().sum().to_string())
    print(f"\nSaved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
