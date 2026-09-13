from pathlib import Path
import csv

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_FILE = PROJECT_ROOT / "data" / "raw" / "all_ECB_speeches.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_FILE = PROCESSED_DIR / "ecb_speeches_1999_2025.csv"


def load_raw_data() -> pd.DataFrame:
    """Load the official ECB speeches dataset."""

    df = pd.read_csv(
        RAW_FILE,
        sep="|",
        encoding="utf-8",
        quoting=csv.QUOTE_NONE,
    )

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    return df


def clean_speeches(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct the baseline analysis sample.

    Inclusion criteria:
    1. Speech date is between 1999-01-01 and 2025-12-31.
    2. Full speech contents are non-missing and non-empty.
    """

    df = df.copy()

    # Basic derived variables
    df["year"] = df["date"].dt.year
    df["quarter"] = df["date"].dt.to_period("Q").astype(str)

    # Define usable text
    df["has_usable_content"] = (
        df["contents"]
        .fillna("")
        .str.strip()
        .ne("")
    )

    # Explicit sample-period indicator
    df["in_sample_period"] = df["year"].between(1999, 2025)

    # Baseline inclusion rule
    df["include_baseline"] = (
        df["in_sample_period"]
        & df["has_usable_content"]
    )

    # Keep only included observations
    sample = df.loc[df["include_baseline"]].copy()

    # Stable speech identifier
    sample = sample.sort_values(
        ["date", "speakers", "title"]
    ).reset_index(drop=True)

    sample["speech_id"] = [
        f"ecb_{i:04d}"
        for i in range(1, len(sample) + 1)
    ]

    # Put identifier first
    columns = [
        "speech_id",
        "date",
        "year",
        "quarter",
        "speakers",
        "title",
        "subtitle",
        "contents",
    ]

    sample = sample[columns]

    return sample


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df = load_raw_data()

    df["year"] = df["date"].dt.year

    df["has_usable_content"] = (
        df["contents"]
        .fillna("")
        .str.strip()
        .ne("")
    )

    outside_period = ~df["year"].between(1999, 2025)

    missing_content_in_period = (
        df["year"].between(1999, 2025)
        & ~df["has_usable_content"]
    )

    print("\n=== EXCLUSION REASONS ===")
    print(f"Outside 1999–2025: {outside_period.sum():,}")
    print(
        "No usable contents within 1999–2025: "
        f"{missing_content_in_period.sum():,}"
    )

    print("\nExamples with no usable contents:")
    print(
        df.loc[
            missing_content_in_period,
            ["date", "speakers", "title", "subtitle"]
        ]
        .head(15)
        .to_string(index=False)
    )

    sample = clean_speeches(df)

    print(
        "\nUnique speech IDs:",
        sample["speech_id"].is_unique
    )
    print("\n=== CLEANING SUMMARY ===")
    print(f"Raw observations: {len(df):,}")
    print(f"Baseline sample: {len(sample):,}")

    print(
        "Excluded observations: "
        f"{len(df) - len(sample):,}"
    )

    print("\nSample period:")
    print(f"Start: {sample['date'].min().date()}")
    print(f"End:   {sample['date'].max().date()}")

    print("\nMissing values in cleaned sample:")
    print(sample.isna().sum())

    print("\nSpeeches by year:")
    print(
        sample.groupby("year")
        .size()
        .rename("n_speeches")
        .to_string()
    )

    sample.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(f"\nSaved cleaned dataset to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()