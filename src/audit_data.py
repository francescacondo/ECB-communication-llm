from pathlib import Path
import csv

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = PROJECT_ROOT / "data" / "raw" / "all_ECB_speeches.csv"
AUDIT_DIR = PROJECT_ROOT / "outputs" / "audit"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"


def load_data() -> pd.DataFrame:
    """Load the official ECB speeches dataset."""

    df = pd.read_csv(
        DATA_FILE,
        sep="|",
        encoding="utf-8",
        quoting=csv.QUOTE_NONE,
    )

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    return df


def main() -> None:

    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()

    # --------------------------------------------------
    # 1. Basic structure
    # --------------------------------------------------

    print("\n=== DATASET STRUCTURE ===")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print(f"Column names: {list(df.columns)}")

    print("\nFirst 5 observations:")
    print(
        df[
            ["date", "speakers", "title", "subtitle"]
        ].head().to_string()
    )

    # --------------------------------------------------
    # 2. Date coverage
    # --------------------------------------------------

    print("\n=== DATE COVERAGE ===")
    print(f"Earliest date: {df['date'].min()}")
    print(f"Latest date:   {df['date'].max()}")
    print(f"Invalid dates: {df['date'].isna().sum():,}")

    # --------------------------------------------------
    # 3. Missing values
    # --------------------------------------------------

    print("\n=== MISSING VALUES ===")

    missing = pd.DataFrame(
        {
            "missing_n": df.isna().sum(),
            "missing_share": df.isna().mean(),
        }
    )

    print(missing)

    missing.to_csv(
        AUDIT_DIR / "missing_values.csv"
    )

    # --------------------------------------------------
    # 4. Duplicates
    # --------------------------------------------------

    print("\n=== DUPLICATES ===")

    exact_duplicates = df.duplicated().sum()

    title_date_duplicates = df.duplicated(
        subset=["date", "title"]
    ).sum()

    print(f"Exact duplicated rows: {exact_duplicates:,}")
    print(
        f"Duplicated date-title combinations: "
        f"{title_date_duplicates:,}"
    )

    duplicates = df[

        df.duplicated(

            subset=["date", "title"],

            keep=False

        )

    ]

    print("\nDuplicated date-title observations:")

    print(

        duplicates[

            ["date", "speakers", "title", "subtitle"]

        ].to_string(index=False)
    )

    # --------------------------------------------------
    # 5. Speech text length
    # --------------------------------------------------

    df["text_chars"] = df["contents"].fillna("").str.len()

    df["text_words"] = (
        df["contents"]
        .fillna("")
        .str.split()
        .str.len()
    )

    print("\n=== TEXT LENGTH ===")

    print("\nCharacters:")
    print(df["text_chars"].describe())

    print("\nApproximate words:")
    print(df["text_words"].describe())

    # --------------------------------------------------
    # 6. Speeches by year
    # --------------------------------------------------

    df["year"] = df["date"].dt.year

    yearly = (
        df.groupby("year")
        .size()
        .rename("n_speeches")
        .reset_index()
    )

    yearly["n_with_content"] = (
        df[df["contents"].notna()]
        .groupby("year")
        .size()
        .reindex(yearly["year"])
        .fillna(0)
        .astype(int)
        .values
    )

    yearly["share_with_content"] = (
        yearly["n_with_content"]
        / yearly["n_speeches"]
    )

    print("\n=== SPEECHES BY YEAR ===")
    print(yearly.to_string(index=False))

    yearly.to_csv(
        AUDIT_DIR / "speeches_by_year.csv",
        index=False,
    )

    # --------------------------------------------------
    # 7. Most frequent speakers
    # --------------------------------------------------

    print("\n=== MOST COMMON SPEAKER ENTRIES ===")

    speakers = (
        df["speakers"]
        .value_counts(dropna=False)
        .head(20)
    )

    print(speakers)

    speakers.to_csv(
        AUDIT_DIR / "top_speakers.csv",
        header=["n_speeches"],
    )

    # --------------------------------------------------
    # 8. Basic coverage figure
    # --------------------------------------------------

    plot_data = yearly.dropna(subset=["year"])

    plt.figure(figsize=(10, 5))

    plt.bar(
        plot_data["year"].astype(int),
        plot_data["n_speeches"],
    )

    plt.xlabel("Year")
    plt.ylabel("Number of speeches")
    plt.title("ECB speeches in official dataset by year")

    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "speeches_by_year.png",
        dpi=200,
    )

    plt.close()

    # --------------------------------------------------

    # 9. Provisional analysis sample

    # --------------------------------------------------

    df["has_usable_content"] = (

        df["contents"]

        .fillna("")

        .str.strip()

        .ne("")

    )

    sample = df[

        df["date"].dt.year.between(1999, 2025)

        & df["has_usable_content"]

    ]

    print("\n=== PROVISIONAL ANALYSIS SAMPLE ===")

    print(f"Usable speeches, 1999–2025: {len(sample):,}")

    print(

        "\nSaved figure to "

        "outputs/figures/speeches_by_year.png"

    )

if __name__ == "__main__":
    main()


