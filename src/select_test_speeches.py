from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ecb_speeches_1999_2025.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "test_sample"

CANDIDATE_OUTPUT_FILE = OUTPUT_DIR / "candidate_speeches.csv"
TEST_OUTPUT_FILE = OUTPUT_DIR / "test_speeches_full.csv"
PILOT_OUTPUT_FILE = OUTPUT_DIR / "pilot_20_speeches_full.csv"


TEST_IDS = [
    "ecb_0066",  # payments/control
    "ecb_0102",  # uncertainty
    "ecb_0112",  # financial stability
    "ecb_0163",  # growth/outlook
    "ecb_0175",  # inflation
]


PILOT_IDS = [
    # Inflationary / upside pressure: 2021-2023
    "ecb_2468",
    "ecb_2513",
    "ecb_2517",
    "ecb_2567",

    # Low inflation / disinflation: 2014-2016
    "ecb_1629",
    "ecb_1716",
    "ecb_1813",
    "ecb_1827",

    # Weak growth / severe downturn
    "ecb_0944",
    "ecb_0953",
    "ecb_2314",
    "ecb_2335",

    # Recovery / stronger activity
    "ecb_1882",
    "ecb_1904",
    "ecb_1900",
    "ecb_1952",

    # Controls / other communication topics
    "ecb_1732",
    "ecb_1802",
    "ecb_1997",
    "ecb_2061",
]


def make_excerpt(text: str, max_words: int = 2000) -> str:
    """Return the first max_words words of a speech."""

    words = str(text).split()
    return " ".join(words[:max_words])


def show_candidates(
    df: pd.DataFrame,
    start_year: int,
    end_year: int,
    keywords: str,
    n: int = 20,
) -> pd.DataFrame:
    """Return targeted candidate speeches for a period and keyword set."""

    subset = df[
        df["year"].between(start_year, end_year)
        & (
            df["title"].str.contains(
                keywords,
                case=False,
                na=False,
                regex=True,
            )
            | df["subtitle"].str.contains(
                keywords,
                case=False,
                na=False,
                regex=True,
            )
        )
    ].copy()

    return subset.head(n)


def print_candidate_block(
    label: str,
    subset: pd.DataFrame,
) -> None:
    """Print a compact candidate table."""

    columns = [
        "speech_id",
        "date",
        "speakers",
        "title",
    ]

    print(f"\n=== {label} ===")

    if subset.empty:
        print("No candidates found.")
    else:
        print(
            subset[columns]
            .to_string(index=False)
        )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_FILE)

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
    )

    if "year" not in df.columns:
        df["year"] = df["date"].dt.year

    # --------------------------------------------------
    # 1. Five-speech diagnostic test sample
    # --------------------------------------------------

    test_sample = (
        df[df["speech_id"].isin(TEST_IDS)]
        .set_index("speech_id")
        .loc[TEST_IDS]
        .reset_index()
    )

    test_sample["excerpt"] = (
        test_sample["contents"]
        .apply(make_excerpt)
    )

    test_sample["excerpt_words"] = (
        test_sample["excerpt"]
        .str.split()
        .str.len()
    )

    print("\n=== FIVE-SPEECH TEST SAMPLE ===")

    print(
        test_sample[
            [
                "speech_id",
                "date",
                "speakers",
                "title",
                "excerpt_words",
            ]
        ].to_string(index=False)
    )

    test_sample.to_csv(
        TEST_OUTPUT_FILE,
        index=False,
    )

    print(
        f"\nSaved five-speech test sample to: "
        f"{TEST_OUTPUT_FILE}"
    )

    # --------------------------------------------------
    # 2. Twenty-speech diagnostic pilot
    # --------------------------------------------------

    pilot_sample = (
        df[df["speech_id"].isin(PILOT_IDS)]
        .set_index("speech_id")
        .loc[PILOT_IDS]
        .reset_index()
    )

    pilot_sample["excerpt"] = (
        pilot_sample["contents"]
        .apply(make_excerpt)
    )

    pilot_sample["excerpt_words"] = (
        pilot_sample["excerpt"]
        .str.split()
        .str.len()
    )

    print("\n=== TWENTY-SPEECH PILOT SAMPLE ===")

    print(
        pilot_sample[
            [
                "speech_id",
                "date",
                "speakers",
                "title",
                "excerpt_words",
            ]
        ].to_string(index=False)
    )

    pilot_sample.to_csv(
        PILOT_OUTPUT_FILE,
        index=False,
    )

    print(
        f"\nSaved 20-speech pilot sample to: "
        f"{PILOT_OUTPUT_FILE}"
    )

    # --------------------------------------------------
    # 3. Broad candidate file
    # --------------------------------------------------

    broad_keywords = (
        "inflation|growth|outlook|financial stability|"
        "banking|uncertainty|crisis|digital euro|payments"
    )

    candidates = df[
        df["title"].str.contains(
            broad_keywords,
            case=False,
            na=False,
            regex=True,
        )
        | df["subtitle"].str.contains(
            broad_keywords,
            case=False,
            na=False,
            regex=True,
        )
    ].copy()

    candidate_columns = [
        "speech_id",
        "date",
        "speakers",
        "title",
        "subtitle",
    ]

    candidates[candidate_columns].to_csv(
        CANDIDATE_OUTPUT_FILE,
        index=False,
    )

    print(
        f"\nBroad candidate speeches: "
        f"{len(candidates):,}"
    )

    print(
        f"Saved broad candidate file to: "
        f"{CANDIDATE_OUTPUT_FILE}"
    )

    # --------------------------------------------------
    # 4. Targeted candidate pools
    # --------------------------------------------------

    inflation_upside = show_candidates(
        df=df,
        start_year=2021,
        end_year=2023,
        keywords=(
            "inflation|prices|price stability|"
            "price pressures|wages"
        ),
        n=20,
    )

    low_inflation = show_candidates(
        df=df,
        start_year=2014,
        end_year=2016,
        keywords=(
            "inflation|prices|price stability|"
            "deflation|low inflation"
        ),
        n=20,
    )

    weak_growth_crisis = show_candidates(
        df=df,
        start_year=2008,
        end_year=2009,
        keywords=(
            "growth|outlook|economy|"
            "economic activity|crisis|recession"
        ),
        n=20,
    )

    weak_growth_pandemic = show_candidates(
        df=df,
        start_year=2020,
        end_year=2020,
        keywords=(
            "growth|outlook|economy|"
            "pandemic|recession|recovery"
        ),
        n=20,
    )

    stronger_growth = show_candidates(
        df=df,
        start_year=2016,
        end_year=2018,
        keywords=(
            "growth|recovery|outlook|"
            "expansion|economy|economic activity"
        ),
        n=20,
    )

    controls = show_candidates(
        df=df,
        start_year=2015,
        end_year=2025,
        keywords=(
            "payments|digital euro|"
            "payment systems|CBDC|"
            "market infrastructure"
        ),
        n=20,
    )

    print_candidate_block(
        "INFLATION UPSIDE CANDIDATES: 2021-2023",
        inflation_upside,
    )

    print_candidate_block(
        "LOW INFLATION / DISINFLATION CANDIDATES: 2014-2016",
        low_inflation,
    )

    print_candidate_block(
        "WEAK GROWTH / CRISIS CANDIDATES: 2008-2009",
        weak_growth_crisis,
    )

    print_candidate_block(
        "WEAK GROWTH / PANDEMIC CANDIDATES: 2020",
        weak_growth_pandemic,
    )

    print_candidate_block(
        "STRONGER GROWTH / RECOVERY CANDIDATES: 2016-2018",
        stronger_growth,
    )

    print_candidate_block(
        "CONTROL / OTHER-TOPIC CANDIDATES: 2015-2025",
        controls,
    )


if __name__ == "__main__":
    main()