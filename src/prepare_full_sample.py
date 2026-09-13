from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ecb_speeches_1999_2025.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "production"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "ecb_speeches_1999_2025_excerpts.csv"
)


def make_excerpt(
    text: str,
    max_words: int = 2000,
) -> str:
    """Return the first max_words words of a speech."""

    words = str(text).split()

    return " ".join(
        words[:max_words]
    )


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.read_csv(
        INPUT_FILE
    )

    df["excerpt"] = (
        df["contents"]
        .apply(make_excerpt)
    )

    df["excerpt_words"] = (
        df["excerpt"]
        .str.split()
        .str.len()
    )

    output_columns = [
        "speech_id",
        "date",
        "year",
        "quarter",
        "speakers",
        "title",
        "subtitle",
        "excerpt_words",
        "excerpt",
    ]

    output = df[
        output_columns
    ].copy()

    print("\n=== FULL EXTRACTION SAMPLE ===")
    print(
        f"Speeches: "
        f"{len(output):,}"
    )

    print("\nExcerpt length:")
    print(
        output["excerpt_words"]
        .describe()
    )

    print(
        "\nSpeeches shorter than "
        "2,000 words:"
    )

    print(
        (
            output["excerpt_words"]
            < 2000
        ).sum()
    )

    print(
        "\nMissing excerpts:"
    )

    print(
        output["excerpt"]
        .isna()
        .sum()
    )

    output.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(
        f"\nSaved production input to: "
        f"{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()