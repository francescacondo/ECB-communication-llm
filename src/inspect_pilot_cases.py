from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SPEECH_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "test_sample"
    / "pilot_20_speeches_full.csv"
)

RESULT_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "extracted_signals"
    / "pilot_20_extractions_groq.csv"
)

CASES_TO_INSPECT = [
    "ecb_2517",  # high inflation environment -> inflation_outlook = -1
    "ecb_2314",  # pandemic crisis -> growth_outlook = +1
    "ecb_1904",  # recovery speech -> growth_outlook = 0
    "ecb_2567",  # stubborn inflation -> financial stability attention = 1
]


def main() -> None:
    speeches = pd.read_csv(SPEECH_FILE)
    results = pd.read_csv(RESULT_FILE)

    merged = speeches.merge(
        results,
        on="speech_id",
        how="inner",
        suffixes=("_speech", "_result"),
    )

    cases = (
        merged[
            merged["speech_id"].isin(CASES_TO_INSPECT)
        ]
        .set_index("speech_id")
        .loc[CASES_TO_INSPECT]
        .reset_index()
    )

    for _, row in cases.iterrows():

        print("\n" + "=" * 100)
        print(f"SPEECH ID: {row['speech_id']}")
        print(f"DATE: {row['date_speech']}")
        print(f"SPEAKER: {row['speakers_speech']}")
        print(f"TITLE: {row['title_speech']}")

        print("\nMODEL CLASSIFICATION")
        print("-" * 100)

        print(
            f"inflation_attention: "
            f"{row['inflation_attention']}"
        )

        print(
            f"inflation_outlook: "
            f"{row['inflation_outlook']}"
        )

        print(
            f"growth_attention: "
            f"{row['growth_attention']}"
        )

        print(
            f"growth_outlook: "
            f"{row['growth_outlook']}"
        )

        print(
            f"financial_stability_attention: "
            f"{row['financial_stability_attention']}"
        )

        print(
            f"uncertainty_attention: "
            f"{row['uncertainty_attention']}"
        )

        print("\nEXCERPT SUPPLIED TO MODEL")
        print("-" * 100)

        print(row["excerpt"])

        print("\n" + "=" * 100)


if __name__ == "__main__":
    main()