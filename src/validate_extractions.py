from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "production"
    / "ecb_speeches_1999_2025_excerpts.csv"
)

RESULT_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "extracted_signals"
    / "ecb_signals_gpt5mini_v1.csv"
)


BINARY_COLUMNS = [
    "inflation_attention",
    "growth_attention",
    "financial_stability_attention",
    "uncertainty_attention",
]

OUTLOOK_COLUMNS = [
    "inflation_outlook",
    "growth_outlook",
]

EXPECTED_MODEL = "gpt-5-mini"
EXPECTED_PROMPT_VERSION = "v1"
EXPECTED_REASONING_EFFORT = "low"


def check(condition: bool, message: str) -> None:
    """
    Print PASS/FAIL for one validation condition.
    """

    status = "PASS" if condition else "FAIL"

    print(
        f"{status}: {message}"
    )


def main() -> None:
    input_df = pd.read_csv(INPUT_FILE)
    results = pd.read_csv(RESULT_FILE)

    print("\n=== EXTRACTION VALIDATION ===\n")

    # --------------------------------------------------
    # 1. Row counts
    # --------------------------------------------------

    check(
        len(results) == len(input_df),
        (
            f"Result rows match input rows "
            f"({len(results):,} vs {len(input_df):,})"
        ),
    )

    check(
        len(results) == 2770,
        f"Expected 2,770 result rows ({len(results):,})",
    )

    # --------------------------------------------------
    # 2. Unique speech IDs
    # --------------------------------------------------

    check(
        results["speech_id"].is_unique,
        "speech_id is unique in results",
    )

    check(
        input_df["speech_id"].is_unique,
        "speech_id is unique in production input",
    )

    input_ids = set(
        input_df["speech_id"].astype(str)
    )

    result_ids = set(
        results["speech_id"].astype(str)
    )

    missing_ids = sorted(
        input_ids - result_ids
    )

    extra_ids = sorted(
        result_ids - input_ids
    )

    check(
        len(missing_ids) == 0,
        f"No missing speech IDs ({len(missing_ids)} missing)",
    )

    check(
        len(extra_ids) == 0,
        f"No unexpected speech IDs ({len(extra_ids)} extra)",
    )

    if missing_ids:
        print(
            "\nMissing IDs:"
        )
        print(
            missing_ids
        )

    if extra_ids:
        print(
            "\nUnexpected IDs:"
        )
        print(
            extra_ids
        )

    # --------------------------------------------------
    # 3. Missing values
    # --------------------------------------------------

    signal_columns = (
        BINARY_COLUMNS
        + OUTLOOK_COLUMNS
    )

    missing_signals = (
        results[signal_columns]
        .isna()
        .sum()
    )

    check(
        missing_signals.sum() == 0,
        "No missing values in extracted signal variables",
    )

    print(
        "\nMissing values by signal:"
    )

    print(
        missing_signals
    )

    # --------------------------------------------------
    # 4. Allowed values
    # --------------------------------------------------

    for column in BINARY_COLUMNS:

        observed = set(
            results[column]
            .dropna()
            .unique()
        )

        check(
            observed.issubset({0, 1}),
            (
                f"{column} contains only "
                f"allowed values {{0, 1}} "
                f"(observed={sorted(observed)})"
            ),
        )

    for column in OUTLOOK_COLUMNS:

        observed = set(
            results[column]
            .dropna()
            .unique()
        )

        check(
            observed.issubset({-1, 0, 1}),
            (
                f"{column} contains only "
                f"allowed values {{-1, 0, 1}} "
                f"(observed={sorted(observed)})"
            ),
        )

    # --------------------------------------------------
    # 5. Logical constraints
    # --------------------------------------------------

    inflation_violations = results[
        (results["inflation_attention"] == 0)
        & (results["inflation_outlook"] != 0)
    ]

    growth_violations = results[
        (results["growth_attention"] == 0)
        & (results["growth_outlook"] != 0)
    ]

    check(
        len(inflation_violations) == 0,
        (
            "inflation_outlook = 0 whenever "
            "inflation_attention = 0"
        ),
    )

    check(
        len(growth_violations) == 0,
        (
            "growth_outlook = 0 whenever "
            "growth_attention = 0"
        ),
    )

    # --------------------------------------------------
    # 6. Run metadata
    # --------------------------------------------------

    model_values = set(
        results["model"]
        .dropna()
        .astype(str)
        .unique()
    )

    prompt_values = set(
        results["prompt_version"]
        .dropna()
        .astype(str)
        .unique()
    )

    reasoning_values = set(
        results["reasoning_effort"]
        .dropna()
        .astype(str)
        .unique()
    )

    check(
        model_values == {EXPECTED_MODEL},
        (
            f"All rows use model "
            f"{EXPECTED_MODEL} "
            f"(observed={sorted(model_values)})"
        ),
    )

    check(
        prompt_values == {EXPECTED_PROMPT_VERSION},
        (
            f"All rows use prompt version "
            f"{EXPECTED_PROMPT_VERSION} "
            f"(observed={sorted(prompt_values)})"
        ),
    )

    check(
        reasoning_values == {EXPECTED_REASONING_EFFORT},
        (
            f"All rows use reasoning effort "
            f"{EXPECTED_REASONING_EFFORT} "
            f"(observed={sorted(reasoning_values)})"
        ),
    )

    # --------------------------------------------------
    # 7. Distributions
    # --------------------------------------------------

    print(
        "\n=== SIGNAL DISTRIBUTIONS ==="
    )

    for column in signal_columns:

        print(
            f"\n{column}"
        )

        counts = (
            results[column]
            .value_counts()
            .sort_index()
        )

        shares = (
            results[column]
            .value_counts(
                normalize=True
            )
            .sort_index()
        )

        distribution = pd.DataFrame(
            {
                "count": counts,
                "share": shares,
            }
        )

        print(
            distribution
        )

    # --------------------------------------------------
    # 8. Short-excerpt check
    # --------------------------------------------------

    print(
        "\n=== EXCERPT LENGTH CHECK ==="
    )

    print(
        results["excerpt_words"]
        .describe()
    )

    very_short = results[
        results["excerpt_words"] < 200
    ]

    print(
        f"\nSpeeches with fewer than "
        f"200 excerpt words: "
        f"{len(very_short)}"
    )

    if not very_short.empty:

        print(
            very_short[
                [
                    "speech_id",
                    "date",
                    "speakers",
                    "title",
                    "excerpt_words",
                ]
            ]
            .sort_values(
                "excerpt_words"
            )
            .to_string(
                index=False
            )
        )

    # --------------------------------------------------
    # 9. Final summary
    # --------------------------------------------------

    print(
        "\n=== VALIDATION COMPLETE ==="
    )

    print(
        f"Input speeches: {len(input_df):,}"
    )

    print(
        f"Successful extractions: {len(results):,}"
    )

    print(
        f"Unique result IDs: "
        f"{results['speech_id'].nunique():,}"
    )


if __name__ == "__main__":
    main()