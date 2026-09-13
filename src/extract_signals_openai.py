from pathlib import Path
from datetime import datetime, timezone
import json
import os
import time

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError, APIError

from schema import SpeechSignals


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "production"
    / "ecb_speeches_1999_2025_excerpts.csv"
)

PROMPT_FILE = (
    PROJECT_ROOT
    / "prompts"
    / "extraction_prompt.txt"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "extracted_signals"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "ecb_signals_gpt5mini_v1.csv"
)

ERROR_FILE = (
    OUTPUT_DIR
    / "ecb_signals_gpt5mini_v1_errors.csv"
)


MODEL = "gpt-5-mini"

PROMPT_VERSION = "v1"

REASONING_EFFORT = "low"

MAX_RETRIES = 6

BASE_RETRY_WAIT_SECONDS = 2

PAUSE_BETWEEN_CALLS_SECONDS = 0.2


# ============================================================
# Structured-output schema
# ============================================================

SIGNAL_SCHEMA = {
    "type": "object",
    "properties": {
        "inflation_attention": {
            "type": "integer",
            "enum": [0, 1],
        },
        "inflation_outlook": {
            "type": "integer",
            "enum": [-1, 0, 1],
        },
        "growth_attention": {
            "type": "integer",
            "enum": [0, 1],
        },
        "growth_outlook": {
            "type": "integer",
            "enum": [-1, 0, 1],
        },
        "financial_stability_attention": {
            "type": "integer",
            "enum": [0, 1],
        },
        "uncertainty_attention": {
            "type": "integer",
            "enum": [0, 1],
        },
    },
    "required": [
        "inflation_attention",
        "inflation_outlook",
        "growth_attention",
        "growth_outlook",
        "financial_stability_attention",
        "uncertainty_attention",
    ],
    "additionalProperties": False,
}


# ============================================================
# API setup
# ============================================================

def load_client() -> OpenAI:
    """Load the OpenAI API key and initialize the client."""

    load_dotenv(PROJECT_ROOT / ".env")

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY was not found. "
            "Check that it exists in the project .env file."
        )

    return OpenAI(api_key=api_key)


# ============================================================
# Prompt handling
# ============================================================

def load_prompt() -> str:
    """Load the versioned extraction prompt."""

    return PROMPT_FILE.read_text(
        encoding="utf-8"
    )


def build_prompt(
    prompt_template: str,
    title: str,
    subtitle: str,
    text: str,
) -> str:
    """Insert speech metadata and excerpt into the prompt."""

    subtitle = "" if pd.isna(subtitle) else str(subtitle)

    return prompt_template.format(
        title=str(title),
        subtitle=subtitle,
        text=str(text),
    )


# ============================================================
# One API extraction
# ============================================================

def extract_one(
    client: OpenAI,
    prompt: str,
) -> SpeechSignals:
    """
    Send one speech to OpenAI using strict structured output,
    then validate the result locally with Pydantic.
    """

    response = client.responses.create(
        model=MODEL,
        input=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        reasoning={
            "effort": REASONING_EFFORT,
        },
        text={
            "format": {
                "type": "json_schema",
                "name": "speech_signals",
                "schema": SIGNAL_SCHEMA,
                "strict": True,
            }
        },
    )

    raw_output = response.output_text

    if not raw_output:
        raise ValueError(
            "OpenAI returned an empty structured-output response."
        )

    parsed = json.loads(raw_output)

    return SpeechSignals.model_validate(parsed)


# ============================================================
# Resume functionality
# ============================================================

def load_completed_ids() -> set[str]:
    """
    Return speech IDs that already have successful extractions.
    """

    if not OUTPUT_FILE.exists():
        return set()

    existing = pd.read_csv(OUTPUT_FILE)

    if existing.empty:
        return set()

    return set(
        existing["speech_id"]
        .astype(str)
        .tolist()
    )


# ============================================================
# Saving
# ============================================================

def append_result(result: dict) -> None:
    """
    Append one successful extraction immediately.

    This ensures progress survives interruptions.
    """

    row = pd.DataFrame([result])

    write_header = not OUTPUT_FILE.exists()

    row.to_csv(
        OUTPUT_FILE,
        mode="a",
        header=write_header,
        index=False,
    )


def append_error(error_record: dict) -> None:
    """Append a failed extraction to the error log."""

    row = pd.DataFrame([error_record])

    write_header = not ERROR_FILE.exists()

    row.to_csv(
        ERROR_FILE,
        mode="a",
        header=write_header,
        index=False,
    )


# ============================================================
# Error helpers
# ============================================================

def get_retry_wait(attempt: int) -> int:
    """
    Exponential backoff:
    2, 4, 8, 16, 32 seconds.
    """

    return BASE_RETRY_WAIT_SECONDS * (2 ** (attempt - 1))


# ============================================================
# Main pipeline
# ============================================================

def main() -> None:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    client = load_client()
    prompt_template = load_prompt()

    df = pd.read_csv(INPUT_FILE)

    required_columns = {
        "speech_id",
        "date",
        "speakers",
        "title",
        "subtitle",
        "excerpt_words",
        "excerpt",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            "Input file is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    completed_ids = load_completed_ids()

    remaining = df[
        ~df["speech_id"]
        .astype(str)
        .isin(completed_ids)
    ].copy()

    print("\n=== PRODUCTION EXTRACTION RUN ===")
    print(f"Model: {MODEL}")
    print(f"Prompt version: {PROMPT_VERSION}")
    print(f"Reasoning effort: {REASONING_EFFORT}")
    print(f"Input speeches: {len(df):,}")
    print(f"Already completed: {len(completed_ids):,}")
    print(f"Remaining: {len(remaining):,}")
    print()

    # --------------------------------------------------------
    # Extraction loop
    # --------------------------------------------------------

    for _, row in remaining.iterrows():

        speech_id = str(row["speech_id"])

        print(
            f"Processing {speech_id}: "
            f"{row['title']}"
        )

        prompt = build_prompt(
            prompt_template=prompt_template,
            title=row["title"],
            subtitle=row["subtitle"],
            text=row["excerpt"],
        )

        signals = None
        final_error = None

        # ----------------------------------------------------
        # Retry loop
        # ----------------------------------------------------

        for attempt in range(
            1,
            MAX_RETRIES + 1,
        ):

            try:
                signals = extract_one(
                    client=client,
                    prompt=prompt,
                )

                break

            except RateLimitError as error:

                final_error = error

                print(
                    f"Rate limit reached "
                    f"(attempt {attempt}/{MAX_RETRIES})."
                )

                if attempt < MAX_RETRIES:

                    wait_seconds = get_retry_wait(
                        attempt
                    )

                    print(
                        f"Waiting "
                        f"{wait_seconds} seconds..."
                    )

                    time.sleep(
                        wait_seconds
                    )

            except APIError as error:

                final_error = error

                print(
                    f"OpenAI API error "
                    f"(attempt {attempt}/{MAX_RETRIES}): "
                    f"{error}"
                )

                if attempt < MAX_RETRIES:

                    wait_seconds = get_retry_wait(
                        attempt
                    )

                    print(
                        f"Waiting "
                        f"{wait_seconds} seconds..."
                    )

                    time.sleep(
                        wait_seconds
                    )

            except json.JSONDecodeError as error:

                final_error = error

                print(
                    f"Invalid JSON returned "
                    f"(attempt {attempt}/{MAX_RETRIES}): "
                    f"{error}"
                )

                if attempt < MAX_RETRIES:

                    wait_seconds = get_retry_wait(
                        attempt
                    )

                    print(
                        f"Waiting "
                        f"{wait_seconds} seconds before retry..."
                    )

                    time.sleep(
                        wait_seconds
                    )

            except Exception as error:

                final_error = error

                print(
                    f"Unexpected error: "
                    f"{error}"
                )

                break

        # ----------------------------------------------------
        # Failure handling
        # ----------------------------------------------------

        if signals is None:

            error_record = {
                "speech_id": speech_id,
                "date": row["date"],
                "speakers": row["speakers"],
                "title": row["title"],
                "model": MODEL,
                "prompt_version": PROMPT_VERSION,
                "excerpt_words": row["excerpt_words"],
                "reasoning_effort": REASONING_EFFORT,
                "error_type": (
                    type(final_error).__name__
                    if final_error is not None
                    else "UnknownError"
                ),
                "error_message": str(final_error),
                "timestamp_utc": (
                    datetime.now(timezone.utc)
                    .isoformat()
                ),
            }

            append_error(
                error_record
            )

            print(
                f"FAILED: {speech_id}"
            )

            print()

            continue

        # ----------------------------------------------------
        # Successful extraction
        # ----------------------------------------------------

        result = {
            "speech_id": speech_id,
            "date": row["date"],
            "speakers": row["speakers"],
            "title": row["title"],
            "model": MODEL,
            "prompt_version": PROMPT_VERSION,
            "excerpt_words": row["excerpt_words"],
            "reasoning_effort": REASONING_EFFORT,
            "timestamp_utc": (
                datetime.now(timezone.utc)
                .isoformat()
            ),
            **signals.model_dump(),
        }

        append_result(
            result
        )

        completed_ids.add(
            speech_id
        )

        print(
            signals.model_dump()
        )

        print(
            f"Saved successfully. "
            f"Progress: "
            f"{len(completed_ids):,}/{len(df):,}"
        )

        print()

        time.sleep(
            PAUSE_BETWEEN_CALLS_SECONDS
        )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    final_completed = load_completed_ids()

    print("\n=== RUN FINISHED ===")

    print(
        f"Successful extractions: "
        f"{len(final_completed):,}/{len(df):,}"
    )

    if ERROR_FILE.exists():

        errors = pd.read_csv(
            ERROR_FILE
        )

        print(
            f"Logged error rows: "
            f"{len(errors):,}"
        )

    print(
        f"Results file: "
        f"{OUTPUT_FILE}"
    )

    if ERROR_FILE.exists():

        print(
            f"Error file: "
            f"{ERROR_FILE}"
        )


if __name__ == "__main__":
    main()