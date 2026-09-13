from pathlib import Path
from datetime import datetime, timezone
import json
import os
import time

import pandas as pd
from dotenv import load_dotenv
from groq import Groq, RateLimitError

from schema import SpeechSignals


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "test_sample"
    / "pilot_20_speeches_full.csv"
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
    / "pilot_20_extractions_groq_predecessor.csv"
)

ERROR_FILE = (
    OUTPUT_DIR
    / "pilot_20_errors_groq_predecessor.csv"
)

MODEL = "openai/gpt-oss-20b"

REASONING_EFFORT = "low"

MAX_RETRIES = 5

BASE_RETRY_WAIT_SECONDS = 2

PAUSE_BETWEEN_CALLS_SECONDS = 1

PROMPT_VERSION = "predecessor"


# ============================================================
# API setup
# ============================================================

def load_client() -> Groq:
    """Load the Groq API key and initialize the client."""

    load_dotenv(PROJECT_ROOT / ".env")

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY was not found. "
            "Check the project .env file."
        )

    return Groq(api_key=api_key)


# ============================================================
# Prompt handling
# ============================================================

def load_prompt() -> str:
    """Load the extraction prompt."""

    return PROMPT_FILE.read_text(
        encoding="utf-8"
    )


def build_prompt(
    prompt_template: str,
    title: str,
    subtitle: str,
    text: str,
) -> str:
    """Insert speech information into the prompt."""

    return prompt_template.format(
        title=title,
        subtitle=subtitle,
        text=text,
    )


# ============================================================
# Structured extraction
# ============================================================

def extract_one(
    client: Groq,
    prompt: str,
) -> SpeechSignals:
    """Send one speech to the model and validate the result."""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        reasoning_effort=REASONING_EFFORT,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "speech_signals",
                "strict": True,
                "schema": {
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
                },
            },
        },
    )

    raw_output = response.choices[0].message.content

    parsed = json.loads(raw_output)

    return SpeechSignals.model_validate(parsed)


# ============================================================
# Existing results / resume functionality
# ============================================================

def load_completed_ids() -> set[str]:
    """
    Return speech IDs already successfully processed.

    This allows interrupted runs to resume without repeating
    successful API calls.
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

    The file is written after every successful API call so
    progress is preserved if the script is interrupted.
    """

    result_df = pd.DataFrame([result])

    write_header = not OUTPUT_FILE.exists()

    result_df.to_csv(
        OUTPUT_FILE,
        mode="a",
        header=write_header,
        index=False,
    )


def append_error(error_record: dict) -> None:
    """Append one failed extraction to the error log."""

    error_df = pd.DataFrame([error_record])

    write_header = not ERROR_FILE.exists()

    error_df.to_csv(
        ERROR_FILE,
        mode="a",
        header=write_header,
        index=False,
    )


# ============================================================
# Main extraction pipeline
# ============================================================

def main() -> None:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    client = load_client()

    prompt_template = load_prompt()

    df = pd.read_csv(INPUT_FILE)

    completed_ids = load_completed_ids()

    remaining = df[
        ~df["speech_id"]
        .astype(str)
        .isin(completed_ids)
    ].copy()

    print("\n=== EXTRACTION RUN ===")

    print(f"Model: {MODEL}")
    print(f"Prompt version: {PROMPT_VERSION}")
    print(f"Reasoning effort: {REASONING_EFFORT}")

    print(
        f"Total speeches in input: "
        f"{len(df)}"
    )

    print(
        f"Already completed: "
        f"{len(completed_ids)}"
    )

    print(
        f"Remaining: "
        f"{len(remaining)}"
    )

    print()

    # --------------------------------------------------------
    # Process speeches
    # --------------------------------------------------------

    for _, row in remaining.iterrows():

        speech_id = str(
            row["speech_id"]
        )

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

                    wait_seconds = (
                        BASE_RETRY_WAIT_SECONDS
                        * attempt
                    )

                    print(
                        f"Waiting "
                        f"{wait_seconds} seconds..."
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
        # Failed speech
        # ----------------------------------------------------

        if signals is None:

            error_record = {
                "speech_id": speech_id,
                "date": row["date"],
                "title": row["title"],
                "model": MODEL,
                "prompt_version": PROMPT_VERSION,
                "error_type": (
                    type(final_error).__name__
                    if final_error
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
        # Successful speech
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
            f"{len(completed_ids)}/{len(df)}"
        )

        print()

        # Reduce pressure on Groq's TPM limit.
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
        f"{len(final_completed)}/{len(df)}"
    )

    if ERROR_FILE.exists():

        errors = pd.read_csv(
            ERROR_FILE
        )

        print(
            f"Logged errors: "
            f"{len(errors)}"
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