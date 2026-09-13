"""
Compare an independent second coder against the production extraction.

This exists to make a hand-coding session efficient, not to validate
anything. The second coder here is another large language model, and
the distinction matters enough to be stated in the code rather than
only in a README:

    A second LLM is NOT a substitute for the human coder. It shares
    training data and inductive biases with the model being scored, so
    its errors are correlated with that model's errors by an unknown
    amount. Agreement between two language models is evidence about
    cross-model robustness. It is not evidence about accuracy, and it
    cannot estimate the human-human ceiling that the second validation
    round exists to establish.

What it is good for is triage. Cells where two independently prompted
models disagree are disproportionately the genuinely ambiguous or
genuinely wrong ones. Ordering the coding sheet so those come first
means scarce human attention goes where it is informative rather than
to the long tail of speeches that are trivially zero on everything.

Outputs a triage sheet ordered by the number of contested cells.

Run from the project root:

    python src/triage_second_coder.py
"""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

AUDIT_DIR = PROJECT_ROOT / "outputs" / "audit"

SECOND_CODER_FILE = AUDIT_DIR / "llm_second_coder_r2.csv"

KEY_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "validation"
    / "validation_sample_r2_key.csv"
)

BLIND_FILE = AUDIT_DIR / "validation_sample_r2_blind.csv"

TRIAGE_FILE = AUDIT_DIR / "validation_r2_triage.csv"

SUMMARY_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "validation"
    / "second_coder_agreement.csv"
)

ATTENTION_VARS = [
    "inflation_attention",
    "growth_attention",
    "financial_stability_attention",
    "uncertainty_attention",
]

OUTLOOK_VARS = ["inflation_outlook", "growth_outlook"]

ALL_VARS = ATTENTION_VARS + OUTLOOK_VARS


def load() -> pd.DataFrame:
    """Merge the second coder's codes onto the model's."""

    second = pd.read_csv(SECOND_CODER_FILE)
    key = pd.read_csv(KEY_FILE)
    blind = pd.read_csv(BLIND_FILE)

    merged = second.merge(
        key,
        on="speech_id",
        how="left",
        suffixes=("", "_model"),
        validate="one_to_one",
    )

    if merged[ALL_VARS].isna().any().any():
        raise ValueError(
            "Second-coder rows without a matching key entry; the "
            "coding sheet and the key have diverged."
        )

    return merged.merge(
        blind[["speech_id", "title", "excerpt_words"]],
        on="speech_id",
        how="left",
    )


def compare(data: pd.DataFrame) -> pd.DataFrame:
    """Per-variable agreement between the two coders."""

    rows = []

    for variable in ALL_VARS:
        model = data[variable]
        second = data[f"{variable}_c"]

        rows.append(
            {
                "variable": variable,
                "n": len(data),
                "n_agree": int((model == second).sum()),
                "agreement": float((model == second).mean()),
                "model_mean": float(model.mean()),
                "second_mean": float(second.mean()),
            }
        )

    return pd.DataFrame(rows)


def triage(data: pd.DataFrame) -> pd.DataFrame:
    """
    One row per speech, listing the contested cells.

    Speeches are ordered by how many cells the two coders disagree on,
    so the top of the sheet is where a human reading changes the most.
    """

    rows = []

    for _, record in data.iterrows():
        contested = []

        for variable in ALL_VARS:
            model = record[variable]
            second = record[f"{variable}_c"]

            if model != second:
                contested.append(
                    f"{variable}: model={int(model):+d} "
                    f"second={int(second):+d}"
                )

        rows.append(
            {
                "speech_id": record["speech_id"],
                "date": record["date"],
                "speaker": record["speaker"],
                "title": record["title"],
                "n_contested": len(contested),
                "contested_cells": "; ".join(contested),
                "second_coder_confidence": record["confidence"],
                "second_coder_note": record["note"],
                "priority": (
                    "high"
                    if len(contested) >= 2
                    else "medium"
                    if len(contested) == 1
                    else "low"
                ),
            }
        )

    frame = pd.DataFrame(rows)

    return frame.sort_values(
        ["n_contested", "speech_id"], ascending=[False, True]
    ).reset_index(drop=True)


def main() -> None:
    data = load()

    print("\n=== SECOND-CODER PASS ===")
    print(
        f"Speeches coded: {len(data)} of 60 drawn "
        f"({len(data) / 60:.0%} of the sample)"
    )
    print(
        "Second coder is a language model, not a human. These numbers "
        "measure\ncross-model agreement and must not be reported as "
        "validation."
    )

    summary = compare(data)

    print("\n=== AGREEMENT BY VARIABLE ===")
    print(summary.round(3).to_string(index=False))

    total_cells = len(data) * len(ALL_VARS)
    total_agree = int(summary["n_agree"].sum())

    print(
        f"\nPooled cells: {total_agree}/{total_cells} = "
        f"{total_agree / total_cells:.3f}"
    )

    sheet = triage(data)

    print("\n=== TRIAGE ===")
    print(
        sheet["priority"].value_counts().rename("speeches").to_string()
    )

    contested = sheet[sheet["n_contested"] > 0]

    print(f"\nContested speeches: {len(contested)} of {len(sheet)}")
    print(
        contested[
            ["speech_id", "date", "n_contested", "contested_cells"]
        ].to_string(index=False)
    )

    sheet.to_csv(TRIAGE_FILE, index=False)
    summary.to_csv(SUMMARY_FILE, index=False)

    print(f"\nTriage sheet: {TRIAGE_FILE}")
    print(f"Agreement summary: {SUMMARY_FILE}")


if __name__ == "__main__":
    main()
