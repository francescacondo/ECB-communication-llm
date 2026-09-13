"""
Build the hand-coding sheet for the second validation round.

Emits one CSV carrying the full speech text and empty code columns, plus
one plain-text file per speech for the cases a spreadsheet cannot hold.

**Full speech, not excerpt.** The first round coded the same 2,000-word
excerpt the model was shown -- all thirty rows of
outputs/audit/manual_audit_sample.csv match the production excerpt
exactly, and twenty-five of those speeches were truncated. That design
cannot observe a truncation-induced miss: if a speech discusses
inflation only after word 2,000, the model does not see it, the coder
did not see it either, and the two agree on a code that is wrong about
the speech. The reported sensitivity of 1.000 is therefore a statement
about the model given its input, not about the pipeline given the
speech.

The published measures are meant to describe speeches, so the reference
for the correction should be the speech. This sheet accordingly carries
the full text as the thing to code. The excerpt is carried alongside as
a diagnostic: when the coder and the model disagree, comparing the two
separates a model error from a truncation artefact. That check costs
nothing until a disagreement actually occurs.

**Still blind.** No model codes appear in the sheet, in the row order,
or in the file names. They are in the sealed key written by
src/select_validation_sample.py.

Run from the project root:

    python src/build_coding_sheet.py
"""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ROUND = 4

BLIND_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "audit"
    / f"validation_sample_r{ROUND}_blind.csv"
)

SPEECH_FILE = (
    PROJECT_ROOT / "data" / "processed" / "ecb_speeches_1999_2025.csv"
)

SHEET_FILE = (
    PROJECT_ROOT / "outputs" / "audit" / f"coding_sheet_r{ROUND}_full.csv"
)

TEXT_DIR = PROJECT_ROOT / "outputs" / "audit" / f"speech_texts_r{ROUND}"

CODE_VARS = [
    "inflation_attention",
    "inflation_outlook",
    "growth_attention",
    "growth_outlook",
    "financial_stability_attention",
    "uncertainty_attention",
]

# Excel silently truncates any cell beyond this many characters. Four
# speeches in the sample exceed it, which is why the plain-text files
# exist alongside the CSV.
EXCEL_CELL_LIMIT = 32_767


def build() -> pd.DataFrame:
    sample = pd.read_csv(BLIND_FILE)
    speeches = pd.read_csv(SPEECH_FILE)

    merged = sample.merge(
        speeches[["speech_id", "contents"]],
        on="speech_id",
        how="left",
        validate="one_to_one",
    )

    if merged["contents"].isna().any():
        raise ValueError(
            "Sampled speeches missing full text: "
            f"{merged.loc[merged['contents'].isna(), 'speech_id'].tolist()}"
        )

    merged["words_full"] = merged["contents"].str.split().str.len()

    merged = merged.rename(
        columns={
            "excerpt": "excerpt_model_saw",
            "excerpt_words": "words_model_saw",
            "contents": "speech_full",
        }
    )

    merged["was_truncated"] = (
        merged["words_full"] > merged["words_model_saw"]
    )

    merged["text_file"] = merged["speech_id"] + ".txt"

    for variable in CODE_VARS:
        merged[variable] = ""

    merged["borderline_vars"] = ""
    merged["notes"] = ""

    columns = (
        [
            "speech_id",
            "date",
            "speaker",
            "title",
            "words_full",
            "words_model_saw",
            "was_truncated",
            "text_file",
        ]
        + CODE_VARS
        + ["borderline_vars", "notes", "excerpt_model_saw", "speech_full"]
    )

    leaked = [c for c in columns if c.endswith("_model") or c == "stratum"]

    if leaked:
        raise ValueError(f"Key columns reached the sheet: {leaked}")

    return merged[columns]


def write_texts(sheet: pd.DataFrame) -> None:
    """One readable text file per speech, for the oversized cases."""

    TEXT_DIR.mkdir(parents=True, exist_ok=True)

    for _, row in sheet.iterrows():
        header = (
            f"{row['speech_id']} | {row['date'][:10]} | {row['speaker']}\n"
            f"{row['title']}\n"
            f"Full speech: {row['words_full']} words. "
            f"Model saw: {row['words_model_saw']} words"
            f"{' (TRUNCATED)' if row['was_truncated'] else ''}.\n"
            + "=" * 76
            + "\n\n"
        )

        (TEXT_DIR / row["text_file"]).write_text(
            header + str(row["speech_full"]), encoding="utf-8"
        )


def main() -> None:
    sheet = build()

    write_texts(sheet)

    sheet.to_csv(SHEET_FILE, index=False)

    oversized = sheet[
        sheet["speech_full"].str.len() > EXCEL_CELL_LIMIT
    ]

    print("\n=== CODING SHEET ===")
    print(f"Speeches: {len(sheet)}")
    print(
        f"Truncated at extraction: {int(sheet['was_truncated'].sum())} "
        f"({sheet['was_truncated'].mean():.0%})"
    )
    print(
        "Median words the model did not see, among truncated: "
        f"{int((sheet.loc[sheet['was_truncated'], 'words_full'] - sheet.loc[sheet['was_truncated'], 'words_model_saw']).median())}"
    )

    print("\n=== CODE COLUMNS (left empty; sheet is blind) ===")
    print("  attention vars: 0 or 1")
    print("  outlook vars:  -1, 0 or 1  (must be 0 if attention is 0)")
    print("  borderline_vars: comma-separated names you found ambiguous")

    if not oversized.empty:
        print(
            f"\n=== {len(oversized)} SPEECHES EXCEED THE EXCEL CELL "
            "LIMIT ==="
        )
        print(
            "Excel will silently cut these in the speech_full column. "
            "Read them\nfrom the text files instead:"
        )

        for _, row in oversized.iterrows():
            print(
                f"  {row['text_file']}  "
                f"({len(row['speech_full']):,} chars, "
                f"{row['words_full']:,} words)"
            )

    print(f"\nSheet: {SHEET_FILE}")
    print(f"Speech texts: {TEXT_DIR}")
    print(f"Sealed key (open only after coding): outputs/validation/")


if __name__ == "__main__":
    main()
