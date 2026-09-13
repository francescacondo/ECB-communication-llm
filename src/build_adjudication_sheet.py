"""
Build the adjudication sheet for the round-3 disagreements.

Two coders working independently from docs/coding_rules.md disagreed on
119 cells across 53 speeches. Round 3 established that this is a
codebook problem rather than a model problem: on identical text the two
agree on only 0.694 of cells, and financial stability, inflation outlook
and uncertainty have human-human kappas of -0.154, 0.059 and 0.182.

Adjudication is how that gets fixed. The two coders resolve each
disagreement together and, crucially, record *why* -- the decision rule
the resolution implies. Those rules become the revision to
docs/coding_rules.md. The resolved codes are a by-product; the rules are
the output.

Three design choices.

**The model's codes are excluded.** They are available and they are
deliberately not in this sheet. The adjudicated codes will become the
reference against which sensitivity and specificity are re-estimated,
and if the model's own codes influenced the adjudication, those rates
would be biased upwards by construction. A reference standard has to be
built without reference to the thing it will judge.

**Identical-text disagreements come first.** On the twelve speeches
that were never truncated, both coders read exactly the same words, so
any disagreement is pure interpretation and speaks directly to the
rules. On the other forty-one, coder B read only the excerpt, so a
disagreement may reflect the missing text rather than a difference of
reading. Those are still worth adjudicating, but they are weaker
evidence about the codebook and should not drive a rule change on their
own.

**Variables are ordered by how badly the rules fail.** Financial
stability, inflation outlook and uncertainty first, because those are
where the human-human kappa says the rules do not determine the code.

Run from the project root:

    python src/build_adjudication_sheet.py
"""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CODER_A_FILE = PROJECT_ROOT / "data" / "human" / "round2_coder_a_full.csv"

CODER_B_FILE = PROJECT_ROOT / "data" / "human" / "round3_coder_b_excerpt.csv"

SHEET_FILE = (
    PROJECT_ROOT / "outputs" / "audit" / "coding_sheet_r2_full.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT / "outputs" / "audit" / "adjudication_sheet.csv"
)

# Ordered by human-human kappa from round 3, worst first. These are the
# variables the written rules fail to determine.
VARIABLE_ORDER = [
    "financial_stability_attention",
    "inflation_outlook",
    "uncertainty_attention",
    "growth_attention",
    "inflation_attention",
    "growth_outlook",
]

CEILING_KAPPA = {
    "financial_stability_attention": -0.154,
    "inflation_outlook": 0.059,
    "uncertainty_attention": 0.182,
    "growth_attention": 0.500,
    "inflation_attention": 0.636,
    "growth_outlook": 0.714,
}


def load() -> pd.DataFrame:
    a = pd.read_csv(CODER_A_FILE)
    b = pd.read_csv(CODER_B_FILE)

    sheet = pd.read_csv(SHEET_FILE)[
        ["speech_id", "date", "speaker", "title", "was_truncated",
         "words_full", "words_model_saw", "text_file"]
    ]

    columns = ["speech_id", "borderline_vars", "notes"] + VARIABLE_ORDER

    merged = (
        b[columns]
        .merge(a[columns], on="speech_id", suffixes=("_B", "_A"),
               validate="one_to_one")
        .merge(sheet, on="speech_id", validate="one_to_one")
    )

    if len(merged) != len(b):
        raise ValueError(
            f"Coder B has {len(b)} rows; only {len(merged)} aligned."
        )

    return merged


def build(data: pd.DataFrame) -> pd.DataFrame:
    """One row per disagreeing cell."""

    rows = []

    for _, record in data.iterrows():
        identical = not bool(record["was_truncated"])

        for rank, variable in enumerate(VARIABLE_ORDER):
            code_a = record[f"{variable}_A"]
            code_b = record[f"{variable}_B"]

            if code_a == code_b:
                continue

            flagged = variable in str(record["borderline_vars_A"])

            rows.append(
                {
                    "priority": (0 if identical else 1) * 10 + rank,
                    "evidence": (
                        "identical text"
                        if identical
                        else "B saw excerpt only"
                    ),
                    "variable": variable,
                    "ceiling_kappa": CEILING_KAPPA[variable],
                    "speech_id": record["speech_id"],
                    "date": record["date"],
                    "speaker": record["speaker"],
                    "title": record["title"],
                    "code_A": int(code_a),
                    "code_B": int(code_b),
                    "A_flagged_borderline": flagged,
                    "words_full": record["words_full"],
                    "words_in_excerpt": record["words_model_saw"],
                    "text_file": record["text_file"],
                    "note_A": record["notes_A"],
                    "note_B": record["notes_B"],
                    "resolved_code": "",
                    "rule_implication": "",
                    "rule_is_new": "",
                }
            )

    frame = pd.DataFrame(rows)

    return frame.sort_values(
        ["priority", "variable", "speech_id"]
    ).drop(columns=["priority"]).reset_index(drop=True)


def main() -> None:
    data = load()
    sheet = build(data)

    identical = sheet[sheet["evidence"] == "identical text"]

    print("\n=== ADJUDICATION SHEET ===")
    print(f"Disagreeing cells: {len(sheet)}")
    print(
        f"  on identical text, pure interpretation: {len(identical)}"
    )
    print(
        f"  where B saw the excerpt only: "
        f"{len(sheet) - len(identical)}"
    )

    print("\n=== BY VARIABLE (worst human-human kappa first) ===")

    summary = (
        sheet.groupby("variable", sort=False)
        .agg(
            ceiling_kappa=("ceiling_kappa", "first"),
            total=("speech_id", "size"),
            identical_text=(
                "evidence",
                lambda s: int((s == "identical text").sum()),
            ),
        )
        .reindex(VARIABLE_ORDER)
    )

    print(summary.to_string())

    print("\n=== HOW TO WORK THROUGH IT ===")
    print(
        "Start at the top: the 22 identical-text rows are the ones that\n"
        "speak to the rules, because both coders read the same words.\n"
        "For each, agree a code, then write the general rule that\n"
        "resolution implies in rule_implication. If the rule is not\n"
        "already in docs/coding_rules.md, mark rule_is_new as yes.\n"
    )
    print(
        "The model's codes are deliberately absent. Do not consult them:\n"
        "these adjudicated codes will be the reference for re-estimating\n"
        "the model's error rates, and a reference contaminated by the\n"
        "model's own output cannot measure it."
    )

    sheet.to_csv(OUTPUT_FILE, index=False)

    print(f"\nSheet: {OUTPUT_FILE}")
    print(
        f"Speech texts: "
        f"{PROJECT_ROOT / 'outputs' / 'audit' / 'speech_texts_r2'}"
    )


if __name__ == "__main__":
    main()
