"""
Build the excerpt re-coding sheet, to decompose the round-2 specificity drop.

Three things changed between validation rounds at once: the coder became
blind, the text became the full speech rather than the 2,000-word
excerpt, and the codes came from a different reading. The recovered
specificities fell from 1.000 to 0.57 and 0.61 for two variables, and
attributing that fall requires separating the channels.

Splitting round 2's positive predictive value by whether the model saw
the whole speech already suggests truncation carries much of it:
financial-stability flags are 0.929 confirmed when the model read
everything and 0.594 when it did not. But that split compares different
speeches. This sheet compares the *same* speeches read two ways.

The coder re-codes the excerpt the model actually saw. Their full-text
codes for the same speeches are already recorded. Any difference is the
effect of the extra text.

Two design points.

**Controls are included.** Speeches that were never truncated appear
too, and for those the excerpt is the whole speech. Any code that
changes on a control is pure coder inconsistency between sittings, not
a truncation effect, which is the only way to tell the two apart. The
sheet does not say which is which, and it omits both the truncation
flag and the full word count so that the coder cannot infer how much
text is missing.

**Order is random, so any prefix is a valid sample.** Stopping after
twenty-five rows gives an unbiased estimate on twenty-five speeches
rather than a biased one on a selected subset. There is no need to
finish the sheet for the exercise to be analysable.

One limitation cannot be designed away: the coder has already read these
speeches in full and will remember some of them. Recall pushes the
second reading towards reproducing the first, which biases the measured
truncation effect *downwards*. Whatever effect survives is therefore a
lower bound.

Run from the project root:

    python src/build_excerpt_recode_sheet.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FULL_SHEET_FILE = (
    PROJECT_ROOT / "outputs" / "audit" / "coding_sheet_r2_full.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT / "outputs" / "audit" / "coding_sheet_r3_excerpt.csv"
)

# Distinct from the sampling seed so the two draws are independent.
SEED = 20260827

# Every truncated speech is included; the controls are drawn from the
# rest. Twelve is enough to see coder inconsistency if it is material
# without doubling the work.
N_CONTROLS = 12

CODE_VARS = [
    "inflation_attention",
    "inflation_outlook",
    "growth_attention",
    "growth_outlook",
    "financial_stability_attention",
    "uncertainty_attention",
]


def build() -> pd.DataFrame:
    sheet = pd.read_csv(FULL_SHEET_FILE)

    truncated = sheet[sheet["was_truncated"]]
    intact = sheet[~sheet["was_truncated"]]

    if len(intact) < N_CONTROLS:
        raise ValueError(
            f"Only {len(intact)} untruncated speeches available for "
            f"{N_CONTROLS} controls."
        )

    rng = np.random.default_rng(SEED)

    chosen = rng.choice(len(intact), size=N_CONTROLS, replace=False)

    combined = pd.concat(
        [truncated, intact.iloc[np.sort(chosen)]], ignore_index=True
    )

    order = rng.permutation(len(combined))
    combined = combined.iloc[order].reset_index(drop=True)

    out = combined[
        ["speech_id", "date", "speaker", "title", "words_model_saw"]
    ].copy()

    for variable in CODE_VARS:
        out[variable] = ""

    out["borderline_vars"] = ""
    out["notes"] = ""

    out["excerpt"] = combined["excerpt_model_saw"]

    # The sheet must not reveal how much text is missing, or which rows
    # are controls, or what was coded before.
    forbidden = {
        "was_truncated",
        "words_full",
        "speech_full",
        "text_file",
    }

    leaked = forbidden & set(out.columns)

    if leaked:
        raise ValueError(f"Sheet reveals the treatment: {sorted(leaked)}")

    return out


def main() -> None:
    sheet = build()

    source = pd.read_csv(FULL_SHEET_FILE)

    included = source[source["speech_id"].isin(sheet["speech_id"])]

    n_truncated = int(included["was_truncated"].sum())

    print("\n=== EXCERPT RE-CODING SHEET ===")
    print(f"Speeches: {len(sheet)}")
    print(
        f"  truncated (the comparison): {n_truncated}\n"
        f"  controls, excerpt = full speech: "
        f"{len(sheet) - n_truncated}"
    )
    print(
        "\nThe sheet does not say which is which, and carries neither "
        "the truncation\nflag nor the full word count."
    )

    print(
        "\nMedian words withheld from the model, among the truncated: "
        f"{int((included.loc[included['was_truncated'], 'words_full'] - included.loc[included['was_truncated'], 'words_model_saw']).median())}"
    )

    print("\n=== HOW TO USE IT ===")
    print(
        "Code the excerpt on its own terms, as though the speech ended "
        "where the text\nends. Do not try to recall the full version. "
        "Row order is random, so you may\nstop at any point and what "
        "you have coded is still an unbiased sample."
    )

    print(
        "\nRecall from the first reading biases the measured truncation "
        "effect down,\nso whatever survives is a lower bound."
    )

    sheet.to_csv(OUTPUT_FILE, index=False)

    print(f"\nSheet: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
