"""
Line-level diff between the production prompt and the revised one.

prompts/extraction_prompt.txt is version 1 and is what produced all
2,770 extractions. prompts/extraction_prompt_v2.txt mirrors the revised
codebook for any future re-extraction. The write-up claims the second
is the first plus a fixed set of rules, and Section 4 rests on that
claim: the round-4 model-versus-human comparison is interpretable only
because the difference between the two specifications is known and
small.

The two prompts diverged silently once already -- against the codebook
rather than each other, but by exactly the mechanism a hand-maintained
description invites -- so the difference is computed here rather than
described. If someone edits either file, this script's output changes
and the appendix table can be checked against it.

Reports the diff by prompt section, so a change can be attributed to
the variable it governs rather than to a line number.

Run from the project root:

    python src/diff_prompts.py
"""

from difflib import SequenceMatcher
from pathlib import Path
import re

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

OLD_FILE = PROJECT_ROOT / "prompts" / "extraction_prompt.txt"

NEW_FILE = PROJECT_ROOT / "prompts" / "extraction_prompt_v2.txt"

OUT_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "validation"
    / "prompt_v1_to_v2_diff.csv"
)

# A section begins at a numbered variable, or at one of the unnumbered
# headings. Everything between headings is attributed to the last one.
VARIABLE_HEADING = re.compile(r"^(\d+)\.\s+`?(\w+)`?\s*$")

PLAIN_HEADINGS = {
    "General rules:": "General rules",
    "Required output structure:": "Output structure",
    "Speech metadata and text follow below.": "Template",
}


def sections_for(lines: list[str]) -> list[str]:
    """Label each line with the prompt section it belongs to."""

    labels = []
    current = "Preamble"

    for line in lines:
        stripped = line.strip()

        match = VARIABLE_HEADING.match(stripped)

        if match:
            current = f"{match.group(1)}. {match.group(2)}"
        elif stripped in PLAIN_HEADINGS:
            current = PLAIN_HEADINGS[stripped]

        labels.append(current)

    return labels


def diff() -> pd.DataFrame:
    """One row per changed line, attributed to its section."""

    old = OLD_FILE.read_text(encoding="utf-8").split("\n")
    new = NEW_FILE.read_text(encoding="utf-8").split("\n")

    old_sections = sections_for(old)
    new_sections = sections_for(new)

    rows = []

    matcher = SequenceMatcher(None, old, new, autojunk=False)

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue

        # Pair removals with insertions inside a replacement so a
        # reworded line reads as one change rather than two.
        span = max(i2 - i1, j2 - j1)

        for offset in range(span):
            old_index = i1 + offset if i1 + offset < i2 else None
            new_index = j1 + offset if j1 + offset < j2 else None

            if old_index is None and new_index is None:
                continue

            if old_index is None:
                change = "added"
            elif new_index is None:
                change = "removed"
            else:
                change = "modified"

            section = (
                new_sections[new_index]
                if new_index is not None
                else old_sections[old_index]
            )

            rows.append(
                {
                    "change": change,
                    "section": section,
                    "v1_line": (
                        old_index + 1 if old_index is not None else ""
                    ),
                    "v2_line": (
                        new_index + 1 if new_index is not None else ""
                    ),
                    "v1_text": (
                        old[old_index].strip()
                        if old_index is not None
                        else ""
                    ),
                    "v2_text": (
                        new[new_index].strip()
                        if new_index is not None
                        else ""
                    ),
                }
            )

    frame = pd.DataFrame(rows)

    # Blank lines shift when text is inserted and carry no content.
    substantive = (
        frame["v1_text"].str.len() + frame["v2_text"].str.len()
    ) > 0

    return frame[substantive].reset_index(drop=True)


def main() -> None:
    changes = diff()

    print("\n=== PROMPT v1 -> v2 ===")
    print(f"old: {OLD_FILE.relative_to(PROJECT_ROOT)}")
    print(f"new: {NEW_FILE.relative_to(PROJECT_ROOT)}")

    print("\n=== BY SECTION ===")
    print(
        changes.groupby(["section", "change"])
        .size()
        .rename("n")
        .reset_index()
        .to_string(index=False)
    )

    print("\n=== EVERY CHANGE ===")

    for row in changes.itertuples():
        print(f"\n[{row.change}] {row.section}")

        if row.v1_text:
            print(f"  v1 (line {row.v1_line}): {row.v1_text}")

        if row.v2_text:
            print(f"  v2 (line {row.v2_line}): {row.v2_text}")

    removed = int((changes["change"] == "removed").sum())
    modified = int((changes["change"] == "modified").sum())

    print(
        f"\n{len(changes)} substantive changes: "
        f"{int((changes['change'] == 'added').sum())} added, "
        f"{removed} removed, {modified} modified."
    )

    if removed == 0 and modified == 0:
        print(
            "Version 2 is version 1 plus additions. No existing rule "
            "was altered or\nwithdrawn, which is what licenses reading "
            "the round-4 model scores as the\nmodel being held to a "
            "superset of the rules it was given."
        )

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    changes.to_csv(OUT_FILE, index=False)

    print(f"\nSaved to: {OUT_FILE}")


if __name__ == "__main__":
    main()
