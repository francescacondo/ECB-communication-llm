"""
Inter-coder reliability, and what it does to the round-2 error rates.

Two people coded overlapping samples from docs/coding_rules.md without
consulting each other. Coder A read the full speech; coder B read the
2,000-word excerpt the model was shown. That difference is a nuisance
for some comparisons and an advantage for others, because it determines
which pairs read *identical text*:

    model vs B   all 53 speeches. The model was shown the excerpt and
                 B coded the excerpt, so the text is identical
                 throughout. This is the best-powered clean comparison
                 in the project.

    A vs B       only the 12 speeches that were never truncated, where
                 the excerpt is the whole speech. This is the
                 human-human ceiling.

    model vs A   the same 12, for the same reason.

The ceiling is what docs/measurement_notes.md lists as its second
limitation: without it, an agreement rate of 0.944 or 0.767 cannot be
judged, because nobody knows what two careful readers achieve on the
same task.

The design also still answers the question it was built for. Comparing
A with B on truncated speeches mixes a coder difference with a text
difference; the controls measure the coder difference alone, so the
excess disagreement on truncated speeches is what the withheld text
contributes.

Run from the project root:

    python analysis/18_coder_reliability.py
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from agreement import (  # noqa: E402
    bootstrap_kappa_ci,
    categories_for,
    clopper_pearson_ci,
    cohen_kappa,
)

CODER_A_FILE = PROJECT_ROOT / "data" / "human" / "round2_coder_a_full.csv"

CODER_B_FILE = PROJECT_ROOT / "data" / "human" / "round3_coder_b_excerpt.csv"

KEY_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "validation"
    / "validation_sample_r2_key.csv"
)

SHEET_FILE = (
    PROJECT_ROOT / "outputs" / "audit" / "coding_sheet_r2_full.csv"
)

VALIDATION_DIR = PROJECT_ROOT / "outputs" / "validation"
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"

ATTENTION_VARS = [
    "inflation_attention",
    "growth_attention",
    "financial_stability_attention",
    "uncertainty_attention",
]

ALL_VARS = ATTENTION_VARS + ["inflation_outlook", "growth_outlook"]

LABELS = {
    "inflation_attention": "Inflation attention",
    "growth_attention": "Growth attention",
    "financial_stability_attention": "Financial stability",
    "uncertainty_attention": "Uncertainty",
    "inflation_outlook": "Inflation outlook",
    "growth_outlook": "Growth outlook",
}


def load() -> pd.DataFrame:
    """Align both coders, the model, and the truncation flag."""

    a = pd.read_csv(CODER_A_FILE)[["speech_id"] + ALL_VARS]
    b = pd.read_csv(CODER_B_FILE)[["speech_id"] + ALL_VARS]
    key = pd.read_csv(KEY_FILE)[["speech_id"] + ALL_VARS]
    sheet = pd.read_csv(SHEET_FILE)[
        ["speech_id", "was_truncated", "words_full", "words_model_saw"]
    ]

    merged = (
        b.merge(a, on="speech_id", suffixes=("_B", "_A"), validate="one_to_one")
        .merge(key, on="speech_id", validate="one_to_one")
        .merge(sheet, on="speech_id", validate="one_to_one")
    )

    if len(merged) != len(b):
        raise ValueError(
            f"Coder B has {len(b)} rows but only {len(merged)} aligned."
        )

    for column in [f"{v}_A" for v in ALL_VARS] + [f"{v}_B" for v in ALL_VARS]:
        if merged[column].isna().any():
            raise ValueError(f"{column} has uncoded cells.")

    return merged


def base_rate_table(data: pd.DataFrame) -> pd.DataFrame:
    """
    How often each coder, and the model, assigns a positive code.

    For the attention variables this is the share coded one. The outlook
    variables are ternary, so the comparable quantity is the share coded
    directionally rather than neutral.

    This is the evidence that the two coders differ by a threshold
    rather than by a scatter of individual judgements: on three
    variables coder B codes one in roughly seven speeches out of eight,
    where coder A codes one in between a third and three fifths.
    """

    rows = []

    for variable in ALL_VARS:
        if variable.endswith("outlook"):
            a = float((data[f"{variable}_A"] != 0).mean())
            b = float((data[f"{variable}_B"] != 0).mean())
            m = float((data[variable] != 0).mean())
            quantity = "share directional"
        else:
            a = float(data[f"{variable}_A"].mean())
            b = float(data[f"{variable}_B"].mean())
            m = float(data[variable].mean())
            quantity = "share coded 1"

        rows.append(
            {
                "variable": variable,
                "label": LABELS[variable],
                "quantity": quantity,
                "coder_a": a,
                "coder_b": b,
                "model": m,
                "b_minus_a": b - a,
            }
        )

    return pd.DataFrame(rows)


def write_base_rate_tex(table: pd.DataFrame, path: Path) -> None:
    """Emit the base-rate table used in the corrupted-codebook section."""

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Positive classification rates by coder, "
        r"fifty-three shared speeches}",
        r"\label{tab:base-rates}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Variable & Coder A & Coder B & Model \\",
        r"\midrule",
    ]

    for _, row in table.iterrows():
        if row["variable"] == "inflation_outlook":
            lines.append(r"\addlinespace")

        emphasis = row["coder_b"] > 0.85

        b = (
            f"\\textbf{{{row['coder_b']:.3f}}}"
            if emphasis
            else f"{row['coder_b']:.3f}"
        )

        lines.append(
            f"{row['label']} & {row['coder_a']:.3f} & {b} "
            f"& {row['model']:.3f} \\\\"
        )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"",
        r"\vspace{0.5em}",
        r"\begin{minipage}{0.9\textwidth}",
        r"{\footnotesize The first four rows report the share of "
        r"speeches coded one. The outlook variables are ternary, so the "
        r"last two report the share coded directionally rather than "
        r"neutral. Bold marks the three variables on which coder B "
        r"codes one in $86.8\%$ of speeches, against $32.1$ to "
        r"$58.5\%$ for coder A. On forty-one of the fifty-three "
        r"speeches coder B read only the 2,000-word excerpt while coder "
        r"A read the full speech, so the gap mixes a difference in "
        r"threshold with a difference in text --- but coder B saw "
        r"\emph{less} text and still coded \emph{more} ones, which "
        r"runs against the text explanation rather than with it.}",
        r"\end{minipage}",
        r"\end{table}",
    ]

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def pair_table(
    data: pd.DataFrame,
    left: str,
    right: str,
    label: str,
) -> pd.DataFrame:
    """Agreement and kappa for one pair of coders."""

    rows = []

    for variable in ALL_VARS:
        x = data[f"{variable}{left}"].to_numpy()
        y = data[f"{variable}{right}"].to_numpy()

        cats = categories_for(variable)

        observed, kappa = cohen_kappa(x, y, cats)
        lo, hi = bootstrap_kappa_ci(x, y, cats)

        n_agree = int((x == y).sum())
        ci_lo, ci_hi = clopper_pearson_ci(n_agree, len(x))

        rows.append(
            {
                "pair": label,
                "variable": variable,
                "label": LABELS[variable],
                "n": len(x),
                "agreement": observed,
                "agree_lo": ci_lo,
                "agree_hi": ci_hi,
                "kappa": kappa,
                "kappa_lo": lo,
                "kappa_hi": hi,
            }
        )

    return pd.DataFrame(rows)


def pooled(data: pd.DataFrame, left: str, right: str) -> tuple[int, int]:
    """Pooled agreeing cells over all six variables."""

    agree = sum(
        int((data[f"{v}{left}"] == data[f"{v}{right}"]).sum())
        for v in ALL_VARS
    )

    return agree, len(data) * len(ALL_VARS)


def decompose(data: pd.DataFrame) -> pd.DataFrame:
    """
    Separate the coder effect from the withheld-text effect.

    On controls the two coders read identical words, so disagreement is
    coder difference alone. On truncated speeches B additionally saw
    less text. The excess is what the withheld text contributes, on the
    assumption that the coder difference is the same in both groups.
    """

    controls = data[~data["was_truncated"]]
    truncated = data[data["was_truncated"]]

    rows = []

    for variable in ALL_VARS:
        control_disagree = float(
            (controls[f"{variable}_A"] != controls[f"{variable}_B"]).mean()
        )

        truncated_disagree = float(
            (truncated[f"{variable}_A"] != truncated[f"{variable}_B"]).mean()
        )

        rows.append(
            {
                "variable": variable,
                "label": LABELS[variable],
                "coder_effect": control_disagree,
                "coder_plus_text": truncated_disagree,
                "text_effect": truncated_disagree - control_disagree,
                "n_controls": len(controls),
                "n_truncated": len(truncated),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    data = load()

    controls = data[~data["was_truncated"]]

    print("\n=== SAMPLE ===")
    print(f"Coder B coded {len(data)} speeches from the excerpt.")
    print(
        f"Coder A had already coded the same speeches in full. "
        f"{len(controls)} were never\ntruncated, so all three coders "
        "read identical words on those."
    )

    # --------------------------------------------------
    # Best-powered clean comparison
    # --------------------------------------------------

    model_b = pair_table(data, "", "_B", "model vs B (all 53)")

    agree, total = pooled(data, "", "_B")

    print("\n=== MODEL vs CODER B, IDENTICAL TEXT, n = 53 ===")
    print(
        "Both read the 2,000-word excerpt. This is the largest clean "
        "comparison available."
    )
    print(
        model_b[["label", "agreement", "kappa", "kappa_lo", "kappa_hi"]]
        .round(3)
        .to_string(index=False)
    )
    print(f"\nPooled: {agree}/{total} = {agree / total:.3f}")

    # --------------------------------------------------
    # The ceiling
    # --------------------------------------------------

    print("\n=== THREE-WAY ON IDENTICAL TEXT, n = 12 ===")

    frames = []

    for left, right, label in [
        ("_A", "_B", "human A vs human B"),
        ("", "_A", "model vs human A"),
        ("", "_B", "model vs human B"),
    ]:
        table = pair_table(controls, left, right, label)
        frames.append(table)

        agree, total = pooled(controls, left, right)
        lo, hi = clopper_pearson_ci(agree, total)

        print(
            f"  {label:20s} {agree:2d}/{total} = {agree / total:.3f} "
            f"[{lo:.3f}, {hi:.3f}]"
        )

    three_way = pd.concat(frames, ignore_index=True)

    print("\n=== PER VARIABLE: HUMAN CEILING vs MODEL ===")

    pivot = three_way.pivot(
        index="label", columns="pair", values="kappa"
    ).round(3)

    print(pivot.to_string())

    ceiling = three_way[three_way["pair"] == "human A vs human B"]

    weakest = ceiling.nsmallest(2, "kappa")

    print(
        "\nWeakest human-human agreement: "
        + ", ".join(
            f"{row['label']} (kappa {row['kappa']:.3f})"
            for _, row in weakest.iterrows()
        )
    )
    print(
        "These are variables the written rules do not pin down, not "
        "variables the model\nuniquely struggles with. Tightening the "
        "rules has to come before re-estimating\nany correction on them."
    )

    # --------------------------------------------------
    # Decomposition
    # --------------------------------------------------

    rates = base_rate_table(data)

    print("\n=== POSITIVE CLASSIFICATION RATES ===")
    print(
        rates[["label", "quantity", "coder_a", "coder_b", "model",
               "b_minus_a"]]
        .round(3)
        .to_string(index=False)
    )

    split = decompose(data)

    print("\n=== DECOMPOSITION OF A-B DISAGREEMENT ===")
    print(
        split[["label", "coder_effect", "coder_plus_text", "text_effect"]]
        .round(3)
        .to_string(index=False)
    )

    coder = float(split["coder_effect"].mean())
    text = float(split["text_effect"].mean())

    print(
        f"\nAveraged over variables: coder difference {coder:.3f} of "
        f"cells, withheld text\nadds {text:.3f}. Truncation is real and "
        "second-order; coder indeterminacy dominates."
    )

    print(
        "\nCaution: 12 control speeches, 72 cells. Every interval above "
        "is wide, and the\ndecomposition assumes the coder difference "
        "is the same on both groups."
    )

    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    rates.to_csv(
        VALIDATION_DIR / "coder_base_rates.csv", index=False
    )

    write_base_rate_tex(rates, TABLE_DIR / "coder_base_rates.tex")

    three_way.to_csv(
        VALIDATION_DIR / "coder_reliability_three_way.csv", index=False
    )
    model_b.to_csv(
        VALIDATION_DIR / "coder_reliability_model_vs_B.csv", index=False
    )
    split.to_csv(
        VALIDATION_DIR / "coder_reliability_decomposition.csv", index=False
    )

    print(f"\nSaved to: {VALIDATION_DIR}")
    print(f"Table: {TABLE_DIR / 'coder_base_rates.tex'}")


if __name__ == "__main__":
    main()
