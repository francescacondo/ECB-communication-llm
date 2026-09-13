"""
The human-human ceiling under the revised coding rules.

Round 3 established a ceiling of 0.694 on twelve speeches, and showed
that the two worst variables owed their disagreement to a corrupted
codebook. That corruption was repaired and six decision rules were added
by adjudication, producing version 2 of docs/coding_rules.md.

Round 4 tests whether the revision worked. Fifty speeches nobody had
coded before, drawn with flat weights so the sample represents the
corpus, coded independently and blind by both coders from the same
written rules on identical full text.

Two comparisons matter and they are not equally clean.

**Per-variable kappa against round 3** is the test the revision was
designed to pass. Financial stability, inflation outlook and uncertainty
had kappas of -0.154, 0.059 and 0.182. If the six rules made the task
determinate, those should move.

**Pooled agreement against round 3's 0.694** is weaker evidence, because
three things changed at once: the rules, the speeches, and the sampling
weights. Round 3's sample deliberately over-represented speeches the
model codes one, which is where disagreement concentrates, so some
improvement is expected from the sampling change alone. The script
reports round 4 restricted to high-attention-count speeches as a partial
control.

The model's codes are also compared, with a caveat that must travel with
the number: the extraction ran under version 1 and never saw these
rules, so it is being scored against a specification it was not given.

Run from the project root:

    python analysis/19_round4_ceiling.py
"""

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from agreement import (  # noqa: E402
    bootstrap_kappa_ci,
    categories_for,
    clopper_pearson_ci,
    cohen_kappa,
)

CODER_1_FILE = PROJECT_ROOT / "data" / "human" / "round4_coder_1.csv"

CODER_2_FILE = PROJECT_ROOT / "data" / "human" / "round4_coder_2.csv"

KEY_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "validation"
    / "validation_sample_r4_key.csv"
)

SHEET_FILE = (
    PROJECT_ROOT / "outputs" / "audit" / "coding_sheet_r4_full.csv"
)

VALIDATION_DIR = PROJECT_ROOT / "outputs" / "validation"

FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"

PANEL_FILE = PROJECT_ROOT / "outputs" / "panel" / "speech_panel.csv"

# The speeches round 4 could not draw, because rounds 1 and 2 had
# already been coded on them. Mirrors ROUNDS[4]["exclude"] in
# src/select_validation_sample.py; if that list changes, this must.
EXCLUDED_FILES = [
    PROJECT_ROOT / "data" / "human" / "human_audit_30.csv",
    PROJECT_ROOT / "outputs" / "audit" / "validation_sample_r2_blind.csv",
]

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

# Round 3, twelve speeches, identical text, corrupted rules.
ROUND3_KAPPA = {
    "financial_stability_attention": -0.154,
    "inflation_outlook": 0.059,
    "uncertainty_attention": 0.182,
    "growth_attention": 0.500,
    "inflation_attention": 0.636,
    "growth_outlook": 0.714,
}

ROUND3_POOLED = 0.694


def load() -> pd.DataFrame:
    one = pd.read_csv(CODER_1_FILE)
    two = pd.read_csv(CODER_2_FILE)
    key = pd.read_csv(KEY_FILE)
    sheet = pd.read_csv(SHEET_FILE)[["speech_id", "was_truncated"]]

    for name, frame in [("coder 1", one), ("coder 2", two)]:
        if frame[ALL_VARS].isna().any().any():
            raise ValueError(f"{name} has uncoded cells.")

        for variable in ALL_VARS:
            allowed = set(categories_for(variable))
            found = set(frame[variable].unique())

            if not found <= allowed:
                raise ValueError(
                    f"{name}: {variable} has values outside "
                    f"{sorted(allowed)}"
                )

        for outlook, attention in [
            ("inflation_outlook", "inflation_attention"),
            ("growth_outlook", "growth_attention"),
        ]:
            bad = frame[(frame[attention] == 0) & (frame[outlook] != 0)]

            if not bad.empty:
                raise ValueError(
                    f"{name} violates the outlook constraint for "
                    f"{bad['speech_id'].tolist()}"
                )

    merged = (
        one[["speech_id"] + ALL_VARS]
        .merge(two[["speech_id"] + ALL_VARS], on="speech_id",
               suffixes=("_1", "_2"), validate="one_to_one")
        .merge(key[["speech_id"] + ALL_VARS], on="speech_id",
               validate="one_to_one")
        .merge(sheet, on="speech_id", validate="one_to_one")
    )

    if len(merged) != len(one):
        raise ValueError("Coders disagree on which speeches were coded.")

    merged["attention_count"] = merged[ATTENTION_VARS].sum(axis=1)

    return merged


def pooled(data: pd.DataFrame, left: str, right: str) -> tuple[int, int]:
    agree = sum(
        int((data[f"{v}{left}"] == data[f"{v}{right}"]).sum())
        for v in ALL_VARS
    )

    return agree, len(data) * len(ALL_VARS)


def ceiling_table(data: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for variable in ALL_VARS:
        x = data[f"{variable}_1"].to_numpy()
        y = data[f"{variable}_2"].to_numpy()

        cats = categories_for(variable)

        observed, kappa = cohen_kappa(x, y, cats)
        lo, hi = bootstrap_kappa_ci(x, y, cats)

        rows.append(
            {
                "variable": variable,
                "label": LABELS[variable],
                "agreement": observed,
                "kappa_r4": kappa,
                "kappa_lo": lo,
                "kappa_hi": hi,
                "kappa_r3": ROUND3_KAPPA[variable],
                "change": kappa - ROUND3_KAPPA[variable],
            }
        )

    return pd.DataFrame(rows)


def model_table(data: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for variable in ALL_VARS:
        cats = categories_for(variable)
        model = data[variable].to_numpy()

        for coder in ("1", "2"):
            human = data[f"{variable}_{coder}"].to_numpy()
            observed, kappa = cohen_kappa(human, model, cats)

            rows.append(
                {
                    "variable": variable,
                    "label": LABELS[variable],
                    "coder": coder,
                    "agreement": observed,
                    "kappa": kappa,
                }
            )

    return pd.DataFrame(rows)


def frame_rates(data: pd.DataFrame) -> pd.DataFrame:
    """
    Model-positive rates in the round-4 sample, in the frame it was
    drawn from, and in the baseline corpus.

    Round 4 used flat weights, so the sample should look like the frame
    it was drawn from. Checking that is the only evidence available
    that the realised draw is not accidentally concentrated somewhere
    unrepresentative -- the round has no stratification to appeal to.

    The frame and the baseline are reported separately because they are
    not the same population and differ in the third decimal. The frame
    is the baseline less the eighty-nine speeches already coded in
    rounds 1 and 2, and it is the frame, not the corpus, that a flat
    draw is representative of. Reporting one and calling it the other
    is an easy mistake to make and an invisible one to catch.
    """

    if not PANEL_FILE.exists():
        raise FileNotFoundError(
            f"{PANEL_FILE} is missing. Run analysis/10_panel.py first."
        )

    panel = pd.read_csv(PANEL_FILE)

    missing = [path for path in EXCLUDED_FILES if not path.exists()]

    if missing:
        raise FileNotFoundError(
            "Cannot rebuild the round-4 frame; these exclusion files "
            f"are absent: {[str(path) for path in missing]}"
        )

    already = pd.concat(
        [pd.read_csv(path)["speech_id"] for path in EXCLUDED_FILES]
    ).drop_duplicates()

    baseline = panel[panel["in_baseline"]]

    frame = baseline[~baseline["speech_id"].isin(already)]

    rows = []

    for variable in ATTENTION_VARS:
        rows.append(
            {
                "variable": variable,
                "label": LABELS[variable],
                "sample": float(data[variable].mean()),
                "frame": float(frame[variable].mean()),
                "baseline": float(baseline[variable].mean()),
                "sample_minus_frame": float(
                    data[variable].mean() - frame[variable].mean()
                ),
            }
        )

    out = pd.DataFrame(rows)

    out.attrs["n_sample"] = len(data)
    out.attrs["n_frame"] = len(frame)
    out.attrs["n_baseline"] = len(baseline)

    return out


def plot_ceiling(ceiling: pd.DataFrame, path: Path) -> None:
    """
    Round-3 and round-4 human-human kappa per variable, with round 4's
    interval drawn.

    The table this replaces gives every variable the same visual
    weight, and two of its rows should not have it. Financial stability
    moves +0.714 and uncertainty +0.098, but uncertainty's round-4
    interval runs from below zero to 0.544, so the small move and the
    large one are not known with remotely the same precision. On a
    table both are one cell high and a reader will take the point
    estimates at face value. Drawing the interval is the whole purpose
    of the figure; the arrows are secondary.

    Round 3 has no interval because it rests on twelve speeches and
    seventy-two cells, which is too few to be worth drawing next to a
    fifty-speech estimate. Its absence is the point: the comparison
    across rounds is descriptive, and the figure should not dress it up
    as two estimates of the same quantity.
    """

    order = ceiling.sort_values("kappa_r4")

    y = np.arange(len(order))

    figure, axis = plt.subplots(figsize=(9, 4.6))

    axis.axvline(0, color="0.75", linewidth=0.8)

    axis.hlines(
        y,
        order["kappa_lo"],
        order["kappa_hi"],
        color="#4c72b0",
        linewidth=2.4,
        alpha=0.45,
        zorder=1,
    )

    for row, position in zip(order.itertuples(), y):
        axis.annotate(
            "",
            xy=(row.kappa_r4, position),
            xytext=(row.kappa_r3, position),
            arrowprops=dict(
                arrowstyle="->",
                color="0.55",
                linewidth=0.9,
                shrinkA=0,
                shrinkB=4,
            ),
            zorder=2,
        )

    axis.scatter(
        order["kappa_r3"], y,
        s=42, color="0.45", marker="o",
        label="Round 3 (12 speeches, pre-revision rules)",
        zorder=3,
    )

    axis.scatter(
        order["kappa_r4"], y,
        s=58, color="#c44e52", marker="D",
        label="Round 4 (50 speeches, revised rules), 95% interval",
        zorder=4,
    )

    axis.set_yticks(y)
    axis.set_yticklabels(order["label"])

    axis.set_xlabel("Cohen's $\\kappa$, coder 1 against coder 2")
    axis.set_title(
        "Inter-coder reliability by variable, before and after the "
        "codebook revision",
        fontsize=11,
    )

    axis.grid(axis="x", alpha=0.2, linewidth=0.5)
    # Upper left: the highest-kappa variable sorts to the top, so that
    # corner is the only one no interval reaches.
    axis.legend(frameon=False, fontsize=8, loc="upper left")

    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def main() -> None:
    data = load()

    print("\n=== ROUND 4 ===")
    print(f"Speeches: {len(data)}, coded independently by two coders.")
    print(
        f"Truncated at extraction: "
        f"{int(data['was_truncated'].sum())} "
        f"({data['was_truncated'].mean():.0%}) -- both coders read the "
        "full speech."
    )

    rates = frame_rates(data)

    print("\n=== IS THE FLAT DRAW REPRESENTATIVE? ===")
    print(
        f"Model-positive rates. Sample n = {rates.attrs['n_sample']}, "
        f"frame n = {rates.attrs['n_frame']:,}, "
        f"baseline n = {rates.attrs['n_baseline']:,}."
    )
    print(
        "The frame is the baseline less the speeches rounds 1 and 2 had "
        "already used;\nit, not the corpus, is what a flat draw "
        "represents."
    )
    print(
        rates.drop(columns=["variable"]).round(3).to_string(index=False)
    )

    agree, total = pooled(data, "_1", "_2")
    lo, hi = clopper_pearson_ci(agree, total)

    print("\n=== THE CEILING ===")
    print(
        f"Coder 1 vs coder 2: {agree}/{total} = {agree / total:.3f} "
        f"[{lo:.3f}, {hi:.3f}]"
    )
    print(
        f"Round 3, twelve speeches, corrupted rules: {ROUND3_POOLED:.3f}"
    )

    dense = data[data["attention_count"] >= 3]

    d_agree, d_total = pooled(dense, "_1", "_2")

    print(
        f"\nRestricted to speeches the model codes 1 on three or more "
        f"topics,\nwhich is where round 3's sample was concentrated: "
        f"{d_agree}/{d_total} = {d_agree / d_total:.3f} "
        f"(n = {len(dense)})"
    )
    print(
        "The sampling change therefore accounts for part of any "
        "improvement,\nand this row is the fairer comparison with "
        "round 3."
    )

    ceiling = ceiling_table(data)

    print("\n=== PER VARIABLE: DID THE SIX RULES WORK? ===")
    print(
        ceiling[
            ["label", "agreement", "kappa_r3", "kappa_r4",
             "kappa_lo", "kappa_hi", "change"]
        ]
        .round(3)
        .to_string(index=False)
    )

    broken = ["financial_stability_attention", "inflation_outlook",
              "uncertainty_attention"]

    fixed = ceiling[ceiling["variable"].isin(broken)]

    print(
        "\nThe three variables the revision targeted moved from "
        f"{fixed['kappa_r3'].mean():.3f} to "
        f"{fixed['kappa_r4'].mean():.3f} on average."
    )

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    figure_path = FIGURE_DIR / "round4_ceiling.png"

    plot_ceiling(ceiling, figure_path)

    print(f"\nSaved figure to: {figure_path}")

    model = model_table(data)

    print("\n=== MODEL AGAINST EACH CODER ===")
    print(
        "The extraction ran under version 1 and never saw the six "
        "rules below which\nthese humans coded. It is being scored "
        "against a specification it was not given."
    )
    print(
        model.pivot(index="label", columns="coder", values="kappa")
        .round(3)
        .to_string()
    )

    for coder in ("1", "2"):
        m_agree, m_total = pooled(data, "", f"_{coder}")
        print(
            f"  pooled, model vs coder {coder}: "
            f"{m_agree}/{m_total} = {m_agree / m_total:.3f}"
        )

    pd.DataFrame(
        [
            {
                "sample": "all speeches",
                "n_speeches": len(data),
                "cells_agreeing": agree,
                "cells": total,
                "agreement": agree / total,
                "ci_low": lo,
                "ci_high": hi,
            },
            {
                "sample": "attention count >= 3",
                "n_speeches": len(dense),
                "cells_agreeing": d_agree,
                "cells": d_total,
                "agreement": d_agree / d_total,
                "ci_low": float("nan"),
                "ci_high": float("nan"),
            },
        ]
    ).to_csv(VALIDATION_DIR / "round4_pooled_agreement.csv", index=False)

    ceiling.to_csv(
        VALIDATION_DIR / "round4_ceiling.csv", index=False
    )
    model.to_csv(
        VALIDATION_DIR / "round4_model_vs_coders.csv", index=False
    )

    rates.to_csv(
        VALIDATION_DIR / "round4_frame_rates.csv", index=False
    )

    print(f"\nSaved to: {VALIDATION_DIR}")


if __name__ == "__main__":
    main()
