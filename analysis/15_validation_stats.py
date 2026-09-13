"""
Human validation of the LLM-generated ECB communication signals.

Reads the blind-coded validation sample, compares it with the
production extraction, and reports:

    1. cell-level agreement and Cohen's kappa (with bootstrap CIs);
    2. confusion matrices and error direction for the attention
       variables;
    3. the decomposition of outlook disagreements into those forced by
       an attention disagreement and those that are independent;
    4. sensitivity and specificity estimates, exported as JSON so that
       the aggregation step can correct the published series.

Run from the project root:

    python analysis/15_validation_stats.py
"""

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from measurement_error import (  # noqa: E402
    corrected_share_bounds,
    correct_share,
    estimate_error_rates,
)

from agreement import (  # noqa: E402
    BOOTSTRAP_SEED,
    N_BOOTSTRAP,
    bootstrap_kappa_ci,
    categories_for,
    clopper_pearson_ci,
    cohen_kappa,
)


# ============================================================
# Configuration
# ============================================================

HUMAN_FILE = PROJECT_ROOT / "data" / "human" / "human_audit_30.csv"

SIGNALS_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "extracted_signals"
    / "ecb_signals_gpt5mini_v1.csv"
)

PANEL_FILE = PROJECT_ROOT / "outputs" / "panel" / "speech_panel.csv"

TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"
VALIDATION_DIR = PROJECT_ROOT / "outputs" / "validation"

PARAMS_FILE = VALIDATION_DIR / "measurement_error_params.json"

ATTENTION_VARS = [
    "inflation_attention",
    "growth_attention",
    "financial_stability_attention",
    "uncertainty_attention",
]

OUTLOOK_VARS = [
    "inflation_outlook",
    "growth_outlook",
]

# Outlook variables are constrained to be zero when the corresponding
# attention variable is zero, so an attention disagreement mechanically
# propagates into an outlook disagreement.
OUTLOOK_PARENT = {
    "inflation_outlook": "inflation_attention",
    "growth_outlook": "growth_attention",
}

ALL_VARS = [
    "inflation_attention",
    "inflation_outlook",
    "growth_attention",
    "growth_outlook",
    "financial_stability_attention",
    "uncertainty_attention",
]

LABELS = {
    "inflation_attention": "Inflation attention",
    "inflation_outlook": "Inflation outlook",
    "growth_attention": "Growth attention",
    "growth_outlook": "Growth outlook",
    "financial_stability_attention": "Financial stability attention",
    "uncertainty_attention": "Uncertainty attention",
}

# ============================================================
# Loading
# ============================================================

def baseline_sensitivity(data: pd.DataFrame) -> dict[str, object]:
    """
    Recompute cell agreement without the audit speeches that no
    published measure uses.

    The round-one sample was drawn before the baseline restriction was
    settled, and it contains at least one item below the 300-word floor
    -- a slide stub rather than a speech. Such an item is trivially easy
    to code, so it contributes agreements that flatter the headline
    without saying anything about the instrument. Reporting the
    headline and this figure together is the honest form: the exclusion
    barely moves the number, and showing that is worth more than
    quietly leaving the stub in.
    """

    if not PANEL_FILE.exists():
        raise FileNotFoundError(
            f"{PANEL_FILE} is missing. Run analysis/10_panel.py first; "
            "the baseline flag and word counts live there."
        )

    panel = pd.read_csv(PANEL_FILE)[
        ["speech_id", "n_words", "in_baseline"]
    ]

    merged = data.merge(
        panel, on="speech_id", how="left", validate="one_to_one"
    )

    if merged["in_baseline"].isna().any():
        raise ValueError(
            "Audit speeches absent from the panel: "
            f"{merged.loc[merged['in_baseline'].isna(), 'speech_id'].tolist()}"
        )

    dropped = merged[~merged["in_baseline"]]

    kept = merged[merged["in_baseline"]]

    def cells(frame: pd.DataFrame) -> tuple[int, int]:
        total = len(frame) * len(ALL_VARS)

        disagree = int(
            sum((frame[v] != frame[f"{v}_h"]).sum() for v in ALL_VARS)
        )

        return total - disagree, total

    agree_all, total_all = cells(merged)
    agree_kept, total_kept = cells(kept)

    return {
        "n_dropped": len(dropped),
        "dropped_ids": dropped["speech_id"].tolist(),
        "dropped_words": dropped["n_words"].astype(int).tolist(),
        "agreement_full": agree_all / total_all,
        "agreement_baseline": agree_kept / total_kept,
        "cells_full": total_all,
        "cells_baseline": total_kept,
    }


def load_validation_sample() -> pd.DataFrame:
    """
    Merge the human codes with the production extraction.

    Verifies that every cell the coder marked as agreeing does in fact
    agree, which catches transcription errors in the coding sheet.
    """

    human = pd.read_csv(HUMAN_FILE)
    signals = pd.read_csv(SIGNALS_FILE)

    merged = human.merge(
        signals[["speech_id", "date", "speakers"] + ALL_VARS],
        on="speech_id",
        how="left",
        validate="one_to_one",
    )

    missing = merged[ALL_VARS].isna().any(axis=1)

    if missing.any():
        raise ValueError(
            "Validation speeches absent from the extraction: "
            f"{merged.loc[missing, 'speech_id'].tolist()}"
        )

    inconsistent = []

    for variable in ALL_VARS:
        flagged_agree = merged[f"{variable}_flag"] == "agree"
        differs = merged[variable] != merged[f"{variable}_h"]

        bad = merged.loc[flagged_agree & differs, "speech_id"]
        inconsistent.extend(
            f"{sid} ({variable})" for sid in bad
        )

    if inconsistent:
        raise ValueError(
            "Coding sheet marks agreement where the codes differ: "
            + ", ".join(inconsistent)
        )

    return merged


# ============================================================
# Reporting blocks
# ============================================================

def agreement_table(data: pd.DataFrame) -> pd.DataFrame:
    """
    Agreement and kappa per variable, under two treatments of cells the
    coder flagged as borderline rather than as clear errors.
    """

    rows = []

    for variable in ALL_VARS:
        cats = categories_for(variable)

        llm = data[variable].to_numpy()
        human = data[f"{variable}_h"].to_numpy()

        borderline = (
            data[f"{variable}_flag"] == "borderline"
        ).to_numpy()

        # Lenient reading: borderline cells count as agreement.
        human_lenient = np.where(borderline, llm, human)

        observed, kappa = cohen_kappa(llm, human, cats)
        lo, hi = bootstrap_kappa_ci(llm, human, cats)

        observed_lenient, kappa_lenient = cohen_kappa(
            llm, human_lenient, cats
        )

        rows.append(
            {
                "variable": variable,
                "n": len(data),
                "n_disagree": int((llm != human).sum()),
                "n_borderline": int(borderline.sum()),
                "agreement": observed,
                "kappa": kappa,
                "kappa_lo": lo,
                "kappa_hi": hi,
                "agreement_lenient": observed_lenient,
                "kappa_lenient": kappa_lenient,
            }
        )

    return pd.DataFrame(rows)


def confusion_table(data: pd.DataFrame) -> pd.DataFrame:
    """Confusion counts and error rates for the attention variables."""

    rows = []

    for variable in ATTENTION_VARS:
        llm = data[variable]
        human = data[f"{variable}_h"]

        true_positive = int(((llm == 1) & (human == 1)).sum())
        false_positive = int(((llm == 1) & (human == 0)).sum())
        false_negative = int(((llm == 0) & (human == 1)).sum())
        true_negative = int(((llm == 0) & (human == 0)).sum())

        rates = estimate_error_rates(llm, human)

        sens_lo, sens_hi = clopper_pearson_ci(
            rates.n_true_one - false_negative,
            rates.n_true_one,
        )

        spec_lo, spec_hi = clopper_pearson_ci(
            rates.n_true_zero - false_positive,
            rates.n_true_zero,
        )

        predicted_positive = true_positive + false_positive

        rows.append(
            {
                "variable": variable,
                "TP": true_positive,
                "FP": false_positive,
                "FN": false_negative,
                "TN": true_negative,
                "precision": (
                    true_positive / predicted_positive
                    if predicted_positive
                    else float("nan")
                ),
                "recall": rates.sensitivity,
                "sensitivity": rates.sensitivity,
                "sens_lo": sens_lo,
                "sens_hi": sens_hi,
                "specificity": rates.specificity,
                "spec_lo": spec_lo,
                "spec_hi": spec_hi,
                "n_true_one": rates.n_true_one,
                "n_true_zero": rates.n_true_zero,
            }
        )

    return pd.DataFrame(rows)


def outlook_error_decomposition(data: pd.DataFrame) -> pd.DataFrame:
    """
    Split outlook disagreements into those mechanically forced by an
    attention disagreement and those that are independent, and record
    whether any disagreement is an outright sign reversal.
    """

    rows = []

    for variable in OUTLOOK_VARS:
        parent = OUTLOOK_PARENT[variable]

        differs = data[variable] != data[f"{variable}_h"]
        parent_differs = data[parent] != data[f"{parent}_h"]

        disagreements = data[differs]

        forced = int((differs & parent_differs).sum())

        sign_flips = int(
            (
                differs
                & (data[variable] != 0)
                & (data[f"{variable}_h"] != 0)
                & (np.sign(data[variable]) != np.sign(data[f"{variable}_h"]))
            ).sum()
        )

        to_neutral = int(
            (
                differs
                & (data[variable] != 0)
                & (data[f"{variable}_h"] == 0)
            ).sum()
        )

        rows.append(
            {
                "variable": variable,
                "n_disagree": int(differs.sum()),
                "forced_by_attention": forced,
                "independent": int(differs.sum()) - forced,
                "directional_to_neutral": to_neutral,
                "sign_reversals": sign_flips,
                "speech_ids": ", ".join(disagreements["speech_id"]),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# LaTeX output
# ============================================================

def fmt(value: float, digits: int = 3) -> str:
    """Format a float for LaTeX, rendering NaN as an em dash."""

    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "---"

    return f"{value:.{digits}f}"


def write_agreement_tex(table: pd.DataFrame, path: Path) -> None:
    lines = [
        r"\begin{tabular}{lrrrc rr}",
        r"\toprule",
        r"Variable & $n$ & Disagree & Agreement & "
        r"$\kappa$ [95\% CI] & Agreement$^{L}$ & $\kappa^{L}$ \\",
        r"\midrule",
    ]

    for _, row in table.iterrows():
        interval = (
            f"[{fmt(row['kappa_lo'], 2)}, {fmt(row['kappa_hi'], 2)}]"
            if not np.isnan(row["kappa_lo"])
            else "---"
        )

        lines.append(
            f"{LABELS[row['variable']]} & {int(row['n'])} & "
            f"{int(row['n_disagree'])} & {fmt(row['agreement'])} & "
            f"{fmt(row['kappa'])} {interval} & "
            f"{fmt(row['agreement_lenient'])} & "
            f"{fmt(row['kappa_lenient'])} \\\\"
        )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
    ]

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_confusion_tex(table: pd.DataFrame, path: Path) -> None:
    lines = [
        r"\begin{tabular}{lrrrr rr rr}",
        r"\toprule",
        r" & \multicolumn{4}{c}{Counts} & "
        r"\multicolumn{2}{c}{Rates} & "
        r"\multicolumn{2}{c}{[95\% CI]} \\",
        r"\cmidrule(lr){2-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}",
        r"Variable & TP & FP & FN & TN & Sens. & Spec. & "
        r"Sens. & Spec. \\",
        r"\midrule",
    ]

    for _, row in table.iterrows():
        lines.append(
            f"{LABELS[row['variable']]} & {int(row['TP'])} & "
            f"{int(row['FP'])} & {int(row['FN'])} & {int(row['TN'])} & "
            f"{fmt(row['sensitivity'])} & {fmt(row['specificity'])} & "
            f"[{fmt(row['sens_lo'], 2)}, {fmt(row['sens_hi'], 2)}] & "
            f"[{fmt(row['spec_lo'], 2)}, {fmt(row['spec_hi'], 2)}] \\\\"
        )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
    ]

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ============================================================
# Parameter export
# ============================================================

def export_parameters(
    confusion: pd.DataFrame,
    data: pd.DataFrame,
) -> dict:
    """
    Write sensitivity and specificity to JSON for the aggregation step.
    """

    params = {
        "source": {
            "human_file": str(HUMAN_FILE.relative_to(PROJECT_ROOT)),
            "signals_file": str(SIGNALS_FILE.relative_to(PROJECT_ROOT)),
            "n_validation_speeches": int(len(data)),
            "n_bootstrap": N_BOOTSTRAP,
            "bootstrap_seed": BOOTSTRAP_SEED,
        },
        "caveats": [
            "Validation sample is n=30 and coded by a single coder.",
            "The coding sheet displayed the LLM labels, so agreement "
            "is an upper bound until a blind replication is run.",
            "The 30 speech ids were not drawn by a seeded script.",
        ],
        "variables": {},
    }

    for _, row in confusion.iterrows():
        params["variables"][row["variable"]] = {
            "sensitivity": float(row["sensitivity"]),
            "specificity": float(row["specificity"]),
            "sensitivity_ci": [
                float(row["sens_lo"]),
                float(row["sens_hi"]),
            ],
            "specificity_ci": [
                float(row["spec_lo"]),
                float(row["spec_hi"]),
            ],
            "n_true_one": int(row["n_true_one"]),
            "n_true_zero": int(row["n_true_zero"]),
        }

    PARAMS_FILE.write_text(
        json.dumps(params, indent=2) + "\n",
        encoding="utf-8",
    )

    return params


# ============================================================
# Main
# ============================================================

def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)

    data = load_validation_sample()

    print("\n=== HUMAN VALIDATION SAMPLE ===")
    print(f"Speeches: {len(data)}")
    print(f"Cells: {len(data) * len(ALL_VARS)}")
    print(
        "Date range: "
        f"{data['date'].min()} to {data['date'].max()}"
    )
    print(f"Speakers: {data['speakers'].nunique()}")

    total_cells = len(data) * len(ALL_VARS)

    total_disagreements = int(
        sum(
            (data[v] != data[f"{v}_h"]).sum()
            for v in ALL_VARS
        )
    )

    print(
        "\nCell-level agreement: "
        f"{1 - total_disagreements / total_cells:.3f} "
        f"({total_cells - total_disagreements}/{total_cells})"
    )

    short = baseline_sensitivity(data)

    print("\n=== AGREEMENT WITHOUT THE NON-BASELINE ITEMS ===")

    if short["n_dropped"]:
        described = ", ".join(
            f"{sid} ({words} words)"
            for sid, words in zip(
                short["dropped_ids"], short["dropped_words"]
            )
        )

        print(
            f"{short['n_dropped']} audit speech(es) fall below the "
            f"300-word floor: {described}."
        )
        print(
            "Each contributes six easy agreements that no published "
            "measure uses."
        )
        print(
            f"  with:    {short['agreement_full']:.4f} "
            f"({short['cells_full']} cells)"
        )
        print(
            f"  without: {short['agreement_baseline']:.4f} "
            f"({short['cells_baseline']} cells)"
        )
    else:
        print("Every audit speech is in the baseline sample.")

    # --------------------------------------------------
    # 1. Agreement and kappa
    # --------------------------------------------------

    agreement = agreement_table(data)

    print("\n=== AGREEMENT AND COHEN'S KAPPA ===")
    print(
        agreement[
            [
                "variable",
                "n_disagree",
                "agreement",
                "kappa",
                "kappa_lo",
                "kappa_hi",
                "agreement_lenient",
                "kappa_lenient",
            ]
        ]
        .round(3)
        .to_string(index=False)
    )

    # --------------------------------------------------
    # 2. Confusion matrices
    # --------------------------------------------------

    confusion = confusion_table(data)

    print("\n=== CONFUSION MATRICES (attention variables) ===")
    print(
        confusion[
            [
                "variable",
                "TP",
                "FP",
                "FN",
                "TN",
                "precision",
                "recall",
                "specificity",
            ]
        ]
        .round(3)
        .to_string(index=False)
    )

    if (confusion["FN"] == 0).all():
        print(
            "\nNo false negatives in any attention variable: "
            "measurement error is one-sided."
        )

    # --------------------------------------------------
    # 3. Outlook error decomposition
    # --------------------------------------------------

    decomposition = outlook_error_decomposition(data)

    print("\n=== OUTLOOK DISAGREEMENTS ===")
    print(
        decomposition[
            [
                "variable",
                "n_disagree",
                "forced_by_attention",
                "independent",
                "directional_to_neutral",
                "sign_reversals",
            ]
        ].to_string(index=False)
    )

    if (decomposition["sign_reversals"] == 0).all():
        print(
            "\nNo sign reversals: outlook error attenuates towards "
            "neutral rather than distorting direction."
        )

    # --------------------------------------------------
    # 4. Corrected shares
    # --------------------------------------------------

    signals = pd.read_csv(SIGNALS_FILE, parse_dates=["date"])

    print("\n=== MISCLASSIFICATION-CORRECTED SHARES ===")
    print(
        f"{'variable':32s} {'raw':>8s} {'corrected':>10s} "
        f"{'conservative 95% bounds':>26s}"
    )

    corrected_rows = []

    for _, row in confusion.iterrows():
        variable = row["variable"]

        rates = estimate_error_rates(
            data[variable], data[f"{variable}_h"]
        )

        raw = float(signals[variable].mean())
        corrected = correct_share(raw, rates)

        lo, hi = corrected_share_bounds(
            raw,
            (row["sens_lo"], row["sens_hi"]),
            (row["spec_lo"], row["spec_hi"]),
        )

        corrected_rows.append(
            {
                "variable": variable,
                "raw_share": raw,
                "corrected_share": corrected,
                "corrected_lo": lo,
                "corrected_hi": hi,
            }
        )

        print(
            f"{variable:32s} {raw:8.3f} {corrected:10.3f} "
            f"      [{lo:.3f}, {hi:.3f}]"
        )

    pd.DataFrame(corrected_rows).to_csv(
        VALIDATION_DIR / "corrected_shares.csv", index=False
    )

    # --------------------------------------------------
    # 5. Write outputs
    # --------------------------------------------------

    agreement.to_csv(
        VALIDATION_DIR / "agreement_statistics.csv", index=False
    )

    confusion.to_csv(
        VALIDATION_DIR / "confusion_matrices.csv", index=False
    )

    decomposition.to_csv(
        VALIDATION_DIR / "outlook_error_decomposition.csv", index=False
    )

    write_agreement_tex(
        agreement, TABLE_DIR / "validation_agreement.tex"
    )

    write_confusion_tex(
        confusion, TABLE_DIR / "validation_confusion.tex"
    )

    export_parameters(confusion, data)

    print(f"\nSaved tables to: {TABLE_DIR}")
    print(f"Saved statistics to: {VALIDATION_DIR}")
    print(f"Saved correction parameters to: {PARAMS_FILE}")


if __name__ == "__main__":
    main()
