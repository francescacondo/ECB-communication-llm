"""
Draw the second-round validation sample.

The first audit (data/human/human_audit_30.csv) has four documented
weaknesses: it was not blind, it had one coder, it was small, and it
was not drawn by a seeded script. This module fixes the first and the
fourth, and sizes the draw so that the third is materially improved.
The second requires a person other than the coder and cannot be fixed
in code.

Three design choices carry the work.

**Stratified on the model's attention count, oversampling the top.**
The audit found sensitivity of 1.000 for every attention variable, so
every observed error is a false positive. False positives can only be
found among speeches the model codes one. Specificity is therefore the
parameter the data constrain least -- the interval for financial
stability runs from 0.715 to 1.000 despite thirty agreements out of
thirty -- and the sample is allocated towards speeches where the model
codes more topics present. This is a deliberate departure from
proportional sampling and means the raw agreement rate in this sample
is NOT an estimate of corpus-wide agreement. It is designed to
estimate the error rates, which is a different quantity.

**Crossed with era.** Without this the draw could concentrate in one
period, and speech style changes enough across presidencies that a
period-specific quirk could be mistaken for a general error rate.

**Blind by construction.** The coding sheet carries the speech text and
empty code columns and no model codes anywhere. The model codes go to a
separate key file that the coder does not open until coding is
complete. Nothing enforces that discipline except the discipline
itself, but at least the sheet cannot leak the answer by accident, as
the first round's did.

The thirty speeches already coded are excluded. They become a
development set: if the prompt is ever revised, accuracy should be
reported on this new sample and not on the one used to find the
problem.

**How this sample must be analysed.** Because selection depends on the
model's codes, the naive sample proportions are NOT unbiased estimates
of sensitivity and specificity, and scoring this sample the way the
first round was scored would give the wrong answer. Selection is
independent of the human code given the model code, so what the sample
estimates without bias is the other conditional direction:

    PPV = Pr(human = 1 | model = 1),
    NPV = Pr(human = 0 | model = 0).

The rates of interest are recovered from these by Bayes, using the
model's marginal base rate q = Pr(model = 1), which is not estimated
but known exactly from all 2,717 baseline speeches:

    p     = PPV * q + (1 - NPV) * (1 - q)
    alpha = PPV * q / p
    beta  = NPV * (1 - q) / (1 - p)

This is what makes the oversampling legitimate rather than merely
convenient: the design buys precision where the errors are, and the
known marginal buys back the unbiasedness that stratifying on the model
code would otherwise cost. The scoring script for this round must
implement the three lines above; it must not reuse
estimate_error_rates() from src/measurement_error.py, which assumes a
sample drawn without reference to the model codes.

Run from the project root:

    python src/select_validation_sample.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PANEL_FILE = PROJECT_ROOT / "outputs" / "panel" / "speech_panel.csv"

EXCERPT_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "production"
    / "ecb_speeches_1999_2025_excerpts.csv"
)

# Each round excludes every speech coded in earlier rounds, so that the
# ceiling is estimated on speeches neither coder has seen. Round 2's
# parameters are kept so its draw stays reproducible.
ROUNDS = {
    2: {
        "seed": 20260826,
        "size": 60,
        "exclude": ["data/human/human_audit_30.csv"],
        # Oversampled speeches the model codes one, because the round's
        # purpose was estimating specificity and every observed error
        # was a false positive.
        "weights": {0: 0.9, 1: 0.9, 2: 1.0, 3: 1.3, 4: 1.6},
    },
    4: {
        "seed": 20260828,
        "size": 50,
        "exclude": [
            "data/human/human_audit_30.csv",
            "outputs/audit/validation_sample_r2_blind.csv",
        ],
        # Flat. This round estimates the human-human ceiling, which is a
        # property of the coding task rather than of the model, so the
        # sample should represent the corpus rather than concentrate on
        # cells where the model is most likely wrong. Stratifying on the
        # model's codes would also make the ceiling conditional on them,
        # which is precisely the dependence this round exists to remove.
        "weights": {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0},
    },
}

ACTIVE_ROUND = 4

# The sheet carries ECB speech text, so it lives with the other audit
# material rather than in the tracked data directory.
AUDIT_DIR = PROJECT_ROOT / "outputs" / "audit"

BLIND_FILE = AUDIT_DIR / f"validation_sample_r{ACTIVE_ROUND}_blind.csv"

KEY_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "validation"
    / f"validation_sample_r{ACTIVE_ROUND}_key.csv"
)

# Committed here so the draw is reproducible. Changing it after seeing
# the sample would defeat the purpose of fixing it at all.
SEED = ROUNDS[ACTIVE_ROUND]["seed"]

SAMPLE_SIZE = ROUNDS[ACTIVE_ROUND]["size"]

ATTENTION_VARS = [
    "inflation_attention",
    "growth_attention",
    "financial_stability_attention",
    "uncertainty_attention",
]

OUTLOOK_VARS = ["inflation_outlook", "growth_outlook"]

ALL_VARS = ATTENTION_VARS + OUTLOOK_VARS

ERA_BREAK_YEAR = 2012

# Relative sampling weight by the number of attention variables the
# model codes one, set per round above because the right choice depends
# on what the round is for.
# The weights are deliberately mild. Steeper weights were tried and
# rejected: at {0.5, 0.75, 1.0, 1.5, 2.0} the draw left only three
# speeches with growth attention coded zero, which is too few to
# estimate the negative predictive value for that variable at all.
# The coverage check below is what caught it.
COUNT_WEIGHTS = ROUNDS[ACTIVE_ROUND]["weights"]

# The draw must leave enough cells of each kind to estimate anything.
MIN_PER_ATTENTION_CELL = 8
MIN_PER_OUTLOOK_CELL = 4


def load_frame() -> pd.DataFrame:
    """Panel, excerpt text, and exclusion of the first-round sample."""

    panel = pd.read_csv(PANEL_FILE, parse_dates=["date"])
    excerpts = pd.read_csv(EXCERPT_FILE)

    frame = panel.merge(
        excerpts[["speech_id", "excerpt", "excerpt_words"]],
        on="speech_id",
        how="left",
        validate="one_to_one",
    )

    missing = frame["excerpt"].isna()

    if missing.any():
        raise ValueError(
            "Speeches without an excerpt: "
            f"{frame.loc[missing, 'speech_id'].tolist()[:10]}"
        )

    already = pd.concat(
        [
            pd.read_csv(PROJECT_ROOT / path)["speech_id"]
            for path in ROUNDS[ACTIVE_ROUND]["exclude"]
        ]
    ).drop_duplicates()

    # Exclude against the whole panel rather than the baseline. One
    # first-round speech (ecb_2682, 69 words) is a slide stub that the
    # baseline drops, so intersecting with the baseline would leave it
    # eligible for redrawing. Checking here also documents the
    # anomaly: a speech was hand-coded that no published measure uses.
    in_panel = frame["speech_id"].isin(already)

    if int(in_panel.sum()) != len(already):
        raise ValueError(
            f"Only {int(in_panel.sum())} of {len(already)} first-round "
            "speeches were found in the panel; the exclusion would be "
            "incomplete."
        )

    non_baseline = frame.loc[in_panel & ~frame["in_baseline"], "speech_id"]

    if len(non_baseline):
        print(
            f"Note: {len(non_baseline)} first-round audit speech(es) "
            f"fall outside the baseline sample: "
            f"{non_baseline.tolist()}"
        )

    frame = frame[frame["in_baseline"] & ~in_panel].copy()

    frame["attention_count"] = frame[ATTENTION_VARS].sum(axis=1)

    frame["era"] = np.where(
        frame["date"].dt.year < ERA_BREAK_YEAR, "early", "late"
    )

    frame["stratum"] = (
        frame["era"] + "_n" + frame["attention_count"].astype(str)
    )

    return frame


def allocate(frame: pd.DataFrame, n_total: int) -> dict[str, int]:
    """
    Split the sample across strata by size and weight.

    Allocation is proportional to stratum size times the attention-count
    weight, rounded, then adjusted so the parts sum to the target. No
    stratum may be allocated more speeches than it contains.
    """

    sizes = frame.groupby("stratum").size()

    weights = frame.groupby("stratum")["attention_count"].first().map(
        COUNT_WEIGHTS
    )

    if weights.isna().any():
        raise ValueError(
            "A stratum has an attention count outside 0-4; the weight "
            "table is incomplete."
        )

    raw = sizes * weights
    target = (raw / raw.sum() * n_total).round().astype(int)

    target = target.clip(upper=sizes)

    # Rounding and clipping rarely hit the target exactly. Distribute
    # the remainder to the strata with the most headroom, largest
    # first, so the correction does not concentrate in a tiny stratum.
    while target.sum() != n_total:
        headroom = sizes - target

        if target.sum() < n_total:
            candidates = headroom[headroom > 0]

            if candidates.empty:
                raise ValueError(
                    "Cannot reach the requested sample size; the "
                    "corpus has too few eligible speeches."
                )

            target[candidates.idxmax()] += 1
        else:
            candidates = target[target > 0]
            target[candidates.idxmax()] -= 1

    return target.to_dict()


def draw(frame: pd.DataFrame, n_total: int, seed: int) -> pd.DataFrame:
    """Draw the stratified sample and shuffle the result."""

    rng = np.random.default_rng(seed)

    allocation = allocate(frame, n_total)

    parts = []

    for stratum, n in sorted(allocation.items()):
        if n == 0:
            continue

        pool = frame[frame["stratum"] == stratum]

        chosen = rng.choice(len(pool), size=n, replace=False)

        parts.append(pool.iloc[np.sort(chosen)])

    sample = pd.concat(parts, ignore_index=True)

    # Shuffle so that row order carries no information about the
    # stratum, and therefore none about the model's codes.
    order = rng.permutation(len(sample))

    return sample.iloc[order].reset_index(drop=True)


def check_coverage(sample: pd.DataFrame) -> pd.DataFrame:
    """
    Verify that every code value appears often enough to be estimable,
    and fail loudly rather than handing over a sheet that cannot answer
    the question it was drawn for.
    """

    rows = []
    problems = []

    for variable in ALL_VARS:
        values = [0, 1] if variable in ATTENTION_VARS else [-1, 0, 1]

        floor = (
            MIN_PER_ATTENTION_CELL
            if variable in ATTENTION_VARS
            else MIN_PER_OUTLOOK_CELL
        )

        for value in values:
            count = int((sample[variable] == value).sum())

            rows.append(
                {
                    "variable": variable,
                    "model_code": value,
                    "n": count,
                    "floor": floor,
                    "ok": count >= floor,
                }
            )

            if count < floor:
                problems.append(
                    f"{variable}={value} has {count} speeches "
                    f"(floor {floor})"
                )

    if problems:
        raise ValueError(
            "The draw does not cover every code value often enough: "
            + "; ".join(problems)
            + ". Increase SAMPLE_SIZE or revisit COUNT_WEIGHTS."
        )

    return pd.DataFrame(rows)


def write_sheets(sample: pd.DataFrame) -> None:
    """
    Write the blind coding sheet and the sealed key.

    The sheet contains the text and empty columns. It contains no model
    codes: not in a column, not in the row order, not in the file name.
    """

    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    KEY_FILE.parent.mkdir(parents=True, exist_ok=True)

    sheet = sample[
        ["speech_id", "date", "speaker", "title", "excerpt_words", "excerpt"]
    ].copy()

    for variable in ALL_VARS:
        sheet[f"{variable}_h"] = ""
        sheet[f"{variable}_note"] = ""

    leaked = [c for c in sheet.columns if c in ALL_VARS]

    if leaked:
        raise ValueError(
            f"Model code columns reached the blind sheet: {leaked}"
        )

    sheet.to_csv(BLIND_FILE, index=False)

    key = sample[["speech_id", "date", "speaker", "stratum"] + ALL_VARS]

    key.to_csv(KEY_FILE, index=False)


def main() -> None:
    frame = load_frame()

    print(f"\n=== ROUND {ACTIVE_ROUND}: ELIGIBLE POOL ===")
    print(
        "Excluding speeches coded in earlier rounds: "
        + ", ".join(ROUNDS[ACTIVE_ROUND]["exclude"])
    )
    print(f"Eligible baseline speeches: {len(frame):,}")

    sample = draw(frame, SAMPLE_SIZE, SEED)

    print("\n=== DRAW ===")
    print(f"Sample size: {len(sample)}  (seed {SEED})")
    print(
        f"Period: {sample['date'].min().date()} to "
        f"{sample['date'].max().date()}"
    )
    print(f"Distinct speakers: {sample['speaker'].nunique()}")

    print("\n=== ALLOCATION BY STRATUM ===")
    print(
        sample.groupby("stratum")
        .size()
        .rename("n")
        .to_frame()
        .to_string()
    )

    coverage = check_coverage(sample)

    print("\n=== COVERAGE OF MODEL CODE VALUES ===")
    print(coverage.to_string(index=False))

    print("\n=== BASE RATES: SAMPLE AGAINST CORPUS ===")
    for variable in ATTENTION_VARS:
        print(
            f"{variable:32s} sample {sample[variable].mean():.3f}  "
            f"corpus {frame[variable].mean():.3f}"
        )

    if any(w != 1.0 for w in COUNT_WEIGHTS.values()):
        print(
            "\nThis round over-represents speeches the model codes one."
            "\nRaw agreement is therefore not an estimate of "
            "corpus-wide agreement;\nthe recovered error rates are."
        )
    else:
        print(
            "\nWeights are flat, so the sample represents the corpus "
            "and raw agreement\nbetween coders is interpretable "
            "directly. Selection does not depend on\nthe model's "
            "codes, so the ceiling it estimates is not conditional "
            "on them."
        )

    write_sheets(sample)

    print(f"\nBlind coding sheet: {BLIND_FILE}")
    print(f"Sealed key (do not open until coding is done): {KEY_FILE}")


if __name__ == "__main__":
    main()
