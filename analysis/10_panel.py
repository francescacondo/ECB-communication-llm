"""
Build the speech-level analysis panel.

Merges the LLM signals with speech metadata and derives the variables
needed for aggregation and for the robustness checks that the data
audit flagged: text length, language, and speaker composition.

The six original signals are carried through untouched. Everything
derived here is added alongside them.

Run from the project root:

    python analysis/10_panel.py
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SPEECH_FILE = (
    PROJECT_ROOT / "data" / "processed" / "ecb_speeches_1999_2025.csv"
)

SIGNALS_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "extracted_signals"
    / "ecb_signals_gpt5mini_v1.csv"
)

PANEL_DIR = PROJECT_ROOT / "outputs" / "panel"

PANEL_FILE = PANEL_DIR / "speech_panel.csv"

SIGNALS = [
    "inflation_attention",
    "inflation_outlook",
    "growth_attention",
    "growth_outlook",
    "financial_stability_attention",
    "uncertainty_attention",
]

# Speeches below this length are slide decks, press-briefing background
# notes and agenda items rather than speeches. They code near-zero on
# every signal and dilute aggregate shares.
MIN_SPEECH_WORDS = 300

# The excerpt cap used at extraction time. Attention rises with text
# length, so whether a speech was truncated is a robustness control.
EXCERPT_WORD_CAP = 2000


# ============================================================
# ECB office holders
# ============================================================

# Presidency terms. Start dates are the first day in office; the final
# term is open-ended.
PRESIDENTS = [
    ("Willem F. Duisenberg", "1998-06-01", "2003-10-31"),
    ("Jean-Claude Trichet", "2003-11-01", "2011-10-31"),
    ("Mario Draghi", "2011-11-01", "2019-10-31"),
    ("Christine Lagarde", "2019-11-01", None),
]

# Executive Board member responsible for Economics, conventionally the
# ECB's chief economist.
CHIEF_ECONOMISTS = [
    ("Otmar Issing", "1998-06-01", "2006-05-31"),
    ("Jürgen Stark", "2006-06-01", "2011-12-31"),
    ("Peter Praet", "2012-01-01", "2019-05-31"),
    ("Philip R. Lane", "2019-06-01", None),
]


def assign_office(
    dates: pd.Series,
    terms: list[tuple[str, str, str | None]],
) -> pd.Series:
    """Return the office holder in post on each date."""

    holder = pd.Series(pd.NA, index=dates.index, dtype="object")

    for name, start, end in terms:
        in_term = dates >= pd.Timestamp(start)

        if end is not None:
            in_term &= dates <= pd.Timestamp(end)

        holder = holder.mask(in_term, name)

    return holder


# ============================================================
# Text characteristics
# ============================================================

ENGLISH_MARKERS = re.compile(
    r"\b(the|of|and|to|in|that|is|for|with|this)\b",
    re.IGNORECASE,
)

NON_ENGLISH_MARKERS = re.compile(
    r"\b(der|die|das|und|ist|nicht|eine|sich|auch|het|van|een|niet"
    r"|les|des|une|nous|pour|dans|est|della|dello|degli|sono|questo"
    r"|anche|que|para|los|las|con|del)\b",
    re.IGNORECASE,
)

LANGUAGE_SAMPLE_CHARS = 4000


def flag_non_english(contents: pd.Series) -> pd.DataFrame:
    """
    Heuristic language flag from function-word frequencies.

    Counts English and non-English function words in the opening of
    each speech and flags the speech when the latter dominates. This is
    a screen, not a language identifier: it is reliable on speeches
    delivered wholly in German, French, Italian, Spanish or Dutch, and
    unreliable on short texts and on documents that mix an English
    abstract with a non-English body. The flag exists so that language
    can be used as a robustness control, not so that speeches can be
    dropped silently.
    """

    opening = contents.fillna("").str.slice(0, LANGUAGE_SAMPLE_CHARS)

    english = opening.str.count(ENGLISH_MARKERS)
    non_english = opening.str.count(NON_ENGLISH_MARKERS)

    return pd.DataFrame(
        {
            "english_markers": english,
            "non_english_markers": non_english,
            "non_english": non_english > english,
        }
    )


# ============================================================
# Panel construction
# ============================================================

def build_panel() -> pd.DataFrame:
    speeches = pd.read_csv(SPEECH_FILE, parse_dates=["date"])
    signals = pd.read_csv(SIGNALS_FILE, parse_dates=["date"])

    panel = speeches.merge(
        signals[["speech_id"] + SIGNALS],
        on="speech_id",
        how="left",
        validate="one_to_one",
    )

    unmatched = panel[SIGNALS].isna().any(axis=1)

    if unmatched.any():
        raise ValueError(
            "Speeches without extracted signals: "
            f"{panel.loc[unmatched, 'speech_id'].tolist()[:10]}"
        )

    # --------------------------------------------------
    # Time
    # --------------------------------------------------

    panel["year"] = panel["date"].dt.year
    panel["month"] = panel["date"].dt.month
    panel["ym"] = panel["date"].dt.to_period("M").astype(str)
    panel["yq"] = panel["date"].dt.to_period("Q").astype(str)

    # --------------------------------------------------
    # Text
    # --------------------------------------------------

    panel["n_words"] = panel["contents"].str.split().str.len()

    panel["log_words"] = np.log(panel["n_words"].astype(float))

    panel["truncated"] = panel["n_words"] > EXCERPT_WORD_CAP
    panel["is_short"] = panel["n_words"] < MIN_SPEECH_WORDS

    panel = panel.join(flag_non_english(panel["contents"]))

    # --------------------------------------------------
    # Speakers and office
    # --------------------------------------------------

    panel["n_speakers"] = (
        panel["speakers"].str.count(",").fillna(0).astype(int) + 1
    )

    panel["speaker"] = (
        panel["speakers"].str.split(",").str[0].str.strip()
    )

    panel["president"] = assign_office(panel["date"], PRESIDENTS)

    panel["chief_economist"] = assign_office(
        panel["date"], CHIEF_ECONOMISTS
    )

    # True when the speech was given by whoever held the office on the
    # date of the speech, so a former president's later speeches do not
    # count.
    panel["by_president"] = panel.apply(
        lambda r: str(r["president"]) in str(r["speakers"]),
        axis=1,
    )

    panel["by_chief_economist"] = panel.apply(
        lambda r: str(r["chief_economist"]) in str(r["speakers"]),
        axis=1,
    )

    # --------------------------------------------------
    # Baseline sample
    # --------------------------------------------------

    panel["in_baseline"] = ~panel["is_short"]

    columns = [
        "speech_id",
        "date",
        "year",
        "month",
        "ym",
        "yq",
        "speakers",
        "speaker",
        "n_speakers",
        "title",
        "subtitle",
        "president",
        "chief_economist",
        "by_president",
        "by_chief_economist",
        "n_words",
        "log_words",
        "truncated",
        "is_short",
        "english_markers",
        "non_english_markers",
        "non_english",
        "in_baseline",
    ] + SIGNALS

    return panel[columns]


def length_trend(panel: pd.DataFrame) -> dict[str, float]:
    """
    Mean speech length overall and its decline to the trough.

    Attention is measured on a fixed 2,000-word excerpt, so a topic has
    more chance of being discussed substantively in a long speech than
    in a short one. If mean length trends, measured attention can move
    for reasons that have nothing to do with what the ECB chose to talk
    about, which is why the write-up quotes this decline when it warns
    against reading trends off the attention series.

    The trough is reported rather than the final year deliberately.
    Length falls to a minimum in the middle of the sample and then
    partly recovers, so first-to-last understates the variation the
    attention measures are exposed to, and "declines by X% over the
    sample" would be the wrong summary if X were the first-to-trough
    figure. Both are returned so neither can be quoted as the other.
    """

    yearly = panel.groupby("year")["n_words"].mean()

    first_year = int(yearly.index.min())
    last_year = int(yearly.index.max())
    trough_year = int(yearly.idxmin())

    return {
        "mean_words": float(panel["n_words"].mean()),
        "first_year": first_year,
        "first_year_mean": float(yearly.loc[first_year]),
        "trough_year": trough_year,
        "trough_mean": float(yearly.loc[trough_year]),
        "last_year": last_year,
        "last_year_mean": float(yearly.loc[last_year]),
        "decline_to_trough": float(
            yearly.loc[trough_year] / yearly.loc[first_year] - 1.0
        ),
        "decline_to_last": float(
            yearly.loc[last_year] / yearly.loc[first_year] - 1.0
        ),
    }


def validate_offices(panel: pd.DataFrame) -> None:
    """
    Check that every office holder in the lookup tables actually
    appears as a speaker, so that a misspelling fails loudly rather
    than silently producing an all-false indicator.
    """

    speakers = set()

    for entry in panel["speakers"].dropna():
        speakers.update(name.strip() for name in entry.split(","))

    for label, terms in [
        ("president", PRESIDENTS),
        ("chief economist", CHIEF_ECONOMISTS),
    ]:
        for name, _, _ in terms:
            if name not in speakers:
                raise ValueError(
                    f"{label} '{name}' never appears in the speaker "
                    "field; check the spelling in the lookup table."
                )


def main() -> None:
    PANEL_DIR.mkdir(parents=True, exist_ok=True)

    panel = build_panel()
    validate_offices(panel)

    print("\n=== SPEECH PANEL ===")
    print(f"Speeches: {len(panel):,}")
    print(
        f"Period: {panel['date'].min().date()} to "
        f"{panel['date'].max().date()}"
    )
    print(f"Speakers: {panel['speaker'].nunique()}")

    print("\n=== SAMPLE FLAGS ===")
    print(
        f"Below {MIN_SPEECH_WORDS} words (excluded from baseline): "
        f"{int(panel['is_short'].sum())}"
    )
    print(
        f"Truncated at {EXCERPT_WORD_CAP} words: "
        f"{int(panel['truncated'].sum())} "
        f"({panel['truncated'].mean():.1%})"
    )
    print(
        "Flagged non-English: "
        f"{int(panel['non_english'].sum())} "
        f"({panel['non_english'].mean():.1%})"
    )
    print(f"Baseline sample: {int(panel['in_baseline'].sum()):,}")

    print("\n=== SPEECHES BY PRESIDENCY ===")
    print(
        panel.groupby("president", sort=False)
        .agg(
            n=("speech_id", "size"),
            by_president=("by_president", "sum"),
            by_chief_econ=("by_chief_economist", "sum"),
            mean_words=("n_words", "mean"),
        )
        .round(0)
        .to_string()
    )

    length = length_trend(panel)

    print("\n=== SPEECH LENGTH ===")
    print(f"Mean words, whole corpus: {length['mean_words']:,.0f}")
    print(
        f"{length['first_year']}: {length['first_year_mean']:,.0f} -> "
        f"{length['trough_year']}: {length['trough_mean']:,.0f} "
        f"({length['decline_to_trough']:+.1%}, the trough)"
    )
    print(
        f"{length['first_year']}: {length['first_year_mean']:,.0f} -> "
        f"{length['last_year']}: {length['last_year_mean']:,.0f} "
        f"({length['decline_to_last']:+.1%}, end of sample)"
    )
    print(
        "Quote the first for the fall to the trough and the second for "
        "the change\nacross the sample; they are not the same number."
    )

    panel.to_csv(PANEL_FILE, index=False)

    print(f"\nSaved speech panel to: {PANEL_FILE}")


if __name__ == "__main__":
    main()
