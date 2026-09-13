"""
Keyword-dictionary implementation of the coding scheme.

This is the benchmark the LLM extraction is measured against. The
question it answers is the one any reader of the project will ask
first: what did the language model buy that counting words would not
have given for free?

For that comparison to mean anything the dictionary has to be a
good-faith effort rather than a strawman, so three things are held
fixed between the two instruments:

    1. The same input text. Both read the truncated excerpts in
       outputs/production/, not the full speeches, so the comparison is
       not confounded by one instrument seeing more words.

    2. The same codebook. Every term below is drawn from the concepts
       docs/coding_rules.md lists as qualifying evidence. The
       dictionary is given the same instructions the prompt was given,
       in the only form a dictionary can accept them.

    3. The same decision thresholds, in the sense that the cutoffs are
       calibrated to reproduce the LLM's base rates rather than chosen
       to flatter either side. That calibration happens in
       analysis/16_dictionary_benchmark.py.

What the dictionary cannot be given is composition. "Inflation
pressures have receded" and "receding pressures have given way to
inflation" contain the same words. Direction is where a bag of words
is expected to fail, and the benchmark is designed to show whether it
does.

Two standard refinements are included so that the baseline is the
respectable version rather than the naive one: scoring is restricted
to sentences that mention the topic, so directional language about
something else is not attributed to it, and a short negation window
flips the sign of a directional term preceded by a negator.
"""

import re

import numpy as np
import pandas as pd


# Sentence splitting is deliberately crude. Central bank prose is
# well-punctuated, and a full segmenter would add a dependency for a
# marginal gain on a benchmark that is meant to be simple.
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

# A directional term is flipped when a negator appears within this many
# words before it. Three is chosen to catch "not rising" and "no longer
# increasing" without reaching across clauses. It is a choice, not a
# standard: no window length is claimed here to be conventional, and
# the benchmark's sensitivity to it has not been tested.
NEGATION_WINDOW = 3

NEGATORS = [
    "not", "no", "nor", "never", "without", "unlikely", "hardly",
    "scarcely", "barely", "little", "few", "lack", "lacking",
    "absent", "neither", "cannot", "unable",
]


# ============================================================
# Topic vocabularies
# ============================================================

# Terms qualifying a speech as discussing each topic. Drawn from the
# "code 1 when" clauses of docs/coding_rules.md.

TOPIC_TERMS = {
    "inflation": [
        "inflation", "inflationary", "price stability", "price level",
        "prices", "price pressure", "price pressures", "hicp",
        "consumer price", "consumer prices", "cost pressure",
        "cost pressures", "disinflation", "deflation", "deflationary",
        "inflation expectations", "wage pressure", "wage pressures",
        "price developments",
    ],
    "growth": [
        "growth", "economic growth", "gdp", "output",
        "economic activity", "demand",
        "real activity", "domestic demand", "consumption",
        "investment", "recession", "recovery", "expansion",
        "employment", "unemployment", "labour market", "labor market",
        "economic conditions", "business cycle", "demand conditions",
    ],
    "financial_stability": [
        "financial stability", "systemic risk", "banking system",
        "banking sector", "financial system", "financial sector",
        "solvency", "bank capital", "capital buffer",
        "capital buffers", "macroprudential", "contagion",
        "financial fragility", "vulnerabilities", "financial stress",
        "market stress", "bank resilience", "financial imbalances",
        "credit risk", "liquidity risk",
    ],
    "uncertainty": [
        "uncertain", "uncertainty", "uncertainties", "unpredictable",
        "unpredictability", "unclear", "ambiguous", "volatility",
        "volatile", "risks", "risk", "downside risks",
        "upside risks", "alternative scenarios", "wide range",
        "difficult to assess", "hard to predict", "unforeseen",
    ],
}


# ============================================================
# Directional vocabularies
# ============================================================

# Direction is scored only inside sentences that mention the topic.
# The inflation and growth lists overlap heavily, as they should:
# "strengthening" means upside for both. What differs is the
# topic-specific vocabulary, which is what the separate lists carry.

DIRECTION_TERMS = {
    "inflation": {
        "up": [
            "rise", "rises", "rising", "rose", "increase", "increases",
            "increasing", "increased", "higher", "upside", "upward",
            "accelerate", "accelerating", "acceleration", "persistent",
            "persistence", "pressure", "pressures", "strong",
            "stronger", "strengthening", "elevated", "above target",
            "overheating", "surge", "surging", "mounting", "picking up",
            "second-round", "de-anchoring", "entrenched",
        ],
        "down": [
            "fall", "falls", "falling", "fell", "decline", "declines",
            "declining", "declined", "decrease", "decreasing", "lower",
            "downside", "downward", "subdued", "weak", "weaker",
            "weakening", "moderate", "moderating", "moderation",
            "ease", "easing", "eased", "slow", "slowing", "slowed",
            "below target", "disinflation", "disinflationary",
            "deflation", "deflationary", "dampen", "dampening",
            "muted", "contained", "receded", "receding",
        ],
    },
    "growth": {
        "up": [
            "recovery", "recovering", "recover", "expansion",
            "expanding", "strengthen", "strengthening", "strengthened",
            "improve", "improving", "improved", "robust", "resilient",
            "resilience", "accelerate", "accelerating", "pick up",
            "picking up", "rebound", "rebounding", "buoyant",
            "upswing", "upside", "stronger", "strong", "rising",
            "increase", "increasing", "higher",
        ],
        "down": [
            "slowdown", "slowing", "slow", "weaken", "weakening",
            "weak", "weaker", "contraction", "contracting", "recession",
            "recessionary", "deteriorate", "deteriorating",
            "deterioration", "downturn", "sluggish", "subdued",
            "decline", "declining", "fall", "falling", "downside",
            "lower", "stagnation", "stagnant", "dampen", "dampening",
            "headwinds",
        ],
    },
}


# ============================================================
# Matching
# ============================================================

def compile_terms(terms: list[str]) -> re.Pattern:
    """
    One alternation regex per vocabulary, with word boundaries.

    Longer phrases are placed first so that "price stability" is
    matched as a phrase rather than consumed by "prices". Terms are
    escaped because several contain hyphens.
    """

    ordered = sorted(set(terms), key=len, reverse=True)

    pattern = "|".join(re.escape(term) for term in ordered)

    return re.compile(rf"\b(?:{pattern})\b", re.IGNORECASE)


TOPIC_PATTERNS = {
    topic: compile_terms(terms) for topic, terms in TOPIC_TERMS.items()
}

DIRECTION_PATTERNS = {
    topic: {
        side: compile_terms(terms)
        for side, terms in sides.items()
    }
    for topic, sides in DIRECTION_TERMS.items()
}

NEGATOR_PATTERN = compile_terms(NEGATORS)

WORD_PATTERN = re.compile(r"\b\w+\b")


def is_negated(sentence: str, position: int) -> bool:
    """
    True when a negator occurs within NEGATION_WINDOW words before the
    match starting at `position`.
    """

    preceding = WORD_PATTERN.findall(sentence[:position])

    window = preceding[-NEGATION_WINDOW:]

    return any(NEGATOR_PATTERN.fullmatch(word) for word in window)


def count_directional(sentence: str, pattern: re.Pattern) -> int:
    """
    Net count of matches in a sentence, negation-adjusted.

    A negated term contributes -1 rather than 0, since "inflation is
    not rising" is evidence for the opposite direction, not merely
    absence of evidence.
    """

    total = 0

    for match in pattern.finditer(sentence):
        total += -1 if is_negated(sentence, match.start()) else 1

    return total


# ============================================================
# Speech-level scoring
# ============================================================

def score_speech(text: str) -> dict[str, float]:
    """
    All dictionary quantities for one speech.

    Attention is a density, matches per thousand words, so that longer
    speeches are not mechanically more attentive. That is the same
    length confound the data audit found in the LLM codes, and the
    dictionary is given the fairer treatment rather than the
    convenient one.

    Direction is a net count over topic sentences, normalised by the
    number of topic sentences, so it is bounded in interpretation and
    comparable across speeches of different lengths.
    """

    text = "" if not isinstance(text, str) else text

    n_words = len(WORD_PATTERN.findall(text))

    result: dict[str, float] = {"dict_n_words": float(n_words)}

    scale = 1000.0 / n_words if n_words else 0.0

    for topic, pattern in TOPIC_PATTERNS.items():
        result[f"dens_{topic}"] = len(pattern.findall(text)) * scale

    sentences = SENTENCE_SPLIT.split(text)

    for topic in DIRECTION_PATTERNS:
        topic_pattern = TOPIC_PATTERNS[topic]

        up_pattern = DIRECTION_PATTERNS[topic]["up"]
        down_pattern = DIRECTION_PATTERNS[topic]["down"]

        net = 0
        n_topic_sentences = 0

        for sentence in sentences:
            if not topic_pattern.search(sentence):
                continue

            n_topic_sentences += 1

            net += count_directional(sentence, up_pattern)
            net -= count_directional(sentence, down_pattern)

        result[f"tone_{topic}"] = (
            net / n_topic_sentences if n_topic_sentences else 0.0
        )

        result[f"n_sent_{topic}"] = float(n_topic_sentences)

    return result


def score_corpus(texts: pd.Series) -> pd.DataFrame:
    """Apply score_speech across a corpus, preserving the index."""

    scored = pd.DataFrame(
        [score_speech(text) for text in texts],
        index=texts.index,
    )

    if scored.isna().any().any():
        raise ValueError(
            "Dictionary scoring produced missing values; a speech "
            "probably has no extractable words."
        )

    return scored


# ============================================================
# Turning scores into codes
# ============================================================

def threshold_at_prevalence(
    scores: np.ndarray,
    target_rate: float,
) -> float:
    """
    Cutoff reproducing a target share of ones.

    The dictionary needs a decision threshold and the LLM does not, so
    any fixed choice would be a researcher degree of freedom that could
    be tuned to win. Calibrating to the LLM's own base rate removes it:
    both instruments then flag the same number of speeches, and any
    difference in agreement with the human codes is about *which*
    speeches, not how many.

    This deliberately grants the dictionary the LLM's prevalence, which
    is information a standalone dictionary would not have. The
    benchmark is therefore generous to the baseline.
    """

    if not 0.0 < target_rate < 1.0:
        raise ValueError(
            f"Target rate must lie strictly in (0, 1), got {target_rate}"
        )

    return float(np.quantile(scores, 1.0 - target_rate))


def code_attention(
    scores: np.ndarray,
    target_rate: float,
) -> np.ndarray:
    """Binary attention codes at the prevalence-matched threshold."""

    cutoff = threshold_at_prevalence(scores, target_rate)

    return (scores > cutoff).astype(int)


def code_outlook(
    tone: np.ndarray,
    attention: np.ndarray,
    neutral_rate: float,
) -> np.ndarray:
    """
    Ternary outlook codes from a continuous tone score.

    The neutral band is set so that the share of zeros matches the
    LLM's, for the same reason the attention threshold is
    prevalence-matched. Speeches not coded as attending to the topic
    are forced to zero, which is the logical constraint the coding
    rules impose and which the LLM was also held to.
    """

    tone = np.asarray(tone, dtype=float)
    attention = np.asarray(attention)

    codes = np.sign(tone).astype(int)

    attending = attention == 1

    if attending.sum() == 0:
        raise ValueError("No speech was coded as attending the topic.")

    # The band is chosen among attending speeches only, since
    # non-attending ones are forced to zero regardless and would
    # otherwise consume the whole neutral quota.
    magnitudes = np.abs(tone[attending])

    share_neutral_forced = 1.0 - attending.mean()

    remaining = neutral_rate - share_neutral_forced

    if remaining <= 0:
        # The attention constraint alone already produces at least the
        # target share of zeros; no additional band is needed.
        band = 0.0
    else:
        within = remaining / attending.mean()

        band = float(np.quantile(magnitudes, min(within, 1.0)))

    codes[np.abs(tone) <= band] = 0

    codes[~attending] = 0

    return codes
