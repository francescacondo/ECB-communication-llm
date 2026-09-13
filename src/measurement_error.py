"""
Misclassification correction for LLM-generated binary measures.

The LLM signals are generated measurements, not ground truth. A
human-coded validation sample identifies how often the model flags a
topic that a human coder would not (false positive) and how often it
misses one (false negative).

Writing alpha for sensitivity and beta for specificity, the observed
share of speeches coded 1 relates to the true share p as

    E[observed] = alpha * p + (1 - beta) * (1 - p)

which inverts to

    p_hat = (observed - (1 - beta)) / (alpha + beta - 1)

The correction is only defined when alpha + beta > 1, i.e. when the
classifier is informative. Corrected shares are clipped to [0, 1]
because the inversion is unbiased but not range-respecting in small
samples.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


MIN_INFORMATIVE_MARGIN = 1e-6


@dataclass(frozen=True)
class ErrorRates:
    """Sensitivity and specificity for one binary measure."""

    sensitivity: float
    specificity: float
    n_true_one: int
    n_true_zero: int

    @property
    def is_informative(self) -> bool:
        return (
            self.sensitivity + self.specificity - 1.0
            > MIN_INFORMATIVE_MARGIN
        )


def estimate_error_rates(
    llm: pd.Series,
    human: pd.Series,
) -> ErrorRates:
    """
    Estimate sensitivity and specificity from a validation sample.

    Sensitivity is the share of human-coded ones the LLM also codes
    one. Specificity is the share of human-coded zeros the LLM also
    codes zero.
    """

    llm = llm.astype(int)
    human = human.astype(int)

    true_one = human == 1
    true_zero = human == 0

    n_one = int(true_one.sum())
    n_zero = int(true_zero.sum())

    sensitivity = (
        float(llm[true_one].mean())
        if n_one > 0
        else float("nan")
    )

    specificity = (
        float((1 - llm[true_zero]).mean())
        if n_zero > 0
        else float("nan")
    )

    return ErrorRates(
        sensitivity=sensitivity,
        specificity=specificity,
        n_true_one=n_one,
        n_true_zero=n_zero,
    )


def correct_share(
    observed_share: float | pd.Series,
    rates: ErrorRates,
) -> float | pd.Series:
    """
    Invert the misclassification model to recover the true share.

    Returns the observed share unchanged when the estimated classifier
    is not informative, so that downstream code never silently divides
    by something near zero.
    """

    if not rates.is_informative:
        return observed_share

    denominator = rates.sensitivity + rates.specificity - 1.0

    corrected = (
        observed_share - (1.0 - rates.specificity)
    ) / denominator

    if isinstance(corrected, pd.Series):
        return corrected.clip(0.0, 1.0)

    return float(np.clip(corrected, 0.0, 1.0))


def correct_series(
    signals: pd.DataFrame,
    variable: str,
    rates: ErrorRates,
    by: pd.Series,
) -> pd.DataFrame:
    """
    Aggregate a speech-level binary signal by some grouping (year,
    month, speaker) and report the raw and corrected shares side by
    side.
    """

    grouped = signals.groupby(by)[variable]

    out = pd.DataFrame(
        {
            "n": grouped.size(),
            "raw": grouped.mean(),
        }
    )

    out["corrected"] = correct_share(out["raw"], rates)

    return out


def corrected_share_bounds(
    observed_share: float,
    sensitivity_ci: tuple[float, float],
    specificity_ci: tuple[float, float],
) -> tuple[float, float]:
    """
    Conservative interval for a corrected share.

    The correction is not monotone in the error rates, because
    sensitivity and specificity enter both the numerator and the
    denominator, so the interval is obtained by evaluating the
    correction at the four corners of the rate confidence box and
    taking the extremes.

    A nonparametric bootstrap over the validation sample is not used
    here: when a variable has no observed errors, every resample also
    has none, and the resulting interval collapses to a point. The
    corner method inherits the exact binomial intervals and therefore
    stays informative at the boundary. It is conservative, since it
    ignores the correlation between the two estimated rates.
    """

    corners = []

    for alpha in sensitivity_ci:
        for beta in specificity_ci:
            if np.isnan(alpha) or np.isnan(beta):
                continue

            rates = ErrorRates(
                sensitivity=alpha,
                specificity=beta,
                n_true_one=0,
                n_true_zero=0,
            )

            if not rates.is_informative:
                continue

            corners.append(correct_share(observed_share, rates))

    if not corners:
        return float("nan"), float("nan")

    return min(corners), max(corners)
