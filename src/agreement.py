"""
Agreement statistics for comparing a coder against a reference.

Factored out of analysis/15_validation_stats.py so that the same
machinery can score any instrument against the human codes, not just
the LLM. The dictionary benchmark in analysis/16_dictionary_benchmark.py
is the reason this is a module: a horse race is only meaningful if both
contestants are scored by identical code.

Nothing here knows anything about LLMs, dictionaries or ECB speeches.
It takes two aligned arrays of codes and a list of admissible
categories.
"""

from math import comb

import numpy as np


# Bootstrap settings are fixed here so that every reported interval is
# reproducible without the caller having to pass them.
N_BOOTSTRAP = 10_000

BOOTSTRAP_SEED = 20250824


def categories_for(variable: str) -> list[int]:
    """
    Admissible values for a project variable.

    Outlook variables are ternary, attention variables binary. Kept
    with the statistics because chance agreement depends on it.
    """

    return [-1, 0, 1] if variable.endswith("outlook") else [0, 1]


def cohen_kappa(
    a: np.ndarray,
    b: np.ndarray,
    cats: list[int],
) -> tuple[float, float]:
    """
    Return observed agreement and Cohen's kappa.

    Kappa is undefined when chance agreement is one, which happens when
    both coders use a single category throughout. In that case only the
    observed agreement is informative.
    """

    observed = float((a == b).mean())

    expected = float(
        sum(
            (a == c).mean() * (b == c).mean()
            for c in cats
        )
    )

    if expected >= 1.0:
        return observed, float("nan")

    return observed, (observed - expected) / (1.0 - expected)


def bootstrap_kappa_ci(
    a: np.ndarray,
    b: np.ndarray,
    cats: list[int],
    n_draws: int = N_BOOTSTRAP,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    """
    Percentile bootstrap confidence interval for Cohen's kappa.

    Draws in which kappa is undefined are dropped rather than treated
    as zero.
    """

    rng = np.random.default_rng(seed)

    n = len(a)
    draws = []

    for _ in range(n_draws):
        idx = rng.integers(0, n, n)
        _, k = cohen_kappa(a[idx], b[idx], cats)

        if not np.isnan(k):
            draws.append(k)

    if not draws:
        return float("nan"), float("nan")

    lo, hi = np.percentile(draws, [2.5, 97.5])

    return float(lo), float(hi)


def binomial_cdf(k: int, n: int, p: float) -> float:
    """P(X <= k) for X ~ Binomial(n, p), computed exactly."""

    if p <= 0.0:
        return 1.0

    if p >= 1.0:
        return 0.0 if k < n else 1.0

    return float(
        sum(
            comb(n, i) * p**i * (1.0 - p) ** (n - i)
            for i in range(k + 1)
        )
    )


def clopper_pearson_ci(
    n_success: int,
    n_total: int,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """
    Exact binomial confidence interval for a proportion.

    Preferred over a nonparametric bootstrap here because the
    validation sample is small and several cells are at the boundary.
    A bootstrap of a proportion with zero observed failures returns a
    degenerate interval of [1, 1], which would misrepresent an
    uninformative estimate as a certain one. The exact interval
    correctly reports, for example, that ten out of ten successes are
    consistent with a true rate as low as 0.69.

    Bounds are found by bisection on the binomial CDF, which is
    monotone in p.
    """

    if n_total == 0:
        return float("nan"), float("nan")

    def bisect(target: float, is_lower: bool) -> float:
        lo, hi = 0.0, 1.0

        for _ in range(200):
            mid = 0.5 * (lo + hi)

            # Lower bound solves P(X >= k) = alpha/2.
            # Upper bound solves P(X <= k) = alpha/2.
            value = (
                1.0 - binomial_cdf(n_success - 1, n_total, mid)
                if is_lower
                else binomial_cdf(n_success, n_total, mid)
            )

            if (value < target) == is_lower:
                lo = mid
            else:
                hi = mid

        return 0.5 * (lo + hi)

    lower = 0.0 if n_success == 0 else bisect(alpha / 2.0, True)
    upper = 1.0 if n_success == n_total else bisect(alpha / 2.0, False)

    return lower, upper


def sensitivity_specificity(
    reference: np.ndarray,
    candidate: np.ndarray,
) -> dict[str, float | int]:
    """
    Score a binary candidate coder against a binary reference.

    Sensitivity is the share of reference ones the candidate also codes
    one; specificity the share of reference zeros it also codes zero.
    Both are reported with exact intervals and with the underlying
    counts, since at these sample sizes the counts are what the reader
    needs to judge the estimate.
    """

    reference = np.asarray(reference)
    candidate = np.asarray(candidate)

    n_true_one = int((reference == 1).sum())
    n_true_zero = int((reference == 0).sum())

    true_positive = int(((reference == 1) & (candidate == 1)).sum())
    true_negative = int(((reference == 0) & (candidate == 0)).sum())

    sensitivity = (
        true_positive / n_true_one if n_true_one else float("nan")
    )

    specificity = (
        true_negative / n_true_zero if n_true_zero else float("nan")
    )

    sens_lo, sens_hi = clopper_pearson_ci(true_positive, n_true_one)
    spec_lo, spec_hi = clopper_pearson_ci(true_negative, n_true_zero)

    return {
        "n_true_one": n_true_one,
        "n_true_zero": n_true_zero,
        "true_positive": true_positive,
        "true_negative": true_negative,
        "false_positive": n_true_zero - true_negative,
        "false_negative": n_true_one - true_positive,
        "sensitivity": sensitivity,
        "sens_lo": sens_lo,
        "sens_hi": sens_hi,
        "specificity": specificity,
        "spec_lo": spec_lo,
        "spec_hi": spec_hi,
    }
