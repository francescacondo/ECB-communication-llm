"""
Inference for correlations between persistent quarterly time series.

The communication measures and the macro series are both strongly
autocorrelated, and the rate-path variable is built from overlapping
two-quarter differences. A textbook correlation standard error of
1/sqrt(n-3) assumes independent observations and would overstate
precision badly in both respects. Everything here exists to avoid that.

The approach is deliberately plain:

    1. A correlation is the OLS slope of the standardised dependent
       variable on the standardised regressor. Estimating it that way
       makes heteroskedasticity- and autocorrelation-consistent
       (Newey-West) standard errors directly available.

    2. A difference in correlations across two eras is the coefficient
       on an interaction term in the pooled, within-era standardised
       regression, so the same machinery tests it.

    3. A moving-block bootstrap provides a second, non-parametric
       interval, because the HAC sandwich is an asymptotic
       approximation and these subsamples have roughly fifty
       observations.

scipy is not a dependency of this project, so the Student-t tail
probability is computed here from the regularised incomplete beta
function rather than imported.
"""

from dataclasses import dataclass
from math import lgamma, exp, log, sqrt, floor

import numpy as np


# Continued-fraction evaluation of the incomplete beta stops when the
# relative change falls below this, or after this many iterations.
BETA_TOLERANCE = 1e-12
BETA_MAX_ITERATIONS = 300

# Guard against dividing by a near-zero denominator in Lentz's
# algorithm, which is the standard numerical-recipes safeguard.
BETA_TINY = 1e-30


# ============================================================
# Student-t tail probability
# ============================================================

def _beta_continued_fraction(a: float, b: float, x: float) -> float:
    """
    Continued fraction for the incomplete beta function, evaluated by
    the modified Lentz algorithm. Converges quickly for
    x < (a + 1) / (a + b + 2); the caller enforces that by symmetry.
    """

    qab = a + b
    qap = a + 1.0
    qam = a - 1.0

    c = 1.0
    d = 1.0 - qab * x / qap

    if abs(d) < BETA_TINY:
        d = BETA_TINY

    d = 1.0 / d
    h = d

    for m in range(1, BETA_MAX_ITERATIONS + 1):
        m2 = 2 * m

        # Even step.
        numerator = m * (b - m) * x / ((qam + m2) * (a + m2))

        d = 1.0 + numerator * d
        if abs(d) < BETA_TINY:
            d = BETA_TINY

        c = 1.0 + numerator / c
        if abs(c) < BETA_TINY:
            c = BETA_TINY

        d = 1.0 / d
        h *= d * c

        # Odd step.
        numerator = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))

        d = 1.0 + numerator * d
        if abs(d) < BETA_TINY:
            d = BETA_TINY

        c = 1.0 + numerator / c
        if abs(c) < BETA_TINY:
            c = BETA_TINY

        d = 1.0 / d
        delta = d * c
        h *= delta

        if abs(delta - 1.0) < BETA_TOLERANCE:
            return h

    raise RuntimeError(
        f"Incomplete beta did not converge for a={a}, b={b}, x={x}. "
        "This should not happen for the arguments used here; do not "
        "silently accept the last iterate."
    )


def regularised_incomplete_beta(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta function I_x(a, b)."""

    if not 0.0 <= x <= 1.0:
        raise ValueError(f"x must lie in [0, 1], got {x}")

    if x in (0.0, 1.0):
        return x

    front = exp(
        lgamma(a + b)
        - lgamma(a)
        - lgamma(b)
        + a * log(x)
        + b * log(1.0 - x)
    )

    if x < (a + 1.0) / (a + b + 2.0):
        return front * _beta_continued_fraction(a, b, x) / a

    return 1.0 - front * _beta_continued_fraction(b, a, 1.0 - x) / b


def two_sided_t_pvalue(t_statistic: float, df: float) -> float:
    """
    P(|T| > |t|) for T with df degrees of freedom.

    Uses the identity P(|T| > t) = I_{df/(df+t^2)}(df/2, 1/2), which
    avoids any numerical integration.
    """

    if df <= 0:
        raise ValueError(f"Degrees of freedom must be positive, got {df}")

    if not np.isfinite(t_statistic):
        return float("nan")

    return regularised_incomplete_beta(
        0.5 * df, 0.5, df / (df + t_statistic**2)
    )


# ============================================================
# Newey-West OLS
# ============================================================

def newey_west_bandwidth(n_obs: int, min_lags: int = 0) -> int:
    """
    Automatic Bartlett bandwidth, floor(4 * (n/100)^(2/9)).

    `min_lags` raises the floor. It is used when the dependent variable
    is an overlapping h-period difference, which induces an MA(h-1)
    residual by construction: a bandwidth below h-1 would ignore
    autocorrelation that is known to be present rather than merely
    suspected.
    """

    if n_obs <= 0:
        raise ValueError(f"n_obs must be positive, got {n_obs}")

    rule = int(floor(4.0 * (n_obs / 100.0) ** (2.0 / 9.0)))

    return max(rule, min_lags)


@dataclass(frozen=True)
class HACRegression:
    """OLS estimates with Newey-West standard errors."""

    coefficients: np.ndarray
    standard_errors: np.ndarray
    n_obs: int
    n_params: int
    lags: int
    r_squared: float

    @property
    def t_statistics(self) -> np.ndarray:
        return self.coefficients / self.standard_errors

    @property
    def p_values(self) -> np.ndarray:
        df = self.n_obs - self.n_params

        return np.array(
            [two_sided_t_pvalue(t, df) for t in self.t_statistics]
        )

    def confidence_interval(
        self,
        index: int,
        level: float = 0.95,
    ) -> tuple[float, float]:
        """
        Symmetric interval for one coefficient, using normal critical
        values. The t quantile would require inverting the incomplete
        beta; with fifty-odd observations and four parameters the
        difference is small relative to the width itself, and the
        choice is recorded in the output so it is not hidden.
        """

        if level != 0.95:
            raise ValueError(
                "Only the 95% interval is implemented; add a critical "
                "value explicitly rather than interpolating."
            )

        half_width = 1.959964 * self.standard_errors[index]

        return (
            float(self.coefficients[index] - half_width),
            float(self.coefficients[index] + half_width),
        )


def newey_west_ols(
    y: np.ndarray,
    X: np.ndarray,
    lags: int,
) -> HACRegression:
    """
    OLS with Bartlett-kernel HAC standard errors.

    X must already contain an intercept column if one is wanted. The
    residual covariance carries the n/(n-k) small-sample scaling, which
    matters at these sample sizes.
    """

    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)

    if y.ndim != 1:
        raise ValueError(f"y must be one-dimensional, got shape {y.shape}")

    if X.ndim != 2 or X.shape[0] != y.shape[0]:
        raise ValueError(
            f"X must be (n, k) matching y; got {X.shape} against "
            f"{y.shape}"
        )

    if not np.isfinite(y).all() or not np.isfinite(X).all():
        raise ValueError(
            "Non-finite values reached the regression. Drop or "
            "investigate them upstream rather than here."
        )

    n, k = X.shape

    if n <= k:
        raise ValueError(
            f"Only {n} observations for {k} parameters; the regression "
            "is not identified."
        )

    if lags < 0 or lags >= n:
        raise ValueError(
            f"Bandwidth {lags} is not usable with {n} observations."
        )

    xtx = X.T @ X

    if np.linalg.matrix_rank(xtx) < k:
        raise ValueError(
            "Regressor matrix is rank deficient; check for a constant "
            "column duplicated by a dummy."
        )

    xtx_inv = np.linalg.inv(xtx)

    beta = xtx_inv @ (X.T @ y)

    residuals = y - X @ beta

    # Bartlett-weighted long-run covariance of the score.
    scores = X * residuals[:, None]

    omega = scores.T @ scores

    for lag in range(1, lags + 1):
        weight = 1.0 - lag / (lags + 1.0)

        gamma = scores[lag:].T @ scores[:-lag]

        omega += weight * (gamma + gamma.T)

    omega *= n / (n - k)

    covariance = xtx_inv @ omega @ xtx_inv

    variances = np.diag(covariance)

    if (variances <= 0).any():
        raise ValueError(
            "Non-positive HAC variance; the truncated kernel estimate "
            "failed. Reduce the bandwidth."
        )

    total = float(((y - y.mean()) ** 2).sum())

    r_squared = (
        1.0 - float((residuals**2).sum()) / total
        if total > 0
        else float("nan")
    )

    return HACRegression(
        coefficients=beta,
        standard_errors=np.sqrt(variances),
        n_obs=n,
        n_params=k,
        lags=lags,
        r_squared=r_squared,
    )


# ============================================================
# Correlation as a standardised regression
# ============================================================

def standardise(values: np.ndarray) -> np.ndarray:
    """Demean and scale to unit variance, failing on a constant series."""

    values = np.asarray(values, dtype=float)

    spread = values.std(ddof=1)

    if not np.isfinite(spread) or spread <= 0:
        raise ValueError(
            "Cannot standardise a constant or degenerate series."
        )

    return (values - values.mean()) / spread


@dataclass(frozen=True)
class CorrelationEstimate:
    """A correlation with autocorrelation-robust inference."""

    correlation: float
    standard_error: float
    t_statistic: float
    p_value: float
    ci_low: float
    ci_high: float
    n_obs: int
    lags: int


def correlation_hac(
    x: np.ndarray,
    y: np.ndarray,
    min_lags: int = 0,
) -> CorrelationEstimate:
    """
    Pearson correlation with a Newey-West standard error.

    The point estimate is the ordinary sample correlation. Its standard
    error comes from regressing standardised y on standardised x, whose
    slope equals that correlation exactly. Treating the standardising
    moments as known understates uncertainty slightly; the effect is
    second order and far smaller than the serial-correlation adjustment
    this function exists to make.
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.shape != y.shape:
        raise ValueError(f"Shape mismatch: {x.shape} against {y.shape}")

    n = x.shape[0]

    lags = newey_west_bandwidth(n, min_lags=min_lags)

    design = np.column_stack([np.ones(n), standardise(x)])

    fit = newey_west_ols(standardise(y), design, lags)

    slope = float(fit.coefficients[1])

    sample = float(np.corrcoef(x, y)[0, 1])

    if not np.isclose(slope, sample, atol=1e-8):
        raise ValueError(
            "Standardised slope does not equal the sample correlation "
            f"({slope:.10f} vs {sample:.10f}); the standardisation is "
            "wrong."
        )

    low, high = fit.confidence_interval(1)

    return CorrelationEstimate(
        correlation=sample,
        standard_error=float(fit.standard_errors[1]),
        t_statistic=float(fit.t_statistics[1]),
        p_value=float(fit.p_values[1]),
        ci_low=low,
        ci_high=high,
        n_obs=n,
        lags=lags,
    )


# ============================================================
# Moving-block bootstrap
# ============================================================

# Seed and draw count are fixed here so that every reported interval is
# reproducible without the caller having to remember to pass them.
BOOTSTRAP_SEED = 20250824
N_BOOTSTRAP = 10_000


def moving_block_bootstrap_ci(
    x: np.ndarray,
    y: np.ndarray,
    block_length: int,
    n_draws: int = N_BOOTSTRAP,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    """
    Percentile interval for a correlation under serial dependence.

    Blocks of consecutive (x, y) pairs are resampled together, which
    preserves both the autocorrelation within each series and the
    contemporaneous relationship between them. An ordinary pairwise
    bootstrap would destroy the former and give intervals as
    misleadingly tight as the iid formula.

    Draws in which either resampled series is constant are discarded
    rather than counted as zero correlation.
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    n = x.shape[0]

    if block_length < 1 or block_length > n:
        raise ValueError(
            f"Block length {block_length} is not usable with {n} "
            "observations."
        )

    rng = np.random.default_rng(seed)

    n_blocks = int(np.ceil(n / block_length))

    last_start = n - block_length

    offsets = np.arange(block_length)

    draws = []

    for _ in range(n_draws):
        starts = rng.integers(0, last_start + 1, n_blocks)

        index = (starts[:, None] + offsets[None, :]).ravel()[:n]

        xb = x[index]
        yb = y[index]

        if xb.std() <= 0 or yb.std() <= 0:
            continue

        draws.append(float(np.corrcoef(xb, yb)[0, 1]))

    if len(draws) < 0.5 * n_draws:
        raise ValueError(
            f"Only {len(draws)} of {n_draws} bootstrap draws were "
            "usable; the series is close to constant."
        )

    low, high = np.percentile(draws, [2.5, 97.5])

    return float(low), float(high)


def bootstrap_block_length(n_obs: int, min_block: int = 1) -> int:
    """
    Block length for the moving-block bootstrap, ceil(n^(1/3)).

    `min_block` raises the floor so that a known MA order from
    overlapping differences is never shorter than the block.
    """

    if n_obs <= 0:
        raise ValueError(f"n_obs must be positive, got {n_obs}")

    return max(int(np.ceil(n_obs ** (1.0 / 3.0))), min_block)


# ============================================================
# Alternatives to the HAC sandwich
# ============================================================

def driscoll_kraay_ols(
    y: np.ndarray,
    X: np.ndarray,
    unit: np.ndarray,
    time: np.ndarray,
    lags: int,
) -> HACRegression:
    """
    Driscoll-Kraay standard errors for a panel.

    The estimator aggregates the score across units at each date and
    applies a Bartlett kernel to the resulting sequence of cross-
    sectional sums, which makes it robust to correlation between units
    as well as over time.

    With a single unit the cross-sectional sum is the score itself, so
    the estimator coincides exactly with Newey-West. That is worth
    knowing before reaching for it: on a single time series it is not an
    alternative to the HAC sandwich, it *is* the HAC sandwich. It buys
    something only when there is a cross-section whose dependence is
    otherwise unmodelled.
    """

    y = np.asarray(y, float)
    X = np.asarray(X, float)

    n, k = X.shape

    xtx_inv = np.linalg.inv(X.T @ X)
    beta = xtx_inv @ (X.T @ y)
    resid = y - X @ beta

    scores = X * resid[:, None]

    # Aggregate the score across units within each date.
    dates = np.unique(time)
    h = np.zeros((len(dates), k))

    for j, d in enumerate(dates):
        h[j] = scores[time == d].sum(axis=0)

    omega = h.T @ h

    for lag in range(1, lags + 1):
        weight = 1.0 - lag / (lags + 1.0)
        gamma = h[lag:].T @ h[:-lag]
        omega += weight * (gamma + gamma.T)

    omega *= n / (n - k)

    cov = xtx_inv @ omega @ xtx_inv
    var = np.diag(cov)

    if (var <= 0).any():
        raise ValueError("Non-positive Driscoll-Kraay variance.")

    total = float(((y - y.mean()) ** 2).sum())

    return HACRegression(
        coefficients=beta,
        standard_errors=np.sqrt(var),
        n_obs=n,
        n_params=k,
        lags=lags,
        r_squared=1.0 - float((resid**2).sum()) / total,
    )


def autocorrelations(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Sample autocorrelations of x at lags 1..max_lag."""

    x = np.asarray(x, float)
    x = x - x.mean()

    denom = float(x @ x)

    return np.array(
        [float(x[l:] @ x[:-l]) / denom for l in range(1, max_lag + 1)]
    )


def effective_sample_size(
    x: np.ndarray,
    y: np.ndarray,
    max_lag: int | None = None,
) -> float:
    """
    Effective sample size for a correlation between two autocorrelated
    series, following Pyper and Peterman (1998).

    Two persistent series carry less independent information than their
    length suggests. The adjustment estimates how much less:

        1/N_eff = 1/N + (2/N) * sum_j r_xx(j) r_yy(j)

    summed to a cutoff conventionally set at N/5. This is a parametric
    alternative to the HAC sandwich: instead of inflating the variance
    of a slope, it deflates the degrees of freedom of a correlation.
    It assumes only that the autocorrelation structure is stationary,
    and involves no kernel, no bandwidth and no resampling.
    """

    n = len(x)

    if max_lag is None:
        max_lag = max(1, n // 5)

    rx = autocorrelations(x, max_lag)
    ry = autocorrelations(y, max_lag)

    inverse = 1.0 / n + (2.0 / n) * float((rx * ry).sum())

    if inverse <= 0:
        return float(n)

    return min(float(n), 1.0 / inverse)


@dataclass(frozen=True)
class AdjustedCorrelation:
    """A correlation tested against a deflated degrees of freedom."""

    correlation: float
    n_obs: int
    n_effective: float
    t_statistic: float
    p_value: float


def correlation_effective_n(
    x: np.ndarray,
    y: np.ndarray,
) -> AdjustedCorrelation:
    """
    Test a correlation using the effective sample size.

    The statistic is the ordinary t for a correlation, but evaluated
    with N_eff in place of N:

        t = r * sqrt((N_eff - 2) / (1 - r^2))

    on N_eff - 2 degrees of freedom.
    """

    x = np.asarray(x, float)
    y = np.asarray(y, float)

    r = float(np.corrcoef(x, y)[0, 1])

    n_eff = effective_sample_size(x, y)

    if n_eff <= 2.0 or abs(r) >= 1.0:
        return AdjustedCorrelation(r, len(x), n_eff, float("nan"), float("nan"))

    t = r * sqrt((n_eff - 2.0) / (1.0 - r**2))

    return AdjustedCorrelation(
        correlation=r,
        n_obs=len(x),
        n_effective=n_eff,
        t_statistic=t,
        p_value=two_sided_t_pvalue(t, n_eff - 2.0),
    )


def prais_winsten_correlation(
    x: np.ndarray,
    y: np.ndarray,
) -> AdjustedCorrelation:
    """
    Correlation inference after removing an AR(1) in the errors.

    Rather than leaving the serial correlation in place and widening the
    standard error, this models it: estimate rho from the OLS residuals,
    quasi-difference both series, and test the transformed regression
    with conventional errors. Parametric where the HAC sandwich is not,
    and correct only if AR(1) is the right description of the
    dependence.

    The reported correlation is still the raw one; only the test
    changes.
    """

    x = np.asarray(x, float)
    y = np.asarray(y, float)

    n = len(x)
    r = float(np.corrcoef(x, y)[0, 1])

    xs, ys = standardise(x), standardise(y)

    design = np.column_stack([np.ones(n), xs])
    beta = np.linalg.lstsq(design, ys, rcond=None)[0]
    resid = ys - design @ beta

    rho = float(resid[1:] @ resid[:-1]) / float(resid[:-1] @ resid[:-1])
    rho = float(np.clip(rho, -0.99, 0.99))

    # Prais-Winsten: quasi-difference, retaining the first observation
    # with the scaling that makes its variance comparable.
    scale = sqrt(1.0 - rho**2)

    yt = np.concatenate([[scale * ys[0]], ys[1:] - rho * ys[:-1]])
    xt = np.concatenate([[scale * xs[0]], xs[1:] - rho * xs[:-1]])
    ct = np.concatenate([[scale], np.full(n - 1, 1.0 - rho)])

    Z = np.column_stack([ct, xt])

    b = np.linalg.lstsq(Z, yt, rcond=None)[0]
    e = yt - Z @ b

    dof = n - 2
    s2 = float(e @ e) / dof
    se = sqrt(s2 * float(np.linalg.inv(Z.T @ Z)[1, 1]))

    t = float(b[1]) / se

    return AdjustedCorrelation(
        correlation=r,
        n_obs=n,
        n_effective=float(dof + 2),
        t_statistic=t,
        p_value=two_sided_t_pvalue(t, dof),
    )
