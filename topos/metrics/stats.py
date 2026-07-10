"""Small statistics helpers shared by the metric modules.

Everything is deliberately elementary — OLS with standard errors, t
confidence intervals, medians, an exponential-decay fit on binned event
rates — so that every number in the report is auditable by hand. Medians
and paired-seed deltas are first-class per the fairness rules (medians
over means; report dispersion).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

from scipy import stats as scipy_stats


@dataclass(frozen=True)
class OLSFit:
    slope: float
    intercept: float
    se_slope: float
    r2: float
    n: int

    @property
    def slope_ci95(self) -> tuple[float, float]:
        if self.n < 3 or not math.isfinite(self.se_slope):
            return (-math.inf, math.inf)
        t = float(scipy_stats.t.ppf(0.975, self.n - 2))
        return (self.slope - t * self.se_slope, self.slope + t * self.se_slope)


def ols(x: Sequence[float], y: Sequence[float]) -> OLSFit:
    """Ordinary least squares y = a + b x with the classic slope SE."""
    n = len(x)
    if n != len(y):
        raise ValueError(f"length mismatch: {n} vs {len(y)}")
    if n < 2:
        return OLSFit(math.nan, math.nan, math.inf, math.nan, n)
    mx = sum(x) / n
    my = sum(y) / n
    sxx = sum((xi - mx) ** 2 for xi in x)
    if sxx == 0.0:
        return OLSFit(math.nan, my, math.inf, math.nan, n)
    sxy = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    slope = sxy / sxx
    intercept = my - slope * mx
    residuals = [yi - (intercept + slope * xi) for xi, yi in zip(x, y)]
    ssr = sum(r * r for r in residuals)
    syy = sum((yi - my) ** 2 for yi in y)
    r2 = 1.0 - ssr / syy if syy > 0 else math.nan
    if n > 2:
        sigma2 = ssr / (n - 2)
        se_slope = math.sqrt(sigma2 / sxx)
    else:
        se_slope = math.inf
    return OLSFit(slope, intercept, se_slope, r2, n)


def median(values: Sequence[float]) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2:
        return float(ordered[mid])
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def iqr(values: Sequence[float]) -> float:
    """Interquartile range — the report's dispersion companion to medians."""
    if len(values) < 2:
        return math.nan
    lo, hi = _quantile(values, 0.25), _quantile(values, 0.75)
    return hi - lo


def _quantile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    pos = q * (len(ordered) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(ordered[lo])
    frac = pos - lo
    return float(ordered[lo] * (1 - frac) + ordered[hi] * frac)


@dataclass(frozen=True)
class TInterval:
    mean: float
    lo: float
    hi: float
    n: int

    def excludes_zero(self) -> bool:
        return (self.lo > 0.0) or (self.hi < 0.0)

    def includes_zero(self) -> bool:
        return not self.excludes_zero()


def t_interval(values: Sequence[float], level: float = 0.95) -> TInterval:
    """Two-sided t confidence interval for the mean of ``values``."""
    finite = [v for v in values if math.isfinite(v)]
    n = len(finite)
    if n == 0:
        return TInterval(math.nan, math.nan, math.nan, 0)
    mean = sum(finite) / n
    if n == 1:
        return TInterval(mean, -math.inf, math.inf, 1)
    var = sum((v - mean) ** 2 for v in finite) / (n - 1)
    se = math.sqrt(var / n)
    t = float(scipy_stats.t.ppf(0.5 + level / 2.0, n - 1))
    return TInterval(mean, mean - t * se, mean + t * se, n)


def paired_deltas(
    a_by_seed: Mapping[int, float], b_by_seed: Mapping[int, float]
) -> list[float]:
    """a - b over the seeds both mappings cover (paired-seed comparison)."""
    shared = sorted(set(a_by_seed) & set(b_by_seed))
    return [a_by_seed[s] - b_by_seed[s] for s in shared]


def paired_ratios(
    a_by_seed: Mapping[int, float], b_by_seed: Mapping[int, float]
) -> list[float]:
    """a / b over shared seeds; a zero denominator yields inf (flagged, not
    silently dropped — a condition that never acts is itself a finding)."""
    shared = sorted(set(a_by_seed) & set(b_by_seed))
    out: list[float] = []
    for s in shared:
        denominator = b_by_seed[s]
        out.append(a_by_seed[s] / denominator if denominator else math.inf)
    return out


@dataclass(frozen=True)
class DecayFit:
    """Exponential-decay estimate rate(t) ~ A * exp(-k t) for an event series.

    ``k`` (per step) > 0 means the rate decays. Estimated by the
    early-vs-late Poisson log-ratio: the series is split into thirds and

        k = ln((c_early + 1/2) / (c_late + 1/2)) / (2 L / 3)

    with c_* the event counts in the first and last thirds and L the
    series length. For a true exponential the ratio of equal-window
    averages is exactly exp(k * window offset), so the estimator is
    consistent; the 1/2 is the standard continuity correction, which
    keeps all-zero windows finite (a series that goes fully quiescent is
    STRONG decay and must not read as 'unfittable' — a log-regression on
    binned rates hits its epsilon floor there and biases k toward 0 for
    exactly the strongest decays). ``se`` is the delta-method Poisson
    standard error sqrt(1/(c_e + 1/2) + 1/(c_l + 1/2)) / (2L/3).
    """

    k: float
    se: float
    n_bins: int
    """Number of comparison windows (always 3: early/middle/late)."""
    r2: float
    """Not defined for the ratio estimator; always NaN."""
    a0: float
    """Observed early-window rate — the babbling amplitude."""


def fit_decay(events_per_step: Sequence[float]) -> DecayFit:
    """Estimate the exponential decay rate of a per-step event series."""
    n = len(events_per_step)
    if n < 3:
        return DecayFit(math.nan, math.inf, 3, math.nan, math.nan)
    third = n // 3
    c_early = float(sum(events_per_step[:third]))
    c_late = float(sum(events_per_step[n - third :]))
    offset = float(n - third)  # distance between the window starts
    k = math.log((c_early + 0.5) / (c_late + 0.5)) / offset
    se = math.sqrt(1.0 / (c_early + 0.5) + 1.0 / (c_late + 0.5)) / offset
    return DecayFit(
        k=k,
        se=se,
        n_bins=3,
        r2=math.nan,
        a0=c_early / third,
    )


def max_drawdown(series: Sequence[float]) -> float:
    """Largest peak-to-trough decline of a cumulative series (>= 0)."""
    peak = -math.inf
    worst = 0.0
    for value in series:
        peak = max(peak, value)
        worst = max(worst, peak - value)
    return worst
