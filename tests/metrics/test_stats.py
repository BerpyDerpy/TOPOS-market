"""Unit tests for the metrics statistics helpers."""

from __future__ import annotations

import math
import random

from topos.metrics.stats import (
    fit_decay,
    iqr,
    max_drawdown,
    median,
    ols,
    paired_deltas,
    paired_ratios,
    t_interval,
)


def test_ols_recovers_line() -> None:
    xs = [float(i) for i in range(50)]
    ys = [3.0 + 2.0 * x for x in xs]
    fit = ols(xs, ys)
    assert abs(fit.slope - 2.0) < 1e-12
    assert abs(fit.intercept - 3.0) < 1e-10
    assert fit.r2 > 0.999999
    lo, hi = fit.slope_ci95
    assert lo <= 2.0 <= hi


def test_ols_degenerate_x_is_nan_not_crash() -> None:
    fit = ols([1.0] * 10, [float(i) for i in range(10)])
    assert math.isnan(fit.slope)
    assert math.isinf(fit.se_slope)


def test_fit_decay_recovers_rate() -> None:
    rng = random.Random(7)
    k_true = 0.02
    series = [
        1.0 if rng.random() < 0.8 * math.exp(-k_true * t) else 0.0
        for t in range(400)
    ]
    fit = fit_decay(series)
    assert 0.5 * k_true < fit.k < 2.0 * k_true
    assert fit.a0 > 0.0
    assert fit.se > 0.0


def test_fit_decay_flat_series_has_k_near_zero() -> None:
    series = [1.0, 0.0] * 200  # constant rate 0.5
    fit = fit_decay(series)
    assert abs(fit.k) < 1e-3


def test_fit_decay_full_quiescence_reads_as_strong_decay() -> None:
    """A series that goes completely quiet is the STRONGEST decay; the
    estimator must not choke on the all-zero tail (the log-regression
    floor bias this estimator replaced)."""
    series = [1.0] * 60 + [0.0] * 240
    fit = fit_decay(series)
    assert fit.k > 3.0 * fit.se > 0.0  # decisively positive
    assert math.isfinite(fit.se)


def test_fit_decay_growth_is_negative_k() -> None:
    series = [0.0] * 200 + [1.0] * 100
    fit = fit_decay(series)
    assert fit.k < 0.0


def test_fit_decay_too_short_is_nan() -> None:
    fit = fit_decay([1.0, 0.0])
    assert math.isnan(fit.k)


def test_max_drawdown() -> None:
    assert max_drawdown([0, 5, 3, 8, 2, 4]) == 6.0
    assert max_drawdown([0, 1, 2, 3]) == 0.0
    assert max_drawdown([]) == 0.0


def test_median_and_iqr() -> None:
    assert median([3.0, 1.0, 2.0]) == 2.0
    assert median([4.0, 1.0, 2.0, 3.0]) == 2.5
    assert math.isnan(median([]))
    assert abs(iqr([1.0, 2.0, 3.0, 4.0, 5.0]) - 2.0) < 1e-12


def test_t_interval_zero_exclusion() -> None:
    tight = t_interval([1.0, 1.1, 0.9, 1.05, 0.95])
    assert tight.excludes_zero()
    loose = t_interval([1.0, -1.2, 0.8, -0.9, 0.3])
    assert loose.includes_zero()
    assert t_interval([]).n == 0


def test_paired_helpers_use_shared_seeds_only() -> None:
    a = {1: 10.0, 2: 20.0, 3: 30.0}
    b = {2: 5.0, 3: 10.0, 4: 99.0}
    assert paired_deltas(a, b) == [15.0, 20.0]
    assert paired_ratios(a, b) == [4.0, 3.0]
    assert paired_ratios({1: 1.0}, {1: 0.0}) == [math.inf]
