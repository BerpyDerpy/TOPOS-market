"""Unit tests for the shared conjugate machinery in topos.beliefs.core."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import special, stats

from topos.beliefs.core import (
    GammaPosterior,
    InverseGammaPosterior,
    SurpriseTracker,
    forget_stats,
    information_gain_terms,
    negative_binomial_log_pmf,
    poisson_entropy_nats,
    quantile_quadrature,
)


# -- the EIG identity ---------------------------------------------------------


def test_information_gain_terms_is_the_mi_identity() -> None:
    weights = np.array([0.25, 0.75])
    conditional = np.array([1.0, 2.0])
    terms = information_gain_terms(3.0, conditional, weights)
    assert terms.expected_conditional_entropy_nats == pytest.approx(1.75)
    assert terms.eig_nats == pytest.approx(3.0 - 1.75)
    assert terms.predictive_entropy_nats == pytest.approx(3.0)
    # The identity: eig = predictive - aleatoric, always.
    assert terms.eig_nats == pytest.approx(
        terms.predictive_entropy_nats - terms.expected_conditional_entropy_nats
    )


def test_information_gain_terms_rejects_bad_weights() -> None:
    with pytest.raises(ValueError):
        information_gain_terms(1.0, np.array([1.0, 1.0]), np.array([0.4, 0.4]))
    with pytest.raises(ValueError):
        information_gain_terms(1.0, np.array([1.0]), np.array([0.5, 0.5]))


# -- quadrature ---------------------------------------------------------------


@pytest.mark.parametrize(
    "a,b", [(1.5, 0.5), (3.0, 2.0), (12.0, 30.0), (60.0, 90.0), (2500.0, 2600.0)]
)
def test_quantile_quadrature_inverse_gamma_log_moment(a: float, b: float) -> None:
    """E[ln c] under InvGamma(a, b) = ln b - psi(a), to high accuracy for
    every concentration regime the modules can reach."""
    nodes, weights = quantile_quadrature(
        lambda u: stats.invgamma.ppf(u, a, scale=b)
    )
    assert weights.sum() == pytest.approx(1.0, abs=1e-12)
    approx = float(np.dot(weights, np.log(nodes)))
    exact = math.log(b) - float(special.digamma(a))
    assert approx == pytest.approx(exact, abs=1e-6)


@pytest.mark.parametrize("a,b", [(1.0, 1.0), (2.0, 0.5), (50.0, 25.0), (4000.0, 1000.0)])
def test_quantile_quadrature_gamma_log_moment(a: float, b: float) -> None:
    nodes, weights = quantile_quadrature(
        lambda u: stats.gamma.ppf(u, a, scale=1.0 / b)
    )
    approx = float(np.dot(weights, np.log(nodes)))
    exact = float(special.digamma(a)) - math.log(b)
    assert approx == pytest.approx(exact, abs=1e-6)


# -- forgetting ---------------------------------------------------------------


def test_forget_stats_moves_toward_prior() -> None:
    out = forget_stats(np.array([10.0, 20.0]), np.array([2.0, 4.0]), 0.5)
    assert out == pytest.approx([6.0, 12.0])


def test_forget_stats_rho_one_is_identity() -> None:
    stats_now = np.array([10.0, 20.0])
    out = forget_stats(stats_now, np.array([2.0, 4.0]), 1.0)
    assert (out == stats_now).all()


@pytest.mark.parametrize("rho", [0.0, -0.5, 1.0001])
def test_forget_stats_rejects_bad_rho(rho: float) -> None:
    with pytest.raises(ValueError):
        forget_stats(np.array([1.0]), np.array([1.0]), rho)


# -- surprise (salience only) -------------------------------------------------


def test_surprise_tracker_warmup_and_outlier() -> None:
    tracker = SurpriseTracker(ewma_decay=0.1)
    assert tracker.score(2.0) == 0.0
    assert tracker.score(2.1) == 0.0
    for _ in range(50):
        tracker.score(2.0 + 0.1 * np.random.default_rng(1).standard_normal())
    calm = abs(tracker.score(2.05))
    shocked = tracker.score(8.0)
    assert shocked > 3.0
    assert shocked > calm
    assert tracker.last_z == shocked


def test_surprise_tracker_validates_parameters() -> None:
    with pytest.raises(ValueError):
        SurpriseTracker(ewma_decay=0.0)
    with pytest.raises(ValueError):
        SurpriseTracker(ewma_decay=1.0)
    with pytest.raises(ValueError):
        SurpriseTracker(warmup=1)


# -- Gamma cell ---------------------------------------------------------------


def test_gamma_posterior_conjugate_update_and_sufficiency() -> None:
    incremental = GammaPosterior(2.0, 0.5)
    rng = np.random.default_rng(3)
    counts = rng.poisson(3.0, 200)
    for y in counts:
        incremental.observe(int(y), 1.0)
    batch = GammaPosterior(2.0, 0.5)
    batch.observe(int(counts.sum()), 200.0)
    assert incremental.a == pytest.approx(batch.a)
    assert incremental.b == pytest.approx(batch.b)


def test_gamma_posterior_predictive_is_normalized() -> None:
    cell = GammaPosterior(2.0, 0.5)
    y = np.arange(4000, dtype=np.float64)
    total = np.exp(cell.predictive_log_pmf(y, 5.0)).sum()
    assert total == pytest.approx(1.0, abs=1e-9)


def test_gamma_posterior_entropy_matches_scipy() -> None:
    cell = GammaPosterior(7.0, 3.0)
    assert cell.entropy_nats() == pytest.approx(
        float(stats.gamma.entropy(7.0, scale=1.0 / 3.0))
    )


def test_gamma_posterior_validates() -> None:
    with pytest.raises(ValueError):
        GammaPosterior(0.0, 1.0)
    cell = GammaPosterior(1.0, 1.0)
    with pytest.raises(ValueError):
        cell.observe(-1, 1.0)
    with pytest.raises(ValueError):
        cell.observe(1, 0.0)
    with pytest.raises(ValueError):
        cell.eig_terms(0.0)


def test_negative_binomial_log_pmf_matches_scipy_for_integer_shape() -> None:
    y = np.arange(30, dtype=np.float64)
    ours = negative_binomial_log_pmf(y, 4.0, 0.3)
    scipys = stats.nbinom.logpmf(y, 4, 0.3)
    assert ours == pytest.approx(scipys)


@pytest.mark.parametrize("mu", [0.5, 3.0, 40.0])
def test_poisson_entropy_matches_scipy(mu: float) -> None:
    ours = float(poisson_entropy_nats(np.array([mu]))[0])
    assert ours == pytest.approx(float(stats.poisson.entropy(mu)), abs=1e-9)


def test_poisson_entropy_of_zero_rate_is_zero() -> None:
    assert poisson_entropy_nats(np.array([0.0]))[0] == 0.0


# -- InverseGamma cell --------------------------------------------------------


def test_inverse_gamma_eig_matches_fully_closed_form() -> None:
    """Quadrature EIG vs the all-closed-form value: Student-t predictive
    entropy minus 0.5*(ln(2*pi*e*s0) + ln b - psi(a))."""
    for a, b, s0 in [(3.0, 6.0, 2.0), (12.0, 30.0, 1.0), (800.0, 900.0, 1.0)]:
        post = InverseGammaPosterior(a, b)
        terms = post.eig_terms_for_gaussian(s0)
        df, t_scale = post.student_t_predictive(s0)
        closed = float(stats.t.entropy(df, scale=t_scale)) - 0.5 * (
            math.log(2.0 * math.pi * math.e * s0)
            + math.log(b)
            - float(special.digamma(a))
        )
        assert terms.eig_nats == pytest.approx(closed, abs=1e-6)


def test_inverse_gamma_validates() -> None:
    with pytest.raises(ValueError):
        InverseGammaPosterior(1.0, 1.0)  # needs a finite mean: a > 1
    post = InverseGammaPosterior(3.0, 2.0)
    with pytest.raises(ValueError):
        post.observe_standardized_square(-0.1)
    with pytest.raises(ValueError):
        post.student_t_predictive(0.0)


def test_inverse_gamma_conjugate_update() -> None:
    post = InverseGammaPosterior(3.0, 2.0)
    post.observe_standardized_square(4.0)
    assert post.a == pytest.approx(3.5)
    assert post.b == pytest.approx(4.0)
