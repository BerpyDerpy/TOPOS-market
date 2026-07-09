"""ImpactModel: conjugate recovery, EIG identity, action dependence.

Ground truth for THIS model in the running system is the harness
counterfactual (P3's ``impact()``), scored in P13 — everything here is
synthetic-only by design: it pins the conjugate arithmetic, the INV-3
identity (closed form == brute-force MI), the own-action conditioning,
and the regressor extraction from observations.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tests.beliefs.conftest import make_obs, null_probe
from tests.selfmodel.conftest import committed_probe, events, plain_obs
from topos.beliefs.core import LOG_2PIE
from topos.contracts.beliefs import ProbeSpec
from topos.contracts.intent import IMPACT
from topos.contracts.market import (
    GTC,
    Ack,
    AckStatus,
    Fill,
    Liquidity,
    PlaceLimit,
    Side,
)
from topos.selfmodel import ImpactModel

ATOL = 0.015
RTOL = 0.03


# =====================================================================
# Conjugate recovery on synthetic data
# =====================================================================


def test_recovers_regression_coefficients() -> None:
    rng = np.random.default_rng(42)
    w_true = np.array([0.5, 0.4, 0.15, 0.8])
    sigma = 0.7
    model = ImpactModel(n_context=1)
    entropy_start = model.posterior_entropy_nats()
    for _ in range(4000):
        aggression = float(rng.integers(-5, 6))
        resting = float(rng.integers(-8, 9))
        context = float(rng.normal())
        x = np.array([1.0, aggression, resting, context])
        y = float(w_true @ x + rng.normal(0.0, sigma))
        model.observe_point(aggression, resting, (context,), y)
    assert np.allclose(model.coef_mean, w_true, atol=0.05)
    assert model.noise_scale_posterior.mean() == pytest.approx(
        sigma * sigma, rel=0.15
    )
    assert model.posterior_entropy_nats() < entropy_start


def test_update_extracts_own_action_regressors() -> None:
    """Crossing buy of 3 lots followed by a +2 mid move must yield the
    data point x = [1, +3, 0, ctx], y = +2 — the own-action conditioning
    the design's Layer-1 depends on."""
    model = ImpactModel(n_context=1)
    model.set_context_regressors((0.5,))
    model.update(plain_obs(0), events(0))  # mid 1000
    place = PlaceLimit(Side.BUY, 1001, 3, tif_steps=GTC)
    ack = Ack(order_id=0, status=AckStatus.ACCEPTED, step=0)
    fill = Fill(order_id=0, price_ticks=1001, size_lots=3,
                liquidity=Liquidity.TAKER, step=0)
    shifted_bids = [(1001 - i, 20) for i in range(10)]
    shifted_asks = [(1003 + i, 20) for i in range(10)]
    obs1 = make_obs(1, shifted_bids, shifted_asks,
                    own_acks=(ack,), own_fills=(fill,))
    model.update(obs1, events(1, messages=(place,), acks=(ack,), fills=(fill,)))
    assert model.last_point is not None
    x, y = model.last_point
    assert x.tolist() == [1.0, 3.0, 0.0, 0.5]
    assert y == pytest.approx(2.0)  # mid 1000 -> 1002


def test_update_counts_resting_size_at_touch() -> None:
    """A passive bid resting at the previous best contributes to the
    resting-at-touch regressor (signed +), with zero aggression."""
    model = ImpactModel(n_context=0)
    model.update(plain_obs(0), events(0))
    place = PlaceLimit(Side.BUY, 999, 4, tif_steps=GTC)
    ack = Ack(order_id=0, status=AckStatus.ACCEPTED, step=0)
    model.update(
        plain_obs(1, own_acks=(ack,)),
        events(1, messages=(place,), acks=(ack,)),
    )
    assert model.last_point is not None
    x, y = model.last_point
    assert x.tolist() == [1.0, 0.0, 4.0]
    assert y == pytest.approx(0.0)
    # Sell-side aggression comes out negative.
    place2 = PlaceLimit(Side.SELL, 999, 2, tif_steps=GTC)
    ack2 = Ack(order_id=1, status=AckStatus.ACCEPTED, step=1)
    fill2 = Fill(order_id=1, price_ticks=999, size_lots=2,
                 liquidity=Liquidity.TAKER, step=1)
    model.update(
        plain_obs(2, own_acks=(ack2,), own_fills=(fill2,)),
        events(2, messages=(place2,), acks=(ack2,), fills=(fill2,)),
    )
    assert model.last_point is not None
    x, _ = model.last_point
    assert x[1] == -2.0


# =====================================================================
# INV-3: eig_nats must BE mutual information
# =====================================================================


def mc_nig_mi(
    m: np.ndarray,
    v: np.ndarray,
    a: float,
    b: float,
    x: np.ndarray,
    rng: np.random.Generator,
    n_outer: int = 30_000,
    n_inner: int = 3_000,
) -> float:
    """Brute-force I((w, sigma^2); Y | x) for the NIG regression: sample
    the parameters, sample Y, estimate the marginal density by a second
    independent Monte Carlo — no Student-t closed form anywhere."""
    d = len(m)
    chol = np.linalg.cholesky(v)

    def draw(n: int) -> tuple[np.ndarray, np.ndarray]:
        sigma2 = b / rng.gamma(a, 1.0, n)
        w = m + (chol @ rng.standard_normal((d, n))).T * np.sqrt(sigma2)[:, None]
        return w, sigma2

    w_outer, sigma2_outer = draw(n_outer)
    y = w_outer @ x + rng.normal(0.0, np.sqrt(sigma2_outer))
    w_inner, sigma2_inner = draw(n_inner)
    mu_inner = w_inner @ x
    log_marginal = np.empty(n_outer)
    for start in range(0, n_outer, 2_000):
        block = y[start : start + 2_000, np.newaxis]
        dens = np.exp(-0.5 * (block - mu_inner) ** 2 / sigma2_inner) / np.sqrt(
            2.0 * math.pi * sigma2_inner
        )
        log_marginal[start : start + 2_000] = np.log(dens.mean(axis=1))
    h_marginal = float(-np.mean(log_marginal))
    h_conditional = float(np.mean(0.5 * (LOG_2PIE + np.log(sigma2_outer))))
    return h_marginal - h_conditional


def assert_close(analytic: float, mc: float, label: str) -> None:
    tol = ATOL + RTOL * abs(analytic)
    assert abs(analytic - mc) <= tol, (
        f"{label}: analytic {analytic:.5f} vs MC {mc:.5f} "
        f"(|diff| {abs(analytic - mc):.5f} > tol {tol:.5f}) — eig_nats is "
        f"not mutual information"
    )


def _seeded_model(n_points: int, rng: np.random.Generator) -> ImpactModel:
    model = ImpactModel(n_context=1)
    w_true = np.array([0.2, 0.3, 0.1, 0.5])
    for _ in range(n_points):
        aggression = float(rng.integers(-4, 5))
        resting = float(rng.integers(-6, 7))
        context = float(rng.normal())
        x = np.array([1.0, aggression, resting, context])
        model.observe_point(
            aggression, resting, (context,), float(w_true @ x + rng.normal(0, 0.8))
        )
    return model


@pytest.mark.parametrize("n_points", [0, 60])
@pytest.mark.parametrize("aggression", [0.0, 5.0])
def test_impact_eig_matches_mc(n_points: int, aggression: float) -> None:
    rng = np.random.default_rng(777)
    model = _seeded_model(n_points, rng)
    model.set_context_regressors((0.4,))
    x = np.array([1.0, aggression, 0.0, 0.4])
    analytic = model.eig_terms_for_x(x).eig_nats
    mc = mc_nig_mi(
        np.asarray(model.coef_mean),
        np.asarray(model.coef_scale_free_cov),
        model.noise_scale_posterior.a,
        model.noise_scale_posterior.b,
        x,
        rng,
    )
    assert_close(analytic, mc, f"NIG n={n_points} aggression={aggression}")


# =====================================================================
# Action dependence and saturation (Layer-1 at the probe level)
# =====================================================================

AGGRESSIVE_PROBE = committed_probe(
    side=1.0, offset_ticks=-1.0, size_frac=1.0, horizon_steps=1, target_id=IMPACT
)


def test_eig_is_action_dependent_and_saturates() -> None:
    """Acting buys extra information along unexplored own-action
    directions (marginal over null > 0); once own aggression is
    well-learned the margin collapses — impact probes stop paying."""
    model = ImpactModel(n_context=0, size_budget_lots=10)
    model.update(plain_obs(0), events(0))
    probe_before = model.eig_nats(AGGRESSIVE_PROBE)
    null_before = model.eig_nats(null_probe(IMPACT, horizon_steps=1))
    margin_before = probe_before - null_before
    assert margin_before > 0.2, "acting must buy information a priori"

    rng = np.random.default_rng(5)
    for _ in range(600):
        aggression = float(rng.integers(-10, 11))
        model.observe_point(
            aggression, 0.0, (), float(0.3 * aggression + rng.normal(0, 0.5))
        )
    probe_after = model.eig_nats(AGGRESSIVE_PROBE)
    null_after = model.eig_nats(null_probe(IMPACT, horizon_steps=1))
    margin_after = probe_after - null_after
    assert probe_after < probe_before
    assert margin_after < 0.05 * margin_before


def test_null_probe_still_earns_positive_eig() -> None:
    """A mid move is observed whether or not the agent acts: the null
    keeps teaching the noise scale and control coefficients (INV-4 —
    world-side information rides the null)."""
    model = ImpactModel(n_context=0)
    model.update(plain_obs(0), events(0))
    assert model.eig_nats(null_probe(IMPACT, horizon_steps=1)) > 0.0


def test_forgetting_reinflates_the_posterior() -> None:
    rng = np.random.default_rng(9)
    model = _seeded_model(500, rng)
    entropy_before = model.posterior_entropy_nats()
    model.forget(0.3)
    assert model.posterior_entropy_nats() > entropy_before
    # Sufficient statistics head toward the prior.
    assert model.noise_scale_posterior.a < 3.0 + 0.5 * 500
    prior = ImpactModel(n_context=1)
    assert model.posterior_entropy_nats() <= prior.posterior_entropy_nats()


def test_validation() -> None:
    model = ImpactModel(n_context=1)
    with pytest.raises(ValueError):
        model.set_context_regressors((1.0, 2.0))
    with pytest.raises(ValueError):
        model.observe_point(0.0, 0.0, (1.0, 2.0), 0.0)
    bad = ProbeSpec(intent=AGGRESSIVE_PROBE.intent, horizon_steps=0)
    with pytest.raises(ValueError):
        model.eig_nats(bad)
    with pytest.raises(ValueError):
        ImpactModel(impact_horizon_steps=0)
