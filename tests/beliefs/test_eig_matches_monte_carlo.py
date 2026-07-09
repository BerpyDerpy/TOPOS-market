"""Drift tripwire for INV-3: eig_nats must BE mutual information.

For every belief module, over a grid of posterior states and probes, the
analytic/quadrature ``eig_nats`` must match a brute-force Monte Carlo
estimate of I(theta; Y): sample theta from the parameter posterior, sample
Y | theta, and estimate MI with no reuse of the module's closed forms —
marginal entropies come from empirical frequencies (discrete Y) or a
double-Monte-Carlo mixture estimate (continuous Y).

If eig_nats were implemented as predictive entropy instead of mutual
information, the MC estimate would disagree by exactly the aleatoric term
(order 1 nat) and every case here would fail. Do NOT weaken the stated
tolerances to make this pass; fix the math.

Stated tolerance: |analytic - MC| <= ATOL + RTOL * analytic, with
ATOL = 0.015 nats and RTOL = 3%, at the sample sizes fixed below (MC
standard error is ~0.002-0.005 nats, so the bound is ~3 sigma slack).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from tests.beliefs.conftest import empty_events, null_probe, obs_for_mid
from topos.beliefs import FairValueKF, FlowIntensity
from topos.contracts.beliefs import ProbeSpec

ATOL = 0.015
RTOL = 0.03
LOG_2PIE = math.log(2.0 * math.pi * math.e)


def mc_gamma_poisson_mi(
    a: float, b: float, exposure: float, rng: np.random.Generator, n: int = 400_000
) -> float:
    """Brute-force I(lambda; Y): empirical marginal entropy (Miller-Madow)
    minus the sampled conditional entropy -E log p(y | lambda)."""
    lam = rng.gamma(a, 1.0 / b, n)
    y = rng.poisson(lam * exposure)
    counts = np.bincount(y)
    p = counts[counts > 0] / n
    h_marginal = float(-np.sum(p * np.log(p)) + (len(p) - 1) / (2.0 * n))
    h_conditional = float(-np.mean(stats.poisson.logpmf(y, lam * exposure)))
    return h_marginal - h_conditional


def mc_scale_mixture_mi(
    a: float,
    b: float,
    scale_free_var: float,
    rng: np.random.Generator,
    n_outer: int = 30_000,
    n_inner: int = 3_000,
) -> float:
    """Brute-force I(c; Y) for Y | c ~ N(0, c * s0), c ~ InvGamma(a, b).

    The marginal density is estimated by a second, independent Monte Carlo
    over c (no Student-t closed form anywhere), chunked to bound memory.
    """
    c_outer = b / rng.gamma(a, 1.0, n_outer)
    y = rng.normal(0.0, np.sqrt(c_outer * scale_free_var))
    c_inner = b / rng.gamma(a, 1.0, n_inner)
    var_inner = c_inner * scale_free_var
    log_marginal = np.empty(n_outer)
    for start in range(0, n_outer, 2_000):
        block = y[start : start + 2_000, np.newaxis]
        dens = np.exp(-0.5 * block * block / var_inner) / np.sqrt(
            2.0 * math.pi * var_inner
        )
        log_marginal[start : start + 2_000] = np.log(dens.mean(axis=1))
    h_marginal = float(-np.mean(log_marginal))
    h_conditional = float(np.mean(0.5 * (LOG_2PIE + np.log(c_outer * scale_free_var))))
    return h_marginal - h_conditional


def mc_gaussian_state_mi(
    sigma_prior: np.ndarray,
    obs_noise_var: float,
    rng: np.random.Generator,
    n_outer: int = 30_000,
    n_inner: int = 3_000,
) -> float:
    """Brute-force I(x; Y) for Y | x ~ N(H x, R), x ~ N(0, Sigma_prior),
    H = [1, 0]; the marginal density again comes from a second MC over x."""
    chol = np.linalg.cholesky(sigma_prior)
    x_outer = (chol @ rng.standard_normal((2, n_outer))).T
    y = x_outer[:, 0] + rng.normal(0.0, math.sqrt(obs_noise_var), n_outer)
    x_inner = (chol @ rng.standard_normal((2, n_inner))).T
    mu_inner = x_inner[:, 0]
    log_marginal = np.empty(n_outer)
    norm = math.sqrt(2.0 * math.pi * obs_noise_var)
    for start in range(0, n_outer, 2_000):
        block = y[start : start + 2_000, np.newaxis]
        dens = np.exp(-0.5 * (block - mu_inner) ** 2 / obs_noise_var) / norm
        log_marginal[start : start + 2_000] = np.log(dens.mean(axis=1))
    h_marginal = float(-np.mean(log_marginal))
    h_conditional = 0.5 * (LOG_2PIE + math.log(obs_noise_var))
    return h_marginal - h_conditional


def assert_close(analytic: float, mc: float, label: str) -> None:
    tol = ATOL + RTOL * abs(analytic)
    assert abs(analytic - mc) <= tol, (
        f"{label}: analytic {analytic:.5f} vs MC {mc:.5f} "
        f"(|diff| {abs(analytic - mc):.5f} > tol {tol:.5f}) — eig_nats is "
        f"not mutual information"
    )


POSTERIOR_GRID = [(2.0, 0.5), (8.0, 2.0), (50.0, 25.0)]
HORIZONS = [1, 5]


@pytest.mark.parametrize("a,b", POSTERIOR_GRID)
@pytest.mark.parametrize("horizon", HORIZONS)
def test_flow_intensity_cell_eig_matches_mc(a: float, b: float, horizon: int) -> None:
    module = FlowIntensity()
    cell = module.cells[next(iter(module.cells))]
    cell.a, cell.b = a, b
    analytic = cell.eig_terms(float(horizon)).eig_nats
    rng = np.random.default_rng(20260709)
    mc = mc_gamma_poisson_mi(a, b, float(horizon), rng)
    assert_close(analytic, mc, f"Gamma-Poisson cell a={a} b={b} dt={horizon}")


def test_flow_intensity_module_eig_matches_mc() -> None:
    """Module-level eig_nats = sum of per-cell MI over heterogeneous cells."""
    module = FlowIntensity()
    rng = np.random.default_rng(42)
    grid = [(2.0, 0.5), (5.0, 1.5), (20.0, 8.0), (80.0, 40.0)]
    for i, cell in enumerate(module.cells.values()):
        cell.a, cell.b = grid[i % len(grid)]
    probe = null_probe(horizon_steps=3)
    analytic = module.eig_nats(probe)
    mc = sum(
        mc_gamma_poisson_mi(cell.a, cell.b, 3.0, rng, n=200_000)
        for cell in module.cells.values()
    )
    assert_close(analytic, mc, "FlowIntensity module (18 cells, h=3)")


def _prepared_kf(a: float, b: float) -> FairValueKF:
    module = FairValueKF(q_level=0.05, q_drift=0.005)
    for step in range(3):
        module.update(obs_for_mid(step, 1000.0 + step), empty_events(step))
    module.noise_scale_posterior.a = a
    module.noise_scale_posterior.b = b
    return module


@pytest.mark.parametrize("a,b", [(3.0, 6.0), (12.0, 30.0), (60.0, 90.0)])
@pytest.mark.parametrize("horizon", HORIZONS)
def test_fair_value_eig_matches_mc(a: float, b: float, horizon: int) -> None:
    module = _prepared_kf(a, b)
    probe = null_probe(horizon_steps=horizon)
    analytic = module.eig_nats(probe)
    _, s_star = module.horizon_prediction(horizon)
    rng = np.random.default_rng(90210)
    mc = mc_scale_mixture_mi(a, b, s_star, rng)
    assert_close(analytic, mc, f"FairValueKF scale a={a} b={b} h={horizon}")


@pytest.mark.parametrize("horizon", HORIZONS)
def test_fair_value_state_eig_matches_mc(horizon: int) -> None:
    """The closed form 0.5*ln(det Sigma_prior / det Sigma_post) must equal
    brute-force I(x; Y). Model shapes (F, Q0) are rebuilt from the module's
    documented constructor arguments."""
    q_level, q_drift = 0.05, 0.005
    module = _prepared_kf(12.0, 30.0)
    probe = null_probe(horizon_steps=horizon)
    analytic = module.state_eig_nats(probe)
    f = np.array([[1.0, 1.0], [0.0, 1.0]])
    q0 = np.diag([q_level, q_drift])
    p_h = module.state_scale_free_cov
    for _ in range(horizon):
        p_h = f @ p_h @ f.T + q0
    c_bar = module.noise_scale_posterior.mean()
    rng = np.random.default_rng(777)
    mc = mc_gaussian_state_mi(c_bar * p_h, c_bar * 1.0, rng)
    assert_close(analytic, mc, f"FairValueKF state EIG h={horizon}")


def test_probe_rejects_nonpositive_horizon() -> None:
    kf = _prepared_kf(3.0, 6.0)
    flow = FlowIntensity()
    bad = ProbeSpec(intent=null_probe().intent, horizon_steps=0)
    with pytest.raises(ValueError):
        kf.eig_nats(bad)
    with pytest.raises(ValueError):
        flow.eig_nats(bad)
