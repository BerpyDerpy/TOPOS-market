"""Forgetting (INV-2-compatible adaptation): after convergence,
forget(rho < 1) strictly increases posterior entropy AND the EIG of
informative probes; forget(1.0) is an exact no-op; entropy never decreases
under forgetting from the prior state.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.beliefs.conftest import empty_events, make_obs, null_probe, obs_for_mid
from topos.beliefs import FairValueKF, FlowIntensity
from topos.contracts.intent import FLOW_INTENSITY
from topos.contracts.market import Side, Trade

CONVERGE_STEPS = 600


def _converged_flow(rng: np.random.Generator) -> FlowIntensity:
    module = FlowIntensity()
    for step in range(CONVERGE_STEPS):
        lots = int(rng.poisson(3.0))
        trades = (
            (Trade(price_ticks=1001, size_lots=lots, aggressor=Side.BUY),)
            if lots > 0
            else ()
        )
        obs = make_obs(
            step,
            [(999 - i, 50) for i in range(10)],
            [(1001 + i, 50) for i in range(10)],
            trades=trades,
        )
        module.update(obs, empty_events(step))
    return module


def _converged_kf(rng: np.random.Generator) -> FairValueKF:
    module = FairValueKF()
    for step in range(CONVERGE_STEPS):
        y = 1000.0 + rng.normal(0.0, 3.0)
        module.update(obs_for_mid(step, y), empty_events(step))
    return module


@pytest.mark.parametrize("horizon", [1, 5])
def test_forgetting_reinflates_flow_intensity(horizon: int) -> None:
    module = _converged_flow(np.random.default_rng(31))
    probe = null_probe(FLOW_INTENSITY, horizon_steps=horizon)
    entropy_before = module.posterior_entropy_nats()
    eig_before = module.eig_nats(probe)
    module.forget(0.7)
    assert module.posterior_entropy_nats() > entropy_before
    assert module.eig_nats(probe) > eig_before


def test_forgetting_reinflates_fair_value() -> None:
    module = _converged_kf(np.random.default_rng(32))
    probe = null_probe(horizon_steps=1)
    entropy_before = module.posterior_entropy_nats()
    eig_before = module.eig_nats(probe)
    state_var_before = module.state_scale_free_cov[0, 0]
    module.forget(0.7)
    assert module.posterior_entropy_nats() > entropy_before
    assert module.eig_nats(probe) > eig_before
    # The state also reinflates toward its diffuse prior.
    assert module.state_scale_free_cov[0, 0] > state_var_before


def test_forget_rho_one_is_exact_noop() -> None:
    flow = _converged_flow(np.random.default_rng(33))
    kf = _converged_kf(np.random.default_rng(34))
    flow_params = [(cell.a, cell.b) for cell in flow.cells.values()]
    kf_scale = (kf.noise_scale_posterior.a, kf.noise_scale_posterior.b)
    kf_state = (kf.state_mean.copy(), kf.state_scale_free_cov.copy())
    flow.forget(1.0)
    kf.forget(1.0)
    assert [(cell.a, cell.b) for cell in flow.cells.values()] == flow_params
    assert (kf.noise_scale_posterior.a, kf.noise_scale_posterior.b) == kf_scale
    assert (kf.state_mean == kf_state[0]).all()
    assert (kf.state_scale_free_cov == kf_state[1]).all()


def test_forgetting_at_prior_leaves_entropy_unchanged() -> None:
    """Non-decreasing entropy, boundary case: with statistics at the prior,
    forgetting has nothing to discount."""
    flow = FlowIntensity()
    kf = FairValueKF()
    flow_entropy = flow.posterior_entropy_nats()
    kf_entropy = kf.posterior_entropy_nats()
    flow.forget(0.5)
    kf.forget(0.5)
    assert flow.posterior_entropy_nats() == pytest.approx(flow_entropy)
    assert kf.posterior_entropy_nats() == pytest.approx(kf_entropy)


@pytest.mark.parametrize("rho", [0.0, -0.2, 1.5])
def test_forget_rejects_out_of_range_rho(rho: float) -> None:
    flow = FlowIntensity()
    kf = FairValueKF()
    with pytest.raises(ValueError):
        flow.forget(rho)
    with pytest.raises(ValueError):
        kf.forget(rho)
