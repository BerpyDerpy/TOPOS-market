"""The noisy-TV property: pure noise stays SURPRISING but stops being
INTERESTING.

A stream with no learnable structure (constant-parameter noise) keeps
generating high per-observation surprise forever, but once the parameter
posteriors converge there is nothing left to learn, so eig_nats collapses
toward 0. A curiosity signal built on surprise or predictive entropy would
stare at the static forever; mutual information on parameter posteriors
(INV-3) looks away.
"""

from __future__ import annotations

import numpy as np

from tests.beliefs.conftest import empty_events, make_obs, null_probe, obs_for_mid
from topos.beliefs import FairValueKF, FlowIntensity
from topos.contracts.intent import FLOW_INTENSITY
from topos.contracts.market import Side, Trade

EARLY_STEP = 30
TOTAL_STEPS = 2000
LATE_WINDOW = 400


def _assert_surprise_stays_elevated(late_z: list[float]) -> None:
    z = np.asarray(late_z)
    assert z.std() > 0.3, "surprise z-scores collapsed — the noise stopped surprising"
    assert np.abs(z).max() > 1.5
    assert np.mean(np.abs(z) > 1.0) > 0.05


def test_noisy_tv_flow_intensity() -> None:
    """Constant-rate Poisson trade prints: no learnable structure once the
    rate posteriors converge => EIG -> 0 while surprise stays elevated."""
    module = FlowIntensity()
    probe = null_probe(FLOW_INTENSITY, horizon_steps=1)
    rng = np.random.default_rng(21)
    early_eig = None
    late_z: list[float] = []
    for step in range(TOTAL_STEPS):
        lots = int(rng.poisson(4.0))
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
        if step == EARLY_STEP:
            early_eig = module.eig_nats(probe)
        if step >= TOTAL_STEPS - LATE_WINDOW:
            late_z.append(module.surprise_z())
    final_eig = module.eig_nats(probe)
    assert early_eig is not None
    assert final_eig < 0.01
    assert final_eig < 0.05 * early_eig
    _assert_surprise_stays_elevated(late_z)


def test_noisy_tv_fair_value() -> None:
    """Static level plus i.i.d. noise: once the scale posterior converges
    the observations carry no parameter information => EIG -> 0, yet the
    surprise stream stays alive AND the state-tracking information rate
    (deliberately excluded from eig_nats) stays bounded away from zero —
    the exact churn eig_nats must not chase."""
    module = FairValueKF()
    probe = null_probe(horizon_steps=1)
    rng = np.random.default_rng(22)
    early_eig = None
    late_z: list[float] = []
    for step in range(TOTAL_STEPS):
        y = 1000.0 + rng.normal(0.0, 3.0)
        module.update(obs_for_mid(step, y), empty_events(step))
        if step == EARLY_STEP:
            early_eig = module.eig_nats(probe)
        if step >= TOTAL_STEPS - LATE_WINDOW:
            late_z.append(module.surprise_z())
    final_eig = module.eig_nats(probe)
    assert early_eig is not None
    assert final_eig < 1e-3
    assert final_eig < 0.05 * early_eig
    assert module.state_eig_nats(probe) > 0.05
    _assert_surprise_stays_elevated(late_z)
