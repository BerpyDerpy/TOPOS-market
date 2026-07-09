"""Anti-churn at unit scale: repeated consistent observations must drive
eig_nats toward 0 for a fixed probe, while the aleatoric term stays finite.

This is the architectural point of computing curiosity as parameter-
posterior mutual information (INV-3): once the parameters are known, an
endlessly repeating world offers nothing left to learn, no matter how
noisy it remains.
"""

from __future__ import annotations

import numpy as np

from tests.beliefs.conftest import empty_events, make_obs, null_probe, obs_for_mid
from topos.beliefs import FairValueKF, FlowIntensity
from topos.contracts.intent import FLOW_INTENSITY
from topos.contracts.market import Side, Trade

CHECKPOINTS = (30, 100, 400, 1500)


def _static_book_with_trades(step: int, rng: np.random.Generator):
    trades = []
    buy_lots = int(rng.poisson(3.0))
    sell_lots = int(rng.poisson(2.0))
    if buy_lots > 0:
        trades.append(Trade(price_ticks=1001, size_lots=buy_lots, aggressor=Side.BUY))
    if sell_lots > 0:
        trades.append(Trade(price_ticks=999, size_lots=sell_lots, aggressor=Side.SELL))
    bids = [(999 - i, 50) for i in range(10)]
    asks = [(1001 + i, 50) for i in range(10)]
    return make_obs(step, bids, asks, trades=tuple(trades))


def test_flow_intensity_eig_saturates() -> None:
    module = FlowIntensity()
    probe = null_probe(FLOW_INTENSITY, horizon_steps=1)
    rng = np.random.default_rng(11)
    eig_at: dict[int, float] = {}
    for step in range(CHECKPOINTS[-1] + 1):
        module.update(_static_book_with_trades(step, rng), empty_events(step))
        if step in CHECKPOINTS:
            eig_at[step] = module.eig_nats(probe)
    values = [eig_at[s] for s in CHECKPOINTS]
    assert all(later < earlier for earlier, later in zip(values, values[1:])), (
        f"eig_nats not decreasing along checkpoints: {eig_at}"
    )
    assert values[-1] < 0.02
    assert values[-1] < 0.05 * values[0]
    terms = module.eig_breakdown(probe)
    # The aleatoric part (expected conditional entropy of the counts given
    # the rates) stays finite and substantial: the world is still noisy.
    assert 0.5 < terms.expected_conditional_entropy_nats < 50.0
    assert np.isfinite(terms.predictive_entropy_nats)


def test_fair_value_eig_saturates() -> None:
    module = FairValueKF(q_level=0.05, q_drift=0.005)
    probe = null_probe(horizon_steps=1)
    rng = np.random.default_rng(12)
    c_true, r0 = 4.0, 1.0
    x = np.array([1000.0, 0.0])
    f = np.array([[1.0, 1.0], [0.0, 1.0]])
    q_chol = np.sqrt(np.array([c_true * 0.05, c_true * 0.005]))
    eig_at: dict[int, float] = {}
    for step in range(CHECKPOINTS[-1] + 1):
        x = f @ x + q_chol * rng.standard_normal(2)
        y = x[0] + rng.normal(0.0, np.sqrt(c_true * r0))
        module.update(obs_for_mid(step, y), empty_events(step))
        if step in CHECKPOINTS:
            eig_at[step] = module.eig_nats(probe)
    values = [eig_at[s] for s in CHECKPOINTS]
    assert all(later < earlier for earlier, later in zip(values, values[1:])), (
        f"eig_nats not decreasing along checkpoints: {eig_at}"
    )
    assert values[-1] < 1e-3
    assert values[-1] < 0.05 * values[0]
    terms = module.eig_breakdown(probe)
    assert 0.5 < terms.expected_conditional_entropy_nats < 10.0
    assert np.isfinite(terms.predictive_entropy_nats)
