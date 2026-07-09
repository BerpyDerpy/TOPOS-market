"""Calibration of the Poisson-Gamma cell against synthetic ground truth.

Rates are drawn from the cell's own prior, so the conjugate posterior is
the exact Bayesian posterior: 90% credible intervals must cover the true
rate at the nominal rate up to replication noise (400 reps => covered
fraction in [0.85, 0.95], about +/-3 binomial sigmas).
"""

from __future__ import annotations

import numpy as np

from tests.beliefs.conftest import empty_events, make_obs
from topos.beliefs import FlowIntensity, GammaPosterior
from topos.contracts.market import Side, Trade

REPS = 400
STEPS = 300
LEVEL = 0.9
PRIOR_A = 2.0
PRIOR_B = 0.5


def test_poisson_calibration() -> None:
    rng = np.random.default_rng(4)
    covered = 0
    for _ in range(REPS):
        lam_true = rng.gamma(PRIOR_A, 1.0 / PRIOR_B)
        cell = GammaPosterior(PRIOR_A, PRIOR_B)
        # One conjugate update from the sufficient statistics (equivalence
        # with per-step updates is pinned in test_core.py).
        cell.observe(int(rng.poisson(lam_true * STEPS)), float(STEPS))
        lo, hi = cell.interval(LEVEL)
        if lo <= lam_true <= hi:
            covered += 1
    rate = covered / REPS
    assert 0.85 <= rate <= 0.95, (
        f"rate 90% credible interval covered truth at {rate:.2%}"
    )


def test_poisson_calibration_through_module_extraction() -> None:
    """End to end: a module fed synthetic trade prints at a known rate
    recovers that rate in the matching cell's posterior."""
    lam_true = 3.5
    module = FlowIntensity(prior_a=PRIOR_A, prior_b=PRIOR_B)
    rng = np.random.default_rng(5)
    steps = 2000
    for step in range(steps):
        lots = int(rng.poisson(lam_true))
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
    cell = module.cells[("market", Side.BUY, "touch")]
    lo, hi = cell.interval(0.999)
    assert lo <= lam_true <= hi
    assert abs(cell.mean() - lam_true) < 0.2
