"""Calibration of FairValueKF against synthetic ground truth.

The generator IS the module's model (common noise scale drawn from the
module's own prior), so the conjugate posterior is the exact Bayesian
posterior and credible intervals must cover truth at nominal rates up to
replication noise: 90% intervals, 120 replications => the covered fraction
lies in [0.82, 0.97] (about +/-3 binomial sigmas) unless the update math
drifted. The quarter-tick book quantization is < 0.1% of the observation
variance at these settings and cannot move coverage measurably.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from tests.beliefs.conftest import empty_events, obs_for_mid
from topos.beliefs import FairValueKF

REPS = 120
STEPS = 500
LEVEL = 0.9
SCALE_PRIOR_A = 3.0
SCALE_PRIOR_B = 200.0  # prior mean scale = 100 ticks^2 => noise sd ~10 ticks
Q_LEVEL = 0.05
Q_DRIFT = 0.001


def test_kalman_calibration() -> None:
    rng = np.random.default_rng(20260709)
    f = np.array([[1.0, 1.0], [0.0, 1.0]])
    scale_covered = 0
    state_covered = 0
    for _ in range(REPS):
        c_true = SCALE_PRIOR_B / rng.gamma(SCALE_PRIOR_A, 1.0)
        module = FairValueKF(
            q_level=Q_LEVEL,
            q_drift=Q_DRIFT,
            scale_prior_a=SCALE_PRIOR_A,
            scale_prior_b=SCALE_PRIOR_B,
        )
        x = np.array([1000.0, 0.0])
        noise_sd = np.sqrt(np.array([c_true * Q_LEVEL, c_true * Q_DRIFT]))
        for step in range(STEPS):
            x = f @ x + noise_sd * rng.standard_normal(2)
            y = x[0] + rng.normal(0.0, np.sqrt(c_true))
            module.update(obs_for_mid(step, y), empty_events(step))
        lo, hi = module.noise_scale_posterior.interval(LEVEL)
        if lo <= c_true <= hi:
            scale_covered += 1
        # Marginal posterior of the level is Student-t (df = 2a) around the
        # filter mean with scale sqrt((b/a) * P*_vv).
        post = module.noise_scale_posterior
        df = 2.0 * post.a
        t_scale = np.sqrt((post.b / post.a) * module.state_scale_free_cov[0, 0])
        half_width = float(stats.t.ppf(0.5 + LEVEL / 2.0, df)) * t_scale
        if abs(module.state_mean[0] - x[0]) <= half_width:
            state_covered += 1
    scale_rate = scale_covered / REPS
    state_rate = state_covered / REPS
    assert 0.82 <= scale_rate <= 0.97, (
        f"noise-scale 90% credible interval covered truth at {scale_rate:.2%}"
    )
    assert 0.82 <= state_rate <= 0.97, (
        f"latent-level 90% credible interval covered truth at {state_rate:.2%}"
    )
