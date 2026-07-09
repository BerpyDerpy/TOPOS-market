"""FairValueKF: linear-Gaussian latent fair value with uncertain noise scale.

Model (a discount-free DLM in West & Harrison's unknown-variance form):

    state  x_t = (value_t, drift_t),  x_{t+1} = F x_t + w_t
    obs    y_t = microprice_t = H x_t + e_t
    w_t ~ N(0, c * Q0),  e_t ~ N(0, c * r0)

F, H, Q0, r0 are fixed known shapes; the single positive scale ``c``
multiplies BOTH the observation-noise and state-noise covariances and is
uncertain, tracked by a conjugate InverseGamma posterior. This common-scale
form is chosen because it is the exactly conjugate treatment: conditional
on c the filter is the standard scale-free Kalman recursion, each
standardized squared innovation nu^2/S* updates the InverseGamma posterior
in closed form, and the predictive is Student-t (see DESIGN.md, Open
questions — separate scales for R and Q admit no exact conjugate update).

Epistemic accounting (INV-3): the PARAMETER posterior is the posterior over
c, so ``posterior_entropy_nats`` and ``eig_nats`` are computed on it. The
Kalman state covariance alone is not sufficient — and it is deliberately
NOT part of ``eig_nats``: a steady-state filter gains a fixed amount of
state information every step forever, so including it would make curiosity
unsaturatable (churn). The closed-form EIG of an observation about the
latent state, 0.5 * ln(det Sigma_prior / det Sigma_post), is exposed
separately as ``state_eig_nats`` for diagnostics and downstream consumers.

EIG here depends only on the probe horizon, never on the probe's intent:
the microprice is observed passively, so placing orders cannot teach this
module anything extra. Marginal EIG over null is therefore 0 for every
order-placing probe — the null action carries this module's EIG (INV-4).
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats

from topos.contracts.beliefs import (
    EntropySnapshot,
    ForecastStats,
    ProbeSpec,
    SelfEvents,
)
from topos.contracts.intent import FAIR_VALUE, HypothesisId
from topos.contracts.market import BookLevel, Observation

from topos.beliefs.core import (
    EIGTerms,
    FloatArray,
    InverseGammaPosterior,
    SurpriseTracker,
)


def _best_level(levels: tuple[BookLevel, ...]) -> BookLevel | None:
    """First non-padded level (best price by construction), if any."""
    for level in levels:
        if level.size_lots > 0:
            return level
    return None


def microprice_from_observation(obs: Observation) -> float | None:
    """Size-weighted best-quote midpoint; falls back to the one-sided best.

    micro = (p_bid * s_ask + p_ask * s_bid) / (s_bid + s_ask), which leans
    toward the side with LESS resting size (the side more likely to trade).
    Padded levels (size_lots == 0) are treated as absent per the book
    convention. An empty book yields None (no fair-value evidence).
    """
    bid = _best_level(obs.bids)
    ask = _best_level(obs.asks)
    if bid is None and ask is None:
        return None
    if bid is None:
        assert ask is not None
        return float(ask.price_ticks)
    if ask is None:
        return float(bid.price_ticks)
    total = bid.size_lots + ask.size_lots
    return (bid.price_ticks * ask.size_lots + ask.price_ticks * bid.size_lots) / total


class FairValueKF:
    """BeliefModule over the latent fair value (hypothesis_id="fair_value")."""

    hypothesis_id: HypothesisId = FAIR_VALUE

    def __init__(
        self,
        *,
        r0: float = 1.0,
        q_level: float = 0.05,
        q_drift: float = 0.005,
        scale_prior_a: float = 3.0,
        scale_prior_b: float = 2.0,
        level_prior_var: float = 1e6,
        drift_prior_var: float = 1.0,
        surprise_ewma_decay: float = 0.05,
    ) -> None:
        if min(r0, q_level, q_drift) <= 0.0:
            raise ValueError("noise shapes r0, q_level, q_drift must be positive")
        self._f = np.array([[1.0, 1.0], [0.0, 1.0]])
        self._h = np.array([1.0, 0.0])
        self._r0 = float(r0)
        self._q0 = np.diag([float(q_level), float(q_drift)])
        self._scale = InverseGammaPosterior(scale_prior_a, scale_prior_b)
        self._m0 = np.zeros(2)
        self._p0 = np.diag([float(level_prior_var), float(drift_prior_var)])
        self._m: FloatArray = self._m0.copy()
        self._p: FloatArray = self._p0.copy()
        self._initialized = False
        self._surprise = SurpriseTracker(ewma_decay=surprise_ewma_decay)
        self._step = 0

    # -- posterior access (public: tests, proposer, metrics read these) ----

    @property
    def noise_scale_posterior(self) -> InverseGammaPosterior:
        """The parameter posterior over the common noise scale c."""
        return self._scale

    @property
    def state_mean(self) -> FloatArray:
        """Posterior mean of (value, drift), in ticks / ticks-per-step."""
        return self._m.copy()

    @property
    def state_scale_free_cov(self) -> FloatArray:
        """Scale-free state covariance P*; the actual covariance is c * P*."""
        return self._p.copy()

    # -- BeliefModule protocol ---------------------------------------------

    def update(self, obs: Observation, self_events: SelfEvents) -> None:
        """One conjugate filter step from the step's microprice, if any.

        With an empty book the latent state still diffuses (time passes:
        predict without correction) but no innovation reaches the scale
        posterior and no surprise is scored.
        """
        self._step = obs.step
        y = microprice_from_observation(obs)
        if not self._initialized:
            if y is not None:
                # Diffuse initialization: the first observation anchors the
                # level and carries no scale information (its standardized
                # innovation is ~0 under the diffuse prior), so it is not
                # counted as an innovation at all.
                self._m = np.array([y, 0.0])
                self._initialized = True
            return
        # Predict.
        self._m = self._f @ self._m
        self._p = self._f @ self._p @ self._f.T + self._q0
        if y is None:
            return
        # Score surprise on the one-step predictive BEFORE updating.
        s_star = float(self._h @ self._p @ self._h + self._r0)
        df, t_scale = self._scale.student_t_predictive(s_star)
        loc = float(self._h @ self._m)
        nll = -float(stats.t.logpdf(y, df, loc=loc, scale=t_scale))
        self._surprise.score(nll)
        # Correct (Joseph form) and update the scale posterior.
        nu = y - loc
        gain = (self._p @ self._h) / s_star
        self._m = self._m + gain * nu
        i_kh = np.eye(2) - np.outer(gain, self._h)
        self._p = i_kh @ self._p @ i_kh.T + self._r0 * np.outer(gain, gain)
        self._scale.observe_standardized_square(nu * nu / s_star)

    def forget(self, rho: float) -> None:
        """Discount both the scale posterior and the state (information form)
        toward their priors; rho == 1 is a no-op (INV-2-compatible)."""
        self._scale.forget(rho)
        if rho == 1.0 or not self._initialized:
            return
        lam = np.linalg.inv(self._p)
        lam0 = np.linalg.inv(self._p0)
        info = lam @ self._m
        info0 = lam0 @ self._m0
        lam_new = rho * lam + (1.0 - rho) * lam0
        info_new = rho * info + (1.0 - rho) * info0
        self._p = np.linalg.inv(lam_new)
        self._m = self._p @ info_new

    def posterior_entropy_nats(self) -> float:
        """Entropy of the PARAMETER posterior (the noise scale c) — INV-3."""
        return self._scale.entropy_nats()

    def predict(self) -> ForecastStats:
        """One-step-ahead Student-t predictive of the microprice."""
        mean, s_star = self.horizon_prediction(1)
        df, t_scale = self._scale.student_t_predictive(s_star)
        if df > 2.0:
            variance = t_scale * t_scale * df / (df - 2.0)
        else:
            variance = math.inf
        return ForecastStats(mean=mean, variance=variance)

    def surprise_z(self) -> float:
        """Salience-only surprise; never feeds EIG or action scoring."""
        return self._surprise.last_z

    def eig_nats(self, probe: ProbeSpec) -> float:
        """I(c; Y | probe): what the microprice at the probe horizon can
        teach about the noise-scale parameter. Via the shared identity,
        with quadrature over the inverse-gamma posterior (INV-3)."""
        return self.eig_breakdown(probe).eig_nats

    def eig_breakdown(self, probe: ProbeSpec) -> EIGTerms:
        """The epistemic/aleatoric decomposition behind ``eig_nats``."""
        _, s_star = self.horizon_prediction(self._validated_horizon(probe))
        return self._scale.eig_terms_for_gaussian(s_star)

    def snapshot_entropy(self) -> EntropySnapshot:
        """Parameter-posterior entropy at this instant (INV-10)."""
        return EntropySnapshot(
            hypothesis_id=self.hypothesis_id,
            step=self._step,
            entropy_nats=self.posterior_entropy_nats(),
        )

    # -- beyond the protocol -------------------------------------------------

    def state_eig_nats(self, probe: ProbeSpec) -> float:
        """Closed-form EIG of the horizon observation about the LATENT STATE:

            0.5 * ln(det Sigma_prior / det Sigma_post)

        with the noise scale fixed at its posterior mean. This is exact
        Gaussian mutual information, but it is NOT the curiosity quantity:
        a steady-state filter earns it every step forever, so it never
        saturates. It is exposed for diagnostics/metrics only.
        """
        horizon = self._validated_horizon(probe)
        c_bar = self._scale.mean()
        p_h = self._p.copy()
        for _ in range(horizon):
            p_h = self._f @ p_h @ self._f.T + self._q0
        sigma_prior = c_bar * p_h
        s = float(self._h @ sigma_prior @ self._h + c_bar * self._r0)
        gain = (sigma_prior @ self._h) / s
        sigma_post = sigma_prior - s * np.outer(gain, gain)
        _, logdet_prior = np.linalg.slogdet(sigma_prior)
        _, logdet_post = np.linalg.slogdet(sigma_post)
        return float(0.5 * (logdet_prior - logdet_post))

    def horizon_prediction(self, horizon_steps: int) -> tuple[float, float]:
        """(mean, scale-free variance) of the microprice ``horizon_steps``
        ahead: y_h | c ~ N(mean, c * scale_free_var)."""
        if horizon_steps < 1:
            raise ValueError(f"horizon_steps must be >= 1, got {horizon_steps}")
        m_h = self._m.copy()
        p_h = self._p.copy()
        for _ in range(horizon_steps):
            m_h = self._f @ m_h
            p_h = self._f @ p_h @ self._f.T + self._q0
        mean = float(self._h @ m_h)
        s_star = float(self._h @ p_h @ self._h + self._r0)
        return mean, s_star

    @staticmethod
    def _validated_horizon(probe: ProbeSpec) -> int:
        if probe.horizon_steps < 1:
            raise ValueError(
                f"probe horizon_steps must be >= 1, got {probe.horizon_steps}"
            )
        return probe.horizon_steps
