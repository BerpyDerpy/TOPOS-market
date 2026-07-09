"""ImpactModel: Bayesian linear regression of mid moves on own actions.

Model (conjugate normal-inverse-gamma, exact closed-form updates — INV-2):

    y_t = w . x_t + eps_t,   eps_t ~ N(0, sigma^2)
    w | sigma^2 ~ N(m, sigma^2 V),   sigma^2 = c ~ InverseGamma(a, b)

where

    y_t = mid(t + h) - mid(t)          (the short-horizon mid move; h is
                                        the model's fixed impact horizon)
    x_t = [ 1,
            own signed executed aggression at t   (+lots bought marketably,
                                                   -lots sold marketably),
            own signed resting size at the touch  (+bid side, -ask side),
            *flow-intensity context regressors ]  (from P4's headline,
                                                   passed in per cycle via
                                                   ``set_context_regressors``)

THE OWN-ACTION REGRESSORS ARE THE POINT (Layer-1 anti-churn): the model
must be able to LEARN what the agent's own aggression and touch presence
do to the mid, controlling for what background flow was going to do
anyway, so that own impact becomes predictable and boring. A mid-move
model without own-action terms silently degenerates into a second
fair-value model and the reflexive story is lost.

Conjugacy bookkeeping: conditional on sigma^2 the regression is linear-
Gaussian, so each point updates (Lambda, eta) = (V^-1, V^-1 m) by
(x x^T, x y), and the standardized squared residual
(y - m.x)^2 / (1 + x^T V x) updates the SAME inverse-gamma cell type the
fair-value filter uses (``InverseGammaPosterior``) — one shared conjugate
mechanism, per the P4 convention. `forget` discounts (Lambda, eta) and
(a, b) toward their priors.

EIG (INV-3): I((w, sigma^2); Y | x) in closed form —
Student-t predictive entropy (df = 2a, scale^2 = b (1 + x^T V x) / a)
minus E[H(Y | w, sigma^2)] = 0.5 (ln 2 pi e + ln b - digamma(a)).
MC-verified in tests. The null action still earns positive EIG here (a
mid move is observed regardless and teaches the noise scale and control
coefficients); what acting BUYS is the extra 0.5 ln(1 + x^T V x) along
own-action directions — the marginal-over-null the proposer scores
(INV-4).

Ground truth for this model is the harness counterfactual divergence
(P3's ``impact()``); tests here are synthetic-only and P13 closes the
loop.

Prior constants (structural, never tuned to outcomes): coefficient prior
covariance V0 = I in tick-per-lot units (one tick per lot of action is
the natural unit scale of a tick grid); noise prior a0 = 3, b0 = 2 — the
same shape as the fair-value scale prior, giving a finite-variance prior
with mean noise scale 1 tick^2.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Deque

import numpy as np
from scipy import special, stats

from topos.contracts.beliefs import (
    EntropySnapshot,
    ForecastStats,
    ProbeSpec,
    SelfEvents,
)
from topos.contracts.intent import IMPACT, HypothesisId, Intent
from topos.contracts.market import Observation

from topos.beliefs.core import (
    LOG_2PIE,
    EIGTerms,
    FloatArray,
    InverseGammaPosterior,
    SurpriseTracker,
    forget_stats,
)
from topos.selfmodel.common import (
    BookContext,
    OwnOrderLedger,
    context_from_observation,
    implied_order,
    offset_band_of,
)


class ImpactModel:
    """BeliefModule over own price impact (hypothesis_id="impact")."""

    hypothesis_id: HypothesisId = IMPACT

    def __init__(
        self,
        *,
        impact_horizon_steps: int = 1,
        n_context: int = 1,
        coef_prior_var: float = 1.0,
        noise_prior_a: float = 3.0,
        noise_prior_b: float = 2.0,
        size_budget_lots: int = 1,
        surprise_ewma_decay: float = 0.05,
    ) -> None:
        """``impact_horizon_steps`` fixes the regressand (the h-step mid
        move); 1 is the shortest measurable horizon on the step grid and
        matches the granularity of the P3 counterfactual windows.
        ``n_context`` fixes the context-regressor dimension at construction
        (fixed functional form, INV-2); the default 1 slot carries the flow
        headline's forecast mean. ``size_budget_lots`` only interprets
        probes (``Intent.size_frac`` is a fraction of it, per contract).
        """
        if impact_horizon_steps < 1:
            raise ValueError(
                f"impact_horizon_steps must be >= 1, got {impact_horizon_steps}"
            )
        if n_context < 0:
            raise ValueError(f"n_context must be >= 0, got {n_context}")
        if coef_prior_var <= 0.0:
            raise ValueError(f"coef_prior_var must be > 0, got {coef_prior_var}")
        if size_budget_lots < 1:
            raise ValueError(
                f"size_budget_lots must be >= 1, got {size_budget_lots}"
            )
        self.impact_horizon_steps = impact_horizon_steps
        self.n_context = n_context
        self._size_budget_lots = size_budget_lots
        d = 3 + n_context  # intercept, aggression, resting-at-touch, context
        self._d = d
        self._lam0: FloatArray = np.eye(d) / coef_prior_var
        self._eta0: FloatArray = np.zeros(d)  # Lambda0 @ m0 with m0 = 0
        self._lam: FloatArray = self._lam0.copy()
        self._eta: FloatArray = self._eta0.copy()
        self._scale = InverseGammaPosterior(noise_prior_a, noise_prior_b)
        self._ledger = OwnOrderLedger()
        self._ctx_regressors: FloatArray = np.zeros(n_context)
        self._prev_ctx: BookContext | None = None
        self._prev_step: int | None = None
        self._mid_by_step: dict[int, float] = {}
        self._pending: Deque[tuple[int, float, FloatArray]] = deque()
        """(anchor step t, mid_t, x_t); resolves against mid at t + h."""
        self._surprise = SurpriseTracker(ewma_decay=surprise_ewma_decay)
        self._step = 0
        self._last_point: tuple[FloatArray, float] | None = None

    # -- posterior access (public: tests, proposer, metrics read these) ----

    @property
    def coef_mean(self) -> FloatArray:
        """Posterior mean of w = [intercept, aggression, resting, *context]."""
        result: FloatArray = np.linalg.solve(self._lam, self._eta)
        return result

    @property
    def coef_scale_free_cov(self) -> FloatArray:
        """Scale-free coefficient covariance V; actual cov is sigma^2 V."""
        result: FloatArray = np.linalg.inv(self._lam)
        return result

    @property
    def noise_scale_posterior(self) -> InverseGammaPosterior:
        """The inverse-gamma posterior over the noise variance sigma^2."""
        return self._scale

    @property
    def last_point(self) -> tuple[FloatArray, float] | None:
        """(x, y) of the most recently folded data point (for inspection)."""
        if self._last_point is None:
            return None
        x, y = self._last_point
        return x.copy(), y

    def set_context_regressors(self, values: tuple[float, ...]) -> None:
        """Set this cycle's flow-intensity control regressors (from the P4
        headline). They ride the data point anchored at this cycle's
        observation. Dimension is fixed at construction."""
        if len(values) != self.n_context:
            raise ValueError(
                f"expected {self.n_context} context regressors, got {len(values)}"
            )
        self._ctx_regressors = np.asarray(values, dtype=np.float64)

    # -- BeliefModule protocol ---------------------------------------------

    def update(self, obs: Observation, self_events: SelfEvents) -> None:
        """Fold one step: build the previous step's regressor row (its
        executed aggression only becomes known now, from this step's own
        fills), then resolve every pending row whose h-step mid target is
        now observable."""
        self._step = obs.step
        ctx = context_from_observation(obs)
        taker_signed = self._ledger.fold(obs, self_events)
        prev_ctx, prev_step = self._prev_ctx, self._prev_step
        if (
            prev_ctx is not None
            and prev_step is not None
            and prev_ctx.mid is not None
            and obs.step > prev_step
        ):
            resting = self._ledger.resting_at_touch_signed(
                prev_ctx.best_bid, prev_ctx.best_ask
            )
            x = np.concatenate(
                (
                    np.array([1.0, float(taker_signed), float(resting)]),
                    self._ctx_regressors,
                )
            )
            self._pending.append((prev_step, prev_ctx.mid, x))
        if ctx.mid is not None:
            self._mid_by_step[obs.step] = ctx.mid
        while self._pending:
            anchor_step, anchor_mid, x = self._pending[0]
            target = anchor_step + self.impact_horizon_steps
            if target > obs.step:
                break
            self._pending.popleft()
            target_mid = self._mid_by_step.get(target)
            if target_mid is not None:
                self.observe_point_raw(x, target_mid - anchor_mid)
        horizon_floor = obs.step - self.impact_horizon_steps - 1
        self._mid_by_step = {
            s: m for s, m in self._mid_by_step.items() if s >= horizon_floor
        }
        self._prev_ctx = ctx
        self._prev_step = obs.step

    def observe_point(
        self,
        aggression_lots: float,
        resting_touch_lots: float,
        context: tuple[float, ...],
        y_mid_move: float,
    ) -> None:
        """Fold one (regressors, mid move) pair directly.

        ``update`` builds these from observations; this entry point exists
        for synthetic tests and P13 scoring against the counterfactual.
        """
        if len(context) != self.n_context:
            raise ValueError(
                f"expected {self.n_context} context regressors, got {len(context)}"
            )
        x = np.concatenate(
            (
                np.array([1.0, aggression_lots, resting_touch_lots]),
                np.asarray(context, dtype=np.float64),
            )
        )
        self.observe_point_raw(x, y_mid_move)

    def observe_point_raw(self, x: FloatArray, y: float) -> None:
        """Exact conjugate NIG update from one (x, y), scoring surprise on
        the pre-update Student-t predictive."""
        m = self.coef_mean
        v_x = np.linalg.solve(self._lam, x)
        s_star = 1.0 + float(x @ v_x)
        loc = float(m @ x)
        df, t_scale = self._scale.student_t_predictive(s_star)
        self._surprise.score(-float(stats.t.logpdf(y, df, loc=loc, scale=t_scale)))
        residual = y - loc
        self._scale.observe_standardized_square(residual * residual / s_star)
        self._lam = self._lam + np.outer(x, x)
        self._eta = self._eta + x * y
        self._last_point = (x.copy(), y)

    def forget(self, rho: float) -> None:
        """Discount (Lambda, eta) and the noise posterior toward the prior."""
        self._scale.forget(rho)
        self._lam = np.asarray(forget_stats(self._lam, self._lam0, rho))
        self._eta = np.asarray(forget_stats(self._eta, self._eta0, rho))

    def posterior_entropy_nats(self) -> float:
        """Joint PARAMETER-posterior entropy of (w, sigma^2):

            H[sigma^2] + 0.5 ln((2 pi e)^d det V) + (d/2) E[ln sigma^2]

        with E[ln sigma^2] = ln b - digamma(a) under IG(a, b) — INV-3's
        epistemic quantity for this hypothesis.
        """
        _, logdet_lam = np.linalg.slogdet(self._lam)
        e_log_sigma2 = math.log(self._scale.b) - float(
            special.digamma(self._scale.a)
        )
        return (
            self._scale.entropy_nats()
            + 0.5 * (self._d * LOG_2PIE - float(logdet_lam))
            + 0.5 * self._d * e_log_sigma2
        )

    def predict(self) -> ForecastStats:
        """Predictive h-step mid move under NO own action, at the current
        context regressors (the background-drift forecast the workspace
        headline wants)."""
        x = self._x_for(aggression_lots=0.0, resting_touch_lots=0.0)
        mean, variance = self._predictive_mean_variance(x)
        return ForecastStats(mean=mean, variance=variance)

    def surprise_z(self) -> float:
        """Salience-only surprise; never feeds EIG or action scoring."""
        return self._surprise.last_z

    def eig_nats(self, probe: ProbeSpec) -> float:
        """I((w, sigma^2); Y | x(probe)) — closed form, see module docstring."""
        return self.eig_breakdown(probe).eig_nats

    def eig_breakdown(self, probe: ProbeSpec) -> EIGTerms:
        """The epistemic/aleatoric decomposition behind ``eig_nats``."""
        if probe.horizon_steps < 1:
            raise ValueError(
                f"probe horizon_steps must be >= 1, got {probe.horizon_steps}"
            )
        aggression, resting = self.own_regressors_for_intent(probe.intent)
        x = self._x_for(aggression_lots=aggression, resting_touch_lots=resting)
        return self.eig_terms_for_x(x)

    def eig_terms_for_x(self, x: FloatArray) -> EIGTerms:
        """Closed-form I((w, sigma^2); Y | x); MC-verified in tests."""
        v_x = np.linalg.solve(self._lam, x)
        s_star = 1.0 + float(x @ v_x)
        df, t_scale = self._scale.student_t_predictive(s_star)
        predictive_entropy = float(stats.t.entropy(df, scale=t_scale))
        e_log_sigma2 = math.log(self._scale.b) - float(
            special.digamma(self._scale.a)
        )
        conditional = 0.5 * (LOG_2PIE + e_log_sigma2)
        return EIGTerms(
            eig_nats=predictive_entropy - conditional,
            predictive_entropy_nats=predictive_entropy,
            expected_conditional_entropy_nats=conditional,
        )

    def snapshot_entropy(self) -> EntropySnapshot:
        """Parameter-posterior entropy at this instant (INV-10)."""
        return EntropySnapshot(
            hypothesis_id=self.hypothesis_id,
            step=self._step,
            entropy_nats=self.posterior_entropy_nats(),
        )

    # -- probe interpretation and shared predictive forms ---------------------

    def own_regressors_for_intent(self, intent: Intent) -> tuple[float, float]:
        """(aggression_lots, resting_touch_lots) an intent would exercise.

        A marketable implied order exercises the aggression channel; a
        touch-banded one the resting channel; deeper resting orders sit in
        neither modeled impact channel (their x is the null's). Existing
        own resting size at the touch is included in both the probe's and
        the null's row, so it cancels in any marginal-over-null score.
        """
        ctx = self._prev_ctx
        if ctx is None:
            ctx = BookContext(mid=None, best_bid=None, best_ask=None, imbalance=0.0)
        base_resting = float(
            self._ledger.resting_at_touch_signed(ctx.best_bid, ctx.best_ask)
        )
        order = implied_order(intent, ctx, self._size_budget_lots)
        if order is None:
            return 0.0, base_resting
        band = offset_band_of(
            order.side, order.price_ticks, ctx.best_bid, ctx.best_ask
        )
        signed = order.side.value * order.size_lots
        if band == "cross":
            return float(signed), base_resting
        if band == "touch":
            return 0.0, base_resting + float(signed)
        return 0.0, base_resting

    def predictive_own_effect(
        self, aggression_lots: float, resting_touch_lots: float
    ) -> tuple[float, float]:
        """(mean, variance) of the own-action CONTRIBUTION to the h-step
        mid move: the posterior of w . dx with dx the own-channel delta
        against the null row. Consumed by the trajectory compiler, so the
        reflexive forecast rides this same posterior.

        Var[w . dx] = dx^T V dx * E[sigma^2] (coefficient uncertainty
        only — the residual mid noise is the fair-value model's account,
        counting it here too would double-book it).
        """
        dx = np.zeros(self._d)
        dx[1] = aggression_lots
        dx[2] = resting_touch_lots
        mean = float(self.coef_mean @ dx)
        v_dx = np.linalg.solve(self._lam, dx)
        variance = float(dx @ v_dx) * self._scale.mean()
        return mean, variance

    def _x_for(
        self, aggression_lots: float, resting_touch_lots: float
    ) -> FloatArray:
        result: FloatArray = np.concatenate(
            (
                np.array([1.0, aggression_lots, resting_touch_lots]),
                self._ctx_regressors,
            )
        )
        return result

    def _predictive_mean_variance(self, x: FloatArray) -> tuple[float, float]:
        v_x = np.linalg.solve(self._lam, x)
        s_star = 1.0 + float(x @ v_x)
        mean = float(self.coef_mean @ x)
        df, t_scale = self._scale.student_t_predictive(s_star)
        if df > 2.0:
            variance = t_scale * t_scale * df / (df - 2.0)
        else:
            variance = math.inf
        return mean, variance
