"""RegimeTracker: Bayesian online changepoint detection over public summary
statistics, converted into per-module forgetting factors (P11, the slow
loop).

hypothesis_id = REGIME (reserved in ``topos.contracts.intent``). REGIME is
PASSIVE-ONLY: no ``Intent`` may ever carry it as ``target_id``, so this
module is never the target of a probe. It observes only public summary
statistics computed from ``Observation``/``WorldSummary`` — trade tempo,
realized vol, book imbalance, mean depth — and NEVER the harness-only
ground-truth regime id (INV-11). Ground truth exists solely for P13 scoring.

Algorithm (Adams & MacKay, 2007, "Bayesian Online Changepoint Detection")
--------------------------------------------------------------------------
At each slow tick the tracker holds a posterior over the current RUN LENGTH
r_t (steps since the last changepoint), represented as a probability vector
over ``r_t in {0, 1, ..., t}``. The observation model is four independent
Gaussians (one per summary statistic) with unknown mean AND variance,
tracked per run-length hypothesis by a conjugate Normal-Inverse-Gamma (NIG)
posterior — the exact conjugate family for that model, matching INV-2's
"conjugate/analytic updates only".

Recursion per tick, given a constant hazard rate H = cfg.hazard (constant
=> geometric prior on segment length, mean 1/H ticks):

    P(r_t = 0      | y_1:t) ∝ Σ_r P(r_{t-1}=r | y_1:t-1) * pred(y_t | r) * H
    P(r_t = r+1    | y_1:t) ∝ P(r_{t-1}=r | y_1:t-1) * pred(y_t | r) * (1-H)

``pred(y_t | r)`` is the run length's current joint Student-t predictive
density (product over the 4 independent dimensions). After scoring, every
surviving hypothesis's NIG posterior folds in y_t (run length r+1 continues
the r's posterior updated with y_t; run length 0 starts fresh from the
prior updated with y_t). The vector is renormalized and, for tractability,
truncated by dropping the lowest-mass tail once it holds < TRUNCATE_MASS
probability (standard BOCPD practice — an unbounded growing vector is not
implementable; this never discards the head, which is what forgetting and
regime_posterior read).

Forgetting map
---------------
``current_rho()`` converts the run-length posterior into a single scalar
the agent loop (P12) applies to every BeliefModule once per slow tick via
``forget(rho)``:

    rho = 1 - (1 - RHO_MIN) * P(run_length < R_recent)

A stable regime keeps P(run_length < R_recent) ~= 0, so rho ~= 1 (no
forgetting, INV-2 no-op). A confident recent changepoint pushes
P(run_length < R_recent) -> 1, so rho -> RHO_MIN: posteriors re-inflate and
EIG reopens (curiosity regenerates from a regime shift, not from time
alone).

This module calls no other module directly (wiring contract): P12 reads
``current_rho()`` and applies it externally.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray
from scipy import special, stats

from topos.contracts.beliefs import (
    EntropySnapshot,
    ForecastStats,
    ProbeSpec,
    SelfEvents,
)
from topos.contracts.intent import REGIME, HypothesisId
from topos.contracts.market import Observation

from topos.beliefs.core import EIGTerms, SurpriseTracker, information_gain_terms

FloatArray = NDArray[np.float64]

N_DIMS: Final = 4
"""trade_tempo, realized_vol, imbalance, mean_depth — this module's fixed
observation dimensionality, matching the four WorldSummary fields it reads."""

STAT_NAMES: tuple[str, ...] = (
    "trade_tempo",
    "realized_vol",
    "imbalance",
    "mean_depth",
)

RHO_MIN: Final = 0.05
"""Floor on the forgetting factor: the most aggressive discount a confident
changepoint can command. Never 0 — a hard reset to the prior would discard
every already-converged parameter estimate at once rather than reopening
curiosity gradually; 0.05 retains a 5% weight on accumulated evidence per
slow tick even under maximal detected regime change (architecture constant,
not tuned against outcomes)."""

R_RECENT: Final = 5
"""Recency window, in SLOW TICKS: how far back a changepoint still counts
as "recent" for both regime_posterior and the forgetting map. Chosen as a
small multiple of 1 so a changepoint detected within the last few ticks
drives forgetting, while one further back (the run has already
re-stabilized statistics) does not — an architecture constant, not tuned
against outcomes."""

TRUNCATE_MASS: Final = 1e-9
"""The run-length posterior's growing tail is dropped once its cumulative
mass falls below this threshold (standard BOCPD practice for a bounded
representation); the head (recent run lengths, where regime_posterior and
forgetting read) is never affected at this threshold."""


@dataclass(frozen=True)
class RegimeConfig:
    """Constant-hazard BOCPD configuration and the NIG observation prior.

    ``hazard`` is P(changepoint at any given tick), constant over time =>
    a geometric prior on segment length with mean ``1 / hazard`` ticks.
    ``mu0`` gives the four dimensions' prior means (trade_tempo,
    realized_vol, imbalance, mean_depth order, matching ``STAT_NAMES``);
    ``kappa0``/``alpha0``/``beta0`` are the shared NIG prior pseudo-counts
    and shape/scale (broad and weakly informative: this module must detect
    genuine regime shifts, not just settle onto one summary-statistic
    scale).
    """

    hazard: float
    mu0: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    kappa0: float = 0.5
    alpha0: float = 1.5
    beta0: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 < self.hazard < 1.0:
            raise ValueError(f"hazard must be in (0, 1), got {self.hazard}")
        if len(self.mu0) != N_DIMS:
            raise ValueError(f"mu0 must have {N_DIMS} entries, got {len(self.mu0)}")
        if self.kappa0 <= 0.0:
            raise ValueError(f"kappa0 must be > 0, got {self.kappa0}")
        if self.alpha0 <= 0.0:
            raise ValueError(f"alpha0 must be > 0, got {self.alpha0}")
        if self.beta0 <= 0.0:
            raise ValueError(f"beta0 must be > 0, got {self.beta0}")


@dataclass
class _NIGState:
    """Per-run-length, per-dimension Normal-Inverse-Gamma sufficient stats.

    Every array is shape (n_hypotheses, N_DIMS) — one independent NIG cell
    per (run length, dimension), which is what "four independent Gaussians"
    means: each dimension has its OWN unknown mean and variance, so kappa
    and alpha are per-dimension too, not shared scalars per row. Row i is
    run length i (i.e. "i steps have elapsed since the last changepoint").
    """

    mu: FloatArray
    kappa: FloatArray
    alpha: FloatArray
    beta: FloatArray

    def predictive_log_pdf(self, y: FloatArray) -> FloatArray:
        """log joint Student-t predictive density of ``y`` (shape (N_DIMS,))
        under every hypothesis row, summed over the independent dimensions.
        """
        df = 2.0 * self.alpha
        scale2 = self.beta * (self.kappa + 1.0) / (self.alpha * self.kappa)
        scale = np.sqrt(scale2)
        z = (y[np.newaxis, :] - self.mu) / scale
        logpdf = stats.t.logpdf(z, df) - np.log(scale)
        result: FloatArray = np.sum(logpdf, axis=1)
        return result

    def folded(self, y: FloatArray) -> "_NIGState":
        """New state with one observation ``y`` conjugate-folded into every
        row (n=1 NIG update, applied to all hypotheses/dimensions at once).
        """
        kappa_new = self.kappa + 1.0
        mu_new = (self.kappa * self.mu + y[np.newaxis, :]) / kappa_new
        alpha_new = self.alpha + 0.5
        delta = y[np.newaxis, :] - self.mu
        beta_new = self.beta + 0.5 * (self.kappa * delta * delta) / kappa_new
        return _NIGState(mu=mu_new, kappa=kappa_new, alpha=alpha_new, beta=beta_new)

    def entropy_nats(self) -> FloatArray:
        """Joint parameter-posterior entropy per row: sum over dimensions of
        H[NIG(mu, kappa, alpha, beta)], the exact differential entropy of
        the Normal-Inverse-Gamma family,

            H = 0.5*ln(2*pi*e/kappa) + H[InvGamma(alpha, beta)]

        (the Gaussian-given-variance term integrates out the conditioning
        variance exactly, leaving the marginal-variance entropy plus the
        kappa-scaled location term).
        """
        loc_term = 0.5 * np.log(2.0 * math.pi * math.e / self.kappa)
        ig_term = stats.invgamma.entropy(self.alpha, scale=self.beta)
        result: FloatArray = np.sum(loc_term + ig_term, axis=1)
        return result

    def __len__(self) -> int:
        return int(self.mu.shape[0])


def _prior_row(cfg: RegimeConfig) -> _NIGState:
    return _NIGState(
        mu=np.array([list(cfg.mu0)], dtype=np.float64),
        kappa=np.full((1, N_DIMS), cfg.kappa0, dtype=np.float64),
        alpha=np.full((1, N_DIMS), cfg.alpha0, dtype=np.float64),
        beta=np.full((1, N_DIMS), cfg.beta0, dtype=np.float64),
    )


def _stack(prior: _NIGState, grown: _NIGState) -> _NIGState:
    return _NIGState(
        mu=np.concatenate([prior.mu, grown.mu], axis=0),
        kappa=np.concatenate([prior.kappa, grown.kappa], axis=0),
        alpha=np.concatenate([prior.alpha, grown.alpha], axis=0),
        beta=np.concatenate([prior.beta, grown.beta], axis=0),
    )


class RegimeTracker:
    """BeliefModule over the slow-loop regime (hypothesis_id=REGIME).

    Passive-only (never a probe target): sees public summary statistics
    every slow tick (every ``cfg`` steps, driven externally by the caller —
    P12 decides the tick cadence and calls ``update`` only on slow ticks).
    """

    hypothesis_id: HypothesisId = REGIME

    def __init__(self, cfg: RegimeConfig, *, surprise_ewma_decay: float = 0.05) -> None:
        self._cfg = cfg
        self._prior = _prior_row(cfg)
        self._state = _prior_row(cfg)
        self._log_run_length_posterior = np.array([0.0])  # log P(r=0) = 1
        self._surprise = SurpriseTracker(ewma_decay=surprise_ewma_decay)
        self._step = 0
        self._n_ticks = 0

    # -- posterior access (public: tests, P12, metrics read these) ---------

    @property
    def run_length_posterior(self) -> FloatArray:
        """P(run_length = r) for r = 0 .. len-1, most-recent tick's state."""
        return np.exp(self._log_run_length_posterior)

    @property
    def n_ticks(self) -> int:
        """Number of slow ticks observed so far."""
        return self._n_ticks

    def prob_changepoint_within(self, window: int) -> float:
        """P(run_length < window) under the current posterior.

        This is the tracker's read of "a changepoint occurred within the
        last ``window`` ticks" — the quantity both ``regime_posterior`` and
        ``current_rho`` are built from.
        """
        if window <= 0:
            return 0.0
        post = self.run_length_posterior
        k = min(window, len(post))
        return float(np.sum(post[:k]))

    def regime_posterior_summary(self) -> tuple[float, ...]:
        """The (P(run_length < R_recent), P(run_length >= R_recent)) pair
        for ``WorldSummary.regime_posterior`` — a compact, capacity-limited
        readout of "just changed" vs "stable", matching the frozen
        contract's ``tuple[float, ...]`` shape."""
        p_recent = self.prob_changepoint_within(R_RECENT)
        return (p_recent, 1.0 - p_recent)

    def current_rho(self) -> float:
        """The forgetting factor for this slow tick (wiring contract): P12
        calls this once per slow tick and applies the result via
        ``forget(rho)`` on every ``BeliefModule`` — never through this
        module directly.

            rho = 1 - (1 - RHO_MIN) * P(run_length < R_recent)

        Stable regime (P ~= 0) => rho ~= 1 (no forgetting). Confident
        recent changepoint (P ~= 1) => rho -> RHO_MIN (near-maximal
        forgetting, reinflating posteriors and reopening EIG).
        """
        p_recent = self.prob_changepoint_within(R_RECENT)
        return 1.0 - (1.0 - RHO_MIN) * p_recent

    # -- BeliefModule protocol ---------------------------------------------

    def update(self, obs: Observation, self_events: SelfEvents) -> None:
        """No-op: RegimeTracker consumes SUMMARY STATISTICS, not raw
        Observations (there is no well-defined per-engine-step summary
        cadence here — the slow loop ticks every M steps). The caller
        (P12) must call ``observe_summary`` directly on slow ticks; this
        method exists only to satisfy the BeliefModule protocol's
        Observation-keyed update signature and intentionally does nothing.
        """
        self._step = obs.step

    def observe_summary(
        self, trade_tempo: float, realized_vol: float, imbalance: float, mean_depth: float
    ) -> None:
        """One Adams-MacKay BOCPD tick from the four public summary stats.

        Order matches ``STAT_NAMES``: (trade_tempo, realized_vol,
        imbalance, mean_depth) — the fields ``WorldSummary`` and
        ``WorldSummary``-shaped harness output carry for exactly this
        purpose. Never pass a harness-only ground-truth regime id here
        (INV-11) — only publicly observable summaries.
        """
        y = np.array(
            [trade_tempo, realized_vol, imbalance, mean_depth], dtype=np.float64
        )

        # Surprise: NLL of y under the current (pre-update) mixture
        # predictive, BEFORE folding y in — a genuine one-step-ahead score.
        log_pred = self._state.predictive_log_pdf(y)
        log_mix = special.logsumexp(self._log_run_length_posterior + log_pred)
        self._surprise.score(-float(log_mix))

        log_hazard = math.log(self._cfg.hazard)
        log_1m_hazard = math.log1p(-self._cfg.hazard)

        # Growth: r -> r+1 with weight pred(y|r) * (1 - H), each row scored
        # by its OWN predictive. Changepoint: mass H * Σ_r P(r) collapses
        # to a fresh run-length-0 hypothesis; the standard Adams-MacKay
        # recursion scores that collapsed mass by the PRIOR predictive at
        # y (the fresh segment has not seen any data yet), not by any
        # existing row's predictive.
        log_joint = self._log_run_length_posterior + log_pred
        log_growth = log_joint + log_1m_hazard
        prior_pred = self._prior.predictive_log_pdf(y)[0]
        log_cp = (
            special.logsumexp(self._log_run_length_posterior) + log_hazard + prior_pred
        )

        log_unnorm = np.concatenate([[log_cp], log_growth])
        log_z = special.logsumexp(log_unnorm)
        log_post = log_unnorm - log_z

        grown_state = self._state.folded(y)
        new_row0 = _NIGState(
            mu=self._prior.mu.copy(),
            kappa=self._prior.kappa.copy(),
            alpha=self._prior.alpha.copy(),
            beta=self._prior.beta.copy(),
        ).folded(y)
        new_state = _stack(new_row0, grown_state)

        # Truncate the low-mass tail for a bounded representation.
        order = np.argsort(log_post)  # ascending
        cum_from_bottom = np.cumsum(np.exp(log_post[order]))
        drop = order[cum_from_bottom < TRUNCATE_MASS]
        if drop.size > 0 and drop.size < len(log_post):
            keep_mask = np.ones(len(log_post), dtype=bool)
            keep_mask[drop] = False
            log_post = log_post[keep_mask]
            log_post = log_post - special.logsumexp(log_post)
            new_state = _NIGState(
                mu=new_state.mu[keep_mask],
                kappa=new_state.kappa[keep_mask],
                alpha=new_state.alpha[keep_mask],
                beta=new_state.beta[keep_mask],
            )

        self._log_run_length_posterior = log_post
        self._state = new_state
        self._n_ticks += 1

    def forget(self, rho: float) -> None:
        """Forgetting is not meaningful for the run-length posterior itself
        (it already reinflates through the changepoint mechanism, which is
        the whole point of this module) — this is a documented no-op.
        Downstream BeliefModules are forgotten by the CALLER via
        ``current_rho()`` + their own ``forget(rho)``, never by this
        module reaching into them (wiring contract).
        """
        if not 0.0 < rho <= 1.0:
            raise ValueError(f"rho must be in (0, 1], got {rho}")

    def posterior_entropy_nats(self) -> float:
        """Expected PARAMETER-posterior entropy, marginalized over the
        run-length posterior: E_r[H[NIG params | run length r]]. This is
        the mixture's average per-hypothesis entropy, not the entropy of
        the run-length distribution itself (which is a STATE variable,
        analogous to FairValueKF's state/parameter split) — INV-3.
        """
        post = self.run_length_posterior
        row_entropy = self._state.entropy_nats()
        return float(np.dot(post, row_entropy))

    def run_length_entropy_nats(self) -> float:
        """Shannon entropy of the run-length distribution itself (a STATE
        quantity — analogous to FairValueKF's ``state_eig_nats`` split —
        exposed for diagnostics, excluded from ``posterior_entropy_nats``.
        """
        post = self.run_length_posterior
        mask = post > 0.0
        return float(-np.dot(post[mask], np.log(post[mask])))

    def predict(self) -> ForecastStats:
        """Mixture predictive summary (mean, variance) of the next
        4-vector's FIRST dimension (trade_tempo), the module's designated
        scalar headline observable; other dimensions are available via
        ``predict_all``."""
        mean, var = self.predict_all()
        return ForecastStats(mean=float(mean[0]), variance=float(var[0]))

    def predict_all(self) -> tuple[FloatArray, FloatArray]:
        """Mixture (mean, variance) per dimension, marginalized over the
        run-length posterior. Each row's predictive is Student-t with
        location ``mu`` and the standard NIG predictive variance
        ``beta*(kappa+1) / (kappa*(alpha-1))`` for alpha > 1 (else
        infinite, reported as such via ``math.inf``)."""
        post = self.run_length_posterior
        mu = self._state.mu
        mean = post @ mu
        with np.errstate(divide="ignore"):
            row_var = np.where(
                self._state.alpha > 1.0,
                self._state.beta
                * (self._state.kappa + 1.0)
                / (self._state.kappa * (self._state.alpha - 1.0)),
                np.inf,
            )
        # Law of total variance: E[Var] + Var[E].
        e_var = post @ row_var
        deviation = mu - mean[np.newaxis, :]
        var_e = post @ (deviation * deviation)
        return mean, e_var + var_e

    def surprise_z(self) -> float:
        """Salience-only surprise; never feeds EIG or action scoring."""
        return self._surprise.last_z

    def eig_nats(self, probe: ProbeSpec) -> float:
        """I(theta; Y) for one prospective slow tick's summary vector,
        marginal over the CURRENT run-length posterior's mixture.

        Like the other world predictors, this is intent-independent:
        REGIME is passive-only (no probe may target it — enforced by
        ``target_id`` never being REGIME in the proposer's menu), and
        public summaries are observed every slow tick regardless of what
        the agent does. ``probe`` is accepted only to satisfy the
        BeliefModule protocol; only ``probe.horizon_steps`` participates,
        and only through the number of tick-equivalents the horizon
        implies (documented as horizon-independent below, mirroring
        FairValueKF's scale-family invariance): the mixture predictive at
        one tick out is used regardless of horizon, since further-out
        changepoint composition is not identifiable from the current
        posterior alone without simulating forward.
        """
        return self.eig_breakdown(probe).eig_nats

    def eig_breakdown(self, probe: ProbeSpec) -> EIGTerms:
        """The epistemic/aleatoric decomposition behind ``eig_nats``.

        H[Y] is the entropy of the mixture-of-Student-t predictive,
        estimated by Gauss-Hermite-free deterministic quadrature over each
        row's Student-t (closed-form component entropies are not additive
        for a mixture, so the mixture entropy itself is estimated by a
        fixed-node numerical integration in each dimension, assuming
        independence across dimensions and across mixture rows for the
        quadrature grid — documented approximation, MC-verified in
        tests/beliefs/test_eig_matches_monte_carlo.py).
        E_theta H[Y | theta] is the mixture-weighted average of each row's
        exact joint Gaussian-given-parameters entropy (sum of per-dimension
        Gaussian entropies at the row's posterior-mean variance estimate,
        i.e. the aleatoric floor conditional on knowing which regime/row
        is active and its point parameter estimate).
        """
        if probe.horizon_steps < 1:
            raise ValueError(
                f"probe horizon_steps must be >= 1, got {probe.horizon_steps}"
            )
        post = self.run_length_posterior
        df = 2.0 * self._state.alpha
        scale2 = (
            self._state.beta
            * (self._state.kappa + 1.0)
            / (self._state.alpha * self._state.kappa)
        )
        # Aleatoric: E_theta H[Y | theta] using the per-row Student-t
        # predictive entropy (marginalizing the row's own NIG uncertainty
        # is already "known theta" at the mixture level — theta here is
        # "which run length", so within a row we still integrate out that
        # row's (mean, var) uncertainty; this is the standard nested-EIG
        # treatment: row identity is the parameter of interest, so the
        # aleatoric floor per row IS that row's predictive entropy).
        row_entropy = np.sum(stats.t.entropy(df, scale=np.sqrt(scale2)), axis=1)

        h_mixture = self._mixture_entropy_nats(post)
        return information_gain_terms(h_mixture, row_entropy, post)

    def _mixture_entropy_nats(self, post: FloatArray) -> float:
        """Monte Carlo estimate of the mixture predictive's differential
        entropy: draw from the mixture (choose a row by ``post``, then a
        4-vector from that row's product-Student-t predictive), evaluate
        the mixture log-density there, average -log density. A fixed seed
        keeps this deterministic run-to-run for a given posterior state
        (INV-8's spirit: no hidden nondeterminism in a value that feeds a
        logged headline), while remaining a genuine Monte Carlo estimate
        of a quantity with no closed form for a Student-t mixture.
        """
        n_samples = 4000
        rng = np.random.default_rng(1_618_033)
        n_rows = len(post)
        rows = rng.choice(n_rows, size=n_samples, p=post)
        df = 2.0 * self._state.alpha
        scale = np.sqrt(
            self._state.beta
            * (self._state.kappa + 1.0)
            / (self._state.alpha * self._state.kappa)
        )
        samples = self._state.mu[rows] + scale[rows] * rng.standard_t(df[rows])
        log_pred_all = np.stack(
            [self._state.predictive_log_pdf(samples[i]) for i in range(n_samples)]
        )
        log_mix = special.logsumexp(
            np.log(post)[np.newaxis, :] + log_pred_all, axis=1
        )
        return float(-np.mean(log_mix))

    def snapshot_entropy(self) -> EntropySnapshot:
        """Parameter-posterior entropy at this instant (INV-10)."""
        return EntropySnapshot(
            hypothesis_id=self.hypothesis_id,
            step=self._step,
            entropy_nats=self.posterior_entropy_nats(),
        )
