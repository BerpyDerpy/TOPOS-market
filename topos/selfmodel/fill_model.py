"""FillModel: Beta-Bernoulli fill probability, conditioned on own action.

One independent conjugate Beta cell per bucket over

    (side, offset band, book-imbalance band)

with the bands defined once in ``topos.selfmodel.common`` (offset edges
inherited from the flow model's depth bands; imbalance tripartition of
[-1, +1]). 2 sides x 4 offset bands x 3 imbalance bands = 24 cells.

THE CONDITIONING IS THE POINT (Layer-1 anti-churn). An unconditional fill
model can never converge — every context change looks like fresh evidence
about a single global rate, own fills stay surprising forever, and the
agent churns to keep "learning" them. Conditioned on own action and
context, each bucket's parameter posterior saturates: the tenth fill of a
well-probed bucket is boring (EIG ~ 0), while a never-probed bucket keeps
its prior EIG — exactly the gradient a curious proposer should feel.

Trial protocol
--------------
Every ACCEPTED own placement opens a trial in the bucket of its DECISION
context (the observation the placement was chosen against, i.e. the one
BEFORE the ack arrives). The trial resolves as one Bernoulli outcome:

* fully filled with the last counted fill stamped <= placement + horizon
  -> success 1;
* still open at the horizon -> the filled fraction in [0, 1] (partial
  fills split the pseudo-trial between the counts — the conjugate reading
  of "partly filled by the horizon");
* canceled/expired AT or after the horizon -> the filled fraction (the
  experiment ran its course);
* canceled/expired STRICTLY BEFORE the horizon -> the trial is DISCARDED.
  The order's fate over the full horizon was never observed (censoring by
  the agent's own hand), and folding censored trials in as failures would
  bias every bucket toward "orders never fill" in proportion to how
  impatient the motor happens to be.

``eig_nats(probe)`` is the mutual information between the probe's
predicted Bernoulli outcome and the PARAMETER of the bucket the probe
would exercise (INV-3), in closed form via digamma
(``BetaPosterior.eig_terms_bernoulli``; MC-verified in tests). The null
action places no order, so it exercises no bucket and its EIG through
this hypothesis is exactly 0 — this module's information can only be
BOUGHT by acting, which is precisely what distinguishes self-model
hypotheses from world predictors (whose EIG rides the null, INV-4).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from topos.contracts.beliefs import (
    EntropySnapshot,
    ForecastStats,
    ProbeSpec,
    SelfEvents,
)
from topos.contracts.intent import FILL_RATE, HypothesisId, Intent
from topos.contracts.market import AckStatus, Observation, Side

from topos.beliefs.core import BetaPosterior, EIGTerms, SurpriseTracker
from topos.selfmodel.common import (
    IMBALANCE_BANDS,
    OFFSET_BANDS,
    BookContext,
    context_from_observation,
    imbalance_band_of,
    implied_order,
    offset_band_of,
    paired_placements,
)

BucketKey = tuple[Side, str, str]
"""(side, offset band, imbalance band)."""


@dataclass
class _Trial:
    bucket: BucketKey
    size_lots: int
    filled_lots: int
    deadline_step: int


class FillModel:
    """BeliefModule over own-order fill probability (hypothesis_id="fill_rate")."""

    hypothesis_id: HypothesisId = FILL_RATE

    def __init__(
        self,
        horizon_steps: int,
        *,
        size_budget_lots: int = 1,
        prior_a: float = 1.0,
        prior_b: float = 1.0,
        surprise_ewma_decay: float = 0.05,
    ) -> None:
        """``horizon_steps`` defines the Bernoulli outcome: filled within
        that many steps of placement. It is part of the hypothesis (what
        "fill probability" means), fixed at construction and wired by the
        agent config — never adapted from outcomes.

        The Beta(1, 1) default prior is the uniform ignorance prior over a
        probability; ``size_budget_lots`` is the motor's per-step size
        budget (the meaning of ``Intent.size_frac`` per contract), needed
        only to interpret probes.
        """
        if horizon_steps < 1:
            raise ValueError(f"horizon_steps must be >= 1, got {horizon_steps}")
        if size_budget_lots < 1:
            raise ValueError(
                f"size_budget_lots must be >= 1, got {size_budget_lots}"
            )
        self.horizon_steps = horizon_steps
        self._size_budget_lots = size_budget_lots
        self._cells: dict[BucketKey, BetaPosterior] = {
            (side, offset_band, imb_band): BetaPosterior(prior_a, prior_b)
            for side in (Side.BUY, Side.SELL)
            for offset_band in OFFSET_BANDS
            for imb_band in IMBALANCE_BANDS
        }
        self._trials: dict[int, _Trial] = {}
        self._ctx: BookContext | None = None
        self._surprise = SurpriseTracker(ewma_decay=surprise_ewma_decay)
        self._step = 0
        self._last_bucket: BucketKey | None = None

    # -- posterior access (public: tests, proposer, metrics read these) ----

    @property
    def cells(self) -> dict[BucketKey, BetaPosterior]:
        """The 24 independent Beta cells, keyed by (side, offset, imbalance)."""
        return self._cells

    @property
    def open_trials(self) -> int:
        """Number of own orders whose horizon outcome is still pending."""
        return len(self._trials)

    def predictive_fill_probability(
        self,
        side: Side,
        price_ticks: int,
        best_bid: int | None,
        best_ask: int | None,
        imbalance: float,
    ) -> float:
        """Posterior-mean fill probability (within the model horizon) for an
        order at the given price in the given context. Consumed by the
        trajectory compiler so its forecasts ride the SAME posterior."""
        key = (
            side,
            offset_band_of(side, price_ticks, best_bid, best_ask),
            imbalance_band_of(imbalance),
        )
        return self._cells[key].mean()

    # -- BeliefModule protocol ---------------------------------------------

    def update(self, obs: Observation, self_events: SelfEvents) -> None:
        """Open trials for new placements, fold fills, resolve outcomes.

        New placements are bucketed against the PREVIOUS observation's
        context — the book the placement decision was actually made in
        (its ack arrives one observation later).
        """
        self._step = obs.step
        decision_ctx = self._ctx if self._ctx is not None else (
            context_from_observation(obs)
        )
        for order_id, place, ack in paired_placements(self_events):
            bucket = (
                place.side,
                offset_band_of(
                    place.side,
                    place.price_ticks,
                    decision_ctx.best_bid,
                    decision_ctx.best_ask,
                ),
                imbalance_band_of(decision_ctx.imbalance),
            )
            self._trials[order_id] = _Trial(
                bucket=bucket,
                size_lots=place.size_lots,
                filled_lots=0,
                deadline_step=ack.step + self.horizon_steps,
            )
            self._last_bucket = bucket
        for fill in obs.own_fills:
            trial = self._trials.get(fill.order_id)
            if trial is None or fill.step > trial.deadline_step:
                continue
            trial.filled_lots = min(
                trial.size_lots, trial.filled_lots + fill.size_lots
            )
            if trial.filled_lots == trial.size_lots:
                self._resolve(fill.order_id)
        for ack in obs.own_acks:
            if ack.status not in (AckStatus.CANCELED, AckStatus.EXPIRED):
                continue
            trial = self._trials.get(ack.order_id)
            if trial is None:
                continue
            if ack.step >= trial.deadline_step:
                self._resolve(ack.order_id)
            else:
                # Censored by own cancellation before the horizon: discard.
                del self._trials[ack.order_id]
        for order_id in [
            oid
            for oid, trial in self._trials.items()
            if obs.step >= trial.deadline_step
        ]:
            self._resolve(order_id)
        self._ctx = context_from_observation(obs)

    def _resolve(self, order_id: int) -> None:
        trial = self._trials.pop(order_id)
        fraction = trial.filled_lots / trial.size_lots
        cell = self._cells[trial.bucket]
        p_bar = cell.mean()
        # Surprise (salience only): Bernoulli nll at the realized fraction.
        nll = -(
            fraction * math.log(max(p_bar, 1e-300))
            + (1.0 - fraction) * math.log(max(1.0 - p_bar, 1e-300))
        )
        self._surprise.score(nll)
        cell.observe(fraction)
        self._last_bucket = trial.bucket

    def forget(self, rho: float) -> None:
        """Discount every bucket's sufficient statistics toward its prior."""
        for cell in self._cells.values():
            cell.forget(rho)

    def posterior_entropy_nats(self) -> float:
        """Joint PARAMETER-posterior entropy: sum over independent cells."""
        return sum(cell.entropy_nats() for cell in self._cells.values())

    def predict(self) -> ForecastStats:
        """Predictive fill probability of the current locus of own activity
        (the most recently exercised bucket; the ignorance prior before any
        own order exists), with the Bernoulli predictive variance."""
        if self._last_bucket is not None:
            p_bar = self._cells[self._last_bucket].mean()
        else:
            any_cell = next(iter(self._cells.values()))
            p_bar = any_cell.prior_a / (any_cell.prior_a + any_cell.prior_b)
        return ForecastStats(mean=p_bar, variance=p_bar * (1.0 - p_bar))

    def surprise_z(self) -> float:
        """Salience-only surprise; never feeds EIG or action scoring."""
        return self._surprise.last_z

    def eig_nats(self, probe: ProbeSpec) -> float:
        """I(p_bucket; fill outcome) for the bucket the probe exercises.

        0 for the null action (no order, no outcome, nothing this
        hypothesis can learn) — see the module docstring.
        """
        terms = self.eig_breakdown(probe)
        return terms.eig_nats if terms is not None else 0.0

    def eig_breakdown(self, probe: ProbeSpec) -> EIGTerms | None:
        """The epistemic/aleatoric decomposition behind ``eig_nats``;
        None when the probe places no order."""
        if probe.horizon_steps < 1:
            raise ValueError(
                f"probe horizon_steps must be >= 1, got {probe.horizon_steps}"
            )
        key = self.bucket_for_intent(probe.intent)
        if key is None:
            return None
        return self._cells[key].eig_terms_bernoulli()

    def bucket_for_intent(self, intent: Intent) -> BucketKey | None:
        """The bucket an intent's implied order would exercise in the
        current context; None for the null/directionless/zero-size case."""
        ctx = self._ctx
        if ctx is None:
            ctx = BookContext(mid=None, best_bid=None, best_ask=None, imbalance=0.0)
        order = implied_order(intent, ctx, self._size_budget_lots)
        if order is None:
            return None
        return (
            order.side,
            offset_band_of(
                order.side, order.price_ticks, ctx.best_bid, ctx.best_ask
            ),
            imbalance_band_of(ctx.imbalance),
        )

    def snapshot_entropy(self) -> EntropySnapshot:
        """Parameter-posterior entropy at this instant (INV-10)."""
        return EntropySnapshot(
            hypothesis_id=self.hypothesis_id,
            step=self._step,
            entropy_nats=self.posterior_entropy_nats(),
        )
