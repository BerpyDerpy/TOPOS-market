"""SelfTrajectory: the reflexive self-forecast compiler.

NOT a learned model. This module owns no posterior and no update rule: it
COMPILES the predictive distribution of the agent's own near future —
(inventory_lots, mark-to-market value change) over a horizon H — out of
the posteriors that already exist elsewhere:

    * fill probabilities   — the FillModel's Beta cells,
    * own price impact     — the ImpactModel's NIG posterior,
    * fair-value motion    — the FairValueKF's state + noise-scale posterior,

and exposes

    self_entropy_nats(intent) = entropy of that joint predictive.

THE INVARIANT THAT MUST NOT DRIFT: this quantity is curiosity applied
reflexively — "how uncertain am I about what I am about to become?" — and
it must remain a genuine entropy of a genuine forecast derived from the
SAME posteriors used everywhere else. It is NOT a risk penalty: there is
no coefficient here calibrated against run outcomes, no account-state
input of any kind (the compiler consumes ``SelfStateCognitive`` only —
INV-5; mechanically pinned by a source scan in the tests), and no term
that exists because it "produced good behavior".

Method: MOMENT MATCHING over an exact fill-outcome enumeration (chosen
over Monte Carlo forward simulation so the compiler is deterministic and
closed-form, in the spirit of INV-2/INV-8):

1. The candidate intent is mapped to the single order it implies (shared
   ``implied_order`` approximation); together with the working orders this
   gives n Bernoulli fill events, each with its FillModel bucket's
   posterior-mean probability, treated as independent (orders merged per
   (side, price) first). All 2^n outcomes are enumerated exactly.
2. Conditional on a fill outcome, end inventory I and the execution edge
   (fill price vs current mark) are deterministic; the horizon mid change
   is the sum of the fair-value H-step predictive and the impact
   posterior's own-effect contribution for the lots that outcome executes.
   Fills are placed at the START of the horizon (exposure upper bound) and
   impact is read as permanent over H — both documented approximations of
   the compiler, not knobs.
3. The mark-to-market value change conditional on I is moment-matched to
   a Gaussian (law of total variance across outcomes with the same I),
   discretized at the tick-lot grid the whole architecture lives on:
   H = 0.5 ln(2 pi e (var + 1/12)) — the 1/12 is the variance of unit-cell
   rounding (a property of the integer tick/lot grid, not a tuning
   constant), and it makes a deterministic forecast carry ~0 extra nats
   instead of a divergent differential entropy.
4. The joint entropy is exact by the chain rule:
   H(I, dV) = H(I) + sum_I P(I) H(dV | I).

Natural behavior (pinned ordinally by tests, no tuned thresholds):
holding inventory in a volatile market => high self-entropy; a flat book
with no working orders => minimal self-entropy; an aggressive intent in
an illiquid book => high self-entropy through fill/impact uncertainty.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from topos.contracts.intent import SELF_TRAJECTORY, HypothesisId, Intent
from topos.contracts.market import Side
from topos.contracts.workspace import SelfStateCognitive, WorldSummary

from topos.beliefs.core import LOG_2PIE
from topos.beliefs.fair_value import FairValueKF
from topos.selfmodel.common import BookContext, implied_order, offset_band_of
from topos.selfmodel.fill_model import FillModel
from topos.selfmodel.impact_model import ImpactModel

UNIT_CELL_VAR = 1.0 / 12.0
"""Variance of rounding to one tick-lot cell: the architecture's value
quantities live on an integer grid, so the compiled value-change forecast
is a lattice distribution; moment-matching it as Gaussian + uniform
rounding adds exactly 1/12 per cell. Structural (a property of the grid),
not a calibration."""


@dataclass(frozen=True)
class TrajectoryForecast:
    """The compiled predictive of (inventory, mark-to-market value change)."""

    horizon_steps: int
    inventory_pmf: tuple[tuple[int, float], ...]
    """(inventory_lots at horizon, probability), ascending by inventory."""
    value_change_mean: float
    value_change_variance: float
    inventory_entropy_nats: float
    value_entropy_nats: float
    """Expected conditional entropy of the value change given inventory."""
    entropy_nats: float
    """Joint entropy: the reflexive-curiosity quantity."""


@dataclass(frozen=True)
class _FillEvent:
    signed_lots: int
    price_ticks: int
    fill_probability: float
    is_marketable: bool


class SelfTrajectory:
    """Forecast compiler over the agent's own trajectory
    (hypothesis_id="self_trajectory"; never a probeable hypothesis — a
    committed intent carrying this id is the flatten intent, see
    contracts)."""

    hypothesis_id: HypothesisId = SELF_TRAJECTORY

    def __init__(
        self,
        fill_model: FillModel,
        impact_model: ImpactModel,
        fair_value: FairValueKF,
        *,
        size_budget_lots: int = 1,
        max_enumerated_orders: int = 12,
    ) -> None:
        """``size_budget_lots`` interprets ``Intent.size_frac`` (fraction of
        the per-step budget, per contract) — a wiring constant shared with
        the motor config. ``max_enumerated_orders`` bounds the exact 2^n
        enumeration; beyond it the smallest orders are treated as
        never-filling (bounded computation, documented coarsening)."""
        if size_budget_lots < 1:
            raise ValueError(
                f"size_budget_lots must be >= 1, got {size_budget_lots}"
            )
        if max_enumerated_orders < 1:
            raise ValueError(
                f"max_enumerated_orders must be >= 1, got {max_enumerated_orders}"
            )
        self._fill = fill_model
        self._impact = impact_model
        self._fair_value = fair_value
        self._size_budget_lots = size_budget_lots
        self._max_enumerated = max_enumerated_orders
        self._cognitive: SelfStateCognitive | None = None
        self._world: WorldSummary | None = None

    def begin_cycle(
        self, cognitive: SelfStateCognitive, world: WorldSummary
    ) -> None:
        """Receive this cycle's broadcast state. The compiler sees the
        COGNITIVE self-state only (INV-5): inventory and working orders,
        never account quantities."""
        self._cognitive = cognitive
        self._world = world

    # -- the reflexive quantity ------------------------------------------------

    def self_entropy_nats(
        self, intent: Intent, horizon_steps: int | None = None
    ) -> float:
        """Entropy (nats) of the compiled (inventory, value-change)
        predictive under the candidate intent — see the module docstring
        for exactly what this is and is not."""
        return self.forecast(intent, horizon_steps).entropy_nats

    def forecast(
        self, intent: Intent, horizon_steps: int | None = None
    ) -> TrajectoryForecast:
        """Compile the predictive distribution of (inventory_lots,
        mark-to-market value change) over the horizon."""
        if self._cognitive is None or self._world is None:
            raise RuntimeError(
                "call begin_cycle(cognitive_state, world_summary) before "
                "compiling a forecast"
            )
        horizon = (
            horizon_steps if horizon_steps is not None
            # Default to the fill model's own horizon: the fill posteriors
            # answer "filled within THAT many steps", so compiling at the
            # same horizon is the only choice that needs no extrapolation.
            else self._fill.horizon_steps
        )
        if horizon < 1:
            raise ValueError(f"horizon_steps must be >= 1, got {horizon}")
        world = self._world
        cognitive = self._cognitive
        mid = world.mid_ticks
        half_spread = 0.5 * world.spread_ticks
        ctx = BookContext(
            mid=mid,
            best_bid=int(round(mid - half_spread)),
            best_ask=int(round(mid + half_spread)),
            imbalance=world.imbalance,
        )
        events = self._fill_events(intent, cognitive, ctx)

        # Horizon mid-change predictive from the fair-value posterior:
        # Student-t moment-matched to (mean, variance); the variance is
        # E[c] * scale-free predictive variance (exact for df > 2).
        fv_mean_level, fv_scale_free_var = self._fair_value.horizon_prediction(
            horizon
        )
        drift_mean = fv_mean_level - mid
        mid_var = self._fair_value.noise_scale_posterior.mean() * fv_scale_free_var

        # Own resting presence at the touch (working orders + a passive
        # intent) is shared by every outcome; executed aggression differs
        # per outcome.
        resting_touch = 0.0
        for event in events:
            if not event.is_marketable and self._at_touch(event, ctx):
                resting_touch += float(event.signed_lots)

        # Exact enumeration of fill outcomes.
        outcome_prob: list[float] = [1.0]
        outcome_inventory: list[int] = [cognitive.inventory_lots]
        outcome_edge: list[float] = [0.0]
        outcome_aggression: list[float] = [0.0]
        for event in events:
            next_prob: list[float] = []
            next_inventory: list[int] = []
            next_edge: list[float] = []
            next_aggression: list[float] = []
            for prob, inv, edge, aggr in zip(
                outcome_prob, outcome_inventory, outcome_edge, outcome_aggression
            ):
                p = event.fill_probability
                next_prob.append(prob * (1.0 - p))
                next_inventory.append(inv)
                next_edge.append(edge)
                next_aggression.append(aggr)
                next_prob.append(prob * p)
                next_inventory.append(inv + event.signed_lots)
                next_edge.append(
                    edge + event.signed_lots * (mid - event.price_ticks)
                )
                next_aggression.append(
                    aggr
                    + (float(event.signed_lots) if event.is_marketable else 0.0)
                )
            outcome_prob = next_prob
            outcome_inventory = next_inventory
            outcome_edge = next_edge
            outcome_aggression = next_aggression

        # Moment-match the value change per outcome, then pool by end
        # inventory (law of total variance within each inventory value).
        by_inventory: dict[int, list[tuple[float, float, float]]] = {}
        for prob, inv, edge, aggr in zip(
            outcome_prob, outcome_inventory, outcome_edge, outcome_aggression
        ):
            if prob <= 0.0:
                continue
            impact_mean, impact_var = self._impact.predictive_own_effect(
                aggr, resting_touch
            )
            mean_o = inv * (drift_mean + impact_mean) + edge
            var_o = float(inv * inv) * (mid_var + impact_var)
            by_inventory.setdefault(inv, []).append((prob, mean_o, var_o))

        inventory_pmf: list[tuple[int, float]] = []
        inventory_entropy = 0.0
        value_entropy = 0.0
        total_mean = 0.0
        total_second_moment = 0.0
        for inv in sorted(by_inventory):
            group = by_inventory[inv]
            p_inv = sum(prob for prob, _, _ in group)
            mean_inv = sum(prob * mean for prob, mean, _ in group) / p_inv
            var_inv = (
                sum(
                    prob * (var + (mean - mean_inv) ** 2)
                    for prob, mean, var in group
                )
                / p_inv
            )
            inventory_pmf.append((inv, p_inv))
            inventory_entropy -= p_inv * math.log(p_inv)
            value_entropy += p_inv * 0.5 * (
                LOG_2PIE + math.log(var_inv + UNIT_CELL_VAR)
            )
            total_mean += p_inv * mean_inv
            total_second_moment += p_inv * (var_inv + mean_inv * mean_inv)

        return TrajectoryForecast(
            horizon_steps=horizon,
            inventory_pmf=tuple(inventory_pmf),
            value_change_mean=total_mean,
            value_change_variance=max(
                0.0, total_second_moment - total_mean * total_mean
            ),
            inventory_entropy_nats=inventory_entropy,
            value_entropy_nats=value_entropy,
            entropy_nats=inventory_entropy + value_entropy,
        )

    # -- internals --------------------------------------------------------------

    def _fill_events(
        self, intent: Intent, cognitive: SelfStateCognitive, ctx: BookContext
    ) -> list[_FillEvent]:
        """Bernoulli fill events for working orders plus the intent's
        implied order, with probabilities from the FillModel posterior."""
        merged: dict[tuple[int, int], int] = {}  # (side sign, price) -> lots
        for view in cognitive.working_orders:
            if view.size_lots_remaining <= 0:
                continue
            key = (view.side.value, view.price_ticks)
            merged[key] = merged.get(key, 0) + view.size_lots_remaining
        order = implied_order(intent, ctx, self._size_budget_lots)
        if order is not None:
            key = (order.side.value, order.price_ticks)
            merged[key] = merged.get(key, 0) + order.size_lots
        events: list[_FillEvent] = []
        for (side_sign, price), lots in merged.items():
            side = Side.BUY if side_sign > 0 else Side.SELL
            band = offset_band_of(side, price, ctx.best_bid, ctx.best_ask)
            probability = self._fill.predictive_fill_probability(
                side, price, ctx.best_bid, ctx.best_ask, ctx.imbalance
            )
            events.append(
                _FillEvent(
                    signed_lots=side_sign * lots,
                    price_ticks=price,
                    fill_probability=probability,
                    is_marketable=(band == "cross"),
                )
            )
        if len(events) > self._max_enumerated:
            events.sort(key=lambda e: abs(e.signed_lots), reverse=True)
            events = events[: self._max_enumerated]
        return events

    @staticmethod
    def _at_touch(event: _FillEvent, ctx: BookContext) -> bool:
        if event.signed_lots > 0:
            return event.price_ticks == ctx.best_bid
        return event.price_ticks == ctx.best_ask
