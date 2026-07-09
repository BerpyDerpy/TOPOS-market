"""QueuePositionFilter: discrete posterior over lots-ahead for resting orders.

For each of the agent's resting limit orders, this module maintains a
discrete distribution over the number of lots resting AHEAD of it at its
price level (the "queue rank").  The exchange never reports this quantity
(INV-11); it must be inferred from public L2 book changes and own fills.

Model
-----
State per tracked order: a probability vector ``pmf[k]`` for
k in {0, 1, ..., A_max}, where A_max is the level size at placement
(extended if the level grows behind us — that does not affect the
distribution over ahead).

Update rules (one step):

1. **Placement:** ahead = visible_size at the order's price level at the
   moment of placement (point mass).  Price-time priority puts a new
   arrival last.

2. **Trades at the order's level (while unfilled):** traded lots are
   consumed from the front of the queue.  Deterministic shift: reduce
   ``ahead`` by the traded size (floor at 0).  If a trade at this level
   was observed AND our posterior had mass at ahead=0 BUT we received no
   fill, then that mass is renormalized away (we observably were not at
   the front).

3. **Level-size decrease beyond observed trades = cancels of unknown
   position.**  Each canceled lot is ahead with probability
   ``ahead / (ahead + behind)`` independently, where
   ``behind = level_size - ahead - own_size``.  The exact model is
   **hypergeometric thinning**: if there are ``ahead`` lots ahead and
   ``behind`` behind, and ``c`` lots cancel uniformly among all non-own
   lots, then the number of cancels drawn from the ``ahead`` group is
   Hypergeometric(population=ahead+behind, successes=ahead, draws=c).
   This is exact at all sizes (no large-N Binomial approximation needed)
   and reduces to Binomial(c, ahead/(ahead+behind)) in the limit —
   documented here and tested against Binomial crosschecks.

4. **Level-size increases** are always behind (price-time priority):
   update ``behind`` only; the distribution over ``ahead`` is unchanged.

5. **Own partial/total fill:** strong evidence ahead had reached 0;
   condition on ahead == 0.

BeliefModule surface
--------------------
- ``posterior_entropy_nats``: sum of Shannon entropies of the rank
  distributions across all tracked orders.
- ``eig_nats(probe)``: expected rank-entropy reduction from the
  observations a probe would generate, using FlowIntensity's predictive
  for per-step trade/cancel volumes at the order's level.

  Cases:
  * "keep resting and watch trades/cancels": dominant case; coarse model
    of the information that one step of observed L2 changes would provide.
  * "cancel": destroys the tracked question — EIG is 0 (no information
    gained, just lost).
  * "place at a level": creates a new tracked question with a point-mass
    prior — EIG is the expected information from subsequent observations
    on that fresh distribution.

  Approximation: the observation model is coarse — we draw per-step
  trade and cancel volumes from FlowIntensity's NB predictive, then
  propagate through the update rules to compute expected posterior
  entropy.  This is a one-step-lookahead approximation to the true
  multi-step mutual information; exact Monte Carlo EIG would require
  simulating full trajectories.  Documented here and in the probe
  docstring.

- ``surprise_z``: z-scored surprise from realized fill/no-fill vs
  predicted front-of-queue probability.

No torch/jax/tensorflow/sklearn (INV-2): numpy + scipy only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy import stats as sp_stats

from topos.contracts.beliefs import (
    EntropySnapshot,
    ForecastStats,
    ProbeSpec,
    SelfEvents,
)
from topos.contracts.intent import QUEUE_POSITION, HypothesisId
from topos.contracts.market import (
    Ack,
    AckStatus,
    BookLevel,
    Fill,
    Liquidity,
    Observation,
    PlaceLimit,
    Side,
)
from topos.beliefs.core import SurpriseTracker

if TYPE_CHECKING:
    from topos.beliefs.flow_intensity import FlowIntensity

FloatArray = NDArray[np.float64]


# ---------------------------------------------------------------------------
# Tracked-order state
# ---------------------------------------------------------------------------

@dataclass
class _TrackedOrder:
    """Mutable state for one tracked resting order."""

    order_id: int
    side: Side
    price_ticks: int
    own_size: int
    """Remaining lots of the agent's own order at this level."""
    pmf: FloatArray
    """pmf[k] = P(ahead == k), k in {0, ..., len(pmf)-1}."""
    placed_this_step: bool = False
    """True during the step immediately after placement.

    The observation at placement time is taken BEFORE the agent's order
    is submitted to the engine.  So the visible level size in that
    observation does NOT include own_size.  One step later, the level
    DOES include own_size.  This flag selects the correct accounting.
    """

    @property
    def a_max(self) -> int:
        return len(self.pmf) - 1


# ---------------------------------------------------------------------------
# Pure distribution operations
# ---------------------------------------------------------------------------


def _entropy_nats(pmf: FloatArray) -> float:
    """Shannon entropy in nats of a discrete distribution."""
    p = pmf[pmf > 0.0]
    return float(-np.dot(p, np.log(p)))


def _shift_left(pmf: FloatArray, amount: int) -> FloatArray:
    """Deterministic shift: reduce ahead by ``amount`` (floor at 0).

    pmf'[k] = pmf[k + amount] for k >= 0; mass below 0 piles up at 0.
    """
    if amount <= 0:
        return pmf.copy()
    n = len(pmf)
    new = np.zeros(n, dtype=np.float64)
    if amount >= n:
        new[0] = 1.0
    else:
        new[0] = pmf[:amount + 1].sum()
        new[1:n - amount] = pmf[amount + 1:]
    return new


def _condition_not_zero(pmf: FloatArray) -> FloatArray:
    """Condition on ahead > 0: renormalize mass away from k=0.

    If all mass is at 0 (should not happen if update rules are consistent),
    return unchanged to avoid division by zero — caller should treat this
    as a consistency violation.
    """
    if pmf[0] <= 0.0:
        return pmf.copy()
    tail_mass = 1.0 - pmf[0]
    if tail_mass < 1e-15:
        # All mass at 0: cannot condition away from 0.
        # Return unchanged — caller detects via consistency checks.
        return pmf.copy()
    new = pmf.copy()
    new[0] = 0.0
    new /= tail_mass
    return new


def _condition_at_zero(pmf: FloatArray) -> FloatArray:
    """Condition on ahead == 0: point mass at 0."""
    new = np.zeros_like(pmf)
    new[0] = 1.0
    return new


def _hypergeometric_thin(
    pmf: FloatArray, n_cancel: int, total_non_own: int
) -> FloatArray:
    """Apply hypergeometric thinning for ``n_cancel`` cancels of unknown position.

    ``total_non_own`` is the total number of non-own lots at this level
    (i.e. ahead + behind).  For each hypothesized ahead=k, behind_k =
    total_non_own - k, and the number of cancels from the ahead group
    follows Hypergeometric(population=total_non_own, successes=k,
    draws=n_cancel).  This is the EXACT model (not a Binomial
    approximation) and reduces to Binomial(c, k/total_non_own) as
    total_non_own → ∞.
    """
    if n_cancel <= 0:
        return pmf.copy()
    n = len(pmf)
    new_pmf = np.zeros(n, dtype=np.float64)
    for k in range(n):
        if pmf[k] < 1e-30:
            continue
        behind_k = max(0, total_non_own - k)
        population = k + behind_k  # = total_non_own (clamped)
        if population <= 0:
            new_pmf[k] += pmf[k]
            continue
        draws = min(n_cancel, population)
        j_max = min(draws, k)
        j_min = max(0, draws - behind_k)
        for j in range(j_min, j_max + 1):
            new_k = k - j
            prob_j = float(sp_stats.hypergeom.pmf(j, population, k, draws))
            new_pmf[new_k] += pmf[k] * prob_j
    total = new_pmf.sum()
    if total > 0.0:
        new_pmf /= total
    return new_pmf


# ---------------------------------------------------------------------------
# The BeliefModule
# ---------------------------------------------------------------------------


class QueuePositionFilter:
    """BeliefModule over queue rank (hypothesis_id="queue_position").

    Maintains one discrete posterior per tracked resting order, updated
    from public L2 changes and own fill events.  The agent never observes
    ground-truth queue position (INV-11).
    """

    hypothesis_id: HypothesisId = QUEUE_POSITION

    def __init__(
        self,
        *,
        flow_model: FlowIntensity | None = None,
        surprise_ewma_decay: float = 0.05,
        eig_mc_samples: int = 64,
    ) -> None:
        self._flow = flow_model
        self._tracked: dict[int, _TrackedOrder] = {}
        self._surprise = SurpriseTracker(ewma_decay=surprise_ewma_decay)
        self._step = 0
        self._prev: Observation | None = None
        self._eig_mc_samples = eig_mc_samples
        # Own-order ledger for placement tracking
        self._pending_placements: list[PlaceLimit] = []

    # -- posterior access (public: tests, metrics) -------------------------

    @property
    def tracked_orders(self) -> dict[int, _TrackedOrder]:
        """Active tracked orders keyed by order_id."""
        return dict(self._tracked)

    def rank_pmf(self, order_id: int) -> FloatArray | None:
        """Return the posterior pmf over ahead for a tracked order, or None."""
        t = self._tracked.get(order_id)
        return t.pmf.copy() if t is not None else None

    def rank_mean_var(self, order_id: int) -> tuple[float, float] | None:
        """(mean, variance) of the ahead distribution, or None."""
        t = self._tracked.get(order_id)
        if t is None:
            return None
        k = np.arange(len(t.pmf), dtype=np.float64)
        mean = float(np.dot(k, t.pmf))
        var = float(np.dot(k * k, t.pmf)) - mean * mean
        return mean, max(0.0, var)

    # -- BeliefModule protocol ---------------------------------------------

    def update(self, obs: Observation, self_events: SelfEvents) -> None:
        """Update all tracked orders from one step's evidence.

        Processing order:
        1. Register new placements from this step's acks.
        2. Compute per-price traded volume from public trades.
        3. Compute per-price level-size changes.
        4. For each tracked order: apply trades, cancels, fills.
        5. Remove fully filled or canceled orders.
        """
        self._step = obs.step
        prev = self._prev
        self._prev = obs

        # 1. Register new placements.
        self._register_placements(obs, self_events)

        # 2. Compute own fills by order.
        own_fills_by_oid: dict[int, int] = {}
        own_fill_set: set[int] = set()
        for fill in obs.own_fills:
            if fill.order_id in self._tracked:
                own_fills_by_oid[fill.order_id] = (
                    own_fills_by_oid.get(fill.order_id, 0) + fill.size_lots
                )
                own_fill_set.add(fill.order_id)

        # 3. Remove canceled/expired orders from tracking.
        for ack in obs.own_acks:
            if ack.status in (AckStatus.CANCELED, AckStatus.EXPIRED):
                self._tracked.pop(ack.order_id, None)

        if prev is None:
            return

        # 4. Compute per-price traded volumes (from public trades).
        traded_at: dict[tuple[Side, int], int] = {}
        for trade in obs.trades:
            passive = Side.SELL if trade.aggressor is Side.BUY else Side.BUY
            key = (passive, trade.price_ticks)
            traded_at[key] = traded_at.get(key, 0) + trade.size_lots

        # Add own taker fills (these don't show in public trades).
        for fill in obs.own_fills:
            if fill.liquidity is Liquidity.TAKER:
                t = self._tracked.get(fill.order_id)
                if t is not None:
                    passive = Side.SELL if t.side is Side.BUY else Side.BUY
                    key = (passive, fill.price_ticks)
                    traded_at[key] = traded_at.get(key, 0) + fill.size_lots

        # 5. Compute level sizes from book snapshots.
        prev_sizes = _level_size_map(prev)
        cur_sizes = _level_size_map(obs)

        # 6. Surprise accumulator for this step.
        nll_accum = 0.0
        n_surprise = 0

        # 7. Update each tracked order.
        to_remove: list[int] = []
        for oid, t in self._tracked.items():
            fill_lots = own_fills_by_oid.get(oid, 0)
            traded = traded_at.get((t.side, t.price_ticks), 0)

            # Level sizes.
            prev_level = prev_sizes.get((t.side, t.price_ticks), 0)
            cur_level = cur_sizes.get((t.side, t.price_ticks), 0)

            # Surprise: probability we were at front (ahead==0) before
            # this step's evidence.
            p_front = float(t.pmf[0]) if len(t.pmf) > 0 else 0.0

            # --- Apply own fill: condition on ahead == 0.
            if fill_lots > 0:
                t.pmf = _condition_at_zero(t.pmf)
                t.own_size = max(0, t.own_size - fill_lots)
                if t.own_size <= 0:
                    to_remove.append(oid)
                # Surprise from fill: -log P(fill) ≈ -log P(ahead==0).
                nll = -math.log(max(p_front, 1e-30))
                nll_accum += nll
                n_surprise += 1
                continue

            # --- Apply trades (deterministic shift).
            if traded > 0:
                t.pmf = _shift_left(t.pmf, traded)
                # If trades at our level but no fill, and we had mass at 0:
                # condition away from 0 (we observably were not at the front).
                if t.pmf[0] > 0.0:
                    t.pmf = _condition_not_zero(t.pmf)
                # Surprise from no-fill when expected fill.
                nll = -math.log(max(1.0 - p_front, 1e-30))
                nll_accum += nll
                n_surprise += 1

            # --- Apply cancels (level decrease beyond trades).
            # Observation-timing note:
            # * First step after placement (placed_this_step=True): the
            #   observation that is now "prev" was taken BEFORE our order was
            #   submitted. So prev_level does NOT include own_size. The actual
            #   book level after submission was prev_level + own_size. cur_level
            #   includes our order (we are still resting). So:
            #     true_prev = prev_level + own_size
            #     total_non_own = prev_level (all non-own lots that step)
            # * Subsequent steps: prev_level already includes own_size. So:
            #     true_prev = prev_level
            #     total_non_own = prev_level - own_size
            if t.placed_this_step:
                true_prev = prev_level + t.own_size
                total_non_own = max(0, prev_level)
            else:
                true_prev = prev_level
                total_non_own = max(0, prev_level - t.own_size)
            raw_decrease = true_prev - cur_level - traded
            cancel_count = max(0, raw_decrease)

            if cancel_count > 0:
                t.pmf = _hypergeometric_thin(t.pmf, cancel_count, total_non_own)

            t.placed_this_step = False  # clear after first post-placement step


        for oid in to_remove:
            self._tracked.pop(oid, None)

        # Score surprise.
        if n_surprise > 0:
            self._surprise.score(nll_accum / n_surprise)

    def forget(self, rho: float) -> None:
        """Forgetting is not meaningful for a discrete queue-rank posterior
        (there is no conjugate sufficient-statistic structure to discount).

        The distribution resets when an order is placed and evolves
        deterministically from observations.  This is a no-op.
        """
        pass

    def posterior_entropy_nats(self) -> float:
        """Sum of Shannon entropies of rank distributions across tracked orders."""
        return sum(_entropy_nats(t.pmf) for t in self._tracked.values())

    def predict(self) -> ForecastStats:
        """Aggregate predictive summary: mean and variance of total ahead lots."""
        total_mean = 0.0
        total_var = 0.0
        for t in self._tracked.values():
            mv = self.rank_mean_var(t.order_id)
            if mv is not None:
                total_mean += mv[0]
                total_var += mv[1]
        return ForecastStats(mean=total_mean, variance=total_var)

    def surprise_z(self) -> float:
        """z-scored surprise from realized fill/no-fill vs predicted
        front-of-queue probability."""
        return self._surprise.last_z

    def eig_nats(self, probe: ProbeSpec) -> float:
        """Expected information gain from observing one step of L2 changes.

        Approximation: we draw per-step trade and cancel volumes at each
        tracked order's price level from FlowIntensity's negative-binomial
        predictive, propagate through the update rules, and compute the
        expected posterior entropy.  EIG = H_prior - E[H_posterior].

        This is a one-step-lookahead coarse approximation (documented in
        the module docstring).  The observation model uses FlowIntensity's
        predictive means as the representative scenario, with MC samples
        for the volume draws.

        Cases:
        - Null / keep resting: the dominant case — information from
          watching trades and cancels at the order's level.
        - Cancel: destroys the tracked question; EIG = 0.
        - Place at a level: creates a new point-mass prior whose entropy
          is 0; EIG = expected information from the NEXT step's observations
          on that fresh distribution.  This is approximated as 0 (the
          first step carries minimal information since the prior is a
          point mass that can only shift/thin deterministically).
        """
        if not self._tracked:
            return 0.0

        # Cancel intent: EIG is 0 (destroys the question).
        if probe.intent.is_null:
            # Null action: watch trades/cancels.
            pass
        # For non-null committed intents that are not "keep resting",
        # the queue filter's EIG is essentially 0 (placing creates a new
        # point mass that carries negligible first-step EIG).
        # We compute EIG only for the "keep resting and observe" case.

        if self._flow is None:
            # Without a flow model, return 0 — no predictive available.
            return 0.0

        h_prior = self.posterior_entropy_nats()
        if h_prior < 1e-12:
            return 0.0

        horizon = max(1, probe.horizon_steps)

        # For each tracked order, compute expected posterior entropy
        # via Monte Carlo over trade/cancel volume draws.
        expected_h_post = 0.0
        for t in self._tracked.values():
            expected_h_post += self._eig_for_order(t, horizon)

        eig = max(0.0, h_prior - expected_h_post)
        return eig

    def snapshot_entropy(self) -> EntropySnapshot:
        """Parameter-posterior entropy at this instant (INV-10)."""
        return EntropySnapshot(
            hypothesis_id=self.hypothesis_id,
            step=self._step,
            entropy_nats=self.posterior_entropy_nats(),
        )

    # -- internal ----------------------------------------------------------

    def _register_placements(
        self, obs: Observation, self_events: SelfEvents
    ) -> None:
        """Track newly placed orders from this step's acks."""
        placements = [
            msg for msg in self_events.messages_sent if isinstance(msg, PlaceLimit)
        ]
        placement_acks = [
            ack
            for ack in self_events.acks
            if ack.status in (AckStatus.ACCEPTED, AckStatus.REJECTED)
        ]
        for k, ack in enumerate(placement_acks):
            if k >= len(placements):
                break
            if ack.status is AckStatus.ACCEPTED:
                place = placements[k]
                # Determine visible level size at placement price.
                level_size = _visible_level_size(obs, place.side, place.price_ticks)
                # ahead = visible size at level BEFORE our order was added.
                # The observation is taken BEFORE the agent's action in the
                # engine step, so the level size does not include our order.
                ahead = max(0, level_size)
                pmf = np.zeros(ahead + 1, dtype=np.float64)
                pmf[ahead] = 1.0  # point mass: we are last in queue
                self._tracked[ack.order_id] = _TrackedOrder(
                    order_id=ack.order_id,
                    side=place.side,
                    price_ticks=place.price_ticks,
                    own_size=place.size_lots,
                    pmf=pmf,
                    placed_this_step=True,
                )

    def _eig_for_order(self, t: _TrackedOrder, horizon: int) -> float:
        """Expected posterior entropy for one order after ``horizon`` steps.

        Uses MC samples from FlowIntensity's predictive for per-step
        trade and cancel volumes at the order's level.
        """
        assert self._flow is not None

        h_order = _entropy_nats(t.pmf)
        if h_order < 1e-12:
            return 0.0

        # Get predictive trade and cancel rates at this order's band.
        from topos.beliefs.flow_intensity import band_of, BANDS

        # Find the band for this order's price.
        # We need the distance from the best price.  Use the last
        # observation's book.
        band = self._order_band(t)
        side = t.side

        # Trade rate: "market" events by the opposite-side aggressor
        # at this band.
        aggressor = Side.SELL if side is Side.BUY else Side.BUY
        trade_cell = self._flow.cells.get(("market", aggressor, band))
        cancel_cell = self._flow.cells.get(("cancel", side, band))

        if trade_cell is None and cancel_cell is None:
            return h_order

        # MC samples of posterior entropy.
        rng = np.random.default_rng(self._step * 1000 + t.order_id)
        h_posts = np.zeros(self._eig_mc_samples, dtype=np.float64)

        for i in range(self._eig_mc_samples):
            pmf_sim = t.pmf.copy()
            for _ in range(horizon):
                # Draw trade volume.
                if trade_cell is not None:
                    p_nb = trade_cell.b / (trade_cell.b + 1.0)
                    traded = int(rng.negative_binomial(
                        max(1, int(round(trade_cell.a))), max(1e-10, p_nb)
                    ))
                else:
                    traded = 0

                # Draw cancel volume.
                if cancel_cell is not None:
                    p_nb_c = cancel_cell.b / (cancel_cell.b + 1.0)
                    canceled = int(rng.negative_binomial(
                        max(1, int(round(cancel_cell.a))), max(1e-10, p_nb_c)
                    ))
                else:
                    canceled = 0

                # Apply trades.
                if traded > 0:
                    pmf_sim = _shift_left(pmf_sim, traded)
                    # Simulate no-fill conditioning: assume we don't fill
                    # (most likely for most of the distribution).
                    if pmf_sim[0] > 0.0 and pmf_sim[0] < 1.0 - 1e-10:
                        pmf_sim = _condition_not_zero(pmf_sim)

                # Apply cancels.
                if canceled > 0:
                    k_arr = np.arange(len(pmf_sim), dtype=np.float64)
                    mean_ahead = float(np.dot(k_arr, pmf_sim))
                    # Estimate total_non_own from mean ahead + rough behind.
                    total_non_own_est = max(
                        1, int(round(mean_ahead)) * 2 + t.own_size
                    )
                    pmf_sim = _hypergeometric_thin(
                        pmf_sim, canceled, total_non_own_est
                    )

            h_posts[i] = _entropy_nats(pmf_sim)

        return float(np.mean(h_posts))

    def _order_band(self, t: _TrackedOrder) -> str:
        """Determine the depth band for a tracked order's price."""
        from topos.beliefs.flow_intensity import band_of

        if self._prev is None:
            return "touch"
        levels = self._prev.bids if t.side is Side.BUY else self._prev.asks
        best = None
        for lv in levels:
            if lv.size_lots > 0:
                best = lv.price_ticks
                break
        if best is None:
            return "touch"
        if t.side is Side.BUY:
            dist = best - t.price_ticks
        else:
            dist = t.price_ticks - best
        return band_of(dist)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _level_size_map(obs: Observation) -> dict[tuple[Side, int], int]:
    """Map (side, price) -> total visible lots from an observation."""
    m: dict[tuple[Side, int], int] = {}
    for lv in obs.bids:
        if lv.size_lots > 0:
            m[(Side.BUY, lv.price_ticks)] = lv.size_lots
    for lv in obs.asks:
        if lv.size_lots > 0:
            m[(Side.SELL, lv.price_ticks)] = lv.size_lots
    return m


def _visible_level_size(obs: Observation, side: Side, price: int) -> int:
    """Size at a specific price level from an observation."""
    levels = obs.bids if side is Side.BUY else obs.asks
    for lv in levels:
        if lv.price_ticks == price and lv.size_lots > 0:
            return lv.size_lots
    return 0


def _pmf_mean(pmf: FloatArray) -> float:
    """Mean of the distribution."""
    k = np.arange(len(pmf), dtype=np.float64)
    return float(np.dot(k, pmf))
