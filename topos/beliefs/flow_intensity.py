"""FlowIntensity: Poisson-Gamma order-flow intensities per side and band.

One independent conjugate cell per (event kind, book side, depth band):

    kind in {arrival, cancel, market}   (limit arrivals, cancels, marketable)
    side in {BUY, SELL}                 (bids book / asks book; for market
                                         orders, the AGGRESSOR side)
    band in {touch, near, deep}         (0, 1-3, >3 ticks from the pre-step
                                         best on that side)

Each cell holds rate lambda ~ Gamma(a, b); counts (in lots) over dt steps
are Poisson(lambda * dt); the predictive is negative binomial; EIG per cell
is I(lambda; Y) via the shared identity — H[Y] under the exact NB
predictive minus E_lambda H[Poisson(lambda * dt)] by quadrature over the
Gamma posterior (INV-3). The module EIG is the sum over cells (independent
posteriors), with exposure dt = the probe horizon.

Like all world predictors, EIG depends only on the probe horizon: public
flow is observed whether or not the agent places orders, so marginal EIG
over null is 0 for order-placing probes and the null action carries this
module's EIG (INV-4).

Event extraction is a best-effort inference from consecutive observations:

* arrivals  = visible size increases at a price (in lots),
* cancels   = visible size decreases not explained by trading at that price,
* market    = public trade prints, keyed by aggressor side and by how deep
  the print sits relative to the pre-step best on the passive side.

The agent's own footprint is subtracted so the cells model BACKGROUND flow:
own accepted placements (net of same-step taker fills) are removed from
arrivals, own cancellations from cancels, and own taker fills are treated
like trades when netting the passive side's decreases — necessary because
agent-caused prints never appear in Observation.trades (committed P1
behavior; see DESIGN.md, Open questions). Own maker fills need no special
handling: their prints are background-caused and public. Message-to-ack
pairing is positional (k-th placement ack answers the k-th PlaceLimit of
the step), matching the engine's sequential processing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from topos.contracts.beliefs import (
    EntropySnapshot,
    ForecastStats,
    ProbeSpec,
    SelfEvents,
)
from topos.contracts.intent import FLOW_INTENSITY, HypothesisId
from topos.contracts.workspace import Focus
from topos.contracts.market import (
    AckStatus,
    BookLevel,
    Liquidity,
    Observation,
    PlaceLimit,
    Side,
)

from topos.beliefs.core import EIGTerms, GammaPosterior, SurpriseTracker

KINDS: tuple[str, ...] = ("arrival", "cancel", "market")
BANDS: tuple[str, ...] = ("touch", "near", "deep")

CellKey = tuple[str, Side, str]

_STEP_DT = 1.0


def band_of(distance_ticks: int) -> str:
    """Depth band from signed distance to the pre-step best (<=0: at or
    inside the best; 1-3: near; deeper otherwise)."""
    if distance_ticks <= 0:
        return "touch"
    if distance_ticks <= 3:
        return "near"
    return "deep"


def _book_map(levels: tuple[BookLevel, ...]) -> dict[int, int]:
    """price -> size for non-padded levels."""
    return {lv.price_ticks: lv.size_lots for lv in levels if lv.size_lots > 0}


def _best_price(levels: tuple[BookLevel, ...]) -> int | None:
    for level in levels:
        if level.size_lots > 0:
            return level.price_ticks
    return None


def _depth_distance(side: Side, best_ref: int, price: int) -> int:
    """Ticks away from the reference best, positive going deeper."""
    return (best_ref - price) if side is Side.BUY else (price - best_ref)


@dataclass
class _OwnOrder:
    side: Side
    price_ticks: int
    remaining_lots: int


class FlowIntensity:
    """BeliefModule over background flow (hypothesis_id="flow_intensity")."""

    hypothesis_id: HypothesisId = FLOW_INTENSITY

    def __init__(
        self,
        *,
        prior_a: float = 1.0,
        prior_b: float = 1.0,
        surprise_ewma_decay: float = 0.05,
    ) -> None:
        self._cells: dict[CellKey, GammaPosterior] = {
            (kind, side, band): GammaPosterior(prior_a, prior_b)
            for kind in KINDS
            for side in (Side.BUY, Side.SELL)
            for band in BANDS
        }
        self._prev: Observation | None = None
        self._ledger: dict[int, _OwnOrder] = {}
        self._surprise = SurpriseTracker(ewma_decay=surprise_ewma_decay)
        self._last_counts: dict[CellKey, int] = {}
        self._step = 0
        # Broadcast conditioning (P9). Standalone modules run at full
        # per-band fidelity; a workspace sets the flag each cycle via the
        # hook. Unfocused evidence is buffered per cell and folded in
        # exactly on the next focused refresh (Gamma-Poisson batches
        # exactly: sum the counts, sum the exposure), while the coarse
        # aggregate posterior over the TOTAL rate stays current every
        # step. Because every cell shares one exposure path, the coarse
        # cell (prior = sum of the cell priors) is not an approximation:
        # it is exactly the posterior the fine cells induce on their sum.
        self._focused = True
        self._pending: dict[CellKey, int] = {}
        self._pending_exposure = 0.0
        self._coarse = GammaPosterior(prior_a * len(self._cells), prior_b)
        self._surprise_coarse = SurpriseTracker(ewma_decay=surprise_ewma_decay)
        self._last_z = 0.0

    # -- posterior access (public: tests, proposer, metrics read these) ----

    @property
    def cells(self) -> dict[CellKey, GammaPosterior]:
        """The 18 independent Gamma cells, keyed by (kind, side, band)."""
        return self._cells

    @property
    def last_counts(self) -> dict[CellKey, int]:
        """Counts extracted by the most recent update (for inspection)."""
        return dict(self._last_counts)

    # -- BeliefModule protocol ---------------------------------------------

    def update(self, obs: Observation, self_events: SelfEvents) -> None:
        """Extract one step's background-flow counts and update every cell.

        The first observation only establishes the reference book: counts
        are diffs between consecutive books, so there is nothing to count
        yet (and no surprise to score).
        """
        self._step = obs.step
        prev = self._prev
        self._prev = obs
        own_resting, own_cancels, own_taker = self._maintain_ledger(
            obs, self_events
        )
        if prev is None:
            return
        counts = self._extract_counts(prev, obs, own_resting, own_cancels, own_taker)
        self._last_counts = counts
        total = sum(counts.get(key, 0) for key in self._cells)
        if self._focused:
            # Fine path: refresh every per-band posterior, score the
            # step's surprise on the full 18-cell joint predictive.
            self._flush_pending()
            nll = 0.0
            for key, cell in self._cells.items():
                count = counts.get(key, 0)
                y = np.array([float(count)])
                nll -= float(cell.predictive_log_pmf(y, _STEP_DT)[0])
                cell.observe(count, _STEP_DT)
            self._last_z = self._surprise.score(nll)
        else:
            # Coarse path (unfocused): buffer the per-cell evidence for
            # the next focused refresh; surprise is scored against the
            # aggregate predictive of the total count, on its own tracker
            # (fine and coarse NLLs live on different scales, and each
            # tracker z-scores against its own history).
            coarse_nll = -float(
                self._coarse.predictive_log_pmf(np.array([float(total)]), _STEP_DT)[0]
            )
            self._last_z = self._surprise_coarse.score(coarse_nll)
            for key in self._cells:
                count = counts.get(key, 0)
                if count:
                    self._pending[key] = self._pending.get(key, 0) + count
            self._pending_exposure += _STEP_DT
        # The aggregate view is maintained every step regardless of focus.
        self._coarse.observe(total, _STEP_DT)

    def forget(self, rho: float) -> None:
        """Discount every cell's sufficient statistics toward its prior.

        Buffered unfocused evidence is folded in first: evidence precedes
        the discount, otherwise a later flush would resurrect statistics
        the regime shift asked to be forgotten.
        """
        self._flush_pending()
        for cell in self._cells.values():
            cell.forget(rho)
        self._coarse.forget(rho)

    def posterior_entropy_nats(self) -> float:
        """Joint PARAMETER-posterior entropy: sum over independent cells.

        While unfocused this is quoted from the last per-band refresh
        (buffered evidence not yet folded in) — the coarse aggregate does
        not decompose into the 18 questions the fine posterior answers.
        """
        return sum(cell.entropy_nats() for cell in self._cells.values())

    def predict(self) -> ForecastStats:
        """Total background events (lots) expected next step.

        Fine path (no buffered evidence): negative-binomial predictive
        mean/variance summed over the per-band cells. Coarse path
        (unfocused, evidence buffered): the same two moments from the
        aggregate posterior over the total rate — current where the fine
        cells are stale, coarse where they are fine.
        """
        if self._pending_exposure > 0.0:
            mean = self._coarse.mean() * _STEP_DT
            variance = mean * (1.0 + _STEP_DT / self._coarse.b)
            return ForecastStats(mean=mean, variance=variance)
        mean = 0.0
        variance = 0.0
        for cell in self._cells.values():
            cell_mean = cell.mean() * _STEP_DT
            mean += cell_mean
            variance += cell_mean * (1.0 + _STEP_DT / cell.b)
        return ForecastStats(mean=mean, variance=variance)

    def surprise_z(self) -> float:
        """Salience-only surprise; never feeds EIG or action scoring.

        The z of whichever channel scored last: the fine 18-cell joint
        while focused, the coarse aggregate otherwise.
        """
        return self._last_z

    def condition_on_focus(self, focus: Focus | None) -> None:
        """Broadcast conditioning hook (P9): finer work when focused.

        Winning focus triggers the per-band refresh immediately, so the
        proposer's refined menu (which runs right after the broadcast)
        interrogates caught-up posteriors. See ``topos.workspace.
        broadcast`` for the pattern.
        """
        self._focused = focus is not None and focus.hypothesis_id == self.hypothesis_id
        if self._focused:
            self._flush_pending()

    def _flush_pending(self) -> None:
        """Fold buffered unfocused evidence into the per-band cells.

        Exact, not approximate: Gamma-Poisson conjugacy makes one batched
        ``observe(sum of counts, sum of exposure)`` identical to the
        per-step updates it stands in for.
        """
        if self._pending_exposure <= 0.0:
            return
        for key, cell in self._cells.items():
            cell.observe(self._pending.get(key, 0), self._pending_exposure)
        self._pending.clear()
        self._pending_exposure = 0.0

    def eig_nats(self, probe: ProbeSpec) -> float:
        """I(lambdas; counts over the probe horizon), summed over cells."""
        return self.eig_breakdown(probe).eig_nats

    def eig_breakdown(self, probe: ProbeSpec) -> EIGTerms:
        """The epistemic/aleatoric decomposition behind ``eig_nats``.

        Computed on the per-band cells; while unfocused (P9 broadcast
        conditioning) those are quoted from the last per-band refresh —
        attention is what buys the fine-grained update.
        """
        if probe.horizon_steps < 1:
            raise ValueError(
                f"probe horizon_steps must be >= 1, got {probe.horizon_steps}"
            )
        exposure = float(probe.horizon_steps) * _STEP_DT
        eig = 0.0
        predictive = 0.0
        aleatoric = 0.0
        for cell in self._cells.values():
            terms = cell.eig_terms(exposure)
            eig += terms.eig_nats
            predictive += terms.predictive_entropy_nats
            aleatoric += terms.expected_conditional_entropy_nats
        return EIGTerms(eig, predictive, aleatoric)

    def snapshot_entropy(self) -> EntropySnapshot:
        """Parameter-posterior entropy at this instant (INV-10)."""
        return EntropySnapshot(
            hypothesis_id=self.hypothesis_id,
            step=self._step,
            entropy_nats=self.posterior_entropy_nats(),
        )

    # -- own-footprint bookkeeping ------------------------------------------

    def _maintain_ledger(
        self, obs: Observation, self_events: SelfEvents
    ) -> tuple[dict[tuple[Side, int], int], dict[tuple[Side, int], int], dict[tuple[Side, int], int]]:
        """Fold this step's own messages/acks/fills into the order ledger.

        Yields three (side, price) -> lots maps for this step: own size
        newly RESTING in the book, own size cancelled/expired out of it,
        and own TAKER volume (which removed size from the opposite side
        without a public print).
        """
        placements = [
            msg for msg in self_events.messages_sent if isinstance(msg, PlaceLimit)
        ]
        placement_acks = [
            ack
            for ack in self_events.acks
            if ack.status in (AckStatus.ACCEPTED, AckStatus.REJECTED)
        ]
        accepted: dict[int, PlaceLimit] = {}
        for k, ack in enumerate(placement_acks):
            if k >= len(placements):
                break
            if ack.status is AckStatus.ACCEPTED:
                accepted[ack.order_id] = placements[k]
                self._ledger[ack.order_id] = _OwnOrder(
                    side=placements[k].side,
                    price_ticks=placements[k].price_ticks,
                    remaining_lots=placements[k].size_lots,
                )
        own_taker: dict[tuple[Side, int], int] = {}
        for fill in obs.own_fills:
            order = self._ledger.get(fill.order_id)
            if order is None:
                continue
            order.remaining_lots = max(0, order.remaining_lots - fill.size_lots)
            if fill.liquidity is Liquidity.TAKER:
                passive = Side.SELL if order.side is Side.BUY else Side.BUY
                key = (passive, fill.price_ticks)
                own_taker[key] = own_taker.get(key, 0) + fill.size_lots
            if order.remaining_lots == 0 and fill.liquidity is Liquidity.TAKER:
                # Fully-consumed marketable order: nothing rests.
                del self._ledger[fill.order_id]
        own_resting: dict[tuple[Side, int], int] = {}
        for order_id, place in accepted.items():
            order = self._ledger.get(order_id)
            if order is None or order.remaining_lots == 0:
                continue
            key = (order.side, order.price_ticks)
            own_resting[key] = own_resting.get(key, 0) + order.remaining_lots
        own_cancels: dict[tuple[Side, int], int] = {}
        for ack in obs.own_acks:
            if ack.status not in (AckStatus.CANCELED, AckStatus.EXPIRED):
                continue
            order = self._ledger.pop(ack.order_id, None)
            if order is None or order.remaining_lots == 0:
                continue
            key = (order.side, order.price_ticks)
            own_cancels[key] = own_cancels.get(key, 0) + order.remaining_lots
        return own_resting, own_cancels, own_taker

    # -- event extraction -----------------------------------------------------

    def _extract_counts(
        self,
        prev: Observation,
        cur: Observation,
        own_resting: dict[tuple[Side, int], int],
        own_cancels: dict[tuple[Side, int], int],
        own_taker: dict[tuple[Side, int], int],
    ) -> dict[CellKey, int]:
        counts = {key: 0 for key in self._cells}
        # Public traded volume per (passive book side, price), plus own
        # taker volume whose prints are absent from Observation.trades.
        traded: dict[tuple[Side, int], int] = dict(own_taker)
        for trade in cur.trades:
            passive = Side.SELL if trade.aggressor is Side.BUY else Side.BUY
            key = (passive, trade.price_ticks)
            traded[key] = traded.get(key, 0) + trade.size_lots
        for side in (Side.BUY, Side.SELL):
            prev_levels = prev.bids if side is Side.BUY else prev.asks
            cur_levels = cur.bids if side is Side.BUY else cur.asks
            best_ref = _best_price(prev_levels)
            if best_ref is None:
                best_ref = _best_price(cur_levels)
            if best_ref is None:
                continue
            prev_map = _book_map(prev_levels)
            cur_map = _book_map(cur_levels)
            for price in set(prev_map) | set(cur_map):
                delta = cur_map.get(price, 0) - prev_map.get(price, 0)
                band = band_of(_depth_distance(side, best_ref, price))
                if delta > 0:
                    arrived = delta - own_resting.get((side, price), 0)
                    counts[("arrival", side, band)] += max(0, arrived)
                elif delta < 0:
                    decreased = -delta
                    decreased -= traded.get((side, price), 0)
                    decreased -= own_cancels.get((side, price), 0)
                    counts[("cancel", side, band)] += max(0, decreased)
            # Marketable flow: public prints that consumed THIS side,
            # keyed by the aggressor (the opposite side).
            aggressor = Side.SELL if side is Side.BUY else Side.BUY
            for trade in cur.trades:
                if trade.aggressor is not aggressor:
                    continue
                band = band_of(_depth_distance(side, best_ref, trade.price_ticks))
                counts[("market", aggressor, band)] += trade.size_lots
        return counts
