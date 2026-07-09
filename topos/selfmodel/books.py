"""Bookkeeping: the agent's account state, folded from own acks/fills ONLY.

The environment never reports account state (INV-11): inventory, cash,
realized/unrealized PnL, working orders and gross exposure are all
reconstructed here from the agent's own message/ack/fill stream, and their
correctness is validated EXTERNALLY against harness ground truth
(``topos.env.harness.assert_agent_bookkeeping``, the P3 hook).

Two views, per contracts (INV-5):

* ``full_view()``  -> ``SelfStateFull``       — drives/ (homeostat) and
  metrics/ only; carries the account fields.
* ``cognitive_view()`` -> ``SelfStateCognitive`` — the ONLY self-state
  arbitration/proposal code ever receive; NO PnL fields (tripwired).

Accounting conventions:

* Integer ticks, integer lots. ``cash_ticks`` is the signed realized
  cashflow: sum over fills of ``-side * price_ticks * size_lots`` — the
  engine's own ground-truth convention.
* Realized PnL uses average-cost lot matching; unrealized PnL marks the
  open position at the mid of the best visible quotes (falling back to the
  last known mark while the book is one-sided/empty). The decomposition
  satisfies ``realized + unrealized == cash + inventory * mark`` — the
  method-independent identity the tests pin.
* Fill timing: a fill stamped step k is folded the moment it is observed,
  so the live views are maximally fresh; the ground-truth claim stream
  (``claims()``) is instead assembled in STAMP order, because "the account
  at the end of engine step k" means exactly "all fills stamped <= k"
  (the P3 hook's contract) and one observation can deliver fills stamped
  both k and k+1.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Mapping

from topos.contracts.beliefs import SelfEvents
from topos.contracts.market import (
    AckStatus,
    Fill,
    Observation,
    PlaceLimit,
    Side,
)
from topos.contracts.workspace import (
    SelfStateCognitive,
    SelfStateFull,
    WorkingOrderView,
)

from topos.selfmodel.common import context_from_observation, paired_placements

RankLookup = Callable[[int], "tuple[float, float] | None"]
"""order_id -> (queue_rank_mean, queue_rank_var), e.g. the P5 filter's
``rank_mean_var``; None when the order is not tracked."""


@dataclass(frozen=True)
class BookkeepingRecord:
    """Self-tracked account state as of the END of one engine step.

    Duck-type-compatible with the harness hook's ``BookkeepingClaim``
    (``assert_agent_bookkeeping`` reads ``step``, ``inventory_lots``,
    ``cash_ticks``).
    """

    step: int
    inventory_lots: int
    cash_ticks: int


@dataclass
class _WorkingOrder:
    side: Side
    price_ticks: int
    remaining_lots: int
    placed_step: int
    prior_rank_mean: float
    prior_rank_var: float


class Books:
    """The agent's self-tracked account, from own acks/fills only (INV-11)."""

    def __init__(self, rank_lookup: RankLookup | None = None) -> None:
        self._rank_lookup = rank_lookup
        self._step = 0
        # Live account state (folds every observed fill immediately).
        self._inventory = 0
        self._cash = 0
        self._avg_entry_price = 0.0
        self._realized = 0.0
        self._mark: float | None = None
        # Working orders and the persistent side ledger (sides survive
        # order removal: the stamp-ordered claim replay still needs them).
        self._working: dict[int, _WorkingOrder] = {}
        self._order_sides: dict[int, Side] = {}
        # Every fill ever observed, in observation order, for claims().
        self._observed_fills: list[Fill] = []

    # -- folding -------------------------------------------------------------

    def update(self, obs: Observation, self_events: SelfEvents) -> None:
        """Fold one step's own events and re-mark the open position.

        Processing order matches the engine's event order within an
        observation window: placements register first (a taker fill can
        arrive in the same window as its order's ACCEPTED ack), then
        fills, then cancel/expiry removals (an order cannot fill after it
        left the book).
        """
        self._step = obs.step
        for order_id, place, ack in paired_placements(self_events):
            self._order_sides[order_id] = place.side
            n_ahead = self._visible_non_own_lots(obs, place)
            self._working[order_id] = _WorkingOrder(
                side=place.side,
                price_ticks=place.price_ticks,
                remaining_lots=place.size_lots,
                placed_step=ack.step,
                # Ignorance prior over queue rank: uniform on {0..N} with
                # N = visible non-own lots at the level — by price-time
                # priority the true rank cannot exceed the lots present.
                # Mean/var are the uniform's moments, not calibrations.
                prior_rank_mean=n_ahead / 2.0,
                prior_rank_var=n_ahead * (n_ahead + 2.0) / 12.0,
            )
        for fill in obs.own_fills:
            side = self._order_sides.get(fill.order_id)
            if side is None:
                continue
            self._observed_fills.append(fill)
            self._apply_fill_to_account(side, fill.price_ticks, fill.size_lots)
            working = self._working.get(fill.order_id)
            if working is not None:
                working.remaining_lots = max(
                    0, working.remaining_lots - fill.size_lots
                )
                if working.remaining_lots == 0:
                    del self._working[fill.order_id]
        for ack in obs.own_acks:
            if ack.status in (AckStatus.CANCELED, AckStatus.EXPIRED):
                self._working.pop(ack.order_id, None)
        ctx = context_from_observation(obs)
        if ctx.best_bid is not None and ctx.best_ask is not None:
            self._mark = 0.5 * (ctx.best_bid + ctx.best_ask)

    def _apply_fill_to_account(self, side: Side, price: int, size: int) -> None:
        """Average-cost fold of one fill into inventory/cash/realized."""
        signed = side.value * size
        self._cash -= side.value * price * size
        if self._inventory == 0 or (self._inventory > 0) == (signed > 0):
            # Opening or extending: average the entry price.
            total = abs(self._inventory) + size
            self._avg_entry_price = (
                self._avg_entry_price * abs(self._inventory) + price * size
            ) / total
            self._inventory += signed
            return
        # Reducing (possibly through zero): realize on the closed lots.
        closed = min(size, abs(self._inventory))
        direction = 1.0 if self._inventory > 0 else -1.0
        self._realized += closed * (price - self._avg_entry_price) * direction
        self._inventory += signed
        if self._inventory == 0:
            self._avg_entry_price = 0.0
        elif (self._inventory > 0) == (signed > 0):
            # Flipped through zero: the residual opened at this fill price.
            self._avg_entry_price = float(price)

    @staticmethod
    def _visible_non_own_lots(obs: Observation, place: PlaceLimit) -> int:
        levels = obs.bids if place.side is Side.BUY else obs.asks
        for level in levels:
            if level.price_ticks == place.price_ticks and level.size_lots > 0:
                return max(0, level.size_lots - place.size_lots)
        return 0

    # -- live account quantities ----------------------------------------------

    @property
    def step(self) -> int:
        return self._step

    @property
    def inventory_lots(self) -> int:
        return self._inventory

    @property
    def cash_ticks(self) -> int:
        return self._cash

    @property
    def mark(self) -> float | None:
        """Mid of the best visible quotes, or the last known one."""
        return self._mark

    @property
    def realized_pnl(self) -> float:
        return self._realized

    @property
    def unrealized_pnl(self) -> float:
        """Open position marked to mid; 0 while flat or never marked."""
        if self._inventory == 0 or self._mark is None:
            return 0.0
        return self._inventory * (self._mark - self._avg_entry_price)

    @property
    def gross_exposure(self) -> float:
        """|inventory| * mark, in tick-lots; 0 while unmarked."""
        if self._mark is None:
            return 0.0
        return abs(self._inventory) * self._mark

    def working_order_views(self) -> tuple[WorkingOrderView, ...]:
        views: list[WorkingOrderView] = []
        for order_id, order in sorted(self._working.items()):
            rank: tuple[float, float] | None = None
            if self._rank_lookup is not None:
                rank = self._rank_lookup(order_id)
            if rank is None:
                rank = (order.prior_rank_mean, order.prior_rank_var)
            views.append(
                WorkingOrderView(
                    order_id=order_id,
                    side=order.side,
                    price_ticks=order.price_ticks,
                    size_lots_remaining=order.remaining_lots,
                    age_steps=max(0, self._step - order.placed_step),
                    queue_rank_mean=rank[0],
                    queue_rank_var=rank[1],
                )
            )
        return tuple(views)

    # -- the two contracted views (INV-5) --------------------------------------

    def full_view(
        self, drive_distances: Mapping[str, float] | None = None
    ) -> SelfStateFull:
        """The account-bearing view — drives/ (homeostat) and metrics/ only.

        ``drive_distances`` is computed by the homeostat (P7) and threaded
        back in by the agent loop; bookkeeping itself has no bands.
        """
        return SelfStateFull(
            inventory_lots=self._inventory,
            working_orders=self.working_order_views(),
            drive_distances=dict(drive_distances or {}),
            realized_pnl=self.realized_pnl,
            unrealized_pnl=self.unrealized_pnl,
            gross_exposure=self.gross_exposure,
        )

    def cognitive_view(
        self, drive_distances: Mapping[str, float] | None = None
    ) -> SelfStateCognitive:
        """The PnL-free view for the workspace/proposer (INV-5), obtained
        exclusively through the contract's own projection."""
        return self.full_view(drive_distances).cognitive_view()

    # -- ground-truth claims (validated by the P3 hook) ------------------------

    def claims(self, through_step: int) -> tuple[BookkeepingRecord, ...]:
        """End-of-step account claims for steps 0..through_step, inclusive.

        Assembled by replaying observed fills in STAMP order (ties keep
        observation order, which is chronological within a stamp): the
        claim at step k covers exactly the fills stamped <= k, matching
        the engine's end-of-step-k account. Claims are only complete
        through the last observed stamp: an action taken at the final
        step of an episode produces fills the agent never gets to see.
        """
        if through_step < 0:
            raise ValueError(f"through_step must be >= 0, got {through_step}")
        by_stamp: dict[int, list[Fill]] = {}
        for fill in self._observed_fills:
            by_stamp.setdefault(fill.step, []).append(fill)
        records: list[BookkeepingRecord] = []
        inventory = 0
        cash = 0
        for step in range(through_step + 1):
            for fill in by_stamp.get(step, ()):
                side = self._order_sides[fill.order_id]
                inventory += side.value * fill.size_lots
                cash -= side.value * fill.price_ticks * fill.size_lots
            records.append(
                BookkeepingRecord(
                    step=step, inventory_lots=inventory, cash_ticks=cash
                )
            )
        return tuple(records)

    def accounting_identity_gap(self) -> float:
        """|realized + unrealized - (cash + inventory * mark)| — 0 up to
        float rounding, by construction; exposed so tests can pin it."""
        if self._mark is None:
            return 0.0
        lhs = self._realized + self.unrealized_pnl
        rhs = self._cash + self._inventory * self._mark
        return math.fabs(lhs - rhs)
