"""Single-instrument limit order book with strict price-time priority.

Integer ticks and lots throughout.  Orders rest in FIFO queues per price
level; levels are kept sorted (bids descending, asks ascending by price).
The book is pure bookkeeping — no randomness (INV-9), no feedback signal (INV-1).

Self-trade prevention is NOT the engine's job (the agent's motor layer
handles it) — see DESIGN.md spec.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Iterator

from topos.contracts.market import Liquidity, Side


@dataclass
class RestingOrder:
    """A single resting limit order in the book.

    Mutable: `remaining_lots` is decremented on partial fills.
    `sequence` provides time-priority within a price level.
    """

    order_id: int
    actor_id: str
    side: Side
    price_ticks: int
    original_lots: int
    remaining_lots: int
    tif_steps: int
    """0 => GTC; >0 => expires after this many engine steps."""
    placed_step: int
    """Engine step at which the order was accepted."""
    sequence: int
    """Globally monotonic insertion counter for time priority."""


@dataclass
class _PriceLevel:
    """FIFO queue of resting orders at one price."""

    price: int
    orders: OrderedDict[int, RestingOrder] = field(default_factory=OrderedDict)

    @property
    def total_lots(self) -> int:
        return sum(o.remaining_lots for o in self.orders.values())


class OrderBook:
    """Single-instrument limit order book: strict price-time priority.

    Bids kept in descending price order; asks in ascending.  Within a level
    orders are FIFO (by globally monotonic sequence number).  Matching uses
    continuous double-auction semantics: a marketable limit crosses the
    opposing side immediately; partial fills are standard.
    """

    def __init__(self) -> None:
        # price -> _PriceLevel, maintained sorted
        self._bids: OrderedDict[int, _PriceLevel] = OrderedDict()
        self._asks: OrderedDict[int, _PriceLevel] = OrderedDict()
        self._sequence: int = 0

    # ------------------------------------------------------------------
    # Read accessors
    # ------------------------------------------------------------------

    @property
    def best_bid(self) -> int | None:
        """Highest bid price, or None if bid side is empty."""
        for price, lvl in self._bids.items():
            if lvl.total_lots > 0:
                return price
        return None

    @property
    def best_ask(self) -> int | None:
        """Lowest ask price, or None if ask side is empty."""
        for price, lvl in self._asks.items():
            if lvl.total_lots > 0:
                return price
        return None

    def bid_levels(self) -> Iterator[tuple[int, int]]:
        """Yield (price, total_lots) for non-empty bid levels, best first."""
        for lvl in self._bids.values():
            total = lvl.total_lots
            if total > 0:
                yield lvl.price, total

    def ask_levels(self) -> Iterator[tuple[int, int]]:
        """Yield (price, total_lots) for non-empty ask levels, best first."""
        for lvl in self._asks.values():
            total = lvl.total_lots
            if total > 0:
                yield lvl.price, total

    def orders_at(self, side: Side, price: int) -> list[RestingOrder]:
        """Return resting orders at the given side/price in FIFO order."""
        book = self._bids if side == Side.BUY else self._asks
        lvl = book.get(price)
        if lvl is None:
            return []
        return [o for o in lvl.orders.values() if o.remaining_lots > 0]

    def get_order(self, actor_id: str, order_id: int) -> RestingOrder | None:
        """Look up a specific resting order by its actor-scoped id."""
        for book in (self._bids, self._asks):
            for lvl in book.values():
                for o in lvl.orders.values():
                    if o.actor_id == actor_id and o.order_id == order_id and o.remaining_lots > 0:
                        return o
        return None

    def all_resting_orders(self) -> list[RestingOrder]:
        """Return every resting order with remaining_lots > 0."""
        result: list[RestingOrder] = []
        for book in (self._bids, self._asks):
            for lvl in book.values():
                for o in lvl.orders.values():
                    if o.remaining_lots > 0:
                        result.append(o)
        return result

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def _next_seq(self) -> int:
        seq = self._sequence
        self._sequence += 1
        return seq

    def insert(self, order: RestingOrder) -> None:
        """Insert a resting order into the book."""
        book = self._bids if order.side == Side.BUY else self._asks
        if order.price_ticks not in book:
            book[order.price_ticks] = _PriceLevel(price=order.price_ticks)
            # Re-sort: bids descending, asks ascending
            if order.side == Side.BUY:
                sorted_items = sorted(book.items(), key=lambda kv: -kv[0])
            else:
                sorted_items = sorted(book.items(), key=lambda kv: kv[0])
            book.clear()
            for k, v in sorted_items:
                book[k] = v
        book[order.price_ticks].orders[order.sequence] = order

    def remove(self, order: RestingOrder) -> None:
        """Remove an order from the book entirely (cancel / full fill)."""
        book = self._bids if order.side == Side.BUY else self._asks
        lvl = book.get(order.price_ticks)
        if lvl is not None and order.sequence in lvl.orders:
            del lvl.orders[order.sequence]

    def make_resting_order(
        self,
        order_id: int,
        actor_id: str,
        side: Side,
        price_ticks: int,
        size_lots: int,
        tif_steps: int,
        placed_step: int,
    ) -> RestingOrder:
        """Create a RestingOrder with the next sequence number."""
        return RestingOrder(
            order_id=order_id,
            actor_id=actor_id,
            side=side,
            price_ticks=price_ticks,
            original_lots=size_lots,
            remaining_lots=size_lots,
            tif_steps=tif_steps,
            placed_step=placed_step,
            sequence=self._next_seq(),
        )

    def queue_position(self, order: RestingOrder) -> int:
        """Lots ahead of this order at its price level (ground truth)."""
        book = self._bids if order.side == Side.BUY else self._asks
        lvl = book.get(order.price_ticks)
        if lvl is None:
            return 0
        lots_ahead = 0
        for o in lvl.orders.values():
            if o.sequence == order.sequence:
                break
            if o.remaining_lots > 0:
                lots_ahead += o.remaining_lots
        return lots_ahead
