"""Shared context extraction and own-order utilities for the self-model.

Everything here exists so that the fill model, the impact model, and the
trajectory compiler condition on the SAME discretization of the agent's own
action and book context — the design's Layer-1 anti-churn requirement. If
each module invented its own banding, a probe scored in one discretization
would be updated in another and fills would stay surprising forever.

Banding conventions (structural, never tuned):

* ``OFFSET_BANDS``: ``cross`` (marketable — the order would trade on
  arrival), ``touch`` (at or inside the current touch), ``near`` (1-3 ticks
  behind the best), ``deep`` (4+ ticks behind). The 0 / 1-3 / 4+ edges are
  inherited verbatim from ``topos.beliefs.flow_intensity.BANDS`` so that the
  self-model and the background-flow model partition the book identically;
  ``cross`` is the extra band the flow model does not need (background
  marketable flow is already a separate event kind there).
* ``IMBALANCE_BANDS``: the uniform tripartition of the imbalance range
  [-1, +1] at +/- 1/3 — three equal-width cells, no calibration.

Own-order pairing follows the committed P4 convention: the k-th placement
ack of a step answers the k-th ``PlaceLimit`` of that step (the engine
processes messages sequentially).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from topos.contracts.beliefs import SelfEvents
from topos.contracts.intent import Intent
from topos.contracts.market import (
    Ack,
    AckStatus,
    BookLevel,
    Liquidity,
    Observation,
    PlaceLimit,
    Side,
)

OFFSET_BANDS: tuple[str, ...] = ("cross", "touch", "near", "deep")
IMBALANCE_BANDS: tuple[str, ...] = ("sell_heavy", "balanced", "buy_heavy")


# ---------------------------------------------------------------------------
# Book context
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BookContext:
    """The slice of one observation the self-model conditions on."""

    mid: float | None
    best_bid: int | None
    best_ask: int | None
    imbalance: float
    """(visible bid lots - visible ask lots) / (bid lots + ask lots)."""


def _best_price(levels: tuple[BookLevel, ...]) -> int | None:
    for level in levels:
        if level.size_lots > 0:
            return level.price_ticks
    return None


def context_from_observation(obs: Observation) -> BookContext:
    """Extract (mid, best quotes, depth imbalance) from one observation.

    Imbalance uses ALL visible lots per side (padded levels are absent by
    the size-0 convention): the full visible window, not just the touch, so
    a thick-but-deep side counts as pressure.
    """
    best_bid = _best_price(obs.bids)
    best_ask = _best_price(obs.asks)
    if best_bid is not None and best_ask is not None:
        mid: float | None = 0.5 * (best_bid + best_ask)
    elif best_bid is not None:
        mid = float(best_bid)
    elif best_ask is not None:
        mid = float(best_ask)
    else:
        mid = None
    bid_lots = sum(level.size_lots for level in obs.bids)
    ask_lots = sum(level.size_lots for level in obs.asks)
    total = bid_lots + ask_lots
    imbalance = (bid_lots - ask_lots) / total if total > 0 else 0.0
    return BookContext(
        mid=mid, best_bid=best_bid, best_ask=best_ask, imbalance=imbalance
    )


# ---------------------------------------------------------------------------
# Banding
# ---------------------------------------------------------------------------


def offset_band_of(
    side: Side, price_ticks: int, best_bid: int | None, best_ask: int | None
) -> str:
    """Which offset band an order at ``price_ticks`` exercises.

    ``cross`` when the order is marketable against the visible opposite
    best; otherwise banded by ticks behind the own-side best with the flow
    model's 0 / 1-3 / 4+ edges (inside-spread improvements count as
    ``touch`` — they become the new best). With no visible reference on
    either relevant side the order is at the front by construction: touch.
    """
    if side is Side.BUY:
        if best_ask is not None and price_ticks >= best_ask:
            return "cross"
        if best_bid is None:
            return "touch"
        distance = best_bid - price_ticks
    else:
        if best_bid is not None and price_ticks <= best_bid:
            return "cross"
        if best_ask is None:
            return "touch"
        distance = price_ticks - best_ask
    if distance <= 0:
        return "touch"
    if distance <= 3:
        return "near"
    return "deep"


def imbalance_band_of(imbalance: float) -> str:
    """Uniform tripartition of [-1, +1] at +/- 1/3."""
    if imbalance < -1.0 / 3.0:
        return "sell_heavy"
    if imbalance > 1.0 / 3.0:
        return "buy_heavy"
    return "balanced"


# ---------------------------------------------------------------------------
# Intent -> implied order
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ImpliedOrder:
    """The single limit order an intent compiles to, to first order.

    This is a FORECASTING approximation shared by the fill/impact/
    trajectory modules for scoring probes — the motor (P10) owns the real
    compilation. Price follows the Intent contract exactly: offset_ticks
    is distance from mid toward passivity, negative means crossing.
    """

    side: Side
    price_ticks: int
    size_lots: int


def implied_order(
    intent: Intent, ctx: BookContext, size_budget_lots: int
) -> ImpliedOrder | None:
    """Map an intent onto the order it implies in the given book context.

    Returns None for the null action, a directionless intent (side == 0 —
    no order direction exists to forecast), a zero-size intent, or a book
    with no mid to anchor the price.
    """
    if intent.is_null or intent.side == 0.0 or ctx.mid is None:
        return None
    size_lots = int(round(intent.size_frac * size_budget_lots))
    if size_lots <= 0:
        return None
    side = Side.BUY if intent.side > 0.0 else Side.SELL
    if side is Side.BUY:
        price = int(round(ctx.mid - intent.offset_ticks))
    else:
        price = int(round(ctx.mid + intent.offset_ticks))
    return ImpliedOrder(side=side, price_ticks=price, size_lots=size_lots)


# ---------------------------------------------------------------------------
# Own-order pairing and ledger
# ---------------------------------------------------------------------------


def paired_placements(
    self_events: SelfEvents,
) -> tuple[tuple[int, PlaceLimit, Ack], ...]:
    """(order_id, message, ack) for every ACCEPTED placement of one step.

    Positional pairing per the committed P4 convention: the k-th placement
    ack (ACCEPTED or REJECTED) answers the k-th PlaceLimit of the step.
    """
    placements = [
        msg for msg in self_events.messages_sent if isinstance(msg, PlaceLimit)
    ]
    placement_acks = [
        ack
        for ack in self_events.acks
        if ack.status in (AckStatus.ACCEPTED, AckStatus.REJECTED)
    ]
    out: list[tuple[int, PlaceLimit, Ack]] = []
    for k, ack in enumerate(placement_acks):
        if k >= len(placements):
            break
        if ack.status is AckStatus.ACCEPTED:
            out.append((ack.order_id, placements[k], ack))
    return tuple(out)


@dataclass
class LedgerOrder:
    """One own working order, reconstructed from acks/fills only (INV-11)."""

    side: Side
    price_ticks: int
    remaining_lots: int


@dataclass
class OwnOrderLedger:
    """Own working orders folded step by step from acks and fills.

    The environment never reports account or order state (INV-11); this
    ledger is rebuilt purely from the agent's own message/ack/fill stream.
    """

    orders: dict[int, LedgerOrder] = field(default_factory=dict)

    def fold(self, obs: Observation, self_events: SelfEvents) -> int:
        """Fold one step's events; the signed TAKER volume is returned.

        The return value is the agent's signed executed aggression for the
        step: +lots bought marketably, -lots sold marketably.
        """
        for order_id, place, _ack in paired_placements(self_events):
            self.orders[order_id] = LedgerOrder(
                side=place.side,
                price_ticks=place.price_ticks,
                remaining_lots=place.size_lots,
            )
        taker_signed = 0
        for fill in obs.own_fills:
            order = self.orders.get(fill.order_id)
            if order is None:
                continue
            order.remaining_lots = max(0, order.remaining_lots - fill.size_lots)
            if fill.liquidity is Liquidity.TAKER:
                taker_signed += order.side.value * fill.size_lots
            if order.remaining_lots == 0:
                del self.orders[fill.order_id]
        for ack in obs.own_acks:
            if ack.status in (AckStatus.CANCELED, AckStatus.EXPIRED):
                self.orders.pop(ack.order_id, None)
        return taker_signed

    def resting_at_touch_signed(
        self, best_bid: int | None, best_ask: int | None
    ) -> int:
        """Signed own lots resting at the touch: +bid-side, -ask-side.

        Sign convention matches price direction: own resting bids at the
        touch add buying pressure (push the mid up), own resting asks the
        opposite.
        """
        total = 0
        for order in self.orders.values():
            if order.side is Side.BUY and order.price_ticks == best_bid:
                total += order.remaining_lots
            elif order.side is Side.SELL and order.price_ticks == best_ask:
                total -= order.remaining_lots
        return total
