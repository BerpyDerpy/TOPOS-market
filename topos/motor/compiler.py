"""Deterministic intent → exchange-message compiler (INV-8).

This is a **coordinate change**, not a decision-maker: it translates a
high-level ``Intent`` into legal ``ExchangeMessage`` tuples as a pure
function of its declared inputs.  No randomness, no hidden state, no
policy beyond what ``MotorConfig`` exposes.

Intent and compiled messages are logged side by side in the
``WorkspaceRecord``, producing a lossless readout (INV-8).

Book conventions
----------------
Book snapshots pad thin books to ``N_LEVELS`` with ``size_lots == 0``
levels; all routines here treat size-0 levels as absent (mid, touch,
crossing checks).
"""

from __future__ import annotations

import math
from collections.abc import Mapping

from topos.contracts.intent import Intent
from topos.contracts.market import (
    BookLevel,
    Cancel,
    ExchangeMessage,
    PlaceLimit,
    Side,
)
from topos.contracts.workspace import WorkingOrderView
from topos.motor.config import MotorConfig


# ---------------------------------------------------------------------------
# Book helpers — treat size_lots == 0 as absent everywhere
# ---------------------------------------------------------------------------

def _live_levels(levels: tuple[BookLevel, ...]) -> list[BookLevel]:
    """Filter to levels with positive size (padded zeros → absent)."""
    return [lv for lv in levels if lv.size_lots > 0]


def _mid_ticks(
    bids: tuple[BookLevel, ...],
    asks: tuple[BookLevel, ...],
) -> float | None:
    """Midpoint of best live bid and best live ask, or None if one side empty."""
    live_bids = _live_levels(bids)
    live_asks = _live_levels(asks)
    if not live_bids or not live_asks:
        return None
    return (live_bids[0].price_ticks + live_asks[0].price_ticks) / 2.0


def _best_bid_tick(bids: tuple[BookLevel, ...]) -> int | None:
    live = _live_levels(bids)
    return live[0].price_ticks if live else None


def _best_ask_tick(asks: tuple[BookLevel, ...]) -> int | None:
    live = _live_levels(asks)
    return live[0].price_ticks if live else None


# ---------------------------------------------------------------------------
# Staleness check
# ---------------------------------------------------------------------------

def _stale_orders(
    working: tuple[WorkingOrderView, ...],
    compiled_price: int | None,
    compiled_side: Side | None,
    patience_steps: int,
) -> list[WorkingOrderView]:
    """Working orders older than patience_steps whose price differs from
    the current compiled quote."""
    stale: list[WorkingOrderView] = []
    for wo in working:
        if wo.age_steps >= patience_steps:
            # If we have a compiled price and this order is on the same
            # side but at a stale price, it qualifies.
            if compiled_price is not None and compiled_side is not None:
                if wo.side == compiled_side and wo.price_ticks != compiled_price:
                    stale.append(wo)
            else:
                # No new order being placed — cancel any stale order
                # regardless of side (pure hygiene).
                if patience_steps < _NEVER_STALE:
                    stale.append(wo)
    return stale


# Sentinel: patience == 1.0 maps to max_patience_steps which means
# "never cancel-replace on staleness alone".  We detect it by value.
_NEVER_STALE = 2**31


# ---------------------------------------------------------------------------
# Self-trade prevention
# ---------------------------------------------------------------------------

def _stp_cancels(
    new_side: Side,
    new_price: int,
    working: tuple[WorkingOrderView, ...],
) -> list[Cancel]:
    """Cancel own resting orders that a new order would cross.

    A BUY at price P crosses any own resting SELL at price <= P.
    A SELL at price P crosses any own resting BUY at price >= P.
    """
    cancels: list[Cancel] = []
    for wo in working:
        if wo.side == new_side:
            continue
        if new_side == Side.BUY and wo.price_ticks <= new_price:
            cancels.append(Cancel(order_id=wo.order_id))
        elif new_side == Side.SELL and wo.price_ticks >= new_price:
            cancels.append(Cancel(order_id=wo.order_id))
    return cancels


# ---------------------------------------------------------------------------
# Veto enforcement (INV-6, last line of defense)
# ---------------------------------------------------------------------------

def _veto_suppresses(
    msg: ExchangeMessage,
    vetoes: Mapping[str, bool],
    inventory_lots: int,
    working: tuple[WorkingOrderView, ...],
) -> bool:
    """Return True if a veto flag requires suppressing this message.

    Suppress any message that would **increase** the vetoed variable's
    excursion.  Flatten-direction messages (those that reduce |inventory|)
    remain allowed.  Cancel messages are always allowed — they can only
    reduce exposure.

    This is the motor's **last line of defense** and must be unconditional.
    """
    if isinstance(msg, Cancel):
        # Cancels always reduce exposure; never suppress.
        return False

    assert isinstance(msg, PlaceLimit)

    # --- inventory veto ---
    if vetoes.get("inventory", False):
        # Would this order increase |inventory|?
        delta = msg.size_lots if msg.side == Side.BUY else -msg.size_lots
        new_inv = inventory_lots + delta
        if abs(new_inv) > abs(inventory_lots):
            return True

    # --- gross_exposure veto ---
    if vetoes.get("gross_exposure", False):
        # Any new order increases gross exposure unless it reduces |inventory|.
        delta = msg.size_lots if msg.side == Side.BUY else -msg.size_lots
        new_inv = inventory_lots + delta
        if abs(new_inv) >= abs(inventory_lots):
            return True

    # --- message_budget veto ---
    if vetoes.get("message_budget", False):
        # Suppress all new orders (they cost a message); cancels already
        # passed above.
        return True

    # --- drawdown veto ---
    if vetoes.get("drawdown", False):
        # Suppress orders that could increase drawdown by adding risk.
        # Only flatten-direction orders (reducing |inventory|) are allowed.
        delta = msg.size_lots if msg.side == Side.BUY else -msg.size_lots
        new_inv = inventory_lots + delta
        if abs(new_inv) >= abs(inventory_lots):
            return True

    return False


# ---------------------------------------------------------------------------
# Flatten path
# ---------------------------------------------------------------------------

def _compile_flatten(
    intent: Intent,
    bids: tuple[BookLevel, ...],
    asks: tuple[BookLevel, ...],
    working: tuple[WorkingOrderView, ...],
    cfg: MotorConfig,
) -> tuple[ExchangeMessage, ...]:
    """Compile a flatten intent: passive-first reduction of |inventory|.

    * Quote at touch on the reducing side.
    * Escalate to marketable only if ``cfg.flatten_urgent``.
    * Must NEVER increase |inventory|.
    """
    # Determine the reducing side from intent.side.
    if intent.side > 0:
        order_side = Side.BUY
    elif intent.side < 0:
        order_side = Side.SELL
    else:
        return ()  # side == 0 means nothing to flatten.

    # Size
    size_lots = round(intent.size_frac * cfg.size_budget_lots)
    if size_lots <= 0:
        return ()

    # Touch price on the reducing side (passive first).
    if order_side == Side.BUY:
        touch = _best_bid_tick(bids)
    else:
        touch = _best_ask_tick(asks)

    if touch is None:
        # No live levels on our side — can't quote passively.
        if cfg.flatten_urgent:
            # Cross: use the other side's best.
            if order_side == Side.BUY:
                touch = _best_ask_tick(asks)
            else:
                touch = _best_bid_tick(bids)
        if touch is None:
            return ()  # Truly empty book.

    if cfg.flatten_urgent:
        # If urgent, use a crossing price to guarantee a fill.
        if order_side == Side.BUY:
            cross = _best_ask_tick(asks)
            if cross is not None:
                touch = cross  # take the ask
        else:
            cross = _best_bid_tick(bids)
            if cross is not None:
                touch = cross  # hit the bid

    price_ticks = touch

    msgs: list[ExchangeMessage] = []

    # STP: cancel own resting orders that we'd cross.
    stp = _stp_cancels(order_side, price_ticks, working)
    msgs.extend(stp)

    msgs.append(
        PlaceLimit(
            side=order_side,
            price_ticks=price_ticks,
            size_lots=size_lots,
            tif_steps=0,
        )
    )
    return tuple(msgs)


# ---------------------------------------------------------------------------
# Main compile entry-point
# ---------------------------------------------------------------------------

def compile(
    intent: Intent,
    bids: tuple[BookLevel, ...],
    asks: tuple[BookLevel, ...],
    working_orders: tuple[WorkingOrderView, ...],
    homeostat_vetoes: Mapping[str, bool],
    cfg: MotorConfig,
    inventory_lots: int = 0,
) -> tuple[ExchangeMessage, ...]:
    """Compile an ``Intent`` into legal ``ExchangeMessage`` tuples.

    This is a **deterministic pure function** of its declared inputs
    (INV-8).  No randomness, no hidden state.

    Parameters
    ----------
    intent:
        The ignited intent from the arbiter.
    bids, asks:
        Book snapshot (exactly ``N_LEVELS`` each, padded with size_lots=0).
    working_orders:
        The agent's current working (resting) orders.
    homeostat_vetoes:
        ``Mapping[str, bool]``: hard veto flags from the homeostat (INV-6).
    cfg:
        Motor configuration (size budget, patience curve, flatten urgency).
    inventory_lots:
        Current net inventory in lots (needed for veto enforcement and
        STP).  Defaults to 0.
    """
    msgs: list[ExchangeMessage] = []

    # ------------------------------------------------------------------
    # 1. Flatten path (special compilation semantics)
    # ------------------------------------------------------------------
    if intent.is_flatten:
        flatten_msgs = _compile_flatten(intent, bids, asks, working_orders, cfg)
        # Apply veto filtering — flatten direction is allowed but we
        # still run the filter unconditionally.
        for m in flatten_msgs:
            if not _veto_suppresses(m, homeostat_vetoes, inventory_lots, working_orders):
                msgs.append(m)
        return tuple(msgs)

    # ------------------------------------------------------------------
    # 2. Compute compiled price and size for the normal path
    # ------------------------------------------------------------------
    mid = _mid_ticks(bids, asks)

    # Determine the order side from intent.side sign.
    if intent.side > 0:
        order_side = Side.BUY
    elif intent.side < 0:
        order_side = Side.SELL
    else:
        order_side = None  # ambiguous — treat as null if committed.

    compiled_price: int | None = None
    compiled_size: int = 0
    place_order = False

    if not intent.is_null and mid is not None and order_side is not None:
        # Price: mid - sign(side) * offset_ticks, snapped to tick (round).
        raw_price = mid - int(order_side) * intent.offset_ticks
        compiled_price = round(raw_price)

        # Size: round(size_frac * size_budget_lots).
        compiled_size = round(intent.size_frac * cfg.size_budget_lots)
        place_order = compiled_size > 0

    # ------------------------------------------------------------------
    # 3. Staleness / cancel-replace hygiene
    # ------------------------------------------------------------------
    patience_steps = cfg.patience_steps(intent.patience)
    # When patience == 1.0, patience_steps == max_patience_steps.
    # The spec says patience=1 => "never cancel-replace on staleness alone".
    # We enforce this by checking: if patience == 1.0 *exactly*, skip
    # staleness cancels.
    if intent.patience >= 1.0:
        stale = []
    else:
        stale = _stale_orders(
            working_orders,
            compiled_price,
            order_side,
            patience_steps,
        )

    for wo in stale:
        msgs.append(Cancel(order_id=wo.order_id))

    # ------------------------------------------------------------------
    # 4. Self-trade prevention + placement
    # ------------------------------------------------------------------
    if place_order and compiled_price is not None and order_side is not None:
        stp = _stp_cancels(order_side, compiled_price, working_orders)
        # Deduplicate against staleness cancels already emitted.
        existing_cancel_ids = {
            m.order_id for m in msgs if isinstance(m, Cancel)
        }
        for c in stp:
            if c.order_id not in existing_cancel_ids:
                msgs.append(c)

        msgs.append(
            PlaceLimit(
                side=order_side,
                price_ticks=compiled_price,
                size_lots=compiled_size,
                tif_steps=0,
            )
        )

    # ------------------------------------------------------------------
    # 5. Veto enforcement (INV-6 — last line of defense, unconditional)
    # ------------------------------------------------------------------
    filtered: list[ExchangeMessage] = []
    for m in msgs:
        if not _veto_suppresses(m, homeostat_vetoes, inventory_lots, working_orders):
            filtered.append(m)
    return tuple(filtered)
