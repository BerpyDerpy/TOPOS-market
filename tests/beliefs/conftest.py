"""Shared builders for belief-module tests."""

from __future__ import annotations

from topos.contracts.beliefs import ProbeSpec, SelfEvents
from topos.contracts.intent import FAIR_VALUE, HypothesisId, Intent
from topos.contracts.market import (
    N_LEVELS,
    Ack,
    BookLevel,
    Fill,
    Observation,
    Trade,
)


def pad_levels(levels: list[tuple[int, int]]) -> tuple[BookLevel, ...]:
    """Pad (price, size) pairs to exactly N_LEVELS with size-0 levels."""
    out = [BookLevel(price_ticks=p, size_lots=s) for p, s in levels[:N_LEVELS]]
    fill_price = out[-1].price_ticks if out else 0
    while len(out) < N_LEVELS:
        out.append(BookLevel(price_ticks=fill_price, size_lots=0))
    return tuple(out)


def make_obs(
    step: int,
    bids: list[tuple[int, int]],
    asks: list[tuple[int, int]],
    trades: tuple[Trade, ...] = (),
    own_acks: tuple[Ack, ...] = (),
    own_fills: tuple[Fill, ...] = (),
) -> Observation:
    return Observation(
        step=step,
        bids=pad_levels(bids),
        asks=pad_levels(asks),
        trades=trades,
        own_acks=own_acks,
        own_fills=own_fills,
    )


def obs_for_mid(
    step: int, y: float, size_lots: int = 5, trades: tuple[Trade, ...] = ()
) -> Observation:
    """A symmetric book whose microprice equals y to within a quarter tick."""
    q = int(round(2.0 * y))
    best_bid = (q - 1) // 2
    best_ask = q - best_bid
    bids = [(best_bid - i, size_lots) for i in range(N_LEVELS)]
    asks = [(best_ask + i, size_lots) for i in range(N_LEVELS)]
    return make_obs(step, bids, asks, trades=trades)


def empty_events(step: int) -> SelfEvents:
    return SelfEvents(step=step, messages_sent=(), acks=(), fills=())


def null_probe(target_id: HypothesisId = FAIR_VALUE, horizon_steps: int = 1) -> ProbeSpec:
    """The null action as a probe (observe, place nothing) — INV-4."""
    intent = Intent(
        side=0.0,
        offset_ticks=0.0,
        size_frac=0.0,
        patience=1.0,
        target_id=target_id,
        commitment=0.0,
    )
    return ProbeSpec(intent=intent, horizon_steps=horizon_steps)


def order_probe(
    target_id: HypothesisId = FAIR_VALUE, horizon_steps: int = 1
) -> ProbeSpec:
    """A committed order-placing probe with the same horizon as the null."""
    intent = Intent(
        side=1.0,
        offset_ticks=1.0,
        size_frac=0.5,
        patience=0.5,
        target_id=target_id,
        commitment=1.0,
    )
    return ProbeSpec(intent=intent, horizon_steps=horizon_steps)
