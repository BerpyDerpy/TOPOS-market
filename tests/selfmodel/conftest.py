"""Shared builders for self-model tests.

Event-timing convention (matches the engine, see tests/env/test_harness.py):
a message submitted during engine step s is answered by acks/fills stamped
s, which the agent sees in the observation stamped s+1. ``SelfEvents``
therefore groups an observation with the messages sent ONE observation
earlier — the messages whose acks it carries.
"""

from __future__ import annotations

from topos.contracts.beliefs import ProbeSpec, SelfEvents
from topos.contracts.intent import FILL_RATE, HypothesisId, Intent
from topos.contracts.market import Ack, Fill, Observation, PlaceLimit, Trade

from tests.beliefs.conftest import make_obs

BALANCED_BIDS = [(999 - i, 20) for i in range(10)]
BALANCED_ASKS = [(1001 + i, 20) for i in range(10)]


def plain_obs(
    step: int,
    bids: list[tuple[int, int]] | None = None,
    asks: list[tuple[int, int]] | None = None,
    trades: tuple[Trade, ...] = (),
    own_acks: tuple[Ack, ...] = (),
    own_fills: tuple[Fill, ...] = (),
) -> Observation:
    """A symmetric 20-lot book around mid 1000 unless overridden."""
    return make_obs(
        step,
        bids if bids is not None else BALANCED_BIDS,
        asks if asks is not None else BALANCED_ASKS,
        trades=trades,
        own_acks=own_acks,
        own_fills=own_fills,
    )


def events(
    step: int,
    messages: tuple[PlaceLimit, ...] = (),
    acks: tuple[Ack, ...] = (),
    fills: tuple[Fill, ...] = (),
) -> SelfEvents:
    return SelfEvents(
        step=step, messages_sent=messages, acks=acks, fills=fills
    )


def committed_intent(
    side: float,
    offset_ticks: float,
    size_frac: float = 1.0,
    target_id: HypothesisId = FILL_RATE,
) -> Intent:
    return Intent(
        side=side,
        offset_ticks=offset_ticks,
        size_frac=size_frac,
        patience=0.5,
        target_id=target_id,
        commitment=1.0,
    )


def committed_probe(
    side: float,
    offset_ticks: float,
    size_frac: float = 1.0,
    horizon_steps: int = 1,
    target_id: HypothesisId = FILL_RATE,
) -> ProbeSpec:
    return ProbeSpec(
        intent=committed_intent(side, offset_ticks, size_frac, target_id),
        horizon_steps=horizon_steps,
    )
