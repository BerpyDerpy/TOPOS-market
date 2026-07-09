"""FlowIntensity module behavior: protocol, event extraction, own-footprint
subtraction, and intent-independence."""

from __future__ import annotations

import pytest

from tests.beliefs.conftest import (
    empty_events,
    make_obs,
    null_probe,
    order_probe,
)
from topos.beliefs import FlowIntensity, band_of
from topos.contracts.beliefs import BeliefModule, SelfEvents
from topos.contracts.intent import FLOW_INTENSITY
from topos.contracts.market import (
    Ack,
    AckStatus,
    Fill,
    Liquidity,
    PlaceLimit,
    Side,
    Trade,
)

BASE_BIDS = [(999, 5), (998, 5), (996, 5)]
BASE_ASKS = [(1001, 5), (1002, 5), (1005, 5)]


def test_conforms_to_belief_module_protocol() -> None:
    module = FlowIntensity()
    assert isinstance(module, BeliefModule)
    assert module.hypothesis_id == FLOW_INTENSITY


def test_band_edges() -> None:
    assert band_of(0) == "touch"
    assert band_of(-1) == "touch"  # inside the previous best
    assert band_of(1) == "near"
    assert band_of(3) == "near"
    assert band_of(4) == "deep"


def test_first_observation_only_establishes_reference() -> None:
    module = FlowIntensity()
    module.update(make_obs(0, BASE_BIDS, BASE_ASKS), empty_events(0))
    assert module.last_counts == {}
    assert module.surprise_z() == 0.0


def test_arrival_counting_by_band() -> None:
    module = FlowIntensity()
    module.update(make_obs(0, BASE_BIDS, BASE_ASKS), empty_events(0))
    # +3 lots at the previous best bid (touch), +2 at 997 (2 ticks: near),
    # +4 at 994 (5 ticks: deep); asks unchanged.
    module.update(
        make_obs(1, [(999, 8), (998, 5), (997, 2), (996, 5), (994, 4)], BASE_ASKS),
        empty_events(1),
    )
    counts = module.last_counts
    assert counts[("arrival", Side.BUY, "touch")] == 3
    assert counts[("arrival", Side.BUY, "near")] == 2
    assert counts[("arrival", Side.BUY, "deep")] == 4
    assert counts[("arrival", Side.SELL, "touch")] == 0
    assert counts[("cancel", Side.BUY, "touch")] == 0


def test_cancel_counting_nets_out_trades() -> None:
    module = FlowIntensity()
    module.update(make_obs(0, BASE_BIDS, BASE_ASKS), empty_events(0))
    # Ask touch drops 5 -> 1: 2 lots traded (public print), so 2 cancelled.
    trade = Trade(price_ticks=1001, size_lots=2, aggressor=Side.BUY)
    module.update(
        make_obs(1, BASE_BIDS, [(1001, 1), (1002, 5), (1005, 5)], trades=(trade,)),
        empty_events(1),
    )
    counts = module.last_counts
    assert counts[("cancel", Side.SELL, "touch")] == 2
    assert counts[("market", Side.BUY, "touch")] == 2
    assert counts[("cancel", Side.SELL, "near")] == 0


def test_market_orders_banded_by_depth_walked() -> None:
    module = FlowIntensity()
    module.update(make_obs(0, BASE_BIDS, BASE_ASKS), empty_events(0))
    trades = (
        Trade(price_ticks=1001, size_lots=5, aggressor=Side.BUY),
        Trade(price_ticks=1002, size_lots=1, aggressor=Side.BUY),
    )
    module.update(
        make_obs(1, BASE_BIDS, [(1002, 4), (1005, 5)], trades=trades),
        empty_events(1),
    )
    counts = module.last_counts
    assert counts[("market", Side.BUY, "touch")] == 5
    assert counts[("market", Side.BUY, "near")] == 1  # 1 tick past prev best ask
    assert counts[("cancel", Side.SELL, "touch")] == 0
    assert counts[("cancel", Side.SELL, "near")] == 0


def test_own_resting_order_is_not_background_arrival() -> None:
    module = FlowIntensity()
    module.update(make_obs(0, BASE_BIDS, BASE_ASKS), empty_events(0))
    place = PlaceLimit(side=Side.BUY, price_ticks=998, size_lots=4, tif_steps=0)
    ack = Ack(order_id=7, status=AckStatus.ACCEPTED, step=1)
    events = SelfEvents(step=1, messages_sent=(place,), acks=(ack,), fills=())
    obs = make_obs(
        1,
        [(999, 5), (998, 9), (996, 5)],  # our 4 lots joined 998
        BASE_ASKS,
        own_acks=(ack,),
    )
    module.update(obs, events)
    assert module.last_counts[("arrival", Side.BUY, "near")] == 0


def test_own_cancel_is_not_background_cancel() -> None:
    module = FlowIntensity()
    module.update(make_obs(0, BASE_BIDS, BASE_ASKS), empty_events(0))
    place = PlaceLimit(side=Side.BUY, price_ticks=998, size_lots=4, tif_steps=0)
    accept = Ack(order_id=7, status=AckStatus.ACCEPTED, step=1)
    module.update(
        make_obs(1, [(999, 5), (998, 9), (996, 5)], BASE_ASKS, own_acks=(accept,)),
        SelfEvents(step=1, messages_sent=(place,), acks=(accept,), fills=()),
    )
    cancel_ack = Ack(order_id=7, status=AckStatus.CANCELED, step=2)
    module.update(
        make_obs(2, BASE_BIDS, BASE_ASKS, own_acks=(cancel_ack,)),
        SelfEvents(step=2, messages_sent=(), acks=(cancel_ack,), fills=()),
    )
    assert module.last_counts[("cancel", Side.BUY, "near")] == 0


def test_own_taker_fill_nets_passive_decrease_without_public_print() -> None:
    """Agent-caused prints never appear in Observation.trades (committed P1
    behavior), so the module must treat its own taker fills as the trades
    they are when netting the passive side's decrease."""
    module = FlowIntensity()
    module.update(make_obs(0, BASE_BIDS, BASE_ASKS), empty_events(0))
    place = PlaceLimit(side=Side.BUY, price_ticks=1001, size_lots=3, tif_steps=0)
    ack = Ack(order_id=9, status=AckStatus.ACCEPTED, step=1)
    fill = Fill(
        order_id=9, price_ticks=1001, size_lots=3, liquidity=Liquidity.TAKER, step=1
    )
    obs = make_obs(
        1,
        BASE_BIDS,
        [(1001, 2), (1002, 5), (1005, 5)],  # ask touch 5 -> 2, no public print
        own_acks=(ack,),
        own_fills=(fill,),
    )
    module.update(obs, SelfEvents(step=1, messages_sent=(place,), acks=(ack,), fills=(fill,)))
    counts = module.last_counts
    assert counts[("cancel", Side.SELL, "touch")] == 0
    # And our aggression is NOT counted as background market-order flow.
    assert counts[("market", Side.BUY, "touch")] == 0


def test_own_maker_fill_keeps_the_background_print() -> None:
    """A background aggressor hitting our resting order IS background flow:
    the public print is counted, and the size decrease it explains is not
    misread as a cancel."""
    module = FlowIntensity()
    place = PlaceLimit(side=Side.BUY, price_ticks=999, size_lots=2, tif_steps=0)
    accept = Ack(order_id=11, status=AckStatus.ACCEPTED, step=0)
    module.update(
        make_obs(0, [(999, 7), (998, 5), (996, 5)], BASE_ASKS, own_acks=(accept,)),
        SelfEvents(step=0, messages_sent=(place,), acks=(accept,), fills=()),
    )
    fill = Fill(
        order_id=11, price_ticks=999, size_lots=2, liquidity=Liquidity.MAKER, step=1
    )
    trade = Trade(price_ticks=999, size_lots=2, aggressor=Side.SELL)
    module.update(
        make_obs(
            1,
            [(999, 5), (998, 5), (996, 5)],
            BASE_ASKS,
            trades=(trade,),
            own_fills=(fill,),
        ),
        SelfEvents(step=1, messages_sent=(), acks=(), fills=(fill,)),
    )
    counts = module.last_counts
    assert counts[("market", Side.SELL, "touch")] == 2
    assert counts[("cancel", Side.BUY, "touch")] == 0


def test_eig_is_intent_independent() -> None:
    """Public flow is observed passively: marginal EIG over null is 0 for
    order-placing probes (INV-4: the null action carries this EIG)."""
    module = FlowIntensity()
    for horizon in (1, 5):
        eig_null = module.eig_nats(null_probe(FLOW_INTENSITY, horizon_steps=horizon))
        eig_order = module.eig_nats(order_probe(FLOW_INTENSITY, horizon_steps=horizon))
        assert eig_order == pytest.approx(eig_null, abs=1e-12)
        assert eig_null > 0.0


def test_snapshot_entropy_mechanics() -> None:
    module = FlowIntensity()
    module.update(make_obs(0, BASE_BIDS, BASE_ASKS), empty_events(0))
    snap = module.snapshot_entropy()
    assert snap.hypothesis_id == FLOW_INTENSITY
    assert snap.step == 0
    assert snap.entropy_nats == module.posterior_entropy_nats()
