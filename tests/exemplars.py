"""One valid exemplar instance of every contract dataclass.

tests/tripwires/test_contracts_frozen.py checks that every dataclass
discovered under topos.contracts has an exemplar here and that mutating it
raises. Adding a contract type without extending this set fails the
tripwire — deliberately, so new contracts cannot dodge the freeze check.
"""

from __future__ import annotations

from topos.contracts.beliefs import (
    EntropySnapshot,
    ForecastStats,
    ProbeSpec,
    SelfEvents,
)
from topos.contracts.intent import FAIR_VALUE, FILL_RATE, Intent
from topos.contracts.market import (
    N_LEVELS,
    Ack,
    AckStatus,
    BookLevel,
    Cancel,
    Fill,
    Liquidity,
    Observation,
    PlaceLimit,
    Side,
    Trade,
)
from topos.contracts.registry import ModuleDecl
from topos.contracts.rng import StreamKey
from topos.contracts.workspace import (
    Focus,
    Headline,
    SelfStateCognitive,
    SelfStateFull,
    WorkingOrderView,
    WorkspaceRecord,
    WorldSummary,
)


def contract_exemplars() -> dict[type, object]:
    """Map every contract dataclass to one valid instance."""
    place = PlaceLimit(side=Side.BUY, price_ticks=1000, size_lots=2, tif_steps=0)
    cancel = Cancel(order_id=7)
    ack = Ack(order_id=7, status=AckStatus.ACCEPTED, step=3)
    fill = Fill(
        order_id=7, price_ticks=1000, size_lots=1, liquidity=Liquidity.MAKER, step=4
    )
    bids = tuple(
        BookLevel(price_ticks=1000 - i, size_lots=5) for i in range(N_LEVELS)
    )
    asks = tuple(
        BookLevel(price_ticks=1001 + i, size_lots=5) for i in range(N_LEVELS)
    )
    trade = Trade(price_ticks=1000, size_lots=1, aggressor=Side.SELL)
    obs = Observation(
        step=4,
        bids=bids,
        asks=asks,
        trades=(trade,),
        own_acks=(ack,),
        own_fills=(fill,),
    )
    intent = Intent(
        side=1.0,
        offset_ticks=2.0,
        size_frac=0.25,
        patience=0.8,
        target_id=FILL_RATE,
        commitment=0.9,
    )
    probe = ProbeSpec(intent=intent, horizon_steps=10)
    self_events = SelfEvents(
        step=4, messages_sent=(place, cancel), acks=(ack,), fills=(fill,)
    )
    forecast = ForecastStats(mean=1000.5, variance=2.0)
    snapshot = EntropySnapshot(hypothesis_id=FAIR_VALUE, step=4, entropy_nats=1.5)
    headline = Headline(
        hypothesis_id=FAIR_VALUE,
        forecast_mean=1000.5,
        forecast_var=2.0,
        epistemic_entropy_nats=1.5,
        best_marginal_eig_nats=0.1,
        last_surprise_z=0.3,
    )
    order_view = WorkingOrderView(
        order_id=7,
        side=Side.BUY,
        price_ticks=999,
        size_lots_remaining=2,
        age_steps=1,
        queue_rank_mean=3.2,
        queue_rank_var=1.1,
    )
    cognitive = SelfStateCognitive(
        inventory_lots=-3,
        working_orders=(order_view,),
        drive_distances={"inventory": 0.4},
    )
    full = SelfStateFull(
        inventory_lots=-3,
        working_orders=(order_view,),
        drive_distances={"inventory": 0.4},
        realized_pnl=-1.0,
        unrealized_pnl=0.5,
        gross_exposure=3.0,
    )
    focus = Focus(hypothesis_id=FAIR_VALUE, salience=0.7, is_homeostatic=False)
    world = WorldSummary(
        mid_ticks=1000.5,
        spread_ticks=1,
        imbalance=0.1,
        depth_profile=tuple(float(5 - i) for i in range(N_LEVELS)),
        trade_tempo=0.5,
        realized_vol=0.02,
        regime_posterior=(0.7, 0.3),
    )
    record = WorkspaceRecord(
        step=4,
        world_summary=world,
        headlines=(headline,),
        self_state=cognitive,
        focus=focus,
        intent=intent,
        eig_promised_nats=0.1,
        compiled_messages=(place, cancel),
    )
    stream_key = StreamKey(actor_id="agent", step=4, purpose="latency")
    decl = ModuleDecl(
        name="fair_value",
        reads=frozenset({"trades"}),
        writes=frozenset({"fair_value"}),
    )
    instances = [
        place,
        cancel,
        ack,
        fill,
        bids[0],
        trade,
        obs,
        intent,
        probe,
        self_events,
        forecast,
        snapshot,
        headline,
        order_view,
        cognitive,
        full,
        focus,
        world,
        record,
        stream_key,
        decl,
    ]
    return {type(instance): instance for instance in instances}
