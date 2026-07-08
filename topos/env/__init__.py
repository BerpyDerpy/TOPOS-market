"""Matching engine, background market, and test harness (P1-P3).

`env.step(action)` returns an `Observation` only: no scalar feedback channel
of any kind exists in agent-facing code (INV-1). Ground-truth queue position
and engine-side account state flow only through harness-only channels into
metrics/validation, never into the agent (INV-11). All environment
randomness flows through named counter-based RNG streams (INV-9); see
`topos.contracts.rng`.
"""

from topos.env.background import (
    DEFAULT_REGIMES,
    REGIME_ACTOR_ID,
    ZI_ACTOR_ID,
    BackgroundConfig,
    BackgroundMarket,
    DrawRecord,
    MMConfig,
    RegimeController,
    RegimeParams,
    RegimeRecord,
    StabilizingMM,
    ZIConfig,
    ZIFlow,
    mm_actor_id,
)
from topos.env.engine import (
    ActorAccount,
    EngineAck,
    EngineEvent,
    EngineFill,
    EngineTrade,
    GroundTruthView,
    MatchingEngine,
)
from topos.env.orderbook import OrderBook, RestingOrder

__all__ = [
    "MatchingEngine",
    "OrderBook",
    "RestingOrder",
    "ActorAccount",
    "GroundTruthView",
    "EngineAck",
    "EngineFill",
    "EngineTrade",
    "EngineEvent",
    "BackgroundMarket",
    "BackgroundConfig",
    "ZIConfig",
    "MMConfig",
    "ZIFlow",
    "StabilizingMM",
    "RegimeController",
    "RegimeParams",
    "RegimeRecord",
    "DrawRecord",
    "DEFAULT_REGIMES",
    "ZI_ACTOR_ID",
    "REGIME_ACTOR_ID",
    "mm_actor_id",
]
