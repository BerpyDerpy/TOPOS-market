"""Frozen data contracts, protocols, module registry, and RNG streams.

Everything in this package is either a frozen dataclass, an enum, a
`Protocol`, or a pure constructor. No market or cognition logic lives here.
The tripwire suite (tests/tripwires/) treats these definitions as the
ground truth for the architectural invariants listed in DESIGN.md.
"""

from topos.contracts.beliefs import (
    BeliefModule,
    EntropySnapshot,
    ForecastStats,
    ProbeSpec,
    SelfEvents,
    realized_information_gain_nats,
)
from topos.contracts.intent import (
    FAIR_VALUE,
    FILL_RATE,
    FLATTEN_INTENT,
    FLOW_INTENSITY,
    IMPACT,
    KNOWN_HYPOTHESIS_IDS,
    NULL_THRESHOLD,
    QUEUE_POSITION,
    SELF_TRAJECTORY,
    HypothesisId,
    Intent,
    flatten_intent,
)
from topos.contracts.market import (
    GTC,
    N_LEVELS,
    Ack,
    AckStatus,
    BookLevel,
    Cancel,
    ExchangeMessage,
    Fill,
    Liquidity,
    Observation,
    PlaceLimit,
    Side,
    Trade,
)
from topos.contracts.registry import ModuleDecl, ModuleRegistry
from topos.contracts.rng import StreamKey, make_rng
from topos.contracts.workspace import (
    Focus,
    Headline,
    SelfStateCognitive,
    SelfStateFull,
    WorkingOrderView,
    WorkspaceRecord,
    WorldSummary,
)

__all__ = [
    # market
    "N_LEVELS",
    "GTC",
    "Side",
    "AckStatus",
    "Liquidity",
    "PlaceLimit",
    "Cancel",
    "ExchangeMessage",
    "Ack",
    "Fill",
    "BookLevel",
    "Trade",
    "Observation",
    # intent
    "HypothesisId",
    "Intent",
    "flatten_intent",
    "FLATTEN_INTENT",
    "NULL_THRESHOLD",
    "KNOWN_HYPOTHESIS_IDS",
    "FAIR_VALUE",
    "FLOW_INTENSITY",
    "FILL_RATE",
    "IMPACT",
    "QUEUE_POSITION",
    "SELF_TRAJECTORY",
    # workspace
    "Headline",
    "WorkingOrderView",
    "SelfStateCognitive",
    "SelfStateFull",
    "Focus",
    "WorldSummary",
    "WorkspaceRecord",
    # beliefs
    "ProbeSpec",
    "SelfEvents",
    "ForecastStats",
    "EntropySnapshot",
    "BeliefModule",
    "realized_information_gain_nats",
    # registry
    "ModuleDecl",
    "ModuleRegistry",
    # rng
    "StreamKey",
    "make_rng",
]
