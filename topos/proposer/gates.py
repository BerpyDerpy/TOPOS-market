"""Hard gates: motor legality, veto suppression, and the soft-band forecast.

INV-5 boundary. The proposer consumes ``SelfStateCognitive`` ONLY. The
homeostat's byproducts reach this module exclusively as EXPORTED VALUES —
the veto flags and the ``DistanceProjector`` injected per cycle — never as
an import of the drives package (there is no ``topos.drives`` import
anywhere under ``topos/proposer/``, pinned by a source-scan test). Account
quantities never appear here in any form; the projector answers in
dimensionless distance-to-bound units only.

Message cost and motor legality come from the motor compiler itself: it is
a deterministic pure function (INV-8), so calling it during proposal is
exact forecasting, not action. The book it compiles against is
reconstructed from the broadcast ``WorldSummary`` (best bid/ask from mid
and spread) — a documented approximation: only touch prices matter to
compilation, and the summary is the proposer's world input by design.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol

from topos.contracts.intent import Intent
from topos.contracts.market import BookLevel, PlaceLimit
from topos.contracts.workspace import SelfStateCognitive, WorldSummary
from topos.motor.compiler import compile as compile_intent
from topos.motor.config import MotorConfig
from topos.selfmodel.self_trajectory import SelfTrajectory

from topos.proposer.config import GATE_DELTA, GATE_FORECAST_HORIZON_STEPS

_NO_VETOES: Mapping[str, bool] = MappingProxyType({})


class DistanceProjector(Protocol):
    """Exported homeostat distance function, injected per cycle (INV-5).

    Maps a hypothetical post-action cognitive state — inventory in lots
    and the messages this action would send — to the homeostat's
    dimensionless distance-to-bound per variable (0.0 inside the soft
    band, growing toward 1.0 at the hard bound). Variables that cannot be
    predicted from the cognitive view are carried forward at their
    current distances (the no-information forecast). Implementations live
    with the homeostat wiring (P12); this package sees only the protocol.
    """

    def predicted_distances(
        self, inventory_lots: int, new_messages: int
    ) -> Mapping[str, float]: ...


def book_from_summary(
    world: WorldSummary,
) -> tuple[tuple[BookLevel, ...], tuple[BookLevel, ...]]:
    """Minimal (bids, asks) consistent with the broadcast summary.

    One live level per side at the touch implied by (mid, spread); the
    motor reads only live prices for mid/touch/crossing logic, so this is
    sufficient for message forecasting. Sizes are a nominal 1 lot (only
    presence matters; the size-0 convention marks absent levels).
    """
    half_spread = 0.5 * world.spread_ticks
    best_bid = int(round(world.mid_ticks - half_spread))
    best_ask = int(round(world.mid_ticks + half_spread))
    return (BookLevel(best_bid, 1),), (BookLevel(best_ask, 1),)


def compiled_messages(
    intent: Intent,
    world: WorldSummary,
    cognitive: SelfStateCognitive,
    vetoes: Mapping[str, bool],
    motor_cfg: MotorConfig,
) -> tuple[object, ...]:
    """The messages the motor would emit for this intent, right now."""
    bids, asks = book_from_summary(world)
    return compile_intent(
        intent,
        bids,
        asks,
        cognitive.working_orders,
        vetoes,
        motor_cfg,
        cognitive.inventory_lots,
    )


@dataclass(frozen=True)
class GateReport:
    """Everything the hard gates measured about one candidate."""

    motor_legal: bool
    """A committed intent must express something the motor can actually
    do (>= 1 placement before veto filtering); the null and any
    null-commitment intent are trivially legal (they demand nothing)."""
    vetoed: bool
    """True when a homeostat veto flag suppresses any of the candidate's
    messages (measured by compiling with and without the flags)."""
    message_cost: int
    """Messages that would actually be emitted (veto filtering applied)."""
    predicted_distances: Mapping[str, float]
    within_soft_confidence: float
    passed: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "predicted_distances",
            MappingProxyType(dict(self.predicted_distances)),
        )


def evaluate_gates(
    intent: Intent,
    world: WorldSummary,
    cognitive: SelfStateCognitive,
    vetoes: Mapping[str, bool],
    motor_cfg: MotorConfig,
    trajectory: SelfTrajectory,
    projector: DistanceProjector,
) -> GateReport:
    """Apply the three hard gates to one candidate intent.

    (a1) motor-legal; (a2) no homeostat veto suppression; (a3) the
    one-step self-forecast keeps every predicted distance inside the soft
    bands with probability >= 1 - GATE_DELTA. The forecast marginalizes
    over the fill outcomes of the trajectory compiler's inventory pmf —
    the same posteriors used everywhere else, never a separate model.
    """
    unfiltered = compiled_messages(intent, world, cognitive, _NO_VETOES, motor_cfg)
    filtered = compiled_messages(intent, world, cognitive, vetoes, motor_cfg)
    if intent.is_null:
        motor_legal = True
    else:
        motor_legal = any(isinstance(msg, PlaceLimit) for msg in unfiltered)
    vetoed = len(filtered) < len(unfiltered)
    message_cost = len(filtered)

    forecast = trajectory.forecast(intent, GATE_FORECAST_HORIZON_STEPS)
    expected: dict[str, float] = {}
    confidence = 0.0
    for inventory_lots, probability in forecast.inventory_pmf:
        distances = projector.predicted_distances(inventory_lots, message_cost)
        if all(u <= 0.0 for u in distances.values()):
            confidence += probability
        for name, u in distances.items():
            expected[name] = expected.get(name, 0.0) + probability * u

    passed = (
        motor_legal
        and not vetoed
        and confidence >= 1.0 - GATE_DELTA - 1e-12
    )
    return GateReport(
        motor_legal=motor_legal,
        vetoed=vetoed,
        message_cost=message_cost,
        predicted_distances=expected,
        within_soft_confidence=confidence,
        passed=passed,
    )
