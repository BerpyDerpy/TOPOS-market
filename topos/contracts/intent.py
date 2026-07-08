"""The Intent contract: the single abstract action the workspace can ignite.

An `Intent` describes a probe aimed at a specific hypothesis in direction /
urgency / patience terms. It is NOT an exchange message: the motor module
compiles it into concrete messages as a deterministic pure function (INV-8).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, TypeAlias

HypothesisId: TypeAlias = str

FAIR_VALUE: Final[HypothesisId] = "fair_value"
FLOW_INTENSITY: Final[HypothesisId] = "flow_intensity"
FILL_RATE: Final[HypothesisId] = "fill_rate"
IMPACT: Final[HypothesisId] = "impact"
QUEUE_POSITION: Final[HypothesisId] = "queue_position"
SELF_TRAJECTORY: Final[HypothesisId] = "self_trajectory"
REGIME: Final[HypothesisId] = "regime"
"""The slow-loop regime tracker. Passive-only: it learns from public
summaries and is never the target of a probe (no Intent carries it)."""

KNOWN_HYPOTHESIS_IDS: Final[tuple[HypothesisId, ...]] = (
    FAIR_VALUE,
    FLOW_INTENSITY,
    FILL_RATE,
    IMPACT,
    QUEUE_POSITION,
    SELF_TRAJECTORY,
    REGIME,
)

NULL_THRESHOLD: Final[float] = 0.5
"""Commitment below this value means the null action: observe, place nothing.

The null action is a first-class candidate with its own expected information
gain in an active market; probes are scored by MARGINAL EIG over null
(INV-4). The numeric value is provisional — see DESIGN.md, Open questions.
"""


@dataclass(frozen=True)
class Intent:
    """A probe the workspace ignited, aimed at exactly one hypothesis."""

    side: float
    """[-1, +1]; sign is direction, magnitude is conviction."""
    offset_ticks: float
    """Distance from mid toward passivity; negative means crossing."""
    size_frac: float
    """[0, 1] of the per-step size budget."""
    patience: float
    """[0, 1]; controls cancel/replace tempo of working orders."""
    target_id: HypothesisId
    """The hypothesis this probe interrogates."""
    commitment: float
    """[0, 1]; below NULL_THRESHOLD => null action (no new order)."""

    def __post_init__(self) -> None:
        if not -1.0 <= self.side <= 1.0:
            raise ValueError(f"side must be in [-1, +1], got {self.side}")
        if not 0.0 <= self.size_frac <= 1.0:
            raise ValueError(f"size_frac must be in [0, 1], got {self.size_frac}")
        if not 0.0 <= self.patience <= 1.0:
            raise ValueError(f"patience must be in [0, 1], got {self.patience}")
        if not 0.0 <= self.commitment <= 1.0:
            raise ValueError(f"commitment must be in [0, 1], got {self.commitment}")
        if not self.target_id:
            raise ValueError("target_id must be a non-empty HypothesisId")

    @property
    def is_null(self) -> bool:
        """True when this intent compiles to the null action (no new order)."""
        return self.commitment < NULL_THRESHOLD

    @property
    def is_flatten(self) -> bool:
        """True when the motor must apply flatten compilation semantics.

        Convention: SELF_TRAJECTORY is a forecast compiler, not a probeable
        hypothesis — no experiment in the proposer's menu ever targets it —
        so a committed intent carrying it can only be a flatten/corrective
        intent (from the homeostat via the arbiter). The motor keys its
        passive-first inventory-reduction path off this property.
        """
        return self.target_id == SELF_TRAJECTORY and not self.is_null


def flatten_intent(inventory_lots: int, size_frac: float = 1.0) -> Intent:
    """Distinguished constructor: reduce |inventory| toward 0, passive-first.

    Direction opposes the current inventory sign; patience is maximal so the
    motor works the book passively before ever crossing. With zero inventory
    there is nothing to flatten, so commitment collapses below
    NULL_THRESHOLD and the intent is null.

    `size_frac` keeps its universal meaning (fraction of the per-step size
    budget): the homeostat sizes partial corrections with it — e.g. shedding
    only the excess over the soft band — and re-evaluates every cycle, so
    flattening to the band happens across cycles, not in one shot.
    """
    if inventory_lots > 0:
        side = -1.0
    elif inventory_lots < 0:
        side = 1.0
    else:
        side = 0.0
    committed = inventory_lots != 0
    return Intent(
        side=side,
        offset_ticks=1.0,
        size_frac=size_frac,
        patience=1.0,
        target_id=SELF_TRAJECTORY,
        commitment=1.0 if committed else 0.0,
    )


FLATTEN_INTENT = flatten_intent
"""Spec-mandated name for the distinguished flatten constructor."""
