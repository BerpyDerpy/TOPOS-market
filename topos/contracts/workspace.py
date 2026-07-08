"""Workspace (blackboard) contracts.

The bounded, typed workspace holds: world summary, hypothesis headlines,
cognitive self-state, the current focus, and the selected intent. The
`WorkspaceRecord` logged each cycle IS the interpretability story.

INV-5 lives here structurally: `SelfStateCognitive` — the ONLY self-state
arbitration and proposal code ever receive — has no PnL fields of any kind.
`SelfStateFull` carries the PnL fields and is consumed ONLY by drives/
(homeostat, as drawdown distance-to-bound) and metrics/. It is deliberately
NOT a subclass of `SelfStateCognitive`, so a PnL-bearing object can never
flow through an interface typed for the cognitive view.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from topos.contracts.intent import HypothesisId, Intent
from topos.contracts.market import ExchangeMessage, Side


@dataclass(frozen=True)
class Headline:
    """One hypothesis's bid for attention, capacity-limited in the workspace."""

    hypothesis_id: HypothesisId
    forecast_mean: float
    forecast_var: float
    epistemic_entropy_nats: float
    """Entropy of the PARAMETER posterior, not of the predictive (INV-3)."""
    best_marginal_eig_nats: float
    """Best available probe's EIG, marginal over the null action (INV-4)."""
    last_surprise_z: float


@dataclass(frozen=True)
class WorkingOrderView:
    """The agent's own belief about one working order.

    Queue rank is a posterior (mean/var of lots ahead at this level), never
    ground truth: the agent cannot observe engine-side queue position
    (INV-11).
    """

    order_id: int
    side: Side
    price_ticks: int
    size_lots_remaining: int
    age_steps: int
    queue_rank_mean: float
    queue_rank_var: float


@dataclass(frozen=True)
class SelfStateCognitive:
    """The self-state visible to arbitration and proposal code.

    NO PnL fields of any kind may ever appear here (INV-5; enforced by
    tests/tripwires/test_cognitive_view_has_no_pnl.py). `drive_distances`
    carries homeostat distances ONLY — dimensionless distance-to-bound per
    drive, not the underlying account quantities.
    """

    inventory_lots: int
    working_orders: tuple[WorkingOrderView, ...]
    drive_distances: Mapping[str, float]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "drive_distances", MappingProxyType(dict(self.drive_distances))
        )


@dataclass(frozen=True)
class SelfStateFull:
    """Cognitive self-state plus account quantities.

    Consumed ONLY by drives/ (homeostat) and metrics/. Deliberately not a
    subclass of `SelfStateCognitive` (see module docstring); the only path
    from here to the workspace is `cognitive_view()`, which strips the
    account fields.
    """

    inventory_lots: int
    working_orders: tuple[WorkingOrderView, ...]
    drive_distances: Mapping[str, float]
    realized_pnl: float
    unrealized_pnl: float
    gross_exposure: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "drive_distances", MappingProxyType(dict(self.drive_distances))
        )

    def cognitive_view(self) -> SelfStateCognitive:
        """Project onto the PnL-free view the workspace is allowed to see."""
        return SelfStateCognitive(
            inventory_lots=self.inventory_lots,
            working_orders=self.working_orders,
            drive_distances=self.drive_distances,
        )


@dataclass(frozen=True)
class Focus:
    """The single question that won the salience competition this cycle."""

    hypothesis_id: HypothesisId
    salience: float
    is_homeostatic: bool


@dataclass(frozen=True)
class WorldSummary:
    mid_ticks: float
    spread_ticks: int
    imbalance: float
    depth_profile: tuple[float, ...]
    trade_tempo: float
    realized_vol: float
    regime_posterior: tuple[float, ...]


@dataclass(frozen=True)
class WorkspaceRecord:
    """One cycle's complete broadcast — the interpretability story.

    Intent and compiled messages are logged side by side (INV-8);
    `eig_promised_nats` is the prospective EIG the arbiter acted on, to be
    compared against realized information gain from entropy snapshots
    (INV-10).
    """

    step: int
    world_summary: WorldSummary
    headlines: tuple[Headline, ...]
    self_state: SelfStateCognitive
    focus: Focus | None
    intent: Intent | None
    eig_promised_nats: float | None
    compiled_messages: tuple[ExchangeMessage, ...]
