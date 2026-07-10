"""Structural wiring: the module dependency graph and the distance projector.

Both artifacts here are things only the integration layer can own: the
registry declares who reads what across ALL packages (INV-7's
consequence-weights derive from it, once, at startup), and the
``DistanceProjector`` implementation lives with the homeostat wiring
(DESIGN item 23) so that the proposer sees only the protocol and never
imports drives.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from topos.contracts.intent import (
    FAIR_VALUE,
    FILL_RATE,
    FLOW_INTENSITY,
    IMPACT,
    KNOWN_HYPOTHESIS_IDS,
    QUEUE_POSITION,
    REGIME,
    SELF_TRAJECTORY,
)
from topos.contracts.registry import ModuleRegistry
from topos.drives.config import HomeostatConfig, VariableBounds


def default_registry() -> ModuleRegistry:
    """The integrated agent's declared dependency graph (INV-7).

    Edges reflect the ACTUAL dataflow of the loop, declared once at
    startup and never revisited: forecasts feed the trajectory compiler
    and the workspace; the flow forecast additionally conditions the queue
    filter's observation model and the impact model's context regressors;
    the regime tracker's forgetting factor gates every belief module; the
    broadcast focus conditions the two hook-bearing world modules.
    Hypothesis-owning modules register under their hypothesis_id (the
    registry-key convention); ``books`` and ``workspace`` are mechanical
    registrants whose weights the salience competition never looks up.
    """
    registry = ModuleRegistry()
    registry.register(
        FAIR_VALUE,
        reads={"market.book", "regime.rho", "workspace.focus"},
        writes={"fair_value.forecast"},
    )
    registry.register(
        FLOW_INTENSITY,
        reads={"market.book", "market.trades", "self.events", "regime.rho",
               "workspace.focus"},
        writes={"flow.forecast"},
    )
    registry.register(
        QUEUE_POSITION,
        reads={"market.book", "self.events", "flow.forecast", "regime.rho"},
        writes={"queue.rank"},
    )
    registry.register(
        FILL_RATE,
        reads={"market.book", "self.events", "regime.rho"},
        writes={"fill.posterior"},
    )
    registry.register(
        IMPACT,
        reads={"market.book", "self.events", "flow.forecast", "regime.rho"},
        writes={"impact.posterior"},
    )
    registry.register(
        SELF_TRAJECTORY,
        reads={"fill.posterior", "impact.posterior", "fair_value.forecast",
               "self.state"},
        writes={"self.forecast"},
    )
    registry.register(
        REGIME,
        reads={"world.summary"},
        writes={"regime.rho", "regime.posterior"},
    )
    registry.register(
        "books",
        reads={"market.book", "self.events", "queue.rank"},
        writes={"self.state"},
    )
    registry.register(
        "workspace",
        reads={"fair_value.forecast", "flow.forecast", "queue.rank",
               "fill.posterior", "impact.posterior", "self.forecast",
               "self.state", "regime.posterior"},
        writes={"workspace.focus", "world.summary"},
    )
    return registry


def assert_registry_covers_known_ids(registry: ModuleRegistry) -> None:
    """Fail fast at agent construction (adjudication A3): every id in
    KNOWN_HYPOTHESIS_IDS must have a consequence-weight, i.e. every
    hypothesis-owning module registered before the registry froze."""
    weights = registry.centrality_weights()
    missing = [h for h in KNOWN_HYPOTHESIS_IDS if h not in weights]
    if missing:
        raise ValueError(
            "registry centrality weights do not cover every known "
            f"hypothesis id: missing {missing!r} — register each "
            "hypothesis-owning module under its hypothesis_id at startup"
        )


def _excursion(value: float, bounds: VariableBounds) -> float:
    """The homeostat's normalized excursion: 0 inside the soft band,
    (|v| - soft) / (hard - soft) beyond it, 1 at the hard bound."""
    return max(0.0, (abs(value) - bounds.soft) / (bounds.hard - bounds.soft))


@dataclass(frozen=True)
class BandDistanceProjector:
    """The injected ``DistanceProjector`` implementation (DESIGN item 23).

    Maps a hypothetical post-action cognitive state to dimensionless
    distance-to-bound per homeostat variable. Inventory, gross exposure and
    the message budget are predictable from the cognitive view and are
    recomputed against the bands; variables that are not (drawdown) are
    carried forward at their CURRENT distances — the no-information
    forecast — so an account-side breach gates every candidate out without
    the proposer ever seeing account state (INV-5: distances only, never
    the quantities behind them). Rebuilt each cycle from that cycle's mid
    and rolling message count.
    """

    cfg: HomeostatConfig
    mid: float
    rolling_messages: int
    carried: Mapping[str, float]

    def predicted_distances(
        self, inventory_lots: int, new_messages: int
    ) -> Mapping[str, float]:
        return {
            "inventory": _excursion(float(inventory_lots), self.cfg.inventory),
            "gross_exposure": _excursion(
                inventory_lots * self.mid, self.cfg.gross_exposure
            ),
            "message_budget": _excursion(
                float(self.rolling_messages + new_messages),
                self.cfg.message_budget,
            ),
            **dict(self.carried),
        }
