"""Shared builders for workspace tests.

Fake belief modules give the salience/arbitration tests exact control
over the three inputs of the salience formula (weight via the registry
graph, marginal EIG via ``probe_gain``, surprise via ``surprise``), while
still flowing through the REAL ``Proposer`` and the REAL exported
selection rule — the workspace under test must call them verbatim, so
they are never faked. Conditioning-hook tests use the real belief
modules instead (tests/workspace/test_broadcast.py).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from tests.proposer.conftest import BandProjector, make_cognitive, make_world
from topos.contracts.beliefs import (
    EntropySnapshot,
    ForecastStats,
    ProbeSpec,
    SelfEvents,
)
from topos.contracts.intent import KNOWN_HYPOTHESIS_IDS, HypothesisId, Intent
from topos.contracts.market import Observation
from topos.contracts.registry import ModuleRegistry
from topos.contracts.workspace import SelfStateCognitive, WorkspaceRecord, WorldSummary
from topos.motor.config import MotorConfig
from topos.proposer import Proposer
from topos.workspace import Workspace

__all__ = [
    "BUDGET_LOTS",
    "BandProjector",
    "FakeModule",
    "FakeTrajectory",
    "make_cognitive",
    "make_registry",
    "make_workspace",
    "make_world",
    "run_cycle",
]

BUDGET_LOTS = 4
HORIZON = 2


@dataclass
class FakeModule:
    """A controllable BeliefModule: marginal EIG and surprise are dials."""

    hypothesis_id: HypothesisId
    probe_gain: float = 0.1
    """Marginal EIG of every committed probe over the null."""
    null_eig: float = 0.05
    surprise: float = 0.0
    entropy: float = 1.0
    focus_log: list[object] = field(default_factory=list)
    updated_steps: list[int] = field(default_factory=list)

    def update(self, obs: Observation, self_events: SelfEvents) -> None:
        self.updated_steps.append(obs.step)

    def forget(self, rho: float) -> None:
        pass

    def posterior_entropy_nats(self) -> float:
        return self.entropy

    def predict(self) -> ForecastStats:
        return ForecastStats(mean=0.0, variance=1.0)

    def surprise_z(self) -> float:
        return self.surprise

    def eig_nats(self, probe: ProbeSpec) -> float:
        if probe.intent.is_null:
            return self.null_eig
        return self.null_eig + self.probe_gain

    def snapshot_entropy(self) -> EntropySnapshot:
        return EntropySnapshot(
            hypothesis_id=self.hypothesis_id, step=0, entropy_nats=self.entropy
        )

    def condition_on_focus(self, focus: object) -> None:
        self.focus_log.append(focus)


@dataclass(frozen=True)
class _FakeForecast:
    inventory_pmf: tuple[tuple[int, float], ...]
    entropy_nats: float


class FakeTrajectory:
    """Stands in for SelfTrajectory: null is the most self-predictable."""

    def __init__(self) -> None:
        self.cycles: int = 0

    def begin_cycle(
        self, cognitive: SelfStateCognitive, world: WorldSummary
    ) -> None:
        self.cycles += 1

    def self_entropy_nats(
        self, intent: Intent, horizon_steps: int | None = None
    ) -> float:
        return 0.0 if intent.is_null else 1.0

    def forecast(
        self, intent: Intent, horizon_steps: int | None = None
    ) -> _FakeForecast:
        return _FakeForecast(
            inventory_pmf=((0, 1.0),),
            entropy_nats=self.self_entropy_nats(intent, horizon_steps),
        )


def make_registry(
    hypothesis_ids: Sequence[HypothesisId] = (),
    *,
    hub: HypothesisId | None = None,
) -> ModuleRegistry:
    """A registry covering KNOWN_HYPOTHESIS_IDS plus any extra ids.

    With ``hub=None`` the declared graph has no edges, so weights fall
    back to uniform. With ``hub=h``, module h writes a key every other
    module reads — a different graph, hence different weights.
    """
    registry = ModuleRegistry()
    names = list(dict.fromkeys([*KNOWN_HYPOTHESIS_IDS, *hypothesis_ids]))
    for name in names:
        writes = {f"{name}.headline"}
        if name == hub:
            writes.add("hub.broadcast")
        reads = {"hub.broadcast"} if hub is not None and name != hub else set()
        registry.register(name, reads=reads, writes=writes)
    return registry


def make_workspace(
    modules: Mapping[HypothesisId, FakeModule],
    *,
    registry: ModuleRegistry | None = None,
    consumers: Sequence[object] | None = None,
    trajectory: FakeTrajectory | None = None,
    proposer: Proposer | None = None,
    **workspace_kwargs: object,
) -> Workspace:
    motor_cfg = MotorConfig(size_budget_lots=BUDGET_LOTS)
    if proposer is None:
        proposer = Proposer(
            modules=modules,
            trajectory=trajectory or FakeTrajectory(),  # type: ignore[arg-type]
            motor_cfg=motor_cfg,
            probe_horizon_steps=HORIZON,
        )
    return Workspace(
        registry=registry or make_registry(tuple(modules)),
        proposer=proposer,
        modules=modules,
        motor_cfg=motor_cfg,
        consumers=consumers if consumers is not None else (),
        **workspace_kwargs,  # type: ignore[arg-type]
    )


def run_cycle(
    workspace: Workspace,
    *,
    step: int = 0,
    inventory: int = 0,
    drives: Mapping[str, float] | None = None,
    vetoes: Mapping[str, bool] | None = None,
    corrective_intent: Intent | None = None,
) -> WorkspaceRecord:
    return workspace.cycle(
        step=step,
        world=make_world(),
        cognitive=make_cognitive(inventory=inventory),
        drives=dict(drives or {}),
        vetoes=dict(vetoes or {}),
        corrective_intent=corrective_intent,
        projector=BandProjector(),
    )
