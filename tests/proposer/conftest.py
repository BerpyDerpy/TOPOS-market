"""Shared builders for proposer tests.

The seeded module set feeds an active synthetic market (balanced book,
Poisson trade prints — the P4 saturation pattern) into real belief and
self-model modules, so proposer scores flow through exactly the machinery
the agent will use. The projector mirrors the homeostat's normalized
excursion formula without importing drives (the proposer sees only the
protocol, INV-5).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

from tests.beliefs.conftest import empty_events, make_obs
from topos.beliefs import FairValueKF, FlowIntensity
from topos.contracts.beliefs import BeliefModule, ProbeSpec, SelfEvents
from topos.contracts.intent import (
    FAIR_VALUE,
    FILL_RATE,
    FLOW_INTENSITY,
    IMPACT,
    HypothesisId,
    Intent,
    flatten_intent,
)
from topos.contracts.market import N_LEVELS, Observation, Side, Trade
from topos.contracts.workspace import (
    SelfStateCognitive,
    WorkingOrderView,
    WorldSummary,
)
from topos.motor.config import MotorConfig
from topos.proposer import Candidate, Proposer, null_intent
from topos.selfmodel import FillModel, ImpactModel, SelfTrajectory

BUDGET_LOTS = 4
FILL_HORIZON = 2


def active_obs(step: int, rng: np.random.Generator) -> Observation:
    """Balanced 20-lot book around mid 1000 with Poisson trade prints."""
    trades = []
    buy_lots = int(rng.poisson(3.0))
    sell_lots = int(rng.poisson(2.0))
    if buy_lots > 0:
        trades.append(Trade(price_ticks=1001, size_lots=buy_lots, aggressor=Side.BUY))
    if sell_lots > 0:
        trades.append(Trade(price_ticks=999, size_lots=sell_lots, aggressor=Side.SELL))
    bids = [(999 - i, 20) for i in range(N_LEVELS)]
    asks = [(1001 + i, 20) for i in range(N_LEVELS)]
    return make_obs(step, bids, asks, trades=tuple(trades))


def seeded_modules(
    steps: int = 25,
) -> tuple[dict[HypothesisId, BeliefModule], SelfTrajectory]:
    """Real modules fed an active synthetic market for ``steps`` steps."""
    fair_value = FairValueKF()
    flow = FlowIntensity()
    fill = FillModel(horizon_steps=FILL_HORIZON, size_budget_lots=BUDGET_LOTS)
    impact = ImpactModel(size_budget_lots=BUDGET_LOTS)
    rng = np.random.default_rng(7)
    for step in range(steps):
        obs = active_obs(step, rng)
        events: SelfEvents = empty_events(step)
        for module in (fair_value, flow, fill, impact):
            module.update(obs, events)
    trajectory = SelfTrajectory(
        fill, impact, fair_value, size_budget_lots=BUDGET_LOTS
    )
    modules: dict[HypothesisId, BeliefModule] = {
        FAIR_VALUE: fair_value,
        FLOW_INTENSITY: flow,
        FILL_RATE: fill,
        IMPACT: impact,
    }
    return modules, trajectory


def make_proposer(
    modules: Mapping[HypothesisId, BeliefModule], trajectory: SelfTrajectory
) -> Proposer:
    return Proposer(
        modules=modules,
        trajectory=trajectory,
        motor_cfg=MotorConfig(size_budget_lots=BUDGET_LOTS),
        probe_horizon_steps=FILL_HORIZON,
    )


def make_world(
    mid: float = 1000.0, spread: int = 2, imbalance: float = 0.0
) -> WorldSummary:
    return WorldSummary(
        mid_ticks=mid,
        spread_ticks=spread,
        imbalance=imbalance,
        depth_profile=tuple(20.0 for _ in range(N_LEVELS)),
        trade_tempo=5.0,
        realized_vol=1.0,
        regime_posterior=(1.0,),
    )


def make_cognitive(
    inventory: int = 0,
    working: tuple[WorkingOrderView, ...] = (),
    distances: Mapping[str, float] | None = None,
) -> SelfStateCognitive:
    return SelfStateCognitive(
        inventory_lots=inventory,
        working_orders=working,
        drive_distances=dict(distances or {}),
    )


def _excursion(value: float, soft: float, hard: float) -> float:
    return max(0.0, (abs(value) - soft) / (hard - soft))


@dataclass
class BandProjector:
    """Test implementation of the injected DistanceProjector protocol.

    Mirrors the homeostat's normalized-excursion formula for the
    cognitively predictable variables; ``carried`` stands in for the
    variables the projector carries forward at their current distances.
    """

    mid: float = 1000.0
    inventory_soft: float = 10.0
    inventory_hard: float = 20.0
    gross_soft: float = 50_000.0
    gross_hard: float = 100_000.0
    message_soft: float = 50.0
    message_hard: float = 100.0
    rolling_messages: int = 0
    carried: dict[str, float] = field(default_factory=dict)

    def predicted_distances(
        self, inventory_lots: int, new_messages: int
    ) -> Mapping[str, float]:
        return {
            "inventory": _excursion(
                inventory_lots, self.inventory_soft, self.inventory_hard
            ),
            "gross_exposure": _excursion(
                inventory_lots * self.mid, self.gross_soft, self.gross_hard
            ),
            "message_budget": _excursion(
                self.rolling_messages + new_messages,
                self.message_soft,
                self.message_hard,
            ),
            **self.carried,
        }


def saturate(cell, n: int = 400, fraction: float = 0.7) -> None:
    """Drive one Beta cell to a settled posterior (the P6 pattern: many
    resolved trials at a stable rate — the tenth fill is boring)."""
    for _ in range(n):
        cell.observe(fraction)


def probe_candidates(candidates) -> list[Candidate]:
    """The committed experiment candidates (drop the null and flatten)."""
    return [
        c
        for c in candidates
        if not c.probe.intent.is_null and not c.probe.intent.is_flatten
    ]


def mk_candidate(
    kind: str = "probe",
    marginal: float = 0.1,
    self_entropy: float = 1.0,
    gates: bool = True,
    null: bool = False,
    flatten: bool = False,
    vetoed: bool = False,
    confidence: float = 1.0,
    legal: bool = True,
) -> Candidate:
    """Hand-built candidate for selection-rule unit tests."""
    if null:
        intent = null_intent(FILL_RATE)
        marginal = 0.0
    elif flatten:
        intent = flatten_intent(5)
    else:
        intent = Intent(
            side=1.0,
            offset_ticks=1.0,
            size_frac=1.0,
            patience=0.5,
            target_id=FILL_RATE,
            commitment=1.0,
        )
    return Candidate(
        kind=kind,
        probe=ProbeSpec(intent=intent, horizon_steps=FILL_HORIZON),
        eig_nats=max(0.0, marginal),
        null_eig_nats=0.0,
        marginal_eig_nats=marginal,
        self_entropy_nats=self_entropy,
        predicted_distances={},
        within_soft_confidence=confidence,
        message_cost=0 if null else 1,
        motor_legal=legal,
        vetoed=vetoed,
        gates_passed=gates,
    )
