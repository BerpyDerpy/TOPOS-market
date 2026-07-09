"""Candidate generation: the standing coarse menu and the refined menu.

Two-stage design (resolves the focus/menu chicken-and-egg):

1. A STANDING COARSE MENU of action shapes is scored EVERY cycle for EVERY
   hypothesis (cheap, closed-form EIG through each module's own
   machinery): null, passive quote at touch (each side), deep quote (each
   side), small marketable probe (each side), cancel-refresh of each
   working order, flatten. Its output is the per-hypothesis best marginal
   EIG for the workspace headlines (P9 uses these for salience).
2. A REFINED MENU is generated only for the hypothesis currently in
   focus: a finer grid over (offset_ticks, size_frac, patience) around
   the coarse winner, every candidate a full ``Intent`` with
   ``target_id`` = the focus hypothesis. The realized-IG bookkeeping keys
   on that id, so the grid must NEVER retarget (pinned by test).

THE NULL ACTION IS A FIRST-CLASS CANDIDATE (INV-4): its ``ProbeSpec`` is
one step of purely passive market evolution, scored through the SAME
modules and the SAME ``eig_nats`` machinery as every probe — the market
moves and teaches without being poked, so ``EIG_null > 0`` in an active
market. Its intent carries ``commitment = 0.0`` EXACTLY (standing ruling,
DESIGN.md) so logs are unambiguous; probe intents carry full commitment
(1.0 >= NULL_THRESHOLD) or they compile to nothing and fail the
motor-legality gate.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from topos.contracts.beliefs import ProbeSpec
from topos.contracts.intent import (
    FAIR_VALUE,
    REGIME,
    HypothesisId,
    Intent,
    flatten_intent,
)
from topos.contracts.workspace import SelfStateCognitive, WorldSummary

from topos.proposer.config import (
    DEEP_OFFSET_TICKS,
    REFINED_OFFSET_STEPS,
    REFINED_PATIENCE_GRID,
    REFINED_SIZE_FACTORS,
)


def null_intent(target_id: HypothesisId) -> Intent:
    """The null action: observe, place nothing.

    ``commitment`` is 0.0 EXACTLY — not merely below ``NULL_THRESHOLD`` —
    per the standing ruling recorded in DESIGN.md, so a logged null is
    unambiguous. ``target_id`` is bookkeeping only (which question the
    watching is for); the null compiles to no messages and exercises no
    bucket, so no module's scoring reads it. REGIME never appears as an
    ``Intent.target_id`` (adjudication A3): callers scoring the regime
    tracker pass a substitute bookkeeping id (see ``Proposer``).
    """
    if target_id == REGIME:
        raise ValueError(
            "REGIME never appears as an Intent.target_id (adjudication A3); "
            f"use a substitute bookkeeping id, e.g. {FAIR_VALUE!r}"
        )
    return Intent(
        side=0.0,
        offset_ticks=0.0,
        size_frac=0.0,
        patience=1.0,
        target_id=target_id,
        commitment=0.0,
    )


@dataclass(frozen=True)
class ProbeShape:
    """One action shape: intent parameters without a target hypothesis.

    The coarse menu is a set of SHAPES; stage 1 scores every shape through
    every hypothesis module (the same physical action is a different
    experiment for each question), and stage 2 refines the winning shape
    into full intents targeting the focus.
    """

    name: str
    side: float
    offset_ticks: float
    size_frac: float
    patience: float


def intent_for(shape: ProbeShape, target_id: HypothesisId) -> Intent:
    """A fully committed probe intent for one shape and one hypothesis."""
    return Intent(
        side=shape.side,
        offset_ticks=shape.offset_ticks,
        size_frac=shape.size_frac,
        patience=shape.patience,
        target_id=target_id,
        commitment=1.0,
    )


def coarse_shapes(
    world: WorldSummary,
    cognitive: SelfStateCognitive,
    size_budget_lots: int,
) -> tuple[ProbeShape, ...]:
    """The standing coarse menu (minus the null, which is built directly).

    Encodings (structural, never tuned):

    * touch quotes sit at the own-side best (offset = half the spread —
      the ``Intent`` contract measures offsets from the mid);
    * deep quotes sit at the shallowest "deep"-band price
      (half-spread + ``DEEP_OFFSET_TICKS``, the band edge the flow and
      fill models share);
    * the small marketable probe crosses to the opposite best
      (offset = -half-spread) at ONE lot — the size quantum, the smallest
      intervention that still exercises the aggression channel;
    * cancel-refresh re-quotes a working order's remaining size at the
      current touch with patience 0 (the motor's immediate
      cancel-replace tempo);
    * flatten reuses the distinguished constructor's own parameters.

    Quote shapes use patience 1.0 so scoring them never bundles staleness
    cancels of unrelated working orders into the probe.
    """
    if size_budget_lots < 1:
        raise ValueError(f"size_budget_lots must be >= 1, got {size_budget_lots}")
    half_spread = 0.5 * world.spread_ticks
    one_lot = min(1.0, 1.0 / size_budget_lots)
    shapes = [
        ProbeShape("touch_bid", +1.0, half_spread, 1.0, 1.0),
        ProbeShape("touch_ask", -1.0, half_spread, 1.0, 1.0),
        ProbeShape("deep_bid", +1.0, half_spread + DEEP_OFFSET_TICKS, 1.0, 1.0),
        ProbeShape("deep_ask", -1.0, half_spread + DEEP_OFFSET_TICKS, 1.0, 1.0),
        ProbeShape("market_buy", +1.0, -half_spread, one_lot, 1.0),
        ProbeShape("market_sell", -1.0, -half_spread, one_lot, 1.0),
    ]
    for view in cognitive.working_orders:
        if view.size_lots_remaining <= 0:
            continue
        shapes.append(
            ProbeShape(
                name=f"cancel_refresh_{view.order_id}",
                side=float(view.side.value),
                offset_ticks=half_spread,
                size_frac=min(1.0, view.size_lots_remaining / size_budget_lots),
                patience=0.0,
            )
        )
    if cognitive.inventory_lots != 0:
        flat = flatten_intent(cognitive.inventory_lots)
        shapes.append(
            ProbeShape(
                name="flatten",
                side=flat.side,
                offset_ticks=flat.offset_ticks,
                size_frac=flat.size_frac,
                patience=flat.patience,
            )
        )
    return tuple(shapes)


def refined_shapes(
    winner: ProbeShape, size_budget_lots: int
) -> tuple[ProbeShape, ...]:
    """The refined menu: a finer grid around the coarse winner.

    Grid = (offset +/- one tick) x (size halved/kept/doubled, clipped) x
    (patience endpoints and midpoint) — every axis at a resolution the
    architecture already lives on (see ``topos.proposer.config``). Sizes
    that would compile to zero lots are dropped (they could never pass
    the motor-legality gate). The winner itself sits at the grid center.
    """
    shapes: list[ProbeShape] = []
    seen: set[tuple[float, float, float]] = set()
    for offset_step in REFINED_OFFSET_STEPS:
        offset = winner.offset_ticks + offset_step
        for factor in REFINED_SIZE_FACTORS:
            size_frac = min(1.0, winner.size_frac * factor)
            if round(size_frac * size_budget_lots) < 1:
                continue
            for patience in REFINED_PATIENCE_GRID:
                key = (round(offset, 9), round(size_frac, 9), round(patience, 9))
                if key in seen:
                    continue
                seen.add(key)
                shapes.append(
                    ProbeShape(
                        name=(
                            f"refined(offset={offset:g},size={size_frac:g},"
                            f"patience={patience:g})"
                        ),
                        side=winner.side,
                        offset_ticks=offset,
                        size_frac=size_frac,
                        patience=patience,
                    )
                )
    return tuple(shapes)


@dataclass(frozen=True)
class Candidate:
    """One scored candidate: an experiment (or the null, or flatten) with
    its information account and its self-consequences attached.

    The self-consequences (``self_entropy_nats``, ``predicted_distances``,
    ``message_cost``) are attached WITHOUT being scalarized into any
    score: there is deliberately no ``EIG - lambda*entropy - mu*cost``
    quantity anywhere — that would be shaping a maximand through the back
    door. They enter selection only through the hard gates and the
    lexicographic order (``topos.proposer.selection``).
    """

    kind: str
    """Shape name, for the interpretability log ("null", "flatten",
    "touch_bid", "refined(...)", ...)."""
    probe: ProbeSpec
    eig_nats: float
    """EIG_target(candidate), from the target module's own machinery."""
    null_eig_nats: float
    """EIG_target(null): the same module, same machinery, same horizon —
    one step of purely passive market evolution (INV-4)."""
    marginal_eig_nats: float
    """eig_nats - null_eig_nats. Only strictly positive marginals are
    ever eligible to beat the null."""
    self_entropy_nats: float
    """Reflexive self-uncertainty of the candidate, compiled by
    ``SelfTrajectory`` from the same posteriors used everywhere else."""
    predicted_distances: Mapping[str, float]
    """Expected post-action homeostat distances (one-step self-forecast);
    variables not predictable from the cognitive view are carried forward
    at their current values by the projector."""
    within_soft_confidence: float
    """Probability the one-step forecast keeps every predicted distance
    inside the soft bands (the (a)-gate compares it to 1 - GATE_DELTA)."""
    message_cost: int
    """Exchange messages the motor layer would emit for this intent."""
    motor_legal: bool
    vetoed: bool
    gates_passed: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "predicted_distances",
            MappingProxyType(dict(self.predicted_distances)),
        )

    @property
    def intent(self) -> Intent:
        return self.probe.intent


@dataclass(frozen=True)
class Proposal:
    """One cycle's full proposer output.

    ``best_marginal_eig_nats`` is the stage-1 product (per-hypothesis
    headline input for P9 salience); ``candidates`` is the stage-2 menu
    for the focus (always containing the null, plus flatten when there is
    inventory to flatten); ``selected`` is the winner under the exported
    lexicographic rule.
    """

    focus: HypothesisId | None
    null_eig_nats: Mapping[HypothesisId, float]
    best_marginal_eig_nats: Mapping[HypothesisId, float]
    candidates: tuple[Candidate, ...]
    selected: Candidate

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "null_eig_nats", MappingProxyType(dict(self.null_eig_nats))
        )
        object.__setattr__(
            self,
            "best_marginal_eig_nats",
            MappingProxyType(dict(self.best_marginal_eig_nats)),
        )
