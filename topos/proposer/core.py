"""The Proposer: candidate probes, marginal-EIG scoring, self-consequences.

Answers, once per cycle: "which experiment, if any, is worth paying for,
given that WATCHING IS FREE AND INFORMATIVE?"

Scoring contract (INV-3, INV-4):

* ``EIG_h(candidate) = modules[h].eig_nats(ProbeSpec(intent, horizon))`` —
  every number comes from the target module's own parameter-posterior
  machinery, never from anything computed here.
* The null action is scored through the SAME call with the null intent:
  one step of purely passive market evolution. World predictors update
  from passive observation (flow intensities, fair value, queue
  posteriors), so ``EIG_null > 0`` in an active market; self-model
  hypotheses answer 0 for the null because their information must be
  bought by acting. Hardcoding the null's EIG to 0 — or computing it from
  any other observation model — would silently reinstate constant
  trading, and is pinned against by test_null_positive.
* ``score(candidate) = EIG_target(candidate) - EIG_target(null)``. Only
  strictly positive marginal scores are eligible to beat the null.

INV-5: this module consumes ``SelfStateCognitive`` only; homeostat
byproducts arrive as exported values (veto flags, distance projector).
"""

from __future__ import annotations

from collections.abc import Mapping

from topos.contracts.beliefs import BeliefModule, ProbeSpec
from topos.contracts.intent import (
    FAIR_VALUE,
    REGIME,
    SELF_TRAJECTORY,
    HypothesisId,
    Intent,
    flatten_intent,
)
from topos.contracts.workspace import SelfStateCognitive, WorldSummary
from topos.motor.config import MotorConfig
from topos.selfmodel.self_trajectory import SelfTrajectory

from topos.proposer.candidates import (
    Candidate,
    ProbeShape,
    Proposal,
    coarse_shapes,
    intent_for,
    null_intent,
    refined_shapes,
)
from topos.proposer.gates import DistanceProjector, evaluate_gates
from topos.proposer.selection import select_candidate


class Proposer:
    """Two-stage experiment proposer (P8).

    Stage 1 (every cycle, every hypothesis): score the standing coarse
    menu, publish per-hypothesis best marginal EIG for the headlines.
    Stage 2 (focus only): refine the coarse winner into a grid of full
    intents targeting the focus, attach self-consequences, select via the
    exported lexicographic rule.
    """

    def __init__(
        self,
        *,
        modules: Mapping[HypothesisId, BeliefModule],
        trajectory: SelfTrajectory,
        motor_cfg: MotorConfig,
        probe_horizon_steps: int,
    ) -> None:
        """``probe_horizon_steps`` is the single horizon every ProbeSpec
        carries — P12 wires the fill model's horizon, the only horizon the
        fill posteriors answer without extrapolation (the same reasoning
        as the trajectory compiler's default). Marginal scores compare a
        candidate and the null through the same module at the same
        horizon, so the choice cancels out of every comparison that
        matters."""
        if probe_horizon_steps < 1:
            raise ValueError(
                f"probe_horizon_steps must be >= 1, got {probe_horizon_steps}"
            )
        if SELF_TRAJECTORY in modules:
            raise ValueError(
                "self_trajectory is a forecast compiler, never a probeable "
                "hypothesis (adjudication A2); it does not belong in the "
                "proposer's module map"
            )
        self._modules: dict[HypothesisId, BeliefModule] = dict(modules)
        self._trajectory = trajectory
        self._motor_cfg = motor_cfg
        self._horizon = probe_horizon_steps

    def propose(
        self,
        *,
        world: WorldSummary,
        cognitive: SelfStateCognitive,
        focus: HypothesisId | None,
        vetoes: Mapping[str, bool],
        projector: DistanceProjector,
    ) -> Proposal:
        """One cycle: coarse marginals for every hypothesis, a refined
        menu for the focus, and the selected candidate."""
        self._trajectory.begin_cycle(cognitive, world)
        bookkeeping_target = self._bookkeeping_target(focus)
        the_null_intent = null_intent(bookkeeping_target)
        null_spec = ProbeSpec(intent=the_null_intent, horizon_steps=self._horizon)

        # EIG of pure watching, per hypothesis — same modules, same
        # machinery as every probe (INV-4).
        null_eig = {
            hypothesis: module.eig_nats(null_spec)
            for hypothesis, module in self._modules.items()
        }

        # Stage 1: the standing coarse menu, every hypothesis.
        shapes = coarse_shapes(world, cognitive, self._motor_cfg.size_budget_lots)
        best_marginal: dict[HypothesisId, float] = {}
        best_shape: dict[HypothesisId, ProbeShape | None] = {}
        for hypothesis, module in self._modules.items():
            if hypothesis == REGIME:
                # Passive-only (adjudication A3): no probe may target it,
                # so its information rides the null and its headline
                # marginal is 0 by construction.
                best_marginal[hypothesis] = 0.0
                best_shape[hypothesis] = None
                continue
            top = 0.0
            winner: ProbeShape | None = None
            for shape in shapes:
                spec = ProbeSpec(
                    intent=intent_for(shape, hypothesis),
                    horizon_steps=self._horizon,
                )
                marginal = module.eig_nats(spec) - null_eig[hypothesis]
                if marginal > top:
                    top, winner = marginal, shape
            best_marginal[hypothesis] = top
            best_shape[hypothesis] = winner

        # Stage 2: the refined menu for the focus, plus the null and the
        # flatten fallback, all with self-consequences attached.
        focus_module = self._modules.get(focus) if focus is not None else None
        focus_null_eig = (
            null_eig[focus] if focus is not None and focus in null_eig else 0.0
        )
        candidates: list[Candidate] = [
            self._build_candidate(
                kind="null",
                intent=the_null_intent,
                eig=focus_null_eig,
                null_eig=focus_null_eig,
                marginal=0.0,
                world=world,
                cognitive=cognitive,
                vetoes=vetoes,
                projector=projector,
            )
        ]
        if cognitive.inventory_lots != 0:
            candidates.append(
                self._build_candidate(
                    kind="flatten",
                    intent=flatten_intent(cognitive.inventory_lots),
                    # Flatten never competes on EIG (see selection module):
                    # it carries no experiment bookkeeping.
                    eig=0.0,
                    null_eig=focus_null_eig,
                    marginal=0.0,
                    world=world,
                    cognitive=cognitive,
                    vetoes=vetoes,
                    projector=projector,
                )
            )
        if focus is not None and focus != REGIME and focus_module is not None:
            coarse_winner = best_shape[focus]
            if coarse_winner is not None:
                for shape in refined_shapes(
                    coarse_winner, self._motor_cfg.size_budget_lots
                ):
                    # target_id = focus, never anything else: the
                    # realized-IG bookkeeping keys on it.
                    intent = intent_for(shape, focus)
                    eig = focus_module.eig_nats(
                        ProbeSpec(intent=intent, horizon_steps=self._horizon)
                    )
                    candidates.append(
                        self._build_candidate(
                            kind=shape.name,
                            intent=intent,
                            eig=eig,
                            null_eig=focus_null_eig,
                            marginal=eig - focus_null_eig,
                            world=world,
                            cognitive=cognitive,
                            vetoes=vetoes,
                            projector=projector,
                        )
                    )

        selected = select_candidate(candidates, cognitive.drive_distances)
        return Proposal(
            focus=focus,
            null_eig_nats=null_eig,
            best_marginal_eig_nats=best_marginal,
            candidates=tuple(candidates),
            selected=selected,
        )

    # -- internals ----------------------------------------------------------

    def _bookkeeping_target(self, focus: HypothesisId | None) -> HypothesisId:
        """The id the null intent carries — pure bookkeeping ("which
        question the watching is for"). The focus when it is a probeable
        module; FAIR_VALUE (the archetypal passive world hypothesis whose
        information rides the null) when the focus is absent or
        passive-only, since REGIME never appears on an Intent (A3)."""
        if focus is not None and focus in self._modules and focus != REGIME:
            return focus
        return FAIR_VALUE

    def _build_candidate(
        self,
        *,
        kind: str,
        intent: Intent,
        eig: float,
        null_eig: float,
        marginal: float,
        world: WorldSummary,
        cognitive: SelfStateCognitive,
        vetoes: Mapping[str, bool],
        projector: DistanceProjector,
    ) -> Candidate:
        report = evaluate_gates(
            intent=intent,
            world=world,
            cognitive=cognitive,
            vetoes=vetoes,
            motor_cfg=self._motor_cfg,
            trajectory=self._trajectory,
            projector=projector,
        )
        return Candidate(
            kind=kind,
            probe=ProbeSpec(intent=intent, horizon_steps=self._horizon),
            eig_nats=eig,
            null_eig_nats=null_eig,
            marginal_eig_nats=marginal,
            self_entropy_nats=self._trajectory.self_entropy_nats(intent),
            predicted_distances=report.predicted_distances,
            within_soft_confidence=report.within_soft_confidence,
            message_cost=report.message_cost,
            motor_legal=report.motor_legal,
            vetoed=report.vetoed,
            gates_passed=report.passed,
        )
