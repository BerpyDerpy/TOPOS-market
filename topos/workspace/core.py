"""The Workspace: bounded blackboard, competition, arbitration, record.

One call to ``cycle()`` is steps 2-6 of the cognitive cycle (appraise ->
compete -> broadcast -> propose -> ignite intent), and its product — the
``WorkspaceRecord`` — is the system's interpretability contract:
completeness over convenience, emitted every cycle BEFORE the action is
submitted, null cycles included (null cycles are data, not dead air).

INV-5: arbitration receives ``SelfStateCognitive`` only. The homeostat's
byproducts (drive magnitudes, veto flags, the corrective intent, the
distance projector) arrive as exported values — this package never
imports ``topos.drives``, mirroring the proposer's boundary.

INV-7: consequence-weights come from ``ModuleRegistry.centrality_weights()``
exactly once at startup; the constructor asserts they cover every known
hypothesis id, and every cycle re-asserts they have not changed. There is
no code path from any outcome statistic to a weight.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from topos.contracts.beliefs import BeliefModule
from topos.contracts.intent import (
    FAIR_VALUE,
    KNOWN_HYPOTHESIS_IDS,
    HypothesisId,
    Intent,
)
from topos.contracts.registry import ModuleRegistry
from topos.contracts.workspace import (
    Focus,
    Headline,
    SelfStateCognitive,
    WorkspaceRecord,
    WorldSummary,
)
from topos.motor.compiler import compile as compile_intent
from topos.motor.config import MotorConfig
from topos.proposer import (
    Candidate,
    DistanceProjector,
    Proposer,
    book_from_summary,
    null_intent,
)

from topos.workspace.broadcast import FocusConsumer, broadcast_focus, validate_consumers
from topos.workspace.config import GAMMA, K_HEADLINES, S_MIN
from topos.workspace.salience import SalienceBid, compete, hypothesis_salience


class WeightsIntegrityError(RuntimeError):
    """INV-7 violated: consequence-weights differ from the startup snapshot."""


class CoalitionError(RuntimeError):
    """A committed probe reached ignition without its coalition.

    The proposer's exported selection rule guarantees that a committed
    experiment carries a strictly positive marginal EIG AND passed the
    self-model gates. The arbiter re-verifies rather than re-implements:
    seeing this error means the selection rule (or a stand-in for it) is
    broken, not that the arbiter should have patched over it.
    """


class Workspace:
    """Bounded blackboard, salience competition, arbiter, broadcast (P9).

    Constructed once at startup; ``cycle()`` runs once per engine step.

    ``modules`` maps each hypothesis id to its belief module (the
    headline sources). ``consumers`` are the objects that receive the
    broadcast focus each cycle — register every module that implements
    ``condition_on_focus`` (see ``topos.workspace.broadcast`` for the
    pattern P12 must follow).
    """

    def __init__(
        self,
        *,
        registry: ModuleRegistry,
        proposer: Proposer,
        modules: Mapping[HypothesisId, BeliefModule],
        motor_cfg: MotorConfig,
        consumers: Sequence[object] = (),
        k_headlines: int = K_HEADLINES,
        gamma: float = GAMMA,
        s_min: float = S_MIN,
    ) -> None:
        if k_headlines < 1:
            raise ValueError(f"k_headlines must be >= 1, got {k_headlines}")
        if gamma < 0.0:
            raise ValueError(f"gamma must be >= 0, got {gamma}")
        if s_min < 0.0:
            raise ValueError(f"s_min must be >= 0, got {s_min}")
        if not modules:
            raise ValueError("workspace needs at least one hypothesis module")
        # Computed ONCE at startup, from declared reads/writes only
        # (INV-7). The registry freezes on this call; the snapshot below
        # is what every later cycle is checked against.
        weights = dict(registry.centrality_weights())
        missing_known = [h for h in KNOWN_HYPOTHESIS_IDS if h not in weights]
        if missing_known:
            raise ValueError(
                "centrality weights must cover every id in "
                f"KNOWN_HYPOTHESIS_IDS; missing {missing_known!r} — each "
                "hypothesis-owning module registers under its hypothesis_id "
                "before the registry freezes (INV-7)"
            )
        missing_modules = [h for h in modules if h not in weights]
        if missing_modules:
            raise ValueError(
                "every workspace module needs a consequence-weight; the "
                f"registry has none for {missing_modules!r}"
            )
        self._registry = registry
        self._weights: dict[HypothesisId, float] = weights
        self._proposer = proposer
        self._modules: dict[HypothesisId, BeliefModule] = dict(modules)
        self._motor_cfg = motor_cfg
        self._consumers: tuple[FocusConsumer, ...] = validate_consumers(consumers)
        self._k_headlines = k_headlines
        self._gamma = gamma
        self._s_min = s_min

    @property
    def weights(self) -> Mapping[HypothesisId, float]:
        """The startup consequence-weight snapshot (structural, fixed)."""
        return dict(self._weights)

    # -- the cycle -----------------------------------------------------------

    def cycle(
        self,
        *,
        step: int,
        world: WorldSummary,
        cognitive: SelfStateCognitive,
        drives: Mapping[str, float],
        vetoes: Mapping[str, bool],
        corrective_intent: Intent | None,
        projector: DistanceProjector,
    ) -> WorkspaceRecord:
        """Appraise -> compete -> broadcast -> propose -> ignite.

        ``drives``, ``vetoes`` and ``corrective_intent`` are the
        homeostat's exported values for this cycle (INV-5: only exported
        values cross this boundary, never the account state behind them).
        The record is returned BEFORE any action is taken: the caller
        submits ``record.compiled_messages`` after logging the record.
        """
        self._assert_weights_static()

        # Stage 1 of the proposer runs focus-free: the coarse menu's
        # per-hypothesis best marginal EIG is the headline input the
        # competition needs before a focus can exist.
        coarse = self._proposer.propose(
            world=world,
            cognitive=cognitive,
            focus=None,
            vetoes=vetoes,
            projector=projector,
        )

        # Appraise: one headline per hypothesis, one salience bid each.
        headlines: dict[HypothesisId, Headline] = {}
        saliences: dict[HypothesisId, float] = {}
        for hypothesis, module in self._modules.items():
            forecast = module.predict()
            marginal = coarse.best_marginal_eig_nats.get(hypothesis, 0.0)
            surprise = module.surprise_z()
            headlines[hypothesis] = Headline(
                hypothesis_id=hypothesis,
                forecast_mean=forecast.mean,
                forecast_var=forecast.variance,
                epistemic_entropy_nats=module.posterior_entropy_nats(),
                best_marginal_eig_nats=marginal,
                last_surprise_z=surprise,
            )
            saliences[hypothesis] = hypothesis_salience(
                self._weights[hypothesis], marginal, surprise, gamma=self._gamma
            )

        # Compete: hypotheses and drives in the same arena, same units.
        bids = [
            SalienceBid(bid_id=h, salience=s, is_homeostatic=False)
            for h, s in saliences.items()
        ]
        bids.extend(
            SalienceBid(bid_id=name, salience=drive, is_homeostatic=True)
            for name, drive in drives.items()
            if drive > 0.0
        )
        focus = compete(bids, s_min=self._s_min)

        # Broadcast: condition every registered module on the focus
        # BEFORE any further work (the refined menu below, the next
        # update outside).
        broadcast_focus(self._consumers, focus)

        # Ignite: arbitration.
        intent, eig_promised = self._arbitrate(
            focus=focus,
            world=world,
            cognitive=cognitive,
            vetoes=vetoes,
            corrective_intent=corrective_intent,
            projector=projector,
        )

        # The bounded blackboard: top-K headlines by salience, the rest
        # genuinely absent (capacity IS the attention mechanism).
        ranked = sorted(
            headlines.values(),
            key=lambda h: (-saliences[h.hypothesis_id], h.hypothesis_id),
        )
        # Intent and compiled messages logged side by side (INV-8): the
        # motor is a pure function, so compiling here is exact
        # forecasting against the broadcast summary's book, not action.
        book_bids, book_asks = book_from_summary(world)
        messages = compile_intent(
            intent,
            book_bids,
            book_asks,
            cognitive.working_orders,
            vetoes,
            self._motor_cfg,
            cognitive.inventory_lots,
        )
        return WorkspaceRecord(
            step=step,
            world_summary=world,
            headlines=tuple(ranked[: self._k_headlines]),
            self_state=cognitive,
            focus=focus,
            intent=intent,
            eig_promised_nats=eig_promised,
            compiled_messages=messages,
        )

    # -- internals -----------------------------------------------------------

    def _assert_weights_static(self) -> None:
        """INV-7, checked every cycle: the weights the registry answers
        with now must be the startup snapshot, bit for bit."""
        current = dict(self._registry.centrality_weights())
        if current != self._weights:
            raise WeightsIntegrityError(
                "consequence-weights changed across cycles; INV-7 requires "
                "them computed once at startup from the declared dependency "
                f"graph only (was {self._weights!r}, now {current!r})"
            )

    def _arbitrate(
        self,
        *,
        focus: Focus | None,
        world: WorldSummary,
        cognitive: SelfStateCognitive,
        vetoes: Mapping[str, bool],
        corrective_intent: Intent | None,
        projector: DistanceProjector,
    ) -> tuple[Intent, float | None]:
        """One intent per cycle, with the EIG it promises.

        * No focus: the intent is null — a quiet mind watches. The null
          carries FAIR_VALUE as its bookkeeping target (DESIGN.md item
          26: the archetypal hypothesis whose information rides the null).
        * Homeostatic focus: the homeostat's corrective intent, verbatim
          — sized by P7, targeted at self_trajectory by construction. No
          proposer refinement: a drive is not a question. A drive focus
          with no corrective available (e.g. the message budget, whose
          correction is simply to stop sending) resolves to the null.
        * Hypothesis focus: the proposer's refined menu for that
          hypothesis, selected by its exported lexicographic rule — the
          rule is applied inside ``propose``; nothing is re-ranked here.

        ``eig_promised_nats`` is None for a null intent, the selected
        candidate's total EIG through the target module for a committed
        probe (the number realized information gain is compared against,
        INV-10), and 0.0 for corrective/flatten intents, which promise
        action, not information.
        """
        if focus is None:
            return null_intent(FAIR_VALUE), None

        if focus.is_homeostatic:
            intent = corrective_intent
            if intent is None:
                intent = null_intent(FAIR_VALUE)
            return intent, (None if intent.is_null else 0.0)

        refined = self._proposer.propose(
            world=world,
            cognitive=cognitive,
            focus=focus.hypothesis_id,
            vetoes=vetoes,
            projector=projector,
        )
        selected = refined.selected
        self._verify_coalition(selected)
        intent = selected.intent
        if intent.is_null:
            return intent, None
        if intent.is_flatten:
            return intent, 0.0
        return intent, selected.eig_nats

    @staticmethod
    def _verify_coalition(selected: Candidate) -> None:
        """Coalition requirement, restated from the proposer gates: a
        committed probe ignites only with a strictly positive marginal
        EIG AND self-model endorsement (gates passed). The exported
        selection rule already guarantees this; the arbiter verifies
        instead of re-implementing, and refuses to ignite on a violation."""
        intent = selected.intent
        if intent.is_null or intent.is_flatten:
            return
        if not selected.gates_passed:
            raise CoalitionError(
                f"selected candidate {selected.kind!r} did not pass the "
                "self-model gates; a committed probe needs its coalition"
            )
        if not selected.marginal_eig_nats > 0.0:
            raise CoalitionError(
                f"selected candidate {selected.kind!r} has non-positive "
                f"marginal EIG ({selected.marginal_eig_nats}); a committed "
                "probe needs its coalition"
            )
