"""Arbitration: who gets asked what, and the coalition requirement.

* A homeostatic focus takes the homeostat's corrective intent verbatim —
  the proposer is never asked for a refined menu (a drive is not a
  question).
* A hypothesis focus asks the proposer for the refined menu and takes
  the exported lexicographic rule's winner IDENTICALLY — the arbiter
  re-ranks nothing.
* A committed probe without its coalition (positive marginal EIG AND
  gates passed) is a bug the arbiter refuses to ignite.
"""

from __future__ import annotations

from types import MappingProxyType

import pytest

from tests.proposer.conftest import mk_candidate
from tests.workspace.conftest import (
    FakeModule,
    FakeTrajectory,
    make_registry,
    make_workspace,
    run_cycle,
)
from topos.contracts.intent import FAIR_VALUE, HypothesisId, flatten_intent
from topos.motor.config import MotorConfig
from topos.proposer import Proposal, Proposer
from topos.workspace import CoalitionError


class SpyProposer:
    """Delegates to a real Proposer, recording every focus it is asked."""

    def __init__(self, inner: Proposer) -> None:
        self.inner = inner
        self.focus_calls: list[HypothesisId | None] = []
        self.last_proposal: Proposal | None = None

    def propose(self, **kwargs: object) -> Proposal:
        self.focus_calls.append(kwargs["focus"])  # type: ignore[arg-type]
        proposal = self.inner.propose(**kwargs)  # type: ignore[arg-type]
        self.last_proposal = proposal
        return proposal


def _spy_workspace(modules: dict[str, FakeModule]):  # type: ignore[no-untyped-def]
    spy = SpyProposer(
        Proposer(
            modules=modules,
            trajectory=FakeTrajectory(),  # type: ignore[arg-type]
            motor_cfg=MotorConfig(size_budget_lots=4),
            probe_horizon_steps=2,
        )
    )
    workspace = make_workspace(
        modules,
        registry=make_registry(tuple(modules)),
        proposer=spy,  # type: ignore[arg-type]
    )
    return workspace, spy


def test_homeostat_focus_skips_proposer_refinement() -> None:
    modules = {"hyp_a": FakeModule(hypothesis_id="hyp_a", probe_gain=0.1)}
    workspace, spy = _spy_workspace(modules)
    record = run_cycle(
        workspace,
        inventory=5,
        drives={"inventory": 50.0},
        corrective_intent=flatten_intent(5),
    )
    # Only the focus-free stage-1 call happened: no refined menu was
    # requested for anything.
    assert spy.focus_calls == [None]
    assert record.focus is not None and record.focus.is_homeostatic
    assert record.intent == flatten_intent(5)


def test_hypothesis_focus_applies_selection_verbatim() -> None:
    modules = {
        "hyp_a": FakeModule(hypothesis_id="hyp_a", probe_gain=0.3),
        "hyp_b": FakeModule(hypothesis_id="hyp_b", probe_gain=0.1),
    }
    workspace, spy = _spy_workspace(modules)
    record = run_cycle(workspace)
    # Stage 1 focus-free, then exactly one refined request for the winner.
    assert spy.focus_calls == [None, "hyp_a"]
    assert spy.last_proposal is not None
    selected = spy.last_proposal.selected
    # The ignited intent IS the exported rule's winner — same object, no
    # re-ranking, no simplification.
    assert record.intent is selected.intent
    assert record.eig_promised_nats == selected.eig_nats
    assert selected.gates_passed and selected.marginal_eig_nats > 0.0


def test_drive_focus_without_corrective_falls_back_to_null() -> None:
    """A drive whose correction the homeostat cannot express as an order
    (e.g. the message budget: the correction is to stop sending)
    resolves to the null — which is exactly the corrective behavior."""
    modules = {"hyp_a": FakeModule(hypothesis_id="hyp_a", probe_gain=0.1)}
    workspace, spy = _spy_workspace(modules)
    record = run_cycle(
        workspace,
        drives={"message_budget": 50.0},
        corrective_intent=None,
    )
    assert record.focus is not None and record.focus.is_homeostatic
    assert record.focus.hypothesis_id == "message_budget"
    assert record.intent is not None and record.intent.is_null
    assert record.eig_promised_nats is None
    assert spy.focus_calls == [None]


def test_coalition_violation_refuses_to_ignite() -> None:
    """A committed probe that failed the gates (or lost its marginal)
    must never ignite; the arbiter raises rather than 'fixes'."""

    class BrokenSelectionProposer:
        """Emits a committed winner that never had its coalition."""

        def __init__(self, marginal: float, gates: bool) -> None:
            self.bad = mk_candidate(marginal=marginal, gates=gates)
            self.null = mk_candidate(null=True)

        def propose(self, **kwargs: object) -> Proposal:
            return Proposal(
                focus=kwargs["focus"],  # type: ignore[arg-type]
                null_eig_nats=MappingProxyType({"hyp_a": 0.05}),
                best_marginal_eig_nats=MappingProxyType({"hyp_a": 0.2}),
                candidates=(self.null, self.bad),
                selected=self.bad,
            )

    for marginal, gates in ((0.2, False), (0.0, True), (-0.1, True)):
        modules = {"hyp_a": FakeModule(hypothesis_id="hyp_a")}
        workspace = make_workspace(
            modules,
            registry=make_registry(("hyp_a",)),
            proposer=BrokenSelectionProposer(marginal, gates),  # type: ignore[arg-type]
        )
        with pytest.raises(CoalitionError):
            run_cycle(workspace)


def test_null_intent_bookkeeping_target_is_fair_value() -> None:
    """With no focus (and on drive-focus fallbacks) the null carries the
    standing FAIR_VALUE bookkeeping target — DESIGN.md item 26."""
    modules = {"hyp_a": FakeModule(hypothesis_id="hyp_a", probe_gain=0.0)}
    workspace, _ = _spy_workspace(modules)
    record = run_cycle(workspace)
    assert record.intent is not None and record.intent.is_null
    assert record.intent.target_id == FAIR_VALUE
