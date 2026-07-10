"""The bounded blackboard: capacity and per-cycle record completeness.

Capacity IS the attention mechanism: with more hypotheses than slots,
the lowest-salience headlines are genuinely absent from the record — not
summarized, not tucked into a side channel. And the record itself is the
interpretability contract: every cycle yields one schema-valid
``WorkspaceRecord``, null cycles included.
"""

from __future__ import annotations

import math

from tests.workspace.conftest import (
    FakeModule,
    make_registry,
    make_workspace,
    run_cycle,
)
from topos.contracts.intent import FAIR_VALUE, Intent, flatten_intent
from topos.contracts.market import Cancel, PlaceLimit
from topos.contracts.workspace import (
    Focus,
    Headline,
    SelfStateCognitive,
    WorkspaceRecord,
    WorldSummary,
)
from topos.workspace import K_HEADLINES


def test_capacity() -> None:
    """K+3 hypotheses, exactly K headlines broadcast, lowest-salience
    excluded — and the excluded three appear nowhere in the record."""
    k = K_HEADLINES
    names = [f"hyp_{chr(ord('a') + i)}" for i in range(k + 3)]
    # Distinct marginal EIGs => distinct saliences (uniform weights).
    modules = {
        name: FakeModule(hypothesis_id=name, probe_gain=0.1 * (i + 1))
        for i, name in enumerate(names)
    }
    workspace = make_workspace(modules, registry=make_registry(tuple(names)))
    record = run_cycle(workspace)

    assert len(record.headlines) == k
    broadcast_ids = [h.hypothesis_id for h in record.headlines]
    expected_top = list(reversed(names))[:k]  # highest probe_gain first
    assert broadcast_ids == expected_top

    evicted = set(names) - set(expected_top)
    assert len(evicted) == 3
    assert evicted.isdisjoint(broadcast_ids)
    # The evicted are absent from every part of the record, and the focus
    # (argmax salience) is by construction among the broadcast top-K.
    assert record.focus is not None
    assert record.focus.hypothesis_id == names[-1]
    assert record.focus.hypothesis_id in broadcast_ids
    assert record.intent is not None
    assert record.intent.target_id not in evicted

    # Headlines are ranked by salience, descending: the record shows the
    # competition standings, not registration order.
    gains = [modules[h].probe_gain for h in broadcast_ids]
    assert gains == sorted(gains, reverse=True)


def test_capacity_not_binding_when_fewer_hypotheses() -> None:
    modules = {
        name: FakeModule(hypothesis_id=name, probe_gain=0.1)
        for name in ("hyp_a", "hyp_b")
    }
    workspace = make_workspace(
        modules, registry=make_registry(("hyp_a", "hyp_b"))
    )
    record = run_cycle(workspace)
    assert len(record.headlines) == 2


def _assert_schema_valid(record: WorkspaceRecord, step: int) -> None:
    assert isinstance(record, WorkspaceRecord)
    assert record.step == step
    assert isinstance(record.world_summary, WorldSummary)
    assert isinstance(record.self_state, SelfStateCognitive)

    assert isinstance(record.headlines, tuple)
    assert 1 <= len(record.headlines) <= K_HEADLINES
    for headline in record.headlines:
        assert isinstance(headline, Headline)
        assert isinstance(headline.hypothesis_id, str) and headline.hypothesis_id
        for value in (
            headline.forecast_mean,
            headline.forecast_var,
            headline.epistemic_entropy_nats,
            headline.best_marginal_eig_nats,
            headline.last_surprise_z,
        ):
            assert isinstance(value, float) and not math.isnan(value)
        assert headline.best_marginal_eig_nats >= 0.0

    if record.focus is not None:
        assert isinstance(record.focus, Focus)
        assert record.focus.salience > 0.0
        assert isinstance(record.focus.is_homeostatic, bool)

    # The intent is always explicit — a null cycle records the null
    # intent, never a missing one (null cycles are data, not dead air).
    assert isinstance(record.intent, Intent)
    if record.intent.is_null:
        assert record.eig_promised_nats is None
    else:
        assert isinstance(record.eig_promised_nats, float)
        assert record.eig_promised_nats >= 0.0
        assert not math.isnan(record.eig_promised_nats)

    assert isinstance(record.compiled_messages, tuple)
    for message in record.compiled_messages:
        assert isinstance(message, (PlaceLimit, Cancel))
    if record.intent.is_null:
        assert record.compiled_messages == ()


def test_record_completeness() -> None:
    """Every cycle — quiet, curious, or corrective — yields one complete,
    schema-valid WorkspaceRecord, emitted whether or not anything acts."""
    names = ("hyp_a", "hyp_b", "hyp_c")
    modules = {name: FakeModule(hypothesis_id=name) for name in names}
    workspace = make_workspace(modules, registry=make_registry(names))

    for step in range(45):
        scenario = step % 3
        if scenario == 0:  # quiet: nothing clears ignition
            for module in modules.values():
                module.probe_gain = 0.0001
            record = run_cycle(workspace, step=step)
            assert record.focus is None
            assert record.intent is not None and record.intent.is_null
        elif scenario == 1:  # a hypothesis wins focus and probes
            for i, module in enumerate(modules.values()):
                module.probe_gain = 0.1 * (i + 1)
                module.surprise = float(i)
            record = run_cycle(workspace, step=step)
            assert record.focus is not None
            assert not record.focus.is_homeostatic
        else:  # a drive preempts: corrective intent, no refinement
            for module in modules.values():
                module.probe_gain = 0.1
            record = run_cycle(
                workspace,
                step=step,
                inventory=5,
                drives={"inventory": 8.1},
                corrective_intent=flatten_intent(5),
            )
            assert record.focus is not None and record.focus.is_homeostatic
            assert record.intent is not None and record.intent.is_flatten
        _assert_schema_valid(record, step)


def test_null_cycle_record_carries_bookkeeping_target() -> None:
    """The quiet mind's record is still fully populated: the null intent
    carries the standing bookkeeping target and zero commitment exactly."""
    modules = {"hyp_a": FakeModule(hypothesis_id="hyp_a", probe_gain=0.0)}
    workspace = make_workspace(modules, registry=make_registry(("hyp_a",)))
    record = run_cycle(workspace, step=7)
    assert record.focus is None
    assert record.intent is not None
    assert record.intent.commitment == 0.0
    assert record.intent.target_id == FAIR_VALUE
    assert record.eig_promised_nats is None
    assert len(record.headlines) == 1
