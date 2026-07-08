from __future__ import annotations

import dataclasses

import pytest

from tests.exemplars import contract_exemplars
from topos.contracts.workspace import (
    SelfStateCognitive,
    SelfStateFull,
    WorkspaceRecord,
)


def _full() -> SelfStateFull:
    instance = contract_exemplars()[SelfStateFull]
    assert isinstance(instance, SelfStateFull)
    return instance


def test_cognitive_view_projects_and_strips_account_fields() -> None:
    full = _full()
    view = full.cognitive_view()
    assert isinstance(view, SelfStateCognitive)
    assert view.inventory_lots == full.inventory_lots
    assert view.working_orders == full.working_orders
    assert dict(view.drive_distances) == dict(full.drive_distances)
    view_fields = {field.name for field in dataclasses.fields(view)}
    assert not view_fields & {"realized_pnl", "unrealized_pnl", "gross_exposure"}


def test_drive_distances_are_read_only() -> None:
    view = _full().cognitive_view()
    with pytest.raises(TypeError):
        view.drive_distances["inventory"] = 99.0  # type: ignore[index]


def test_workspace_record_allows_idle_cycles() -> None:
    exemplar = contract_exemplars()[WorkspaceRecord]
    assert isinstance(exemplar, WorkspaceRecord)
    idle = WorkspaceRecord(
        step=exemplar.step,
        world_summary=exemplar.world_summary,
        headlines=exemplar.headlines,
        self_state=exemplar.self_state,
        focus=None,
        intent=None,
        eig_promised_nats=None,
        compiled_messages=(),
    )
    assert idle.focus is None and idle.intent is None
    assert idle.compiled_messages == ()
