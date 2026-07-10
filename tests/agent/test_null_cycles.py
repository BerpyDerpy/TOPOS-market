"""Null-intent cycles still produce complete WorkspaceRecords.

Null cycles are data, not dead air (the record IS the interpretability
story): both null paths — pre-market observations with no book to price
against, and vetoed cycles where the homeostat has suppressed every
placement — must log a record with every field populated and an EXPLICIT
null intent (commitment 0.0 exactly, per the standing ruling).
"""

from __future__ import annotations

from tests.agent.conftest import canned_stream, make_agent
from tests.beliefs.conftest import make_obs
from topos.contracts.workspace import WorkspaceRecord


def _assert_complete_null_record(record: WorkspaceRecord, step: int) -> None:
    assert isinstance(record, WorkspaceRecord)
    assert record.step == step
    assert record.world_summary is not None
    assert record.headlines, "headlines must be populated on null cycles too"
    assert record.self_state is not None
    assert record.intent is not None, "a quiet cycle logs an explicit null"
    assert record.intent.is_null
    assert record.intent.commitment == 0.0
    assert record.eig_promised_nats is None
    assert record.compiled_messages == ()


def test_premarket_null_cycles_are_fully_logged() -> None:
    agent = make_agent()
    for step in range(3):
        empty = make_obs(step, [], [])
        record, action = agent.cycle(empty)
        _assert_complete_null_record(record, step)
        assert action is None
        assert record.focus is None

    assert len(agent.records) == 3
    assert agent.ledger.pending is None
    assert agent.ledger.log == ()


def test_vetoed_market_null_cycles_are_fully_logged() -> None:
    agent = make_agent()
    # Breach the message budget's hard bound before the market opens: the
    # veto suppresses every placement, so gates fail, so the selection
    # rule's corrective fallback resolves to the null (no inventory to
    # flatten), while the message drive seizes the workspace.
    window = agent.config.homeostat.message_window_steps
    for _ in range(window):
        agent.homeostat.record_messages(1)
    hard = agent.config.homeostat.message_budget.hard
    assert agent.homeostat.rolling_message_count >= hard

    stream = canned_stream(3)
    for step, obs in enumerate(stream):
        record, action = agent.cycle(obs)
        _assert_complete_null_record(record, step)
        assert action is None
        # The message drive won the competition; its correction is to
        # stop sending, which resolves to the null.
        assert record.focus is not None
        assert record.focus.is_homeostatic
        assert record.focus.hypothesis_id == "message_budget"

    assert len(agent.records) == 3
    assert agent.ledger.log == ()


def test_one_record_per_observation_null_or_not() -> None:
    agent = make_agent()
    stream = canned_stream(25)
    for obs in stream:
        agent.cycle(obs)
    assert len(agent.records) == len(stream)
    assert [record.step for record in agent.records] == [
        obs.step for obs in stream
    ]
    assert all(record.intent is not None for record in agent.records)
