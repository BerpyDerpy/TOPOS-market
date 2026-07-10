"""Determinism (INV-9): the agent is a pure function of
(root_seed, observation stream).

Two independently constructed agents with the same seed, fed the same
canned observation stream, must produce bit-identical WorkspaceRecord and
message logs — every forecast, salience, EIG, snapshot and compiled
message included. The comparison is on the frozen contract dataclasses'
own equality, so any hidden nondeterminism (an unseeded RNG, dict-order
leakage, wall-clock anything) surfaces as a field mismatch.
"""

from __future__ import annotations

from tests.agent.conftest import canned_stream, make_agent, run_canned

N_STEPS = 50


def test_same_seed_same_stream_identical_logs() -> None:
    first = make_agent()
    second = make_agent()

    run_canned(first, canned_stream(N_STEPS))
    run_canned(second, canned_stream(N_STEPS))

    assert len(first.records) == N_STEPS
    assert first.records == second.records
    assert first.message_log == second.message_log
    assert first.ledger.log == second.ledger.log
    assert first.ledger.pending == second.ledger.pending

    # The run must have actually exercised the interesting paths, or the
    # equality above proved little.
    assert any(
        record.intent is not None and not record.intent.is_null
        for record in first.records
    ), "no committed intent in the determinism run"
    assert first.message_log, "no message was ever submitted"
    assert first.ledger.log, "no experiment was ever resolved"
