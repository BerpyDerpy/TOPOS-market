"""The numbered cycle order, asserted on instrumented real components.

Every collaborator the loop touches is shadowed with a recording wrapper
(instance-attribute spies on the REAL objects — nothing is faked out of
the data path), the agent is driven through its own (reset, step) driver
over a canned active market, and the recorded call sequence is segmented
per cycle and checked against the order the spec numbers:

    ingest -> realized-IG resolution (ledger 'before', target update) ->
    remaining belief updates -> slow tick -> homeostat -> workspace
    (appraise/compete/broadcast/propose) -> ledger entry -> act,

including the two orderings that are correctness-critical rather than
merely conventional: every posterior absorbs the observation BEFORE any
EIG/appraisal quantity is computed (stale-posterior EIG double-counts),
and the experiment ledger entry is written BEFORE the action is submitted
(INV-10).
"""

from __future__ import annotations

from typing import Any

from tests.agent.conftest import canned_stream, make_agent, spy
from topos.agent import AgentConfig, ToposAgent
from topos.contracts.market import ExchangeMessage, Observation

N_STEPS = 40

# A fast slow-loop cadence so the run reaches armed regime-gated
# forgetting (n_ticks > R_RECENT) inside the test's horizon.
FAST_TICKS = AgentConfig(slow_tick_every_steps=2)


class _Done(Exception):
    pass


def _drive(agent: ToposAgent, calls: list[str], n_steps: int = N_STEPS) -> None:
    """Run the agent's own driver over a canned stream, recording acts."""
    stream = iter(canned_stream(n_steps))

    def reset() -> Observation:
        return next(stream)

    def step(
        action: ExchangeMessage | None = None,
        workspace_record: object | None = None,
    ) -> Observation:
        assert workspace_record is not None, "record must ride every step call"
        calls.append("act")
        try:
            return next(stream)
        except StopIteration:
            raise _Done from None

    try:
        agent(reset, step)
    except _Done:
        pass


def _instrument(agent: ToposAgent) -> list[str]:
    calls: list[str] = []
    spy(calls, agent.books, "update", "books.update")
    for hypothesis, module in agent.modules.items():
        spy(calls, module, "update", f"update:{hypothesis}")
        spy(calls, module, "predict", f"appraise:{hypothesis}")
        spy(calls, module, "eig_nats", f"eig:{hypothesis}")
        spy(calls, module, "forget", f"forget:{hypothesis}")
    spy(calls, agent.homeostat, "evaluate", "homeostat.evaluate")
    spy(calls, agent.workspace, "cycle", "workspace.cycle")
    spy(calls, agent.ledger, "open", "ledger.open")
    spy(calls, agent.ledger, "resolve_pending", "ledger.resolve")
    spy(calls, agent.regime, "observe_summary", "slow.tick")
    return calls


def _cycles(calls: list[str]) -> list[list[str]]:
    """Segment the flat call log at each ingest (one segment per cycle)."""
    starts = [i for i, label in enumerate(calls) if label == "books.update"]
    assert starts, "no cycle ever ran"
    bounds = starts + [len(calls)]
    return [calls[a:b] for a, b in zip(bounds[:-1], bounds[1:])]


def _indices(cycle: list[str], prefix: str) -> list[int]:
    return [i for i, label in enumerate(cycle) if label.startswith(prefix)]


def test_numbered_cycle_order() -> None:
    agent = make_agent(config=FAST_TICKS)
    calls = _instrument(agent)
    _drive(agent, calls)

    cycles = _cycles(calls)
    assert len(cycles) == N_STEPS
    probes_opened = 0
    resolutions = 0
    forgets = 0

    for cycle in cycles:
        # 1. Ingest is the segment boundary by construction.
        assert cycle[0] == "books.update"

        updates = _indices(cycle, "update:")
        appraisals = _indices(cycle, "appraise:") + _indices(cycle, "eig:")
        assert updates, "a cycle must update its belief modules"
        assert appraisals, "a cycle must appraise (headline inputs)"

        # 3-before-5/8: every posterior absorbs the observation before any
        # forecast or EIG quantity is computed this cycle.
        assert max(updates) < min(appraisals), (
            "appraisal/EIG ran before belief updates finished: "
            f"{cycle}"
        )

        # 2. Realized-IG resolution: from the ledger, before any OTHER
        # update could confound it — first thing after ingest, and the
        # target's update is the very next call.
        resolves = _indices(cycle, "ledger.resolve")
        if resolves and any(
            label.startswith("update:") for label in cycle[resolves[0] + 1 :]
        ):
            resolutions += 1
            resolve_at = resolves[0]
            assert resolve_at < min(updates), (
                "some module updated before the pending experiment was "
                f"resolved: {cycle}"
            )
            assert cycle[resolve_at + 1].startswith("update:"), (
                "the resolution must apply the target module's update "
                f"immediately: {cycle}"
            )

        # 4. Slow tick: after the updates, before the workspace pass; the
        # regime-gated forgetting (when armed) follows the tick.
        ticks = _indices(cycle, "slow.tick")
        workspace_at = cycle.index("workspace.cycle")
        for tick in ticks:
            assert max(updates) < tick < workspace_at
        for forgotten in _indices(cycle, "forget:"):
            forgets += 1
            assert ticks and ticks[0] < forgotten < workspace_at

        # 6-before-7/8: the homeostat's exports exist before the
        # workspace competes/proposes with them.
        homeostat_at = cycle.index("homeostat.evaluate")
        assert max(updates) < homeostat_at < workspace_at

        # 9/10. Ledger entry strictly before the act; exactly one act,
        # and nothing after it in the cycle's segment except the act
        # itself terminating it.
        acts = _indices(cycle, "act")
        assert len(acts) == 1, f"exactly one step submission per cycle: {cycle}"
        for opened in _indices(cycle, "ledger.open"):
            probes_opened += 1
            assert workspace_at < opened < acts[0], (
                f"ledger entry must be written after ignition and BEFORE "
                f"acting (INV-10): {cycle}"
            )
        assert acts[0] == len(cycle) - 1

    # The run must actually have exercised the probe machinery, or the
    # ledger-before-act and resolution-order assertions were vacuous.
    assert probes_opened > 0, "no probe ever ignited; the test proved nothing"
    assert resolutions > 0, "no experiment was ever resolved"
    assert forgets > 0, "regime-gated forgetting never armed"
    assert len(agent.ledger.log) == resolutions


def test_every_cycle_updates_every_module_exactly_once() -> None:
    agent = make_agent()
    calls = _instrument(agent)
    _drive(agent, calls, n_steps=12)

    for cycle in _cycles(calls):
        updated = [label for label in cycle if label.startswith("update:")]
        assert sorted(updated) == sorted(
            f"update:{hypothesis}" for hypothesis in agent.modules
        ), f"each module must absorb each observation exactly once: {cycle}"


def test_records_and_actions_emitted_together() -> None:
    """The record rides the same step call that submits the action, and
    the submitted message is the head of that record's compilation."""
    agent = make_agent()
    stream = canned_stream(30)
    submissions: list[tuple[object, Any]] = []

    it = iter(stream)

    def reset() -> Observation:
        return next(it)

    def step(
        action: ExchangeMessage | None = None,
        workspace_record: object | None = None,
    ) -> Observation:
        submissions.append((workspace_record, action))
        try:
            return next(it)
        except StopIteration:
            raise _Done from None

    try:
        agent(reset, step)
    except _Done:
        pass

    assert len(submissions) == len(agent.records)
    for record, action in submissions:
        assert record in agent.records
        compiled = record.compiled_messages  # type: ignore[attr-defined]
        if compiled:
            assert action == compiled[0]
        else:
            assert action is None
