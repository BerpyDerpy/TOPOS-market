"""End-to-end smoke: 5k steps against the P2/P3 harness.

The long-run existence proof: the full architecture — engine, background
market, every belief module, homeostat, proposer, workspace, motor,
ledger — survives 5000 engine steps without exceptions, the agent's
self-tracked books match engine ground truth at every step (the P3 hook),
and the homeostat's message budget is respected over every rolling window.

This is the suite's slowest test by an order of magnitude (a full
cognitive cycle per engine step); it is deliberately kept in the default
run — the definition of done for P12 includes it.
"""

from __future__ import annotations

from topos.agent import ToposAgent
from topos.env.harness import RunConfig, assert_agent_bookkeeping, run

N_STEPS = 5_000
ROOT_SEED = 20260712


def test_five_thousand_step_smoke() -> None:
    agent = ToposAgent(root_seed=ROOT_SEED)
    config = RunConfig(n_steps=N_STEPS)

    log = run(config, agent, ROOT_SEED)

    # One record per observation (reset + one per step call).
    assert len(log.steps) == N_STEPS
    assert len(agent.records) == N_STEPS + 1
    assert all(record.intent is not None for record in agent.records)

    # Bookkeeping matches engine ground truth at every step (P3 hook).
    claims = agent.books.claims(N_STEPS - 1)
    assert len(claims) == N_STEPS
    assert_agent_bookkeeping(log, claims)

    # Message budget respected: over EVERY rolling window, the number of
    # submitted messages stays within the hard bound (the motor's veto is
    # the last line of defense, and it held).
    window = agent.config.homeostat.message_window_steps
    hard = agent.config.homeostat.message_budget.hard
    counts = [len(record.agent_messages) for record in log.steps]
    running = sum(counts[:window])
    worst = running
    for step in range(window, len(counts)):
        running += counts[step] - counts[step - window]
        worst = max(worst, running)
    assert worst <= hard, f"rolling message count peaked at {worst} > {hard}"

    # The run exercised the whole loop: the agent traded, experiments
    # were promised and resolved, and the slow loop ticked.
    assert agent.message_log, "the agent never submitted a message"
    assert agent.ledger.log, "no experiment was ever resolved"
    assert all(
        entry.step_resolved > entry.step_issued for entry in agent.ledger.log
    )
    expected_ticks = (N_STEPS - 1) // agent.config.slow_tick_every_steps
    assert agent.regime.n_ticks >= expected_ticks
