"""Harness tests (P3).

- test_twin_identity: a never-acting agent's run and its counterfactual are
  bit-identical (book states, events, observations, accounts, draws).
- test_twin_divergence_causality: one deep passive order that never fills
  => divergence begins exactly at the placement step and consists of
  nothing but that order's book-visible footprint.
- test_bookkeeping_hook: a scripted agent with known fills reconstructs its
  books from acks/fills; the hook passes on truth and fails on an
  off-by-one.
- test_no_leak: reflection/gc reachability check — the agent, its handles,
  and its observations cannot reach the engine or any ground-truth view,
  even mid-episode; post-run the handles are dead.
"""

from __future__ import annotations

import gc
import types
from collections import defaultdict
from dataclasses import replace

import pytest

from topos.contracts.market import GTC, Fill, Observation, PlaceLimit, Side
from topos.env import harness as harness_mod
from topos.env.background import BackgroundConfig, BackgroundMarket
from topos.env.engine import ActorAccount, GroundTruthView, MatchingEngine
from topos.env.harness import (
    BookkeepingClaim,
    RunConfig,
    RunLog,
    StepFn,
    ResetFn,
    assert_agent_bookkeeping,
    counterfactual,
    impact,
    null_agent,
    run,
)
from topos.env.orderbook import OrderBook, RestingOrder

ROOT_SEED = 20260708


def calm_config(n_steps: int) -> RunConfig:
    """Single-regime (hazard-free) config so tests are regime-stable."""
    regimes = tuple(replace(r, hazard=0.0) for r in BackgroundConfig().regimes)
    return RunConfig(n_steps=n_steps, background=BackgroundConfig(regimes=regimes))


def best_bid_of(obs: Observation) -> int | None:
    return next((l.price_ticks for l in obs.bids if l.size_lots > 0), None)


def best_ask_of(obs: Observation) -> int | None:
    return next((l.price_ticks for l in obs.asks if l.size_lots > 0), None)


# =====================================================================
# Scripted agents
# =====================================================================

class ObservingNullAgent:
    """Never acts, but exercises the full driver protocol and hoards
    everything it is handed — the strongest hidden-shared-state probe."""

    def __init__(self) -> None:
        self.observations: list[Observation] = []
        self.handles: tuple[ResetFn, StepFn] | None = None

    def __call__(self, reset: ResetFn, step: StepFn) -> None:
        self.handles = (reset, step)
        obs = reset()
        while True:
            self.observations.append(obs)
            obs = step(None)


class OneShotPassiveAgent:
    """Places one deep GTC bid during engine step `act_step`, then idles.

    The k-th step() call executes engine step k-1... precisely: the action
    passed alongside an observation with obs.step == s lands in engine
    step s+1, so triggering on obs.step == act_step - 1 places the order
    during engine step act_step.
    """

    def __init__(self, act_step: int, depth_ticks: int, size_lots: int) -> None:
        assert act_step >= 2, "act_step < 2 is ambiguous with the reset obs"
        self.act_step = act_step
        self.depth_ticks = depth_ticks
        self.size_lots = size_lots
        self.placed_price: int | None = None

    def __call__(self, reset: ResetFn, step: StepFn) -> None:
        obs = reset()
        while True:
            action = None
            if obs.step == self.act_step - 1 and self.placed_price is None:
                best_bid = best_bid_of(obs)
                assert best_bid is not None, "no bid to anchor the deep order"
                self.placed_price = best_bid - self.depth_ticks
                action = PlaceLimit(
                    side=Side.BUY,
                    price_ticks=self.placed_price,
                    size_lots=self.size_lots,
                    tif_steps=GTC,
                )
            obs = step(action)


class SelfBookkeepingAgent:
    """Crosses the spread once, then reconstructs its books from own fills."""

    def __init__(self, act_step: int, size_lots: int) -> None:
        assert act_step >= 2
        self.act_step = act_step
        self.size_lots = size_lots
        self.fills: list[Fill] = []
        self.sent: list[PlaceLimit] = []

    def __call__(self, reset: ResetFn, step: StepFn) -> None:
        obs = reset()
        while True:
            self.fills.extend(obs.own_fills)
            action = None
            if obs.step == self.act_step - 1 and not self.sent:
                best_ask = best_ask_of(obs)
                assert best_ask is not None, "no ask to cross"
                action = PlaceLimit(
                    side=Side.BUY,
                    price_ticks=best_ask + 3,
                    size_lots=self.size_lots,
                    tif_steps=1,
                )
                self.sent.append(action)
            obs = step(action)

    def claims(self, n_steps: int) -> list[BookkeepingClaim]:
        """End-of-step books folded from own fills (all from order_id 0, a BUY)."""
        by_step: dict[int, list[Fill]] = defaultdict(list)
        for fill in self.fills:
            by_step[fill.step].append(fill)
        claims: list[BookkeepingClaim] = []
        inventory = 0
        cash = 0
        for step in range(n_steps):
            for fill in by_step.get(step, ()):
                assert fill.order_id == 0
                inventory += fill.size_lots
                cash -= fill.price_ticks * fill.size_lots
            claims.append(
                BookkeepingClaim(step=step, inventory_lots=inventory, cash_ticks=cash)
            )
        return claims


# =====================================================================
# test_twin_identity
# =====================================================================

def test_twin_identity() -> None:
    config = RunConfig(n_steps=120)  # default regimes, hazard switches allowed
    agent = ObservingNullAgent()
    log = run(config, agent, ROOT_SEED)
    twin = counterfactual(config, log, ROOT_SEED)

    # Bit-identical everything: observations, events, book snapshots,
    # accounts, queue truth, regime chain, and raw draws.
    assert twin.twin_log == log
    assert twin.twin_log.draws == log.draws
    assert twin.twin_log.regimes == log.regimes

    # The divergence series is exactly zero at every step.
    assert len(twin.divergence) == config.n_steps
    for d in twin.divergence:
        assert d.book_identical
        assert d.depth_delta == 0
        assert d.mid_delta is None or d.mid_delta == 0.0

    # The driver protocol really ran: reset obs + one obs per step.
    assert len(agent.observations) == config.n_steps + 1
    assert agent.observations[0] == log.initial_observation
    assert [o.step for o in agent.observations[1:]] == list(range(config.n_steps))


# =====================================================================
# test_twin_divergence_causality
# =====================================================================

def test_twin_divergence_causality() -> None:
    n_steps, act_step, size = 90, 30, 3
    config = calm_config(n_steps)
    agent = OneShotPassiveAgent(act_step=act_step, depth_ticks=12, size_lots=size)
    log = run(config, agent, ROOT_SEED)
    twin = counterfactual(config, log, ROOT_SEED)
    agent_id = config.agent_actor_id

    # Premises: exactly one action, at act_step; the order never filled and
    # rests (visibly, in the book) until the end of the run.
    assert [s.step for s in log.steps if s.agent_messages] == [act_step]
    final = log.steps[-1].account(agent_id)
    assert final.fills == ()
    assert final.inventory_lots == 0
    assert final.open_order_ids == (0,)
    assert agent.placed_price is not None
    for record in log.steps[act_step:]:
        assert len(record.agent_queue_truth) == 1
        assert record.agent_queue_truth[0].price_ticks == agent.placed_price
        assert record.agent_queue_truth[0].remaining_lots == size

    # Nothing the agent cannot cause diverged: draws and regime chain match.
    assert log.draws == twin.twin_log.draws
    assert log.regimes == twin.twin_log.regimes

    # Before the placement step: the runs are bit-identical.
    for run_step, twin_step in zip(
        log.steps[:act_step], twin.twin_log.steps[:act_step]
    ):
        assert run_step == twin_step
    for d in twin.divergence[:act_step]:
        assert d.book_identical
        assert d.depth_delta == 0

    # From the placement step on: divergence is exactly the agent's resting
    # order — one bid level, its size — and nothing else anywhere.
    for run_step, twin_step, d in zip(
        log.steps[act_step:], twin.twin_log.steps[act_step:],
        twin.divergence[act_step:],
    ):
        assert d.bid_level_deltas == ((agent.placed_price, size),)
        assert d.ask_level_deltas == ()
        assert d.depth_delta == size
        assert d.mid_delta == 0.0

        # Only book-visible quantities carry the footprint: public prints,
        # background events, and background accounts are all identical.
        assert run_step.observation.trades == twin_step.observation.trades
        run_bg_events = [
            e for e in run_step.events
            if getattr(e, "actor_id", None) != agent_id
        ]
        assert run_bg_events == list(twin_step.events)
        run_bg_accounts = [v for v in run_step.accounts if v.actor_id != agent_id]
        twin_bg_accounts = [v for v in twin_step.accounts if v.actor_id != agent_id]
        assert run_bg_accounts == twin_bg_accounts


# =====================================================================
# test_bookkeeping_hook
# =====================================================================

def test_bookkeeping_hook() -> None:
    n_steps = 40
    config = calm_config(n_steps)
    agent = SelfBookkeepingAgent(act_step=12, size_lots=5)
    log = run(config, agent, ROOT_SEED)

    # Premise: the crossing order really filled.
    truth = log.steps[-1].account(config.agent_actor_id)
    assert truth.fills
    assert truth.inventory_lots > 0

    # Correct self-tracked books pass against engine truth.
    claims = agent.claims(n_steps)
    assert_agent_bookkeeping(log, claims)

    # ...and the hook has teeth: a single off-by-one fails loudly.
    bad_inventory = list(claims)
    bad_inventory[20] = replace(
        bad_inventory[20], inventory_lots=bad_inventory[20].inventory_lots + 1
    )
    with pytest.raises(AssertionError, match="inventory"):
        assert_agent_bookkeeping(log, bad_inventory)

    bad_cash = list(claims)
    assert bad_cash[30].cash_ticks is not None
    bad_cash[30] = replace(bad_cash[30], cash_ticks=bad_cash[30].cash_ticks - 1)
    with pytest.raises(AssertionError, match="cash"):
        assert_agent_bookkeeping(log, bad_cash)

    # Vacuous or misaligned logs are use errors, not silent passes.
    with pytest.raises(ValueError):
        assert_agent_bookkeeping(log, [])
    with pytest.raises(ValueError):
        assert_agent_bookkeeping(
            log, [BookkeepingClaim(step=n_steps, inventory_lots=0)]
        )


# =====================================================================
# test_no_leak
# =====================================================================

FORBIDDEN_TYPES = (
    MatchingEngine,
    OrderBook,
    RestingOrder,
    ActorAccount,
    GroundTruthView,
    BackgroundMarket,
    RunLog,
    harness_mod._Session,
)


def _reachable_forbidden(*roots: object) -> list[object]:
    """BFS over gc.get_referents from `roots`; returns reachable forbidden
    instances. Non-topos modules are not descended into (their graphs are
    huge and cannot hold our instances); everything else — closures, cells,
    dicts, classes, weakrefs — is walked."""
    seen: set[int] = set()
    stack = list(roots)
    found: list[object] = []
    while stack:
        obj = stack.pop()
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        if isinstance(obj, FORBIDDEN_TYPES):
            found.append(obj)
            continue
        if isinstance(obj, types.ModuleType) and not obj.__name__.startswith("topos"):
            continue
        stack.extend(gc.get_referents(obj))
    return found


class LeakProbeAgent:
    """Stores its handles and observations, then runs the reachability probe
    MID-EPISODE, while the engine is alive and matching."""

    def __init__(self) -> None:
        self.handles: tuple[ResetFn, StepFn] | None = None
        self.observations: list[Observation] = []
        self.mid_run_leaks: list[object] | None = None

    def __call__(self, reset: ResetFn, step: StepFn) -> None:
        self.handles = (reset, step)
        obs = reset()
        self.observations.append(obs)
        obs = step(None)
        self.observations.append(obs)
        self.mid_run_leaks = _reachable_forbidden(self, reset, step, obs)
        while True:
            obs = step(None)


def test_no_leak() -> None:
    config = calm_config(30)
    agent = LeakProbeAgent()
    log = run(config, agent, ROOT_SEED)

    # Control 1: the probe can find a planted engine through a container.
    planted = MatchingEngine()
    assert _reachable_forbidden({"x": planted}) == [planted]
    # Control 2: ground-truth objects really existed during the run.
    assert isinstance(log.steps[-1].account("agent"), GroundTruthView)

    # Mid-run, with the engine live: nothing ground-truth-ish is reachable
    # from the agent, its handles, or its observations.
    assert agent.mid_run_leaks == []

    # Post-run: still nothing reachable, and the handles are dead — the
    # session behind them is gone, so they cannot even be called any more.
    assert _reachable_forbidden(agent) == []
    assert agent.handles is not None
    reset_handle, step_handle = agent.handles
    with pytest.raises(RuntimeError, match="dead"):
        step_handle(None)
    with pytest.raises(RuntimeError, match="dead"):
        reset_handle()


# =====================================================================
# Supporting behavior: passthrough, pairing guards, impact windows
# =====================================================================

def test_workspace_record_passthrough() -> None:
    n_steps = 6
    config = calm_config(n_steps)

    class RecordEmittingAgent:
        def __call__(self, reset: ResetFn, step: StepFn) -> None:
            reset()
            cycle = 0
            while True:
                step(None, workspace_record=("cycle", cycle))
                cycle += 1

    log = run(config, RecordEmittingAgent(), ROOT_SEED)
    assert [s.workspace_record for s in log.steps] == [
        ("cycle", k) for k in range(n_steps)
    ]


def test_run_pads_when_agent_stops_early() -> None:
    config = calm_config(25)

    class QuitterAgent:
        def __call__(self, reset: ResetFn, step: StepFn) -> None:
            reset()
            step(None)  # one step, then walk away

    log = run(config, QuitterAgent(), ROOT_SEED)
    assert len(log.steps) == config.n_steps
    assert all(not s.agent_messages for s in log.steps)

    # A padded quitter run is indistinguishable from a null-agent run.
    assert log == run(config, null_agent, ROOT_SEED)


def test_driver_protocol_guards() -> None:
    config = calm_config(5)

    class ResetTwice:
        def __call__(self, reset: ResetFn, step: StepFn) -> None:
            reset()
            reset()

    class StepBeforeReset:
        def __call__(self, reset: ResetFn, step: StepFn) -> None:
            step(None)

    with pytest.raises(RuntimeError, match="once"):
        run(config, ResetTwice(), ROOT_SEED)
    with pytest.raises(RuntimeError, match="reset"):
        run(config, StepBeforeReset(), ROOT_SEED)


def test_counterfactual_rejects_mismatched_pairing() -> None:
    config = calm_config(10)
    log = run(config, null_agent, ROOT_SEED)

    with pytest.raises(ValueError, match="root_seed"):
        counterfactual(config, log, ROOT_SEED + 1)
    with pytest.raises(ValueError, match="config"):
        counterfactual(calm_config(11), log, ROOT_SEED)


def test_impact_windows_follow_agent_actions() -> None:
    n_steps, act_step, size = 60, 20, 3
    config = calm_config(n_steps)
    agent = OneShotPassiveAgent(act_step=act_step, depth_ticks=12, size_lots=size)
    log = run(config, agent, ROOT_SEED)
    twin = counterfactual(config, log, ROOT_SEED)

    horizon = 10
    records = impact(log, twin.twin_log, horizon)
    assert len(records) == 1
    record = records[0]
    assert record.step == act_step
    assert len(record.window) == horizon + 1
    assert record.window == twin.divergence[act_step : act_step + horizon + 1]
    # The realized footprint of the passive order: pure depth, no mid move.
    assert record.depth_deltas == tuple(size for _ in record.window)
    assert record.mid_deltas == tuple(0.0 for _ in record.window)

    # A window reaching past the end of the run is truncated, not padded.
    late = impact(log, twin.twin_log, n_steps)[0]
    assert len(late.window) == n_steps - act_step

    # Null runs have no impact records; a non-null "twin" is rejected.
    assert impact(twin.twin_log, twin.twin_log, horizon) == ()
    with pytest.raises(ValueError, match="null"):
        impact(log, log, horizon)
