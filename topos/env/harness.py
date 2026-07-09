"""Test harness (P3): experiment runner, ground truth, counterfactual replay.

``run()`` drives one episode of engine + background market around an agent.
Deliberately NOT a Gym-style API: the agent sees ``reset() -> Observation``
and ``step(action) -> Observation`` and nothing else. There is no scalar
feedback value and no info-dict side channel that could carry one (INV-1).

Ground-truth channels (INV-11)
------------------------------
The RunLog records, per step, quantities the agent must never observe: the
regime id + true parameters, engine-side account state for every actor, and
the true queue position of every resting agent order. These flow from the
harness into metrics/validation only. Leak-proofing is structural, not
conventional: the agent is handed exactly two callables built by
``_agent_facade`` that close over a WEAK reference to the episode session,
so neither the harness, the engine, nor any ground-truth view is reachable
from the agent by walking object references — even while the episode is
live — and both handles go dead the moment ``run()`` returns. Verified by
tests/env/test_harness.py::test_no_leak.

Counterfactual replay ("twin run")
----------------------------------
``counterfactual()`` RE-RUNS the identical (config, root_seed) with the
agent's actor replaced by ``null_agent`` (submits nothing). It does not
re-simulate with fresh randomness, and it does not splice the recorded
event log: every background actor re-makes bit-identical raw draws because
all environment randomness is keyed by (actor_id, step, purpose) (INV-9,
verified in P2), so the run and twin book trajectories can diverge only
through the agent's causal footprint. ``impact()`` windows the resulting
divergence series after each agent action; it is the ground truth against
which the agent's impact-model posterior (P6) is scored in P13.

The per-step ``WorkspaceRecord`` the agent emits is carried OPAQUELY
(typed ``object``): the env never imports cognition types, and metrics can
still line up promised vs realized information gain (INV-10) from the log.
"""

from __future__ import annotations

import weakref
from dataclasses import dataclass
from typing import Callable, Iterable, Protocol

from topos.contracts.market import ExchangeMessage, Observation, Side
from topos.env.background import (
    BACKGROUND_ACTOR_PREFIX,
    BackgroundConfig,
    BackgroundMarket,
    DrawRecord,
    RegimeRecord,
)
from topos.env.engine import EngineEvent, GroundTruthView, MatchingEngine


# ---------------------------------------------------------------------------
# The agent-facing protocol: two callables, nothing else
# ---------------------------------------------------------------------------

class EpisodeComplete(Exception):
    """Raised by the step handle once the configured horizon has elapsed.

    Pure control flow — carries no data. A driver may simply let it
    propagate out of its loop; ``run()`` catches it.
    """


ResetFn = Callable[[], Observation]


class StepFn(Protocol):
    """The step handle: submit at most one message, receive one Observation.

    ``workspace_record`` is the agent's per-cycle WorkspaceRecord (or any
    stand-in), logged opaquely into the RunLog. The harness never inspects
    it and nothing flows back through it.
    """

    def __call__(
        self,
        action: ExchangeMessage | None = None,
        workspace_record: object | None = None,
    ) -> Observation: ...


AgentDriver = Callable[[ResetFn, StepFn], None]
"""An agent is a callable that drives its own episode:

    def agent(reset, step):
        obs = reset()                # once, first
        while True:
            obs = step(action, workspace_record=record)

It is handed ONLY these two callables — never the harness, the engine, or
any ground-truth view (INV-11).
"""


def null_agent(reset: ResetFn, step: StepFn) -> None:
    """The counterfactual twin's driver: observes every step, submits nothing."""
    reset()
    while True:  # ends when the step handle raises EpisodeComplete
        step(None)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RunConfig:
    """Full configuration of one harness run."""

    n_steps: int
    background: BackgroundConfig = BackgroundConfig()
    agent_actor_id: str = "agent"

    def __post_init__(self) -> None:
        if self.n_steps <= 0:
            raise ValueError(f"n_steps must be positive, got {self.n_steps}")
        if not self.agent_actor_id:
            raise ValueError("agent_actor_id must be non-empty")
        if self.agent_actor_id.startswith(BACKGROUND_ACTOR_PREFIX):
            raise ValueError(
                f"agent_actor_id must not use the reserved "
                f"{BACKGROUND_ACTOR_PREFIX!r} namespace"
            )


# ---------------------------------------------------------------------------
# Per-step records (harness-only ground truth — INV-11)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BookSnapshot:
    """Full-depth end-of-step book state, harness-side.

    Unlike the agent's Observation (top N_LEVELS per side), this covers
    every non-empty level, so divergence between run and twin is measured
    on the whole book.
    """

    bids: tuple[tuple[int, int], ...]
    """(price_ticks, size_lots) per non-empty bid level, best (highest) first."""
    asks: tuple[tuple[int, int], ...]
    """(price_ticks, size_lots) per non-empty ask level, best (lowest) first."""

    @property
    def best_bid(self) -> int | None:
        return self.bids[0][0] if self.bids else None

    @property
    def best_ask(self) -> int | None:
        return self.asks[0][0] if self.asks else None

    @property
    def mid(self) -> float | None:
        if not self.bids or not self.asks:
            return None
        return (self.bids[0][0] + self.asks[0][0]) / 2.0

    @property
    def total_lots(self) -> int:
        return sum(s for _, s in self.bids) + sum(s for _, s in self.asks)


@dataclass(frozen=True)
class QueueTruth:
    """True queue state of one resting agent order (harness-only, INV-11).

    ``lots_ahead`` is the engine's ground-truth queue position — the
    quantity the agent's queue filter (P5) estimates but never observes.
    """

    order_id: int
    side: Side
    price_ticks: int
    remaining_lots: int
    lots_ahead: int


@dataclass(frozen=True)
class StepRecord:
    """Everything that happened during one engine step.

    ``observation`` is the Observation the engine's builder produced for
    the agent DURING this step (after background events, before the agent
    action); ``agent_messages`` is what the agent submitted during this
    step (decided from the previous step's observation). All ground-truth
    fields are end-of-step state.
    """

    step: int
    observation: Observation
    agent_messages: tuple[ExchangeMessage, ...]
    workspace_record: object | None
    """Opaque passthrough of whatever the agent emitted with its action."""
    events: tuple[EngineEvent, ...]
    # -- harness-only ground truth from here down (INV-11) --
    regime: RegimeRecord
    accounts: tuple[GroundTruthView, ...]
    """Engine-side account state for every actor, sorted by actor_id."""
    agent_queue_truth: tuple[QueueTruth, ...]
    """True queue position of every resting agent order, by order_id."""
    book: BookSnapshot

    def account(self, actor_id: str) -> GroundTruthView:
        for view in self.accounts:
            if view.actor_id == actor_id:
                return view
        raise KeyError(f"no account snapshot for actor {actor_id!r}")


@dataclass(frozen=True)
class RunLog:
    """One complete episode: agent-visible stream + harness-only ground truth."""

    config: RunConfig
    root_seed: int
    initial_observation: Observation
    """What reset() returned: the step-0 pre-market (empty book) snapshot."""
    steps: tuple[StepRecord, ...]
    draws: tuple[DrawRecord, ...]
    """Every background raw draw; bit-identical across twins (INV-9)."""
    regimes: tuple[RegimeRecord, ...]
    """Regime id + true parameters per step; agent-invisible (INV-11)."""

    @property
    def agent_actor_id(self) -> str:
        return self.config.agent_actor_id


# ---------------------------------------------------------------------------
# The episode session (never handed to the agent)
# ---------------------------------------------------------------------------

class _Session:
    """Mutable state of one running episode.

    The agent never receives this object — only the weakref-backed handles
    from ``_agent_facade`` (INV-11 leak-proofing).
    """

    def __init__(self, config: RunConfig, root_seed: int) -> None:
        self._config = config
        self._root_seed = root_seed
        self._engine = MatchingEngine()
        self._market = BackgroundMarket(config.background, root_seed)
        self._initial_obs: Observation | None = None
        self._steps: list[StepRecord] = []

    @property
    def done(self) -> bool:
        return len(self._steps) >= self._config.n_steps

    def reset(self) -> Observation:
        if self._initial_obs is not None:
            raise RuntimeError("reset() may be called only once per episode")
        # Exclusively the engine's observation builder — no other path.
        self._initial_obs = self._engine.observation(self._config.agent_actor_id)
        return self._initial_obs

    def step(
        self, action: ExchangeMessage | None, workspace_record: object | None
    ) -> Observation:
        if self._initial_obs is None:
            raise RuntimeError("call reset() before step()")
        if self.done:
            raise EpisodeComplete(
                f"the {self._config.n_steps}-step episode is over"
            )

        engine = self._engine
        agent_id = self._config.agent_actor_id
        step_index = engine.current_step

        background_events = self._market.events_for_step(engine)
        obs, events = engine.step(
            background_events, agent_id=agent_id, agent_action=action
        )

        self._steps.append(
            StepRecord(
                step=step_index,
                observation=obs,
                agent_messages=(action,) if action is not None else (),
                workspace_record=workspace_record,
                events=tuple(events),
                regime=self._market.regime_log[step_index],
                accounts=self._account_views(),
                agent_queue_truth=self._queue_truth(agent_id),
                book=self._book_snapshot(),
            )
        )
        return obs

    def finish(self) -> RunLog:
        """Drive any remaining steps with no agent action, then assemble the log.

        Padding to the full horizon keeps every RunLog twin-comparable:
        divergence series are only meaningful at equal length.
        """
        if self._initial_obs is None:
            self.reset()
        while not self.done:
            self.step(None, None)
        return RunLog(
            config=self._config,
            root_seed=self._root_seed,
            initial_observation=self._initial_obs,
            steps=tuple(self._steps),
            draws=self._market.draw_log,
            regimes=self._market.regime_log,
        )

    def _account_views(self) -> tuple[GroundTruthView, ...]:
        actor_ids = sorted({self._config.agent_actor_id, *self._market.actor_ids})
        return tuple(self._engine.ground_truth_view(a) for a in actor_ids)

    def _queue_truth(self, agent_id: str) -> tuple[QueueTruth, ...]:
        book = self._engine.book
        own = sorted(
            (o for o in book.all_resting_orders() if o.actor_id == agent_id),
            key=lambda o: o.order_id,
        )
        return tuple(
            QueueTruth(
                order_id=o.order_id,
                side=o.side,
                price_ticks=o.price_ticks,
                remaining_lots=o.remaining_lots,
                lots_ahead=book.queue_position(o),
            )
            for o in own
        )

    def _book_snapshot(self) -> BookSnapshot:
        book = self._engine.book
        return BookSnapshot(
            bids=tuple(book.bid_levels()),
            asks=tuple(book.ask_levels()),
        )


def _agent_facade(session: _Session) -> tuple[ResetFn, StepFn]:
    """Build the ONLY two handles the agent ever receives.

    The closures hold a weak reference to the session, so walking object
    references from the handles (or from an agent that stored them) never
    arrives at the engine, the background market, or any ground-truth view
    (INV-11) — and once ``run()`` drops the session, both handles are dead.
    """
    session_ref = weakref.ref(session)

    def _live() -> _Session:
        live = session_ref()
        if live is None:
            raise RuntimeError("this episode is over; the handle is dead")
        return live

    def reset() -> Observation:
        return _live().reset()

    def step(
        action: ExchangeMessage | None = None,
        workspace_record: object | None = None,
    ) -> Observation:
        return _live().step(action, workspace_record)

    return reset, step


# ---------------------------------------------------------------------------
# run(): the experiment runner
# ---------------------------------------------------------------------------

def run(config: RunConfig, agent: AgentDriver, root_seed: int) -> RunLog:
    """Drive one full episode of engine + background market around `agent`.

    The agent drives itself through the two handles it is given: it calls
    ``reset()`` once, then ``step(action, workspace_record=...)`` repeatedly;
    after ``config.n_steps`` engine steps the step handle raises
    ``EpisodeComplete``, which the agent may simply let propagate. If the
    agent stops calling early, the harness completes the remaining steps
    with no agent action, so the RunLog always spans the full horizon.
    """
    session = _Session(config, root_seed)
    reset_handle, step_handle = _agent_facade(session)
    try:
        agent(reset_handle, step_handle)
    except EpisodeComplete:
        pass
    return session.finish()


# ---------------------------------------------------------------------------
# Counterfactual replay ("twin run") and divergence
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StepDivergence:
    """Run-minus-twin book divergence at one step (aligned by step index)."""

    step: int
    mid_delta: float | None
    """run mid - twin mid; None when either book lacks a defined mid."""
    depth_delta: int
    """Difference in total resting lots across the whole book."""
    bid_level_deltas: tuple[tuple[int, int], ...]
    """(price_ticks, run_lots - twin_lots) for every bid price where they differ."""
    ask_level_deltas: tuple[tuple[int, int], ...]

    @property
    def book_identical(self) -> bool:
        return not self.bid_level_deltas and not self.ask_level_deltas


@dataclass(frozen=True)
class TwinResult:
    """Paired trajectories plus their per-step divergence series."""

    run_log: RunLog
    twin_log: RunLog
    divergence: tuple[StepDivergence, ...]


def _check_paired(run_log: RunLog, twin_log: RunLog) -> None:
    if run_log.config != twin_log.config:
        raise ValueError("logs are not twins: configurations differ")
    if run_log.root_seed != twin_log.root_seed:
        raise ValueError(
            "logs are not twins: root seeds differ — a run under a fresh "
            "seed is a re-simulation, not a replay"
        )
    if len(run_log.steps) != len(twin_log.steps):
        raise ValueError("logs are not twins: step counts differ")


def _level_deltas(
    run_levels: tuple[tuple[int, int], ...],
    twin_levels: tuple[tuple[int, int], ...],
) -> tuple[tuple[int, int], ...]:
    run_map = dict(run_levels)
    twin_map = dict(twin_levels)
    return tuple(
        (price, run_map.get(price, 0) - twin_map.get(price, 0))
        for price in sorted(run_map.keys() | twin_map.keys())
        if run_map.get(price, 0) != twin_map.get(price, 0)
    )


def divergence_series(
    run_log: RunLog, twin_log: RunLog
) -> tuple[StepDivergence, ...]:
    """Per-step book divergence between a run and its twin, aligned by step."""
    _check_paired(run_log, twin_log)
    series: list[StepDivergence] = []
    for run_step, twin_step in zip(run_log.steps, twin_log.steps):
        run_book, twin_book = run_step.book, twin_step.book
        run_mid, twin_mid = run_book.mid, twin_book.mid
        series.append(
            StepDivergence(
                step=run_step.step,
                mid_delta=(
                    run_mid - twin_mid
                    if run_mid is not None and twin_mid is not None
                    else None
                ),
                depth_delta=run_book.total_lots - twin_book.total_lots,
                bid_level_deltas=_level_deltas(run_book.bids, twin_book.bids),
                ask_level_deltas=_level_deltas(run_book.asks, twin_book.asks),
            )
        )
    return tuple(series)


def counterfactual(
    config: RunConfig, run_log: RunLog, root_seed: int
) -> TwinResult:
    """Re-run the identical (config, root_seed) with the agent nulled out.

    This is a re-RUN of every background actor, not a splice of the
    recorded event log: the twin's actors re-make their draws from the same
    named streams, which INV-9 guarantees are bit-identical to the run's,
    so any book divergence is the agent's causal footprint and nothing else.
    """
    if config != run_log.config:
        raise ValueError(
            "counterfactual config must be identical to the run's config"
        )
    if root_seed != run_log.root_seed:
        raise ValueError(
            "counterfactual root_seed must be identical to the run's — a "
            "fresh seed would be a re-simulation, not a twin replay"
        )
    twin_log = run(config, null_agent, root_seed)
    return TwinResult(
        run_log=run_log,
        twin_log=twin_log,
        divergence=divergence_series(run_log, twin_log),
    )


# ---------------------------------------------------------------------------
# Impact measurement (ground truth for the P6 impact model, scored in P13)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ImpactRecord:
    """Realized run-vs-twin divergence following one agent action.

    ``window[0]`` is the divergence at the step the action was applied
    (its footprint lands within that engine step); the window then extends
    ``horizon`` further steps, truncated at the end of the run.
    """

    step: int
    messages: tuple[ExchangeMessage, ...]
    window: tuple[StepDivergence, ...]

    @property
    def mid_deltas(self) -> tuple[float | None, ...]:
        return tuple(d.mid_delta for d in self.window)

    @property
    def depth_deltas(self) -> tuple[int, ...]:
        return tuple(d.depth_delta for d in self.window)


def impact(
    run_log: RunLog, twin_log: RunLog, horizon: int
) -> tuple[ImpactRecord, ...]:
    """Per-agent-action realized mid/book divergence over `horizon` steps."""
    if horizon < 0:
        raise ValueError(f"horizon must be >= 0, got {horizon}")
    if any(record.agent_messages for record in twin_log.steps):
        raise ValueError(
            "twin_log contains agent messages — it must come from a "
            "null-agent counterfactual run"
        )
    series = divergence_series(run_log, twin_log)
    return tuple(
        ImpactRecord(
            step=record.step,
            messages=record.agent_messages,
            window=series[index : index + horizon + 1],
        )
        for index, record in enumerate(run_log.steps)
        if record.agent_messages
    )


# ---------------------------------------------------------------------------
# Ground-truth validation hook (used by P6 tests)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BookkeepingClaim:
    """The agent's self-tracked account state as of the END of one step."""

    step: int
    inventory_lots: int
    cash_ticks: int | None = None
    """Signed realized cashflow (sum of -side * price * size over fills).
    Optional so inventory-only bookkeeping can still be validated; the
    mark-to-market variant derives as cash + inventory * mid, all of which
    the RunLog carries."""


def assert_agent_bookkeeping(
    run_log: RunLog, agent_selfstate_log: Iterable[BookkeepingClaim]
) -> None:
    """Compare the agent's self-tracked books against engine ground truth.

    Entries are duck-typed: anything exposing ``step``, ``inventory_lots``
    and (optionally) ``cash_ticks`` works, so P6 can pass its own record
    type. A claim at step k is compared against the engine-side agent
    account at the end of engine step k (fills stamped step <= k).

    Raises AssertionError listing every mismatch, or ValueError for an
    empty/misaligned log (an empty log would pass vacuously).
    """
    claims = list(agent_selfstate_log)
    if not claims:
        raise ValueError("empty selfstate log: nothing to validate")

    agent_id = run_log.agent_actor_id
    problems: list[str] = []
    for claim in claims:
        step = claim.step
        if not 0 <= step < len(run_log.steps):
            raise ValueError(
                f"claim at step {step} is outside the run's range "
                f"0..{len(run_log.steps) - 1}"
            )
        truth = run_log.steps[step].account(agent_id)
        if claim.inventory_lots != truth.inventory_lots:
            problems.append(
                f"step {step}: claimed inventory {claim.inventory_lots} "
                f"!= engine {truth.inventory_lots}"
            )
        claimed_cash = getattr(claim, "cash_ticks", None)
        if claimed_cash is not None and claimed_cash != truth.cash:
            problems.append(
                f"step {step}: claimed cash {claimed_cash} "
                f"!= engine {truth.cash}"
            )
    if problems:
        raise AssertionError(
            "agent bookkeeping diverges from engine ground truth:\n"
            + "\n".join(problems)
        )
