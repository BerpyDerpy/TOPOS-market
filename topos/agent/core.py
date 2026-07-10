"""The integrated cognitive loop (P12).

One engine step = one call to ``cycle(observation)``, which runs EXACTLY
this order — deviations are correctness bugs:

 1. Ingest the Observation; fold own acks/fills into bookkeeping.
 2. Realized-IG scoring for the previous cycle's probe, if any (INV-10):
    the 'before' snapshot comes from the experiment ledger (captured at
    issuance), the target module absorbs this observation, the 'after'
    snapshot follows immediately — before any other update could confound
    it.
 3. Belief updates for every remaining module: every posterior absorbs the
    current observation BEFORE any EIG is computed this cycle (EIG on a
    stale posterior double-counts information the agent already has).
 4. Slow tick (every M steps): the regime tracker consumes the public
    summary statistics and regime-gated forgetting is applied to every
    belief module.
 5. Build the WorldSummary and run the appraisal (headlines).
 6. Homeostat evaluation (SelfStateFull -> drives, vetoes, distances,
    corrective intent). NOTE on 5/6 ordering: the workspace's committed
    P9 API runs appraise->compete->broadcast->propose->ignite as one
    ``cycle()`` call that CONSUMES the homeostat exports, so the
    evaluation is hoisted immediately before that call. Appraisal reads
    nothing the homeostat writes, so every data dependency of the
    numbered order is preserved.
 7. Salience competition + focus; broadcast conditioning hooks fired.
 8. Refined proposer menu for the focus; lexicographic selection;
    coalition gates => Intent (possibly null).           (5b/7/8 happen
    inside ``Workspace.cycle``.)
 9. If a probe was selected: the experiment ledger entry is written NOW —
    before acting (INV-10).
10. Motor-compiled messages (with vetoes) are logged side by side with
    the intent in the WorkspaceRecord (INV-8) and submitted.

Acting through the P3 harness: the exchange interface accepts AT MOST ONE
message per engine step, so the agent submits the head of the compiled
tuple. The remainder is deliberately NOT queued: the arbiter re-evaluates
every cycle from fresh state, so a cancel-then-replace compilation
completes across consecutive steps as the working-order view updates,
while queued leftovers would execute against a book they were not compiled
for. The record remains lossless: intent and the FULL compilation are
logged together, and what was actually submitted is in the message log.

Event timing (the committed P6 convention, DESIGN item 16): ``SelfEvents``
pairs each observation with the messages whose acks it carries. Because
the harness builds each step's observation BEFORE applying that step's
agent action, the observation returned by ``step(a)`` never answers ``a``:
the acks for the action decided in cycle k arrive in cycle k+2's
observation (one ENGINE step later — the reset observation and the first
step observation share stamp 0). The pairing therefore shifts through two
slots, exactly like the P6 harness test's sent/pending shuffle.

Determinism (INV-9): the agent is a pure function of (root_seed,
observation stream). Every compiled quantity in the loop is closed-form;
if agent-internal sampling is ever needed, ``rng_for`` is the only
sanctioned source (named counter-based streams under actor_id="agent").
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Protocol

import numpy as np

from topos.beliefs import (
    FairValueKF,
    FlowIntensity,
    QueuePositionFilter,
    R_RECENT,
    RegimeTracker,
)
from topos.contracts.beliefs import BeliefModule, SelfEvents
from topos.contracts.intent import (
    FAIR_VALUE,
    FILL_RATE,
    FLOW_INTENSITY,
    IMPACT,
    QUEUE_POSITION,
    REGIME,
    HypothesisId,
)
from topos.contracts.market import N_LEVELS, ExchangeMessage, Observation
from topos.contracts.registry import ModuleRegistry
from topos.contracts.rng import StreamKey, make_rng
from topos.contracts.workspace import Headline, WorkspaceRecord, WorldSummary
from topos.drives import Homeostat, HomeostatOutput
from topos.proposer import (
    DistanceProjector,
    Proposer,
    SelectionRule,
    null_intent,
    select_candidate,
)
from topos.selfmodel import Books, FillModel, ImpactModel, SelfTrajectory
from topos.workspace import Workspace

from topos.agent.ablations import (
    AblationFlags,
    FrozenFillModel,
    FrozenImpactModel,
    NoReflexiveSelection,
    NullDistanceProjector,
    SurpriseAsCuriosity,
    VetoOnlyHomeostat,
)
from topos.agent.config import AgentConfig
from topos.agent.ledger import ExperimentLedger
from topos.agent.summary import WorldSummaryTracker
from topos.agent.wiring import (
    BandDistanceProjector,
    assert_registry_covers_known_ids,
    default_registry,
)

_UPDATE_ORDER: tuple[HypothesisId, ...] = (
    FAIR_VALUE,
    FLOW_INTENSITY,
    QUEUE_POSITION,
    FILL_RATE,
    IMPACT,
    REGIME,
)
"""Canonical per-cycle update order — fixed so the loop is deterministic.
World predictors first (public data), then the self-model (own events);
the regime tracker's per-observation update is a protocol no-op (it
consumes summary statistics on slow ticks only)."""


class StepHandle(Protocol):
    """The harness step callable: at most one message, one Observation back."""

    def __call__(
        self,
        action: ExchangeMessage | None = None,
        workspace_record: object | None = None,
    ) -> Observation: ...


class ToposAgent:
    """The integrated agent: every module, wired behind (reset, step).

    Construction builds and owns all BeliefModules (world: fair_value,
    flow_intensity, regime; the queue filter; self-model: fill_rate,
    impact, and the self_trajectory compiler), the bookkeeping, the
    homeostat, the proposer, the workspace/arbiter, the motor config, and
    the experiment ledger — and asserts (fail fast) that the registry's
    centrality weights cover every id in KNOWN_HYPOTHESIS_IDS.

    Ablation flags are honored by SUBSTITUTING strategy objects at their
    injection points here, never by conditionals in the cycle; with all
    flags off none of the strategy classes is even instantiated
    (see ``topos.agent.ablations``).
    """

    def __init__(
        self,
        config: AgentConfig | None = None,
        *,
        root_seed: int,
        flags: AblationFlags | None = None,
        registry: ModuleRegistry | None = None,
    ) -> None:
        cfg = config if config is not None else AgentConfig()
        self.config = cfg
        self.flags = flags if flags is not None else AblationFlags()
        self._root_seed = int(root_seed)
        budget = cfg.motor.size_budget_lots

        # -- belief modules (NO_SELF_MODEL substitutes the frozen pair) ---
        self.fair_value = FairValueKF()
        self.flow = FlowIntensity()
        self.queue = QueuePositionFilter(flow_model=self.flow)
        if self.flags.no_self_model:
            self.fill: FillModel = FrozenFillModel(
                cfg.fill_horizon_steps, size_budget_lots=budget
            )
            self.impact: ImpactModel = FrozenImpactModel(
                impact_horizon_steps=cfg.impact_horizon_steps,
                size_budget_lots=budget,
            )
        else:
            self.fill = FillModel(
                cfg.fill_horizon_steps, size_budget_lots=budget
            )
            self.impact = ImpactModel(
                impact_horizon_steps=cfg.impact_horizon_steps,
                size_budget_lots=budget,
            )
        self.regime = RegimeTracker(cfg.regime)
        self.modules: dict[HypothesisId, BeliefModule] = {
            FAIR_VALUE: self.fair_value,
            FLOW_INTENSITY: self.flow,
            QUEUE_POSITION: self.queue,
            FILL_RATE: self.fill,
            IMPACT: self.impact,
            REGIME: self.regime,
        }

        # -- self-trajectory compiler and bookkeeping ---------------------
        self.trajectory = SelfTrajectory(
            self.fill, self.impact, self.fair_value, size_budget_lots=budget
        )
        self.books = Books(rank_lookup=self.queue.rank_mean_var)
        self.homeostat = Homeostat(cfg.homeostat)
        self.ledger = ExperimentLedger()
        self._summary = WorldSummaryTracker(cfg.vol_window_steps)

        # -- registry / consequence-weights (INV-7), asserted at startup --
        reg = registry if registry is not None else default_registry()
        assert_registry_covers_known_ids(reg)
        self.registry = reg

        # -- ablation injection points ------------------------------------
        # The proposer's scoring map — the committed P8/P9 shape. REGIME
        # is passive-only (adjudication A3): no probe may target it, so
        # its headline marginal is 0 by construction. QUEUE_POSITION's
        # watch-EIG is intent-independent (the P5 observation model: the
        # rank question is answered by watching the level, and a fresh
        # placement's point-mass prior carries ~0 first-step EIG), so its
        # marginal over null is 0 by the same arithmetic that keeps every
        # world hypothesis out of focus (DESIGN item 28). Both therefore
        # stay out of the scoring map: the workspace reads their headline
        # marginals as the 0 they would compute, without paying the queue
        # filter's Monte-Carlo EIG once per shape per cycle. Their
        # posteriors, headlines, surprise and forgetting are untouched.
        probeable: dict[HypothesisId, BeliefModule] = {
            FAIR_VALUE: self.fair_value,
            FLOW_INTENSITY: self.flow,
            FILL_RATE: self.fill,
            IMPACT: self.impact,
        }
        if self.flags.surprise_curiosity:
            self.scoring_modules: Mapping[HypothesisId, BeliefModule] = {
                hypothesis: SurpriseAsCuriosity(module)
                for hypothesis, module in probeable.items()
            }
        else:
            self.scoring_modules = probeable
        self.selection: SelectionRule = (
            NoReflexiveSelection() if self.flags.no_reflexive else select_candidate
        )
        self.homeostat_filter: VetoOnlyHomeostat | None = (
            VetoOnlyHomeostat() if self.flags.no_homeostat else None
        )
        self.null_projector: NullDistanceProjector | None = (
            NullDistanceProjector() if self.flags.no_homeostat else None
        )

        # -- proposer and workspace ----------------------------------------
        self.proposer = Proposer(
            modules=self.scoring_modules,
            trajectory=self.trajectory,
            motor_cfg=cfg.motor,
            probe_horizon_steps=cfg.fill_horizon_steps,
            selection=self.selection,
        )
        self.workspace = Workspace(
            registry=reg,
            proposer=self.proposer,
            modules=self.modules,
            motor_cfg=cfg.motor,
            consumers=(self.fair_value, self.flow),
        )

        # -- loop state ------------------------------------------------------
        self._pending_sent: tuple[ExchangeMessage, ...] = ()
        """Messages whose acks ride the NEXT observation (two-slot shift)."""
        self._sent_prev: tuple[ExchangeMessage, ...] = ()
        """Messages submitted with the previous cycle (acks one obs later)."""
        self._last_flow_mean = 0.0
        self._records: list[WorkspaceRecord] = []
        self._message_log: list[tuple[int, ExchangeMessage]] = []

    # -- read-only logs (P13 reads these; the env carries records opaquely) --

    @property
    def records(self) -> tuple[WorkspaceRecord, ...]:
        """Every cycle's WorkspaceRecord, in cycle order."""
        return tuple(self._records)

    @property
    def message_log(self) -> tuple[tuple[int, ExchangeMessage], ...]:
        """(observation step, message) for every message actually submitted."""
        return tuple(self._message_log)

    def rng_for(self, step: int, purpose: str) -> np.random.Generator:
        """The ONLY sanctioned source of agent-internal randomness (INV-9).

        Named counter-based stream keyed (actor_id="agent", step, purpose):
        deterministic given the root seed, and incapable of perturbing any
        other actor's draws. Every quantity the loop currently compiles is
        closed-form, so nothing consumes this yet; anything that ever
        samples (e.g. a Monte Carlo trajectory variant) must draw here.
        """
        return make_rng(
            self._root_seed,
            StreamKey(actor_id="agent", step=step, purpose=purpose),
        )

    # -- the harness driver ----------------------------------------------------

    def __call__(
        self, reset: Callable[[], Observation], step: StepHandle
    ) -> None:
        """Drive one full episode through the two harness handles.

        The episode ends when the step handle raises (EpisodeComplete),
        which is allowed to propagate — the harness catches it.
        """
        obs = reset()
        while True:
            record, action = self.cycle(obs)
            obs = step(action, workspace_record=record)

    # -- one cognitive cycle -----------------------------------------------------

    def cycle(
        self, obs: Observation
    ) -> tuple[WorkspaceRecord, ExchangeMessage | None]:
        """Run the full numbered cycle on one observation.

        Returns the cycle's WorkspaceRecord and the (at most one) message
        to submit this engine step. The record is complete on every cycle,
        null cycles included.
        """
        # 1. Ingest: pair the observation with the messages sent one
        # observation earlier (their acks ride this observation).
        self_events = SelfEvents(
            step=obs.step,
            messages_sent=self._pending_sent,
            acks=obs.own_acks,
            fills=obs.own_fills,
        )
        self.books.update(obs, self_events)
        # The impact model's context regressor for rows anchored at this
        # observation: the previous cycle's broadcast flow forecast mean
        # (DESIGN item 19) — set before ANY impact update this cycle.
        self.impact.set_context_regressors((self._last_flow_mean,))

        # 2. Realized-IG scoring for the previous cycle's probe (INV-10):
        # ledger snapshot as 'before', target update from THIS observation,
        # snapshot 'after' — before any other update could confound it.
        pending = self.ledger.pending
        resolved_target: HypothesisId | None = None
        if pending is not None:
            self.ledger.resolve_pending(
                self.modules[pending.target_id], obs, self_events
            )
            resolved_target = pending.target_id

        # 3. Belief updates for every remaining module — all posteriors
        # absorb the observation before any EIG is computed this cycle.
        for hypothesis in _UPDATE_ORDER:
            if hypothesis == resolved_target:
                continue
            self.modules[hypothesis].update(obs, self_events)
        self._summary.fold(obs)

        # 4. Slow tick: one regime observation from the public summary
        # statistics, then regime-gated forgetting on every module. The
        # forgetting map only becomes meaningful once the run-length
        # posterior's support extends past the recency window (before
        # that, every hypothesis is trivially "recent") — a structural
        # warmup, not a calibration.
        if obs.step > 0 and obs.step % self.config.slow_tick_every_steps == 0:
            stats = self._summary.slow_stats()
            self.regime.observe_summary(
                stats.trade_tempo,
                stats.realized_vol,
                stats.imbalance,
                stats.mean_depth,
            )
            if self.regime.n_ticks > R_RECENT:
                rho = self.regime.current_rho()
                for hypothesis in _UPDATE_ORDER:
                    self.modules[hypothesis].forget(rho)

        # 5a. WorldSummary. Pre-market (no two-sided book seen yet) there
        # is no mid to anchor a summary, a probe price, or a mark: the
        # cycle still logs a complete record and holds the explicit null.
        world = self._summary.build(self.regime.regime_posterior_summary())
        if world is None:
            record = self._premarket_record(obs)
            return self._act(record)

        # 6. Homeostat: the only consumer of the account-bearing view
        # (INV-5); exports only from here on. Hoisted immediately before
        # the workspace call, which consumes the exports (see module
        # docstring on 5/6 ordering).
        output = self.homeostat.evaluate(self.books.full_view(), world.mid_ticks)
        exports: HomeostatOutput = (
            self.homeostat_filter.filter(output)
            if self.homeostat_filter is not None
            else output
        )
        cognitive = self.books.cognitive_view(exports.distances)
        projector: DistanceProjector = (
            self.null_projector
            if self.null_projector is not None
            else BandDistanceProjector(
                cfg=self.config.homeostat,
                mid=world.mid_ticks,
                rolling_messages=self.homeostat.rolling_message_count,
                carried={
                    "drawdown": exports.distances.get("drawdown", 0.0)
                },
            )
        )

        # 5b/7/8. Appraise -> compete -> broadcast -> propose -> ignite.
        record = self.workspace.cycle(
            step=obs.step,
            world=world,
            cognitive=cognitive,
            drives=exports.drives,
            vetoes=exports.vetoes,
            corrective_intent=exports.corrective_intent,
            projector=projector,
        )

        # 9. Probe selected => ledger entry NOW, before acting (INV-10):
        # the promised EIG and the target's entropy snapshot at issuance.
        intent = record.intent
        if intent is not None and not intent.is_null and not intent.is_flatten:
            if record.eig_promised_nats is None:
                raise RuntimeError(
                    "committed probe ignited without a promised EIG; the "
                    "workspace record contract (DESIGN item 30) is broken"
                )
            target = self.modules[intent.target_id]
            self.ledger.open(
                step=obs.step,
                target_id=intent.target_id,
                eig_promised_nats=record.eig_promised_nats,
                snapshot_before=target.snapshot_entropy(),
            )

        # 10. Act: log the record, submit the head of the compilation.
        return self._act(record)

    # -- internals ------------------------------------------------------------

    def _act(
        self, record: WorkspaceRecord
    ) -> tuple[WorkspaceRecord, ExchangeMessage | None]:
        """Step 10: bookkeep the cycle's outcome and pick the submission.

        The harness accepts at most one message per engine step; the head
        of the compiled tuple goes out, and the rest is re-derived next
        cycle from fresh state (see module docstring).
        """
        action = record.compiled_messages[0] if record.compiled_messages else None
        # Two-slot shift (see module docstring): the previous cycle's
        # submission is answered by the NEXT observation; this cycle's
        # waits one more.
        self._pending_sent = self._sent_prev
        self._sent_prev = (action,) if action is not None else ()
        self.homeostat.record_messages(0 if action is None else 1)
        self._last_flow_mean = float(self.flow.predict().mean)
        self._records.append(record)
        if action is not None:
            self._message_log.append((record.step, action))
        return record, action

    def _premarket_record(self, obs: Observation) -> WorkspaceRecord:
        """A complete null-cycle record for observations before any
        two-sided book exists: every module still appraises honestly; the
        coarse marginals are 0 because no experiment is priceable yet."""
        headlines: list[Headline] = []
        for hypothesis in _UPDATE_ORDER:
            module = self.modules[hypothesis]
            forecast = module.predict()
            headlines.append(
                Headline(
                    hypothesis_id=hypothesis,
                    forecast_mean=forecast.mean,
                    forecast_var=forecast.variance,
                    epistemic_entropy_nats=module.posterior_entropy_nats(),
                    best_marginal_eig_nats=0.0,
                    last_surprise_z=module.surprise_z(),
                )
            )
        world = WorldSummary(
            mid_ticks=0.0,
            spread_ticks=0,
            imbalance=0.0,
            depth_profile=(0.0,) * N_LEVELS,
            trade_tempo=0.0,
            realized_vol=0.0,
            regime_posterior=self.regime.regime_posterior_summary(),
        )
        return WorkspaceRecord(
            step=obs.step,
            world_summary=world,
            headlines=tuple(headlines),
            self_state=self.books.cognitive_view(),
            focus=None,
            intent=null_intent(FAIR_VALUE),
            eig_promised_nats=None,
            compiled_messages=(),
        )
