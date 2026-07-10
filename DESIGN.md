# TOPOS-Market — Design Document

## 1. Background

TOPOS-Market is a Global-Workspace-Theory-style cognitive architecture instantiated as a
LOB trading agent. Core philosophy: adaptive, curiosity-driven behavior must emerge from
architectural structure. There is NO reinforcement learning, NO reward maximization, NO
profit objective, and NO weight updates during deployment. Adaptation happens exclusively
through Bayesian state estimation inside fixed functional forms: conjugate/analytic
posterior updates plus regime-gated forgetting. Curiosity is prospective EXPECTED
INFORMATION GAIN (EIG) over explicit parameter posteriors — never retrospective prediction
error, never predictive variance. The agent is part of the market it models: it maintains
a self-model (fill rates, impact, queue position, inventory trajectory) so that its own
trading becomes predictable, hence boring. Profit, if it appears, is a measured outcome,
never an input to any decision.

The cognitive cycle: perceive -> appraise -> compete -> broadcast -> propose -> ignite
intent -> act -> observe -> update posteriors. A bounded, typed workspace (blackboard)
holds: world summary, hypothesis headlines (forecast + epistemic entropy + last surprise,
capacity-limited), cognitive self-state, the current focus (the single question that won
salience competition), and the selected intent. The workspace record logged each cycle IS
the interpretability story.

## 2. Invariants

INV-1  No reward channel. env.step(action) returns an Observation only. No scalar feedback
       signal exists anywhere in agent-facing code.

INV-2  No learning frameworks, no gradients. Adaptation = closed-form posterior updates +
       forgetting. torch/jax/tensorflow/sklearn must not be importable from agent packages.

INV-3  Curiosity quantities are computed on PARAMETER posteriors as mutual information
       I(theta; Y | action) — never predictive variance or predictive entropy alone.

INV-4  The null action (observe, place nothing) is a first-class candidate with its own
       EIG > 0 in an active market. Probes are scored by MARGINAL EIG over null.

INV-5  PnL is state, never score. Arbitration and proposal code receive SelfStateCognitive,
       which contains NO PnL fields. PnL reaches only the homeostat (as drawdown
       distance-to-bound) and the metrics package.

INV-6  Homeostat variables are set-points, not maximands: drive is exactly zero inside the
       soft band, grows superlinearly between soft and hard bounds, and a hard veto fires
       at the hard bound. There is no signal that rewards being "extra safe."

INV-7  Salience consequence-weights w_h derive from the module dependency graph (registry
       centrality), computed once at startup — never from outcome statistics of any kind.

INV-8  The motor compiler is a deterministic pure function of its declared inputs. No
       randomness, no hidden state. Intent and compiled messages are logged side by side.

INV-9  All environment randomness flows through named counter-based RNG streams keyed by
       (actor_id, step, purpose). The agent's presence must not perturb any other actor's
       random draws except causally through the visible book state.

INV-10 Realized information gain is computed from entropy snapshots: H(target posterior)
       captured immediately BEFORE the outcome-driven update, minus H immediately AFTER.

INV-11 The agent never observes ground-truth queue position or engine-side account state.
       Those flow only through harness-only channels into metrics/validation.

## 3. Modules

Each module is its own package under `topos/`. The `contracts` package (this task) holds
the frozen data contracts everything else speaks through.

| Package            | Phase(s)     | Responsibility |
|--------------------|--------------|----------------|
| `topos/contracts`  | P0 (this task) | Frozen dataclasses, enums, protocols, module registry, named RNG streams. No market or cognition logic. |
| `topos/env`        | P1–P3        | Matching engine, background market, test harness. `env.step(action)` returns an `Observation` only. |
| `topos/beliefs`    | P4, P5, P11  | World predictors — Kalman fair-value, Poisson-Gamma flow intensities — plus the shared `BeliefModule` protocol and EIG machinery (P4), queue-position filter (P5), regime tracker + forgetting (P11). |
| `topos/selfmodel`  | P6           | Bookkeeping from acks/fills, Beta-Bernoulli fill model, Bayesian-linear impact model, inventory-trajectory forecast, reflexive self-uncertainty. |
| `topos/drives`     | P7           | Homeostat — set-point bands with superlinear drives and hard vetoes. The only agent package that consumes `SelfStateFull`. |
| `topos/proposer`   | P8           | Experiment/probe generation and EIG scoring (marginal over null). |
| `topos/workspace`  | P9           | Blackboard, salience competition, arbiter, broadcast. |
| `topos/motor`      | P10          | Deterministic intent -> message compiler. |
| `topos/agent`      | P12          | The integrated loop. |
| `topos/metrics`    | P13          | Evaluation, ablations, falsification suite — agent code MUST NOT import it (mechanically enforced). |

## 4. The cognitive cycle

One engine step corresponds to one pass through:

1. **Perceive.** The environment delivers an `Observation` (book levels, public trades,
   own acks/fills). The selfmodel folds acks/fills into `SelfEvents` and the working-order
   bookkeeping.
2. **Appraise.** Each belief module performs its closed-form update, then publishes a
   `Headline`: forecast mean/variance, parameter-posterior entropy, best marginal EIG,
   and the z-scored surprise of the last observation.
3. **Compete.** Salience competition over headlines. Consequence-weights come from
   registry centrality (structural, fixed at startup — INV-7); the homeostat can enter
   the competition with a drive-driven, potentially preemptive bid (INV-6).
4. **Broadcast.** The winning `Focus`, the `WorldSummary`, and the `SelfStateCognitive`
   (PnL-free — INV-5) are broadcast to all consumers.
5. **Propose.** The proposer generates candidate `ProbeSpec`s aimed at the focused
   hypothesis. The null action is always a candidate; every probe is scored by marginal
   EIG over null (INV-4), where EIG is mutual information on parameter posteriors (INV-3).
6. **Ignite intent.** The arbiter selects one `Intent` (possibly null; possibly the
   distinguished flatten intent when a hard veto fires) and records `eig_promised_nats`.
7. **Act.** The motor compiles the intent into `ExchangeMessage`s as a deterministic pure
   function (INV-8). Intent and compiled messages are logged side by side.
8. **Observe.** The environment matches, advances the background market (all randomness
   through named RNG streams — INV-9), and produces the next `Observation`.
9. **Update posteriors.** For the probed hypothesis, an `EntropySnapshot` is captured
   immediately before the outcome-driven update and again immediately after; realized
   information gain is their difference (INV-10). Regime-gated forgetting discounts
   sufficient statistics toward the prior.

The `WorkspaceRecord` assembled each cycle (step, world summary, headlines, cognitive
self-state, focus, intent, promised EIG, compiled messages) is the interpretability story:
behavior is a lossless readout of internal state.

## 5. Data contracts

All contracts are frozen dataclasses (or enums/protocols) with integer ticks and integer
lots everywhere. They live in `topos/contracts/`:

- `market.py` — `Side`, `PlaceLimit`, `Cancel`, `ExchangeMessage`, `Ack`, `Fill`,
  `BookLevel`, `Trade`, `Observation` (exactly 6 fields; `N_LEVELS = 10` per side).
- `intent.py` — `HypothesisId`, the six known hypothesis ids, `Intent`,
  `NULL_THRESHOLD`, and the distinguished flatten constructor.
- `workspace.py` — `Headline`, `WorkingOrderView`, `SelfStateCognitive`,
  `SelfStateFull` (+ `cognitive_view()` projection), `Focus`, `WorldSummary`,
  `WorkspaceRecord`.
- `beliefs.py` — `SelfEvents`, `ForecastStats`, `EntropySnapshot`, `ProbeSpec`, the
  `BeliefModule` protocol, and `realized_information_gain_nats`.
- `registry.py` — `ModuleDecl`, `ModuleRegistry` with `centrality_weights()`
  (normalized out-degree, frozen after first computation).
- `rng.py` — `StreamKey`, `make_rng` (SHA-256-keyed numpy Philox/SeedSequence).

## 6. Mechanical enforcement (tripwires)

`tests/tripwires/` enforces the invariants that can be checked without any market or
cognition logic existing yet:

| Tripwire | Enforces |
|----------|----------|
| `test_no_reward_tokens` | INV-1 — feedback-signal vocabulary is absent from all agent-facing source (comments and strings included; `topos/metrics/` exempt). |
| `test_no_learning_frameworks` | INV-2 — importing every agent package pulls in no torch/jax/tensorflow/sklearn; static import scan agrees. |
| `test_metrics_isolation` | metrics boundary — no agent package imports `topos.metrics`, by any import form. |
| `test_cognitive_view_has_no_pnl` | INV-5 — no field name matching `pnl|profit|drawdown|wealth` reachable from `SelfStateCognitive`; `SelfStateFull` is not a subtype of the cognitive view. |
| `test_contracts_frozen` | contract stability — every contract dataclass is frozen and mutation raises. |
| `test_rng_stream_independence` | INV-9 — a stream's draws are identical whether or not any other stream was consumed first. |
| `test_observation_shape` | INV-1/INV-11 — `Observation` has exactly the declared six fields. |

INV-3, INV-4, INV-6, INV-7 (behavioral half), INV-8, and INV-10 acquire executable
tripwires as their modules are implemented (P4+); their structural halves are already
pinned by the contracts (`BeliefModule.eig_nats` signature and docstring, `Headline.
best_marginal_eig_nats`, `ModuleRegistry` freezing, `EntropySnapshot`).

As of P4, INV-3's behavioral tripwire is `tests/beliefs/test_eig_matches_monte_carlo.py`
(eig_nats must equal a brute-force Monte Carlo estimate of parameter–observation mutual
information; predictive entropy alone fails it by the aleatoric term), backed by
`test_eig_saturation.py` and `test_noisy_tv.py` (curiosity saturates; surprise does not
leak into EIG). INV-10's mechanics are pinned by the snapshot tests in
`tests/beliefs/test_fair_value.py` / `test_flow_intensity.py`.

## 7. Open questions

Concerns noted during scaffolding, implemented as specified — recorded here rather than
deviated on:

1. **Spec-internal conflict: the Observation NOTE vs tripwire 1.** The spec asks for a
   code comment on `Observation` enumerating its deliberately-absent fields, one of which
   is the exact token tripwire 1 forbids anywhere under `topos/` (the scan has no
   comment/string exemption, and adding one would weaken the tripwire). Resolution: the
   tripwire stays maximally strict; the code comment paraphrases ("any scalar feedback
   signal"); the verbatim enumeration lives here in DESIGN.md, which the scan excludes.
2. **`NULL_THRESHOLD` value unspecified.** Set to 0.5 provisionally. Nothing downstream
   is calibrated yet; revisit when the arbiter (P9) is implemented.
3. **`FLATTEN_INTENT` is named like a constant but must be parameterized.** Flatten
   direction depends on the sign of current inventory, so a fixed `Intent` value cannot
   express it. Implemented as `flatten_intent(inventory_lots)` with the spec name bound
   as an alias (`FLATTEN_INTENT = flatten_intent`). Zero inventory yields a
   null-commitment intent. Chosen encoding of "passive-first": `patience = 1.0`,
   non-negative `offset_ticks`; the motor (P10) owns the actual passive-then-cross
   escalation.
4. **`SelfEvents`, `ForecastStats`, `EntropySnapshot` referenced but never specified.**
   Defined minimally in `contracts/beliefs.py`: `SelfEvents = (step, messages_sent,
   acks, fills)`; `ForecastStats = (mean, variance)`; `EntropySnapshot =
   (hypothesis_id, step, entropy_nats)`. Extend if P4 needs more.
5. **`WorkingOrderView` fields unspecified** beyond "incl. queue-rank posterior
   mean/var". Chose `(order_id, side, price_ticks, size_lots_remaining, age_steps,
   queue_rank_mean, queue_rank_var)`.
6. **`SelfStateFull` is deliberately NOT a subclass of `SelfStateCognitive`.** The spec
   phrase "SelfStateCognitive fields + ..." could be read as inheritance, but subtyping
   would let a PnL-bearing object satisfy any interface typed `SelfStateCognitive`,
   gutting INV-5 at the type level. Implemented as an independent frozen dataclass with
   the fields repeated plus a `cognitive_view()` projection; a tripwire asserts the
   non-subclass relationship.
7. **Registry centrality choice.** Normalized out-degree (documented option), self-loops
   excluded, uniform fallback when the declared graph has no edges, `ValueError` on an
   empty registry. `centrality_weights()` freezes the registry on first call and caches
   forever — "computed once at startup" is thereby mechanical, not conventional.
8. **Registry keys vs `HypothesisId`.** `centrality_weights()` is typed
   `Mapping[HypothesisId, float]` per spec, but registrants are modules. Assumed
   convention: a hypothesis-owning module registers under its `hypothesis_id`; purely
   mechanical modules (motor, env) get weights too, which the salience competition simply
   never looks up.
9. **`Observation` construction enforces `N_LEVELS` per side** (pad thin books with
   `size_lots = 0` levels). The spec fixes the length but is silent on padding; the
   padding convention is recorded on `BookLevel`.
10. **Intent field validation.** `Intent.__post_init__` rejects out-of-range values
    (`side`, `size_frac`, `patience`, `commitment`, empty `target_id`). The spec gives
    the ranges but does not say whether construction should enforce them; enforcing at
    the contract boundary was judged the conservative reading.

Recorded during P4 (world beliefs, 2026-07-09):

11. **FairValueKF noise uncertainty is a single common scale.** The spec's "uncertain
    observation/state noise scales tracked via conjugate inverse-gamma posteriors" is
    implemented as R = c·r0, Q = c·Q0 with known shapes (r0, Q0) and ONE uncertain scale
    c ~ InverseGamma — the West & Harrison unknown-variance DLM. Both noise scales are
    genuinely uncertain (perfectly correlated a priori); this is the only form with an
    EXACT conjugate inverse-gamma update (standardized squared innovations), exact
    Student-t predictives, and exactly calibrated credible intervals. Independent scales
    for R and Q admit no exact conjugate update — only iterative variational
    approximations, against the spirit of INV-2's "conjugate updates only".
12. **FairValueKF `eig_nats` is PARAMETER EIG only; state EIG is exposed separately.**
    "Parameter uncertainty is the epistemic part; the Kalman state covariance alone is
    not sufficient" (INV-3) is read as: curiosity = I(c; Y). The closed form
    0.5·ln(det Σ_prior/det Σ_post) about the latent state is implemented as
    `state_eig_nats` (MC-verified) but excluded from `eig_nats`: a steady-state filter
    earns constant state information every step forever, so including it would make
    curiosity unsaturatable — the churn the saturation/noisy-TV tests forbid.
13. **World-model EIG is intent-independent (and, for fair value, horizon-independent).**
    Microprice and public flow are observed passively, so an order-placing probe earns
    exactly the EIG of the null at the same horizon: marginal EIG over null is 0 for
    world predictors, and the null action carries their EIG (INV-4). Probes that CAN buy
    information belong to the self-model hypotheses (P5/P6). The probed observable is the
    microprice at horizon end (fair value) / counts aggregated over the horizon (flow);
    for the fair-value scale family, I(c; y_h) is invariant to the h-dependent variance
    multiplier, hence horizon-independent — a property, not a bug (pinned by the MC test
    at h ∈ {1, 5}).
14. **FlowIntensity extraction conventions and the P1 trade-print gap.** Counts are in
    lots; 18 cells = {arrival, cancel, market} × side × {touch ≤0, near 1–3, deep ≥4
    ticks from the pre-step best}. Own footprint is subtracted so cells model BACKGROUND
    flow: own resting placements leave arrivals, own cancellations leave cancels, and own
    TAKER fills net the passive side's decrease — necessary because agent-caused prints
    never appear in `Observation.trades` (committed P1 behavior, now an explicit P4
    decision: the flow model sees background-caused prints only; P13 must score it
    accordingly). Message→ack pairing is positional (k-th placement ack ↔ k-th
    PlaceLimit); level departures/arrivals caused by the 10-level visibility window are
    accepted as extraction noise.
15. **Forgetting monotonicity is a converged-regime property.** With S ← ρS + (1−ρ)S0 as
    specified, a single extreme outlier can leave posterior entropy momentarily ABOVE the
    prior direction of travel (a shocking observation widens an inverse-gamma posterior),
    in which case an immediate forget could reduce entropy. The formula is implemented
    exactly as specified; the non-decreasing property is asserted where it is a theorem —
    after convergence and at the prior — matching the intent (reinflation after regime
    shifts, P11).

Recorded during P6 (self-model, 2026-07-09):

16. **SelfEvents timing convention.** The action submitted alongside the
    observation stamped s executes in engine step s; its acks/fills (stamped s)
    arrive in the observation stamped s+1. `SelfEvents` therefore groups an
    observation with the messages sent ONE observation earlier — the messages
    whose acks it carries — and item 14's positional pairing applies within that
    grouping. P12 must build `SelfEvents` this way (pinned by the P6 harness
    test, which fails under same-observation pairing).
17. **Bookkeeping: live views fold immediately; ground-truth claims replay in
    stamp order.** One observation can deliver fills stamped both s-1 (own
    action at s-1) and s (background maker fills of step s), and "account at end
    of engine step k" means exactly "fills stamped <= k" (the P3 hook's
    contract). `Books.claims()` therefore replays observed fills by stamp while
    the live views stay maximally fresh. Realized PnL is average-cost; the
    method-independent identity realized + unrealized = cash + inventory * mark
    is pinned by tests.
18. **Fill-model trial protocol.** Buckets are (side, cross/touch/near/deep,
    imbalance tripartition at ±1/3), with depth edges inherited verbatim from
    the flow model's BANDS; bucketing uses the DECISION-time context (the
    observation before the ack). Partial-by-horizon updates the Beta cell with
    the filled fraction (one fractional pseudo-trial); own cancel strictly
    before the horizon CENSORS the trial (discarded — counting it as no-fill
    would bias every bucket downward in proportion to motor impatience);
    cancel/expiry at-or-after the horizon resolves at the observed fraction.
    The null action's EIG through fill_rate is exactly 0: self-model
    information must be BOUGHT by acting — the complement of item 13.
19. **ImpactModel is the unknown-variance form again.** The NIG regression
    reuses `InverseGammaPosterior` via standardized squared residuals
    (y - m·x)² / (1 + xᵀVx), so P4's conjugate cell is the single noise-scale
    mechanism in the codebase; EIG is Student-t predictive entropy minus
    ½(ln 2πe + ln b - ψ(a)), closed form, MC-verified. The own-effect variance
    handed to the trajectory compiler is coefficient uncertainty only
    (ΔxᵀVΔx · E[σ²]): the residual mid noise is the fair-value model's account
    and would otherwise be double-booked. Context regressors are a
    fixed-dimension slot (default 1: the flow headline mean) set per cycle.
20. **SelfTrajectory compiles by moment matching over an exact fill-outcome
    enumeration** — not Monte Carlo, so the compiler is deterministic (in the
    spirit of INV-8). Documented approximations: fills at horizon start
    (exposure upper bound), impact permanent over H, per-order independence
    given the bucket means, value-change-given-inventory moment-matched to a
    Gaussian discretized on the tick-lot grid (+1/12 unit-cell variance — a
    property of the integer grid, giving a deterministic forecast ~0 nats
    instead of a divergent differential entropy). Joint entropy is exact by the
    chain rule. Default horizon = the fill model's horizon, the only horizon
    the fill posteriors answer without extrapolation.
21. **Learning does not uniformly lower self-entropy — and should not.** A
    settled fill posterior (p→1) commits forecast weight to the EXPOSED branch,
    which can raise the total self-entropy of an aggressive intent from flat
    relative to an ignorant posterior (p≈0.5) that hedges across branches. The
    ordinal tests therefore isolate the channels: fill ignorance maximizes
    INVENTORY entropy; impact ignorance raises VALUE entropy at a fixed fill
    posterior. Ordering total entropies across posteriors with different fill
    means is not a valid test and was deliberately not asserted.

Recorded during P8 (proposer, 2026-07-09):

22. **Coarse-menu encodings and the proposer's world input.** Offsets are measured
    from the mid per the Intent contract: touch = half-spread, deep = half-spread + 4
    (the shallowest price of the "deep" band the flow and fill models share), small
    marketable = -half-spread at ONE lot (the size quantum — the smallest intervention
    that still exercises the aggression channel). Quote shapes carry patience 1.0 so
    scoring never bundles staleness cancels of unrelated orders into a probe;
    cancel-refresh re-quotes the working order's remaining size at the touch with
    patience 0. Message cost and motor legality come from the motor's pure `compile`
    (INV-8 makes calling it during proposal exact forecasting, not action) against a
    book reconstructed from the broadcast `WorldSummary` (one live level per side at
    the implied touch): the proposer's world input is the broadcast, and only touch
    prices matter to compilation. All ProbeSpecs carry one horizon — the fill model's,
    wired at construction — and marginals compare candidate vs null through the same
    module at the same horizon, so the choice cancels out of every comparison.

23. **Gates and the INV-5 boundary.** The proposer defines a `DistanceProjector`
    protocol and receives an implementation injected per cycle; it imports nothing
    from drives/ (pinned by a package-local source scan that also bans account-state
    vocabulary). Gate (a3): the one-step self-forecast's inventory pmf must keep every
    projected distance inside the soft bands with confidence 1 - delta, delta = 0.05 —
    the per-tail mass of the `interval(0.9)` convention already fixed in the conjugate
    cells. Variables not predictable from the cognitive view are carried forward at
    their current distances by the projector (the no-information forecast), so a
    breach of an account-side band gates every candidate out and rule (d) hands the
    cycle to the corrective fallback without the proposer ever seeing account state.

24. **Churn extinction lives in rule (c), not in a threshold on EIG.** Conjugate EIGs
    approach 0 asymptotically without reaching it, so "strictly positive marginal"
    alone would let a saturated bucket be probed forever. EPSILON_EIG = 0.02 nats —
    the convergence level the P4/P6 saturation tripwires already pin — defines the
    boredom band: once the top marginal falls inside it, the null (marginal exactly 0,
    gates permitting) joins the tie-break pool and wins on minimum self-entropy,
    watching being the most self-predictable action on any menu. The strict-positive
    eligibility test in rule (b) handles world hypotheses, whose probes' marginals are
    exactly 0 (item 13).

25. **Flatten never competes on EIG.** It targets SELF_TRAJECTORY (A2), which carries
    no experiment bookkeeping; if a flatten-shaped action is genuinely the best
    experiment for the focus, the refined grid contains the equivalent probe with
    `target_id` = focus. Flatten enters selection only as the rule-(d) fallback, at
    the constructor's full default size — partial-correction sizing remains the
    homeostat's business (A1). The refined stage runs whenever the coarse winner's
    marginal is strictly positive, however small, so the epsilon logic of item 24 is
    exercised rather than short-circuited.

26. **Null-intent bookkeeping target.** The emitted null carries the focus id when
    the focus is a probeable module; otherwise (no focus yet, or the passive-only
    REGIME, which per A3 never appears on an Intent) it carries FAIR_VALUE, the
    archetypal hypothesis whose information rides the null. The target id on a null is
    bookkeeping only — no module's scoring reads it — and per the standing ruling the
    null's commitment is 0.0 exactly.

Recorded during P9 (workspace and arbiter, 2026-07-10):

27. **S_MIN is derived, not invented.** The ignition threshold is
    `EPSILON_EIG_NATS / len(KNOWN_HYPOTHESIS_IDS)` (= 0.02/7): the salience of a
    boredom-band question carried at uniform structural centrality — both factors are
    scales the architecture already owns (the saturation tripwires' convergence level
    and the registry's no-edges fallback weight). A homeostat drive crosses it at
    u ≈ 0.05, so a hair past the soft band does not seize the workspace. The threshold
    is strict (salience must exceed it); ties in the competition break homeostatic
    first, then lexicographic id.

28. **World hypotheses never win focus — by arithmetic, not by rule.** Salience uses
    the proposer's MARGINAL EIG (the coarse-menu output the spec names), which is
    exactly 0 for fair value, flow, and regime (items 13/22): the workspace attends
    only where information is for sale (self-model hypotheses and drives), and world
    information rides the null. Consequence: in the integrated loop the two
    focus-conditioned world modules run in their coarse/quoted regimes essentially
    always. This is the attention economy working as designed, not a starvation bug —
    their posteriors stay exact (item 31) and their EIG never feeds their own
    marginals.

29. **Two proposer calls per cycle resolve the focus/menu chicken-and-egg.** Stage 1
    must run before the competition (headlines need the coarse marginals) but the
    refined menu needs the winner, so the workspace calls `propose(focus=None)` for
    the headline input and, when a hypothesis wins, `propose(focus=winner)` for the
    refined menu — stage 1 is recomputed inside the second call (cheap, closed-form).
    The arbiter takes `proposal.selected` verbatim; it re-verifies the coalition
    (positive marginal AND gates passed for any committed non-flatten intent) and
    raises `CoalitionError` on violation rather than re-implementing or patching over
    the selection rule.

30. **Record conventions.** `WorkspaceRecord.intent` is never None: a quiet cycle logs
    an explicit null intent (commitment 0.0 exactly, FAIR_VALUE bookkeeping target per
    item 26). `eig_promised_nats` is None for any null intent (spec), the selected
    candidate's TOTAL `eig_nats` for a committed probe (the number INV-10's realized
    IG on the target module is compared against — not the marginal), and 0.0 for
    flatten/corrective intents, which promise action, not information. A homeostatic
    focus whose corrective intent is None (e.g. message budget — its correction is to
    stop sending) resolves to the null. Record messages are compiled against
    `book_from_summary` (the gates' documented approximation): the broadcast is the
    workspace's world input, and the record is emitted before anything is submitted.

31. **Broadcast conditioning changes granularity and timing, never posterior limits.**
    The hook contract (`condition_on_focus(focus)`, explicit registration, validated
    at construction): focus is permission to spend, and must never change what a
    posterior would eventually converge to. FlowIntensity buffers unfocused evidence
    per cell and flushes on refocus — EXACT by Gamma-Poisson batchability (sum counts,
    sum exposure), pinned by a twin test; its coarse aggregate cell (prior = sum of
    cell priors) is exactly the posterior the fine cells induce on the total rate,
    because every cell shares one exposure path; fine and coarse surprise live on
    separate z-trackers; `forget()` flushes first (evidence precedes the discount).
    FairValueKF gates its parameter-EIG quadrature: unfocused curiosity is quoted from
    the last focused refresh. Both modules default to focused, so standalone use (and
    every pre-P9 test) is the full-fidelity path.

Recorded during P12 (integrated agent, 2026-07-10):

32. **One message per engine step; the compilation's head is submitted.**
    The P3 harness accepts at most one `ExchangeMessage` per `step()` call
    (committed P1-P3 behavior). The agent submits the FIRST message of the
    cycle's compiled tuple and deliberately queues nothing: the arbiter
    re-evaluates every cycle from fresh state, so a cancel-then-replace
    compilation completes across consecutive steps as the working-order
    view updates, whereas queued leftovers would execute against a book
    they were not compiled for. INV-8's record stays lossless — the intent
    and the FULL compilation are logged side by side; the message log holds
    what was actually sent.

33. **SelfEvents pairing is a two-slot shift in the integrated loop.** The
    harness builds each step's observation BEFORE applying that step's
    agent action, so the observation returned by `step(a)` never answers
    `a`: acks for the action decided in cycle k arrive in cycle k+2's
    observation (one ENGINE step later — the reset observation and the
    first step observation share stamp 0). Item 16's convention is
    implemented exactly as the P6 harness test's sent/pending shuffle;
    pairing one observation too early silently loses every fill from
    bookkeeping (caught by the P3 hook during P12 bring-up).

34. **The proposer's scoring map is {fair_value, flow_intensity, fill_rate,
    impact} — queue_position and regime are appraised but not scored.**
    REGIME is passive-only (A3). QUEUE_POSITION's watch-EIG is
    intent-independent (the P5 observation model: the rank question is
    answered by watching the level; a fresh placement's point-mass prior
    carries ~0 first-step EIG), so its marginal over null is 0 by the same
    arithmetic that keeps world hypotheses out of focus (item 28). Both
    still update, headline, and forget normally; the workspace reads their
    headline marginals as the 0 they would compute, without paying the
    queue filter's Monte-Carlo EIG once per menu shape per cycle. This is
    also the module-map shape the P8/P9 suites pinned.

35. **Homeostat evaluation is hoisted immediately before the workspace
    call.** The spec's numbered order interleaves appraisal (5) before the
    homeostat (6), but P9's committed `Workspace.cycle()` runs
    appraise->compete->broadcast->propose->ignite atomically and CONSUMES
    the homeostat exports. Appraisal reads nothing the homeostat writes,
    so evaluating between the WorldSummary build (5a) and the workspace
    call preserves every data dependency of the numbered order.

36. **FlowIntensity memoizes its EIG on a sufficient-statistic
    fingerprint.** The module's EIG is intent-independent — a pure function
    of (per-band posteriors, horizon) — and the standing coarse menu asks
    for it once per shape per cycle (~20x per cycle, ~95% of loop runtime
    before the memo). Entries are validated against the tuple of every
    cell's (a, b), so any mutation — update, flush, forget, or a test
    poking a cell directly — forces recomputation. Pure caching; no
    behavioral change (and while unfocused the cells buffer, so the cache
    also implements item 31's "quoted from the last refresh" literally).

37. **Pre-market cycles are explicit null records.** Until a two-sided
    book has ever been seen there is no mid to anchor a WorldSummary, a
    probe price, or a mark; fabricating mid 0 would compile orders at
    price ~0. Such cycles still ingest, update every module, and emit a
    COMPLETE WorkspaceRecord (headlines appraised honestly, marginals 0,
    explicit null intent, no messages). Once a mid exists it is carried
    forward through momentarily one-sided books, mirroring `Books.mark`.

38. **Regime-gated forgetting arms after R_RECENT slow ticks.** BOCPD's
    run-length support is bounded by the number of ticks observed, so for
    the first R_RECENT ticks P(run < R_RECENT) = 1 vacuously and
    `current_rho()` would command near-maximal forgetting every tick,
    erasing all warmup learning. The agent applies `forget(current_rho())`
    only once `n_ticks > R_RECENT` — the earliest tick at which "a recent
    changepoint" is distinguishable from "the market just opened". A
    structural warmup, not a calibration.

39. **Ablation flags substitute strategy objects at construction.** Each
    flag replaces exactly one object at exactly one injection point
    (scoring-map wrappers; frozen self-model subclasses; an alternative
    selection rule — `Proposer` gained a default-preserving `selection`
    constructor parameter as the seam; a homeostat export filter plus null
    projector). With all flags off none of the strategy classes is even
    instantiated. The experiment ledger holds one probe in flight
    (structural: at most one intent per cycle, resolved at the start of
    the next), and its resolution routine is the single place INV-10's
    snapshot order is implemented; the broken orderings live only in the
    P12 test suite as mutants.

Recorded during P13 (metrics, ablation, falsification, 2026-07-11):

40. **Measurement stack and boundary.** Each metric is a pure function
    `RunData(s) -> tidy table + serializable summary`; `RunData` bundles the
    harness RunLog with the agent-side channels P12 committed for P13
    (records, message log, experiment ledger) plus an end-of-run impact-
    posterior snapshot and the null-agent twin replay. A repo-wide tripwire
    (`test_metrics_importers`) allowlists tests/ and experiments/ as the only
    importers of `topos.metrics`. The experiment configs' environment knobs
    (regime hazard lowered to 0.002, scheduled switches at thirds, fixed
    literal seed lists) are measurement design — no agent constant is set
    anywhere in metrics/ or experiments/.

41. **F5 scope: world hypotheses cannot be ledgered under FULL.** Their probe
    marginals are exactly 0 (items 13/28), so the arbiter never opens a
    fair_value/flow_intensity experiment and their information rides the
    untracked null. The falsification spec names those two hypotheses; the
    executable F5 binds where entries exist. SURPRISE_CURIOSITY does ledger
    world hypotheses (surprise-driven commits) and those slopes are reported.

42. **Ledger resolution predates the probe's own evidence for fill_rate.**
    The ledger resolves on the NEXT cycle's observation, but acks arrive two
    observations after the decision (item 33) and a fill trial adds the fill
    horizon on top. fill_rate's ledger realized-IG therefore measures the
    resolution of EARLIER experiments: realized/promised ratio ~0.45, slope
    ~0 — F5's named INV-10-wiring suspicion, confirmed as wiring, not design.
    impact's evidence (the next mid move) does arrive inside the window, so
    F5 binds on impact alone (slope 0.53 at the CI seeds, in band); fill_rate
    is reported both ways (ledger + a records-based windowed estimator that
    is itself contaminated by neighboring probes). Candidate future fix,
    recorded not applied: resolve the ledger when the probe's own acks
    arrive (two-slot, per item 33) — the resolution routine is committed P12
    behavior pinned by its own suite.

43. **The drive lock — the ablation's headline finding.** Drawdown is
    measured against an all-time peak and has no corrective action (the
    homeostat's only corrective is the inventory-excess flatten), so once
    drawdown exceeds its soft band while inventory sits inside its own, the
    drive bids every cycle, outbids curiosity's salience, and resolves to
    null — an ABSORBING quiescent state. Trace: after a regime switch the
    impact marginal reopens to ~1.4 nats and loses the competition to
    drawdown for 600 consecutive cycles. Every homeostat-bearing condition
    locks within a few hundred steps; NO_HOMEOSTAT shows the pure satiation
    curve (probe rate 0.94 -> 0.02 over ~400 steps). Consequences at the
    fixed CI seeds, pinned as falsifications by tests/acceptance/ per the
    fairness rules (results, not bugs): F1 INVERTED (FULL median 113.5
    messages — mostly flatten/corrective churn — vs SURPRISE 35, whose
    z-scored surprise self-normalizes instead of churning); F2's second leg
    falsified (NO_SELF_MODEL's probing decays too — throttled, not satiated;
    both decay CIs exclude 0); F3 REVERSED (NO_REFLEXIVE mean |inventory|
    6.6 vs FULL 9.0); F7 behavioral ratio 1.0 with 16/28 switch windows
    fully quiescent, while the EPISTEMIC EIG-offer ratio is 1.25 — curiosity
    reawakens, behavior cannot follow. F4 (soft-bound excursion time), F5
    (impact slope), and F6 (babbling decay) are confirmed.

44. **Behavioral decay conflates satiation with throttling; measured raw.**
    The message budget binds within the same first window the babbling burst
    occupies (soft 10 per 20 steps at the 1-message/step interface cap), so
    no trim separates budget priming from satiation. The suite fits raw
    decay uniformly and adjudicates cause via NO_HOMEOSTAT (throttle-free)
    and the epistemic series. Estimator: early-vs-late Poisson log-ratio
    with continuity correction — full quiescence must read as the strongest
    decay, where a binned log-OLS hits its epsilon floor and biases k to 0.

45. **Instrument conventions (deliberate, uncorrected biases documented).**
    Flow-calibration truth = twin-run full-depth book-diff lots (background-
    only by construction; the agent extracts from a 10-level window — the
    gap is a finding, not noise). Flow parameter variance recovered exactly
    as NB predictive variance minus mean. Regime-detection window 100 steps
    (BOCPD needs ~3-5 slow ticks; a shorter window files the detection under
    "stable" and inverts the contrast). Reawakening windows 100 steps with
    continuity-corrected ratios — a 0/0 window (slept through the switch) is
    evidence of no reawakening and reads as 1.0, never dropped. Impact
    validation compares the end-of-run posterior against per-action run-vs-
    twin divergence deltas, with deep placements as a placebo group.

### Adjudication (design review, 2026-07-08)

All ten resolutions reviewed against the running code and **accepted**, with items 1, 6,
and 7 confirmed as the intended readings (6 in particular: non-subclassing is the correct
structural enforcement of INV-5 and is now pinned by a tripwire). Three follow-up
amendments were applied during the review — these are deliberate, recorded contract
changes, not drift:

A1. **`flatten_intent` gained a `size_frac` parameter** (default 1.0, universal meaning:
    fraction of per-step size budget). Rationale: the homeostat's corrective intent must
    be *sizable* ("sized to re-enter the soft band", P7) — with `size_frac` hardcoded the
    homeostat could only ever flatten at full budget. Partial corrections shed the excess
    over the soft band across cycles; the homeostat re-evaluates every cycle.

A2. **`Intent.is_flatten` property added**, true iff `target_id == SELF_TRAJECTORY` and
    the intent is committed. Rationale: the motor (P10) applies special flatten
    compilation semantics and needs an explicit key, not an undocumented convention.
    Sound because SELF_TRAJECTORY is a forecast compiler, never a probeable hypothesis:
    no proposer experiment targets it, so a committed intent carrying it is a
    flatten/corrective by construction.

A3. **`REGIME = "regime"` reserved in `contracts/intent.py`** and added to
    `KNOWN_HYPOTHESIS_IDS` (now 7). Rationale: P11 introduces the id; reserving it now
    prevents a name mismatch, and P12 should assert at agent construction that
    `centrality_weights()` covers every id in `KNOWN_HYPOTHESIS_IDS`. Regime is
    passive-only — it never appears as an `Intent.target_id`.

Standing rulings for downstream prompts: item 2's `NULL_THRESHOLD = 0.5` stands until P9;
the proposer (P8) must emit the null candidate with `commitment = 0.0` exactly (not
merely sub-threshold) so logs are unambiguous; item 9's padding convention means all book
consumers (engine invariant tests, world summary, queue filter) must treat
`size_lots == 0` levels as absent.
