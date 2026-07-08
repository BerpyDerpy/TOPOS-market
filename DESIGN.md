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
