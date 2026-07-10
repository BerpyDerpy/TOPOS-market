"""The ablation switch surface (consumed by P13).

Four flags, each implemented as a strategy OBJECT substituted at exactly
one injection point during agent construction — never as a conditional in
the cycle. With every flag off, none of these classes is instantiated, so
nothing here can be consulted (pinned by tests/agent/
test_ablation_isolation.py); with a flag on, only its documented code path
changes:

* ``surprise_curiosity`` — a ``SurpriseAsCuriosity`` wrapper replaces each
  module in the PROPOSER'S scoring map, so every EIG quantity in steps 5-8
  (headline marginals, salience, candidate scores, the promised figure)
  becomes the module's retrospective surprise_z; the null action scores 0.
  Updates, posteriors, snapshots and realized IG stay real: the agent's
  own module map is untouched, which is exactly how the ablation shows up
  as promised-vs-realized miscalibration.
* ``no_self_model`` — ``FrozenFillModel``/``FrozenImpactModel`` replace the
  two self-model modules at construction. Their update pipelines run
  identically (contexts, trials, surprise) but the posterior cells never
  absorb evidence: frozen at priors, bit for bit. The trajectory compiler
  keeps compiling — from the frozen posteriors.
* ``no_reflexive`` — ``NoReflexiveSelection`` is injected as the proposer's
  selection rule: lexicographic tiebreak (c) (the epsilon-band minimum
  self-entropy comparison, and with it the null's boredom-band re-entry)
  is removed; ties break by max marginal EIG then lowest message cost.
  Gates (a), eligibility (b) and the corrective fallback (d) are intact.
* ``no_homeostat`` — ``VetoOnlyHomeostat`` filters the homeostat's exports
  (drives, distances and the corrective intent are silenced) and
  ``NullDistanceProjector`` replaces the band projector (the soft-bound
  gate (a3) passes trivially). Hard vetoes AND motor legality remain: the
  drive is ablated, not the exchange rules.

Each strategy counts its consultations so the isolation tests can assert
"consulted exactly when the flag is on" directly.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from topos.beliefs.core import BetaPosterior, FloatArray, InverseGammaPosterior
from topos.contracts.beliefs import (
    BeliefModule,
    EntropySnapshot,
    ForecastStats,
    ProbeSpec,
    SelfEvents,
)
from topos.contracts.intent import HypothesisId
from topos.contracts.market import Observation
from topos.drives.homeostat import HomeostatOutput
from topos.proposer import Candidate
from topos.selfmodel import FillModel, ImpactModel


@dataclass(frozen=True)
class AblationFlags:
    """The P13 ablation switches. All off = the intact architecture."""

    surprise_curiosity: bool = False
    no_self_model: bool = False
    no_reflexive: bool = False
    no_homeostat: bool = False


# ---------------------------------------------------------------------------
# SURPRISE_CURIOSITY
# ---------------------------------------------------------------------------


class SurpriseAsCuriosity:
    """Scoring adapter: ``eig_nats`` answers with retrospective surprise.

    Wraps one BeliefModule for the proposer's scoring map only. Every
    protocol method delegates unchanged except ``eig_nats``, which ignores
    the parameter posterior entirely and quotes the module's z-scored
    surprise for any committed probe (0 for the null) — the exact
    retrospective signal INV-3 forbids as a curiosity quantity, exposed
    deliberately so P13 can measure what breaks.
    """

    def __init__(self, module: BeliefModule) -> None:
        self._module = module
        self.hypothesis_id: HypothesisId = module.hypothesis_id
        self.consultations = 0

    @property
    def wrapped(self) -> BeliefModule:
        return self._module

    def update(self, obs: Observation, self_events: SelfEvents) -> None:
        self._module.update(obs, self_events)

    def forget(self, rho: float) -> None:
        self._module.forget(rho)

    def posterior_entropy_nats(self) -> float:
        return self._module.posterior_entropy_nats()

    def predict(self) -> ForecastStats:
        return self._module.predict()

    def surprise_z(self) -> float:
        return self._module.surprise_z()

    def eig_nats(self, probe: ProbeSpec) -> float:
        """surprise_z for any committed probe; the null action scores 0."""
        self.consultations += 1
        if probe.intent.is_null:
            return 0.0
        return self._module.surprise_z()

    def snapshot_entropy(self) -> EntropySnapshot:
        return self._module.snapshot_entropy()


# ---------------------------------------------------------------------------
# NO_SELF_MODEL
# ---------------------------------------------------------------------------


class _FrozenBeta(BetaPosterior):
    """A Beta cell that never absorbs evidence: (a, b) stay at the prior."""

    def observe(self, success_fraction: float) -> None:
        if not 0.0 <= success_fraction <= 1.0:
            raise ValueError(
                f"success_fraction must be in [0, 1], got {success_fraction}"
            )

    def forget(self, rho: float) -> None:
        # Already at the prior; a float re-fold could still perturb the
        # last bit, so freezing means freezing.
        pass


class _FrozenInverseGamma(InverseGammaPosterior):
    """An inverse-gamma cell that never absorbs evidence."""

    def observe_standardized_square(self, z: float) -> None:
        if z < 0.0:
            raise ValueError(f"squared innovation must be >= 0, got {z}")

    def forget(self, rho: float) -> None:
        pass


class FrozenFillModel(FillModel):
    """NO_SELF_MODEL fill model: trials, contexts and surprise run exactly
    as in ``FillModel``, but every Beta cell is frozen at its prior."""

    def __init__(
        self,
        horizon_steps: int,
        *,
        size_budget_lots: int = 1,
        prior_a: float = 1.0,
        prior_b: float = 1.0,
        surprise_ewma_decay: float = 0.05,
    ) -> None:
        super().__init__(
            horizon_steps,
            size_budget_lots=size_budget_lots,
            prior_a=prior_a,
            prior_b=prior_b,
            surprise_ewma_decay=surprise_ewma_decay,
        )
        self._cells = {
            key: _FrozenBeta(cell.prior_a, cell.prior_b)
            for key, cell in self._cells.items()
        }


class FrozenImpactModel(ImpactModel):
    """NO_SELF_MODEL impact model: rows are built and surprise is scored
    exactly as in ``ImpactModel``, but neither the coefficient posterior
    (Lambda, eta) nor the noise-scale posterior ever moves off the prior."""

    def __init__(
        self,
        *,
        impact_horizon_steps: int = 1,
        n_context: int = 1,
        coef_prior_var: float = 1.0,
        noise_prior_a: float = 3.0,
        noise_prior_b: float = 2.0,
        size_budget_lots: int = 1,
        surprise_ewma_decay: float = 0.05,
    ) -> None:
        super().__init__(
            impact_horizon_steps=impact_horizon_steps,
            n_context=n_context,
            coef_prior_var=coef_prior_var,
            noise_prior_a=noise_prior_a,
            noise_prior_b=noise_prior_b,
            size_budget_lots=size_budget_lots,
            surprise_ewma_decay=surprise_ewma_decay,
        )
        self._scale = _FrozenInverseGamma(
            self._scale.prior_a, self._scale.prior_b
        )

    def observe_point_raw(self, x: FloatArray, y: float) -> None:
        lam, eta = self._lam, self._eta
        # Scores surprise on the (frozen) prior predictive; the frozen
        # scale cell ignores its conjugate increment.
        super().observe_point_raw(x, y)
        self._lam, self._eta = lam, eta

    def forget(self, rho: float) -> None:
        if not 0.0 < rho <= 1.0:
            raise ValueError(f"rho must be in (0, 1], got {rho}")


# ---------------------------------------------------------------------------
# NO_REFLEXIVE
# ---------------------------------------------------------------------------


class NoReflexiveSelection:
    """The selection rule with lexicographic tiebreak (c) removed.

    (a) hard gates and (b) strictly-positive-marginal eligibility are
    verbatim from the exported rule; among eligible candidates the winner
    is simply the max marginal EIG, ties broken by LOWEST message cost
    (then by candidate order, which is deterministic). There is no epsilon
    band, no self-entropy comparison, and no boredom-band re-entry for the
    null — which is precisely the churn this ablation exists to exhibit.
    Rule (d) is intact: with no eligible candidate, flatten wins if any
    drive distance is nonzero, else the null.
    """

    def __init__(self) -> None:
        self.consultations = 0

    def __call__(
        self,
        candidates: Sequence[Candidate],
        drive_distances: Mapping[str, float],
    ) -> Candidate:
        self.consultations += 1
        null_candidate = next(
            (c for c in candidates if c.probe.intent.is_null), None
        )
        if null_candidate is None:
            raise ValueError(
                "candidate set must contain the null candidate (INV-4)"
            )
        flatten_candidate = next(
            (c for c in candidates if c.probe.intent.is_flatten), None
        )
        eligible = [
            c
            for c in candidates
            if c.gates_passed
            and c.marginal_eig_nats > 0.0
            and not c.probe.intent.is_null
            and not c.probe.intent.is_flatten
        ]
        if eligible:
            return min(
                eligible,
                key=lambda c: (-c.marginal_eig_nats, c.message_cost),
            )
        if flatten_candidate is not None and any(
            u > 0.0 for u in drive_distances.values()
        ):
            return flatten_candidate
        return null_candidate


# ---------------------------------------------------------------------------
# NO_HOMEOSTAT
# ---------------------------------------------------------------------------


class VetoOnlyHomeostat:
    """Export filter: hard vetoes pass through; the drive is silenced.

    Drives never bid salience, distances read as empty (so the corrective
    fallback never triggers and the cognitive view carries no excursions),
    and the corrective intent is withheld. The veto flags — and the motor
    legality they enforce — are untouched: exchange rules survive the
    ablation.
    """

    def __init__(self) -> None:
        self.consultations = 0

    def filter(self, output: HomeostatOutput) -> HomeostatOutput:
        self.consultations += 1
        return HomeostatOutput(
            drives=MappingProxyType({}),
            vetoes=output.vetoes,
            distances=MappingProxyType({}),
            corrective_intent=None,
        )


class NullDistanceProjector:
    """Soft-bound gate (a3) switched off: no distances to breach."""

    def __init__(self) -> None:
        self.consultations = 0

    def predicted_distances(
        self, inventory_lots: int, new_messages: int
    ) -> Mapping[str, float]:
        self.consultations += 1
        return {}
