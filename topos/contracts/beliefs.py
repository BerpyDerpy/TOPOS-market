"""The shared belief-module protocol and its supporting contracts.

Every world predictor (and every self-model component exposed as a
hypothesis) implements `BeliefModule`. Adaptation is exclusively Bayesian
state estimation inside fixed functional forms: conjugate/analytic posterior
updates plus regime-gated forgetting (INV-2). Curiosity is prospective
expected information gain over explicit PARAMETER posteriors — never
retrospective prediction error, never predictive variance (INV-3).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from topos.contracts.intent import HypothesisId, Intent
from topos.contracts.market import Ack, ExchangeMessage, Fill, Observation


@dataclass(frozen=True)
class SelfEvents:
    """The agent's own order-lifecycle events for one step.

    Passed alongside the raw `Observation` so belief modules can condition
    on what the agent did (messages sent) and what came back (acks, fills)
    without ever seeing engine-side state (INV-11).
    """

    step: int
    messages_sent: tuple[ExchangeMessage, ...]
    acks: tuple[Ack, ...]
    fills: tuple[Fill, ...]


@dataclass(frozen=True)
class ForecastStats:
    """Predictive summary of a belief module's observable."""

    mean: float
    variance: float


@dataclass(frozen=True)
class EntropySnapshot:
    """Parameter-posterior entropy captured at a named instant (INV-10).

    Realized information gain = H(immediately BEFORE the outcome-driven
    update) - H(immediately AFTER). Snapshots exist so that realized IG is
    computed from explicit before/after pairs, never reconstructed.
    """

    hypothesis_id: HypothesisId
    step: int
    entropy_nats: float


@dataclass(frozen=True)
class ProbeSpec:
    """Observation model selector for EIG: which experiment, over what horizon."""

    intent: Intent
    horizon_steps: int


@runtime_checkable
class BeliefModule(Protocol):
    """Protocol every hypothesis-owning module implements.

    Implementations hold an explicit parameter posterior with closed-form
    updates. No gradients, no learning frameworks (INV-2).
    """

    hypothesis_id: HypothesisId

    def update(self, obs: Observation, self_events: SelfEvents) -> None:
        """Conjugate/analytic posterior update from one step's evidence."""
        ...

    def forget(self, rho: float) -> None:
        """Discount sufficient statistics toward the prior (regime-gated)."""
        ...

    def posterior_entropy_nats(self) -> float:
        """Entropy of the PARAMETER posterior — not predictive (INV-3)."""
        ...

    def predict(self) -> ForecastStats:
        """Predictive summary for the workspace headline."""
        ...

    def surprise_z(self) -> float:
        """z-scored negative log predictive probability of the last observation."""
        ...

    def eig_nats(self, probe: ProbeSpec) -> float:
        """Prospective expected information gain of a probe.

        I(theta; Y | probe) = H[Y | probe] - E_theta H[Y | theta, probe]
        — mutual information between parameters and outcome, never
        predictive variance or predictive entropy alone (INV-3).
        """
        ...

    def snapshot_entropy(self) -> EntropySnapshot:
        """Capture the current parameter-posterior entropy (for realized IG, INV-10)."""
        ...


def realized_information_gain_nats(
    before: EntropySnapshot, after: EntropySnapshot
) -> float:
    """Realized IG from an explicit before/after snapshot pair (INV-10).

    `before` must be captured immediately BEFORE the outcome-driven update
    and `after` immediately after it, on the same hypothesis.
    """
    if before.hypothesis_id != after.hypothesis_id:
        raise ValueError(
            "entropy snapshots refer to different hypotheses: "
            f"{before.hypothesis_id!r} vs {after.hypothesis_id!r}"
        )
    if after.step < before.step:
        raise ValueError(
            f"'after' snapshot (step {after.step}) precedes 'before' "
            f"snapshot (step {before.step})"
        )
    return before.entropy_nats - after.entropy_nats
