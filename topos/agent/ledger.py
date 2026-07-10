"""The experiment ledger: promised vs realized information gain (INV-10).

One entry is opened per committed probe, at ISSUANCE — after every belief
module has absorbed the cycle's observation, before anything is submitted —
holding the target hypothesis, the EIG the arbiter acted on, and the target
module's entropy snapshot at that instant. It is resolved on the NEXT
cycle's observation (the one carrying the probe's acks/fills), before any
other update could confound it:

    realized IG = H(snapshot taken at issuance, FROM THE LEDGER)
                - H(snapshot taken immediately after the outcome update)

The order is the entire invariant. Re-snapshotting "before" at resolution
time (or snapshotting both after the update) measures the entropy change of
an already-updated posterior — identically ~0 — and silently poisons every
promised-vs-realized calibration downstream. That failure mode is pinned by
a mutation test (tests/agent/test_realized_ig.py keeps the broken variant
in tests/, never here).
"""

from __future__ import annotations

from dataclasses import dataclass

from topos.contracts.beliefs import (
    BeliefModule,
    EntropySnapshot,
    SelfEvents,
    realized_information_gain_nats,
)
from topos.contracts.intent import HypothesisId
from topos.contracts.market import Observation


@dataclass(frozen=True)
class OpenExperiment:
    """A probe in flight: issued, not yet resolved."""

    step_issued: int
    target_id: HypothesisId
    eig_promised_nats: float
    snapshot_before: EntropySnapshot
    """Target parameter-posterior entropy at issuance (INV-10's 'before')."""


@dataclass(frozen=True)
class ResolvedExperiment:
    """One completed (promised, realized) pair — the calibration record."""

    step_issued: int
    step_resolved: int
    target_id: HypothesisId
    eig_promised_nats: float
    snapshot_before: EntropySnapshot
    snapshot_after: EntropySnapshot
    realized_ig_nats: float


class ExperimentLedger:
    """Holds at most one probe in flight, plus the append-only resolution log.

    One slot suffices structurally: the arbiter ignites at most one intent
    per cycle and the pending probe is resolved at the START of the next
    cycle, before a new one could be opened.
    """

    def __init__(self) -> None:
        self._pending: OpenExperiment | None = None
        self._log: list[ResolvedExperiment] = []

    @property
    def pending(self) -> OpenExperiment | None:
        return self._pending

    @property
    def log(self) -> tuple[ResolvedExperiment, ...]:
        return tuple(self._log)

    def open(
        self,
        *,
        step: int,
        target_id: HypothesisId,
        eig_promised_nats: float,
        snapshot_before: EntropySnapshot,
    ) -> OpenExperiment:
        """Record a probe at issuance — BEFORE it is acted on (INV-10).

        ``snapshot_before`` must be captured from the target module at the
        moment of the call, after this cycle's updates: it is the entropy
        the outcome-driven update will be measured against.
        """
        if self._pending is not None:
            raise RuntimeError(
                "an experiment is already in flight (issued at step "
                f"{self._pending.step_issued}); resolve it before opening "
                "another — the arbiter ignites at most one intent per cycle"
            )
        if snapshot_before.hypothesis_id != target_id:
            raise ValueError(
                f"snapshot is for {snapshot_before.hypothesis_id!r}, "
                f"but the probe targets {target_id!r}"
            )
        entry = OpenExperiment(
            step_issued=step,
            target_id=target_id,
            eig_promised_nats=eig_promised_nats,
            snapshot_before=snapshot_before,
        )
        self._pending = entry
        return entry

    def resolve_pending(
        self, module: BeliefModule, obs: Observation, self_events: SelfEvents
    ) -> ResolvedExperiment | None:
        """Apply the outcome observation to the target and log realized IG.

        THE ORDER IS THE INVARIANT (INV-10): the 'before' snapshot comes
        from the ledger entry (taken at issuance — never re-captured here),
        the target module absorbs the outcome observation exactly once, and
        the 'after' snapshot is captured immediately after that update,
        before any other module update this cycle could touch shared state.
        ``realized_information_gain_nats`` validates the hypothesis match
        and the step ordering of the pair.
        """
        entry = self._pending
        if entry is None:
            return None
        if module.hypothesis_id != entry.target_id:
            raise ValueError(
                f"resolution module is {module.hypothesis_id!r}, but the "
                f"pending experiment targets {entry.target_id!r}"
            )
        module.update(obs, self_events)
        snapshot_after = module.snapshot_entropy()
        realized = realized_information_gain_nats(
            entry.snapshot_before, snapshot_after
        )
        resolved = ResolvedExperiment(
            step_issued=entry.step_issued,
            step_resolved=obs.step,
            target_id=entry.target_id,
            eig_promised_nats=entry.eig_promised_nats,
            snapshot_before=entry.snapshot_before,
            snapshot_after=snapshot_after,
            realized_ig_nats=realized,
        )
        self._pending = None
        self._log.append(resolved)
        return resolved
