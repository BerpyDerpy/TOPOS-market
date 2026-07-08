from __future__ import annotations

import pytest

from topos.contracts.beliefs import (
    BeliefModule,
    EntropySnapshot,
    ForecastStats,
    ProbeSpec,
    SelfEvents,
    realized_information_gain_nats,
)
from topos.contracts.intent import FAIR_VALUE, FILL_RATE, HypothesisId
from topos.contracts.market import Observation


class _ConformingStub:
    """Structurally satisfies BeliefModule without any belief math."""

    hypothesis_id: HypothesisId = FAIR_VALUE

    def update(self, obs: Observation, self_events: SelfEvents) -> None:
        pass

    def forget(self, rho: float) -> None:
        pass

    def posterior_entropy_nats(self) -> float:
        return 1.0

    def predict(self) -> ForecastStats:
        return ForecastStats(mean=0.0, variance=1.0)

    def surprise_z(self) -> float:
        return 0.0

    def eig_nats(self, probe: ProbeSpec) -> float:
        return 0.0

    def snapshot_entropy(self) -> EntropySnapshot:
        return EntropySnapshot(hypothesis_id=self.hypothesis_id, step=0, entropy_nats=1.0)


class _NonConformingStub:
    hypothesis_id: HypothesisId = FAIR_VALUE


def test_protocol_is_runtime_checkable() -> None:
    assert isinstance(_ConformingStub(), BeliefModule)
    assert not isinstance(_NonConformingStub(), BeliefModule)


def test_realized_ig_is_before_minus_after() -> None:
    before = EntropySnapshot(hypothesis_id=FAIR_VALUE, step=10, entropy_nats=1.5)
    after = EntropySnapshot(hypothesis_id=FAIR_VALUE, step=10, entropy_nats=1.2)
    assert realized_information_gain_nats(before, after) == pytest.approx(0.3)


def test_realized_ig_may_be_negative() -> None:
    # Forgetting or a surprising outcome can raise posterior entropy;
    # realized IG is a measurement, not a score, so it may go negative.
    before = EntropySnapshot(hypothesis_id=FAIR_VALUE, step=10, entropy_nats=1.0)
    after = EntropySnapshot(hypothesis_id=FAIR_VALUE, step=10, entropy_nats=1.4)
    assert realized_information_gain_nats(before, after) == pytest.approx(-0.4)


def test_realized_ig_rejects_mismatched_hypotheses() -> None:
    before = EntropySnapshot(hypothesis_id=FAIR_VALUE, step=10, entropy_nats=1.5)
    after = EntropySnapshot(hypothesis_id=FILL_RATE, step=10, entropy_nats=1.2)
    with pytest.raises(ValueError):
        realized_information_gain_nats(before, after)


def test_realized_ig_rejects_time_reversed_snapshots() -> None:
    before = EntropySnapshot(hypothesis_id=FAIR_VALUE, step=10, entropy_nats=1.5)
    after = EntropySnapshot(hypothesis_id=FAIR_VALUE, step=9, entropy_nats=1.2)
    with pytest.raises(ValueError):
        realized_information_gain_nats(before, after)
