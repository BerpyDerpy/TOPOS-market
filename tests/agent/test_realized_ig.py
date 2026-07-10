"""Realized information gain: exactness, and the mutation test (INV-10).

The synthetic module is the conjugate case where EXPECTED information gain
equals the ACTUAL entropy drop for every outcome — a Gaussian mean with
known observation noise, whose posterior variance shrinks deterministically
(independently of the observed value). For it, a correctly ordered
before/after snapshot pair must reproduce the promised EIG exactly; any
snapshot-order mistake collapses realized IG to ~0.

The deliberately broken variant (`_mutant_resolve`, kept HERE in tests/ —
never in topos/) re-captures the 'before' snapshot after the update has
been applied, which is the exact failure mode INV-10 exists to prevent:
realized IG identically ~0, silently poisoning all calibration metrics.
The test asserts the healthy path and the mutant disagree, i.e. that this
suite would catch the mutation.
"""

from __future__ import annotations

import math

import pytest

from tests.agent.conftest import canned_stream, make_agent
from tests.beliefs.conftest import empty_events, make_obs, order_probe
from topos.agent import ExperimentLedger
from topos.agent.ledger import OpenExperiment
from topos.contracts.beliefs import (
    EntropySnapshot,
    ForecastStats,
    ProbeSpec,
    SelfEvents,
    realized_information_gain_nats,
)
from topos.contracts.intent import FILL_RATE, HypothesisId
from topos.contracts.market import Observation


class GaussianMeanModule:
    """Unknown mean, known noise: EIG == realized IG for every outcome.

    theta ~ N(m, v); each update observes y | theta ~ N(theta, r). The
    posterior variance contracts as 1/v' = 1/v + 1/r regardless of y, so
    H_before - H_after = 0.5 ln(v/v') deterministically — exactly
    ``eig_nats`` (mutual information of the parameter with one draw).
    """

    hypothesis_id: HypothesisId = FILL_RATE

    def __init__(self, prior_var: float = 4.0, noise_var: float = 1.0) -> None:
        self._var = prior_var
        self._noise = noise_var
        self._step = 0

    def _posterior_var_after_one(self) -> float:
        return 1.0 / (1.0 / self._var + 1.0 / self._noise)

    def update(self, obs: Observation, self_events: SelfEvents) -> None:
        self._step = obs.step
        self._var = self._posterior_var_after_one()

    def forget(self, rho: float) -> None:
        pass

    def posterior_entropy_nats(self) -> float:
        return 0.5 * math.log(2.0 * math.pi * math.e * self._var)

    def predict(self) -> ForecastStats:
        return ForecastStats(mean=0.0, variance=self._var + self._noise)

    def surprise_z(self) -> float:
        return 0.0

    def eig_nats(self, probe: ProbeSpec) -> float:
        return 0.5 * math.log(self._var / self._posterior_var_after_one())

    def snapshot_entropy(self) -> EntropySnapshot:
        return EntropySnapshot(
            hypothesis_id=self.hypothesis_id,
            step=self._step,
            entropy_nats=self.posterior_entropy_nats(),
        )


def _obs(step: int) -> Observation:
    return make_obs(step, [(999, 5)], [(1001, 5)])


def _issue(
    module: GaussianMeanModule, ledger: ExperimentLedger, step: int
) -> float:
    """The agent's issuance protocol: promised EIG and the 'before'
    snapshot captured at issuance, after the cycle's updates."""
    promised = module.eig_nats(order_probe(FILL_RATE))
    ledger.open(
        step=step,
        target_id=module.hypothesis_id,
        eig_promised_nats=promised,
        snapshot_before=module.snapshot_entropy(),
    )
    return promised


def test_realized_ig_positive_and_equal_to_promised() -> None:
    module = GaussianMeanModule()
    ledger = ExperimentLedger()
    # Cycle s: the module absorbs the observation, then the probe is issued.
    module.update(_obs(0), empty_events(0))
    promised = _issue(module, ledger, step=0)
    assert promised > 0.05

    # Cycle s+1: the outcome observation resolves the pending experiment.
    resolved = ledger.resolve_pending(module, _obs(1), empty_events(1))
    assert resolved is not None
    assert resolved.realized_ig_nats > 0.0
    assert resolved.realized_ig_nats == pytest.approx(promised, rel=1e-12)
    assert resolved.snapshot_before.step == 0
    assert resolved.snapshot_after.step == 1
    assert ledger.pending is None
    assert ledger.log == (resolved,)


# ---------------------------------------------------------------------------
# The mutant (kept in tests/, never in topos/)
# ---------------------------------------------------------------------------


def _mutant_resolve(
    entry: OpenExperiment,
    module: GaussianMeanModule,
    obs: Observation,
    self_events: SelfEvents,
) -> float:
    """DELIBERATELY BROKEN resolution: 'before' re-snapshotted after the
    update (equivalently: both snapshots taken after). This is the INV-10
    violation the ledger's ordering exists to make impossible."""
    module.update(obs, self_events)
    snapshot_before = module.snapshot_entropy()  # WRONG: post-update
    snapshot_after = module.snapshot_entropy()
    return realized_information_gain_nats(snapshot_before, snapshot_after)


def test_mutant_snapshot_order_is_caught() -> None:
    module = GaussianMeanModule()
    ledger = ExperimentLedger()
    module.update(_obs(0), empty_events(0))
    promised = _issue(module, ledger, step=0)

    entry = ledger.pending
    assert entry is not None
    mutant_realized = _mutant_resolve(entry, module, _obs(1), empty_events(1))

    # The mutant reports ~0 where the true realized IG equals the promise:
    # the discrepancy IS the detection.
    assert abs(mutant_realized) < 1e-12
    assert mutant_realized != pytest.approx(promised, rel=1e-6)
    assert promised > 0.05


# ---------------------------------------------------------------------------
# The integrated agent's ledger mechanics on a live loop
# ---------------------------------------------------------------------------


def test_agent_ledger_pairs_promised_and_realized() -> None:
    agent = make_agent()
    for obs in canned_stream(40):
        agent.cycle(obs)

    log = agent.ledger.log
    assert log, "no experiment ever ran; the canned market should probe"
    records_by_step = {record.step: record for record in agent.records}
    for entry in log:
        # Snapshots were taken at issuance and resolution, in that order.
        assert entry.snapshot_before.step == entry.step_issued
        assert entry.snapshot_after.step == entry.step_resolved
        assert entry.step_resolved > entry.step_issued
        assert math.isfinite(entry.realized_ig_nats)
        # The promised figure is exactly what the issuing record broadcast.
        record = records_by_step[entry.step_issued]
        assert record.intent is not None
        assert record.intent.target_id == entry.target_id
        assert record.eig_promised_nats == entry.eig_promised_nats

    # A pending entry, if any, is from the very last cycle only.
    if agent.ledger.pending is not None:
        assert agent.ledger.pending.step_issued == agent.records[-1].step
