"""FairValueKF module behavior beyond the shared mandated tests."""

from __future__ import annotations

import numpy as np
import pytest

from tests.beliefs.conftest import (
    empty_events,
    make_obs,
    null_probe,
    obs_for_mid,
    order_probe,
)
from topos.beliefs import FairValueKF, microprice_from_observation
from topos.contracts.beliefs import BeliefModule, realized_information_gain_nats
from topos.contracts.intent import FAIR_VALUE


def test_conforms_to_belief_module_protocol() -> None:
    module = FairValueKF()
    assert isinstance(module, BeliefModule)
    assert module.hypothesis_id == FAIR_VALUE


def test_microprice_weights_toward_the_thin_side() -> None:
    obs = make_obs(0, [(999, 9)], [(1000, 1)])
    # Heavy bid, thin ask: micro = (999*1 + 1000*9) / 10.
    assert microprice_from_observation(obs) == pytest.approx(999.9)


def test_microprice_handles_one_sided_and_empty_books() -> None:
    assert microprice_from_observation(make_obs(0, [(999, 5)], [])) == 999.0
    assert microprice_from_observation(make_obs(0, [], [(1000, 2)])) == 1000.0
    assert microprice_from_observation(make_obs(0, [], [])) is None


def test_empty_book_step_diffuses_state_without_scale_update() -> None:
    module = FairValueKF()
    for step in range(20):
        module.update(obs_for_mid(step, 1000.0), empty_events(step))
    scale_before = (module.noise_scale_posterior.a, module.noise_scale_posterior.b)
    var_before = module.state_scale_free_cov[0, 0]
    module.update(make_obs(20, [], []), empty_events(20))
    assert (module.noise_scale_posterior.a, module.noise_scale_posterior.b) == (
        scale_before
    )
    assert module.state_scale_free_cov[0, 0] > var_before


def test_eig_is_intent_independent() -> None:
    """The microprice is observed passively: an order-placing probe earns
    exactly the same EIG as the null at the same horizon, i.e. its marginal
    EIG over null is 0 (INV-4: the null action carries this module's EIG)."""
    module = FairValueKF()
    rng = np.random.default_rng(51)
    for step in range(50):
        module.update(
            obs_for_mid(step, 1000.0 + rng.normal(0.0, 2.0)), empty_events(step)
        )
    for horizon in (1, 5):
        eig_null = module.eig_nats(null_probe(horizon_steps=horizon))
        eig_order = module.eig_nats(order_probe(horizon_steps=horizon))
        assert eig_order == pytest.approx(eig_null, abs=1e-12)
        assert eig_null > 0.0


def test_forecast_tracks_the_level() -> None:
    module = FairValueKF()
    for step in range(200):
        module.update(obs_for_mid(step, 1000.0), empty_events(step))
    forecast = module.predict()
    assert forecast.mean == pytest.approx(1000.0, abs=1.0)
    assert 0.0 < forecast.variance < np.inf


def test_snapshot_entropy_supports_realized_ig() -> None:
    """INV-10 mechanics: snapshots immediately before/after an outcome
    update difference to the realized information gain."""
    module = FairValueKF()
    rng = np.random.default_rng(52)
    for step in range(100):
        module.update(
            obs_for_mid(step, 1000.0 + rng.normal(0.0, 2.0)), empty_events(step)
        )
    before = module.snapshot_entropy()
    assert before.hypothesis_id == FAIR_VALUE
    assert before.entropy_nats == module.posterior_entropy_nats()
    module.update(obs_for_mid(100, 1001.0), empty_events(100))
    after = module.snapshot_entropy()
    assert after.step == 100
    realized = realized_information_gain_nats(before, after)
    assert realized == pytest.approx(before.entropy_nats - after.entropy_nats)


def test_state_eig_is_positive_and_separate_from_curiosity() -> None:
    module = FairValueKF()
    rng = np.random.default_rng(53)
    for step in range(300):
        module.update(
            obs_for_mid(step, 1000.0 + rng.normal(0.0, 2.0)), empty_events(step)
        )
    probe = null_probe(horizon_steps=1)
    assert module.state_eig_nats(probe) > 0.01
    # Curiosity saturates; state tracking does not — they must be distinct.
    assert module.eig_nats(probe) < module.state_eig_nats(probe)


def test_constructor_validates_shapes() -> None:
    with pytest.raises(ValueError):
        FairValueKF(r0=0.0)
    with pytest.raises(ValueError):
        FairValueKF(q_level=-1.0)
