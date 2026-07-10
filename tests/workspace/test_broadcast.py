"""Broadcast conditioning is real, not decorative.

The focused module does finer work than the unfocused one, on the two
wired exemplars:

* ``FlowIntensity``: per-band posteriors refresh only when focused;
  unfocused evidence is buffered and the flush on refocus is EXACT
  (Gamma-Poisson batches: a stay-focused twin and a defocus/refocus twin
  end with identical sufficient statistics);
* ``FairValueKF``: the parameter-EIG quadrature runs only when focused;
  unfocused curiosity is quoted from the last refresh.

Plus the wiring itself: every registered consumer receives the focus
every cycle, and registering a hook-less consumer fails at construction.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.beliefs.conftest import empty_events
from tests.proposer.conftest import active_obs
from tests.workspace.conftest import (
    FakeModule,
    make_registry,
    make_workspace,
    run_cycle,
)
from topos.beliefs import FairValueKF, FlowIntensity
from topos.contracts.beliefs import ProbeSpec
from topos.contracts.intent import FAIR_VALUE, FLOW_INTENSITY
from topos.contracts.workspace import Focus
from topos.proposer import null_intent

WARM_STEPS = 4
TOTAL_STEPS = 12


def _focus_on(hypothesis_id: str) -> Focus:
    return Focus(hypothesis_id=hypothesis_id, salience=1.0, is_homeostatic=False)


def _stats(module: FlowIntensity) -> dict[object, tuple[float, float]]:
    return {key: (cell.a, cell.b) for key, cell in module.cells.items()}


def test_flow_intensity_defers_per_band_refresh_until_focused() -> None:
    rng = np.random.default_rng(3)
    observations = [active_obs(step, rng) for step in range(TOTAL_STEPS)]
    focused_twin = FlowIntensity()
    lazy_twin = FlowIntensity()

    for obs in observations[:WARM_STEPS]:
        focused_twin.update(obs, empty_events(obs.step))
        lazy_twin.update(obs, empty_events(obs.step))
    assert _stats(lazy_twin) == _stats(focused_twin)

    # Defocused: the per-band posteriors freeze...
    lazy_twin.condition_on_focus(None)
    frozen = _stats(lazy_twin)
    for obs in observations[WARM_STEPS:]:
        focused_twin.update(obs, empty_events(obs.step))
        lazy_twin.update(obs, empty_events(obs.step))
    assert _stats(lazy_twin) == frozen
    assert _stats(focused_twin) != frozen

    # ...while the coarse aggregate stays a live forecast (and surprise
    # is still scored, on the coarse channel).
    coarse_forecast = lazy_twin.predict()
    assert coarse_forecast.mean > 0.0
    assert np.isfinite(lazy_twin.surprise_z())
    # The aggregate view tracks the same total as the fine view's.
    fine_forecast = focused_twin.predict()
    assert coarse_forecast.mean == pytest.approx(fine_forecast.mean)

    # Refocus: the flush is EXACT — batched conjugate evidence equals the
    # per-step updates it stood in for, cell by cell, bit for bit.
    lazy_twin.condition_on_focus(_focus_on(FLOW_INTENSITY))
    assert _stats(lazy_twin) == _stats(focused_twin)


def test_flow_intensity_forget_folds_buffered_evidence_first() -> None:
    rng = np.random.default_rng(4)
    observations = [active_obs(step, rng) for step in range(TOTAL_STEPS)]
    steady = FlowIntensity()
    lazy = FlowIntensity()
    for obs in observations[:WARM_STEPS]:
        steady.update(obs, empty_events(obs.step))
        lazy.update(obs, empty_events(obs.step))
    lazy.condition_on_focus(None)
    for obs in observations[WARM_STEPS:]:
        steady.update(obs, empty_events(obs.step))
        lazy.update(obs, empty_events(obs.step))
    # Evidence precedes the discount: forgetting while unfocused must not
    # differ from forgetting while focused.
    steady.forget(0.5)
    lazy.forget(0.5)
    assert _stats(lazy) == _stats(steady)


def test_fair_value_quadrature_runs_only_when_focused() -> None:
    kf = FairValueKF()
    rng = np.random.default_rng(5)
    for step in range(10):
        kf.update(active_obs(step, rng), empty_events(step))

    posterior = kf.noise_scale_posterior
    quadrature_calls = {"n": 0}
    original = posterior.eig_terms_for_gaussian

    def counting(scale_free_var: float):  # type: ignore[no-untyped-def]
        quadrature_calls["n"] += 1
        return original(scale_free_var)

    posterior.eig_terms_for_gaussian = counting  # type: ignore[method-assign]
    probe = ProbeSpec(intent=null_intent(FAIR_VALUE), horizon_steps=1)

    kf.condition_on_focus(None)
    first_quote = kf.eig_nats(probe)
    assert quadrature_calls["n"] == 1  # cache fill: some answer must exist
    assert kf.eig_nats(probe) == first_quote
    assert quadrature_calls["n"] == 1  # unfocused: quoted, not recomputed

    kf.update(active_obs(10, rng), empty_events(10))
    assert kf.eig_nats(probe) == first_quote
    assert quadrature_calls["n"] == 1  # still the stale-but-honest quote

    kf.condition_on_focus(_focus_on(FAIR_VALUE))
    kf.eig_nats(probe)
    assert quadrature_calls["n"] == 2  # focused: fresh quadrature
    kf.eig_nats(probe)
    assert quadrature_calls["n"] == 3

    # Focus on someone ELSE is not focus: the quote freezes again.
    kf.condition_on_focus(_focus_on(FLOW_INTENSITY))
    kf.eig_nats(probe)
    assert quadrature_calls["n"] == 3


def test_focus_reaches_every_registered_consumer() -> None:
    modules = {
        "hyp_a": FakeModule(hypothesis_id="hyp_a", probe_gain=0.2),
        "hyp_b": FakeModule(hypothesis_id="hyp_b", probe_gain=0.0001),
    }
    workspace = make_workspace(
        modules,
        registry=make_registry(tuple(modules)),
        consumers=tuple(modules.values()),
    )
    record = run_cycle(workspace)
    assert record.focus is not None
    for module in modules.values():
        assert module.focus_log == [record.focus]

    # Quiet cycles broadcast too: None is information ("nothing won").
    for module in modules.values():
        module.probe_gain = 0.0
    quiet = run_cycle(workspace, step=1)
    assert quiet.focus is None
    for module in modules.values():
        assert module.focus_log == [record.focus, None]


def test_hookless_consumer_is_rejected_at_construction() -> None:
    modules = {"hyp_a": FakeModule(hypothesis_id="hyp_a")}
    with pytest.raises(TypeError, match="condition_on_focus"):
        make_workspace(
            modules,
            registry=make_registry(("hyp_a",)),
            consumers=(object(),),
        )
