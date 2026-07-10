"""INV-7's behavioral tripwire: consequence-weights are structural, static,
and blind to every outcome statistic.

``test_weights_static_and_structural`` is the required form: identical
across 1000 synthetic cycles under continuous perturbation of every
outcome-shaped input, unchanged when outcome statistics differ wildly
between two runs, and CHANGED when (and only when) the declared
dependency graph changes.
"""

from __future__ import annotations

import pytest

from tests.workspace.conftest import (
    FakeModule,
    make_registry,
    make_workspace,
    run_cycle,
)
from topos.contracts.intent import (
    FAIR_VALUE,
    FILL_RATE,
    FLOW_INTENSITY,
    IMPACT,
    KNOWN_HYPOTHESIS_IDS,
    QUEUE_POSITION,
)
from topos.contracts.registry import ModuleRegistry
from topos.workspace import WeightsIntegrityError, Workspace

HYPOTHESES = (FAIR_VALUE, FLOW_INTENSITY, FILL_RATE, IMPACT, QUEUE_POSITION)


def _modules() -> dict[str, FakeModule]:
    return {h: FakeModule(hypothesis_id=h) for h in HYPOTHESES}


def test_weights_static_and_structural() -> None:
    modules = _modules()
    workspace = make_workspace(modules)
    startup = workspace.weights

    # 1000 synthetic cycles, every outcome-shaped input perturbed each
    # cycle: EIGs, surprises, entropies, inventory, drive magnitudes.
    # Weights must not move by a bit. (cycle() also re-asserts this
    # internally; the loop would raise WeightsIntegrityError if not.)
    for step in range(1000):
        for index, module in enumerate(modules.values()):
            module.probe_gain = 0.2 * ((step * 7 + index * 3) % 10) / 10.0
            module.surprise = float((step * 13 + index) % 5)
            module.entropy = 1.0 + 0.1 * (step % 9)
        record = run_cycle(
            workspace,
            step=step,
            drives={"inventory": 0.3 * (step % 4) / 3.0},
        )
        assert record.step == step
        assert workspace.weights == startup

    # Outcome statistics perturbed wildly vs. benign: same graph, same
    # weights.
    benign = make_workspace(_modules())
    wild_modules = {
        h: FakeModule(hypothesis_id=h, probe_gain=9.0, surprise=1e6, entropy=50.0)
        for h in HYPOTHESES
    }
    wild = make_workspace(wild_modules)
    run_cycle(wild, drives={"drawdown": 100.0})
    assert benign.weights == wild.weights == startup

    # A different declared dependency graph — and nothing else — changes
    # the weights.
    structural = make_workspace(_modules(), registry=make_registry(hub=FAIR_VALUE))
    assert structural.weights != startup
    assert structural.weights[FAIR_VALUE] > startup[FAIR_VALUE]


def test_weight_drift_raises() -> None:
    """A registry that answers differently on a later cycle is an INV-7
    violation the workspace refuses to run past."""

    class DriftingRegistry(ModuleRegistry):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def centrality_weights(self):  # type: ignore[no-untyped-def]
            self.calls += 1
            weights = dict(super().centrality_weights())
            if self.calls > 1:
                first = next(iter(weights))
                weights[first] += 0.25
            return weights

    registry = DriftingRegistry()
    for name in KNOWN_HYPOTHESIS_IDS:
        registry.register(name, reads=(), writes=(f"{name}.headline",))
    workspace = make_workspace(_modules(), registry=registry)
    with pytest.raises(WeightsIntegrityError):
        run_cycle(workspace)


def test_startup_requires_full_known_id_coverage() -> None:
    """Every id in KNOWN_HYPOTHESIS_IDS must carry a weight at startup."""
    registry = ModuleRegistry()
    registry.register(FAIR_VALUE, reads=(), writes=("fv.headline",))
    with pytest.raises(ValueError, match="KNOWN_HYPOTHESIS_IDS"):
        make_workspace(_modules(), registry=registry)


def test_post_freeze_registration_raises() -> None:
    """Constructing the workspace freezes the registry (INV-7: computed
    once); late registration is impossible, not merely discouraged."""
    registry = make_registry()
    make_workspace(_modules(), registry=registry)
    with pytest.raises(Exception, match="frozen"):
        registry.register("latecomer", reads=(), writes=("x",))
