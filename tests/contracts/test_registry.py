from __future__ import annotations

import pytest

from topos.contracts.registry import ModuleRegistry, RegistryFrozenError


def _example_registry() -> ModuleRegistry:
    registry = ModuleRegistry()
    # fair_value is read by two modules; flow_intensity by one; proposer by none.
    registry.register("fair_value", reads={"trades", "book"}, writes={"fair_value"})
    registry.register(
        "flow_intensity", reads={"trades"}, writes={"flow_intensity"}
    )
    registry.register(
        "proposer",
        reads={"fair_value", "flow_intensity"},
        writes={"candidate_probes"},
    )
    registry.register("motor", reads={"fair_value"}, writes={"messages"})
    return registry


def test_dependency_edges_follow_reads_of_writes() -> None:
    registry = _example_registry()
    assert registry.dependency_edges() == frozenset(
        {
            ("fair_value", "proposer"),
            ("fair_value", "motor"),
            ("flow_intensity", "proposer"),
        }
    )


def test_centrality_is_normalized_out_degree() -> None:
    weights = _example_registry().centrality_weights()
    assert weights["fair_value"] == pytest.approx(2 / 3)
    assert weights["flow_intensity"] == pytest.approx(1 / 3)
    assert weights["proposer"] == 0.0
    assert weights["motor"] == 0.0
    assert sum(weights.values()) == pytest.approx(1.0)


def test_self_loops_are_excluded() -> None:
    registry = ModuleRegistry()
    registry.register("a", reads={"x"}, writes={"x"})
    registry.register("b", reads=set(), writes=set())
    assert registry.dependency_edges() == frozenset()


def test_computed_once_registry_freezes_and_caches() -> None:
    registry = _example_registry()
    first = registry.centrality_weights()
    assert registry.frozen
    assert registry.centrality_weights() is first
    with pytest.raises(RegistryFrozenError):
        registry.register("late_module", reads=set(), writes=set())


def test_weights_mapping_is_read_only() -> None:
    weights = _example_registry().centrality_weights()
    with pytest.raises(TypeError):
        weights["fair_value"] = 0.99  # type: ignore[index]


def test_duplicate_registration_raises() -> None:
    registry = ModuleRegistry()
    registry.register("a", reads=set(), writes=set())
    with pytest.raises(ValueError):
        registry.register("a", reads=set(), writes=set())


def test_edgeless_graph_falls_back_to_uniform() -> None:
    registry = ModuleRegistry()
    registry.register("a", reads=set(), writes={"x"})
    registry.register("b", reads=set(), writes={"y"})
    weights = registry.centrality_weights()
    assert weights == {"a": 0.5, "b": 0.5}


def test_empty_registry_rejects_centrality() -> None:
    with pytest.raises(ValueError):
        ModuleRegistry().centrality_weights()
