"""Module registry and structural consequence-weights (INV-7).

Modules declare, once at startup, their name and the blackboard keys they
read and write. The registry builds the directed dependency graph — edge
A -> B iff B reads at least one key that A writes — and derives the salience
consequence-weights w_h from that graph's structure alone.

Why outcome statistics are forbidden here: a consequence-weight says "this
hypothesis matters because many downstream modules depend on what it
writes". That is a fact about the architecture, knowable before the market
opens, and it never changes with experience. The moment w_h is allowed to
adapt to outcomes — fill rates, PnL, surprise history, realized information
gain, anything measured — attention becomes an optimization channel and a
de-facto value signal enters through the back door, violating the premise
that adaptive behavior emerges from structure, not from feedback tuning
(INV-1, INV-7). Hence: declarations in, weights out, computed exactly once;
the registry freezes on first computation and further registration raises.

Centrality choice (documented per spec): normalized out-degree. A module's
weight is the number of distinct modules that read what it writes
(self-loops excluded), normalized so weights sum to 1. Eigenvector
centrality is an acceptable substitute; out-degree is chosen for being
trivially auditable by hand.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

from topos.contracts.intent import HypothesisId


@dataclass(frozen=True)
class ModuleDecl:
    """A module's startup declaration: what it reads and writes."""

    name: str
    reads: frozenset[str]
    writes: frozenset[str]


class RegistryFrozenError(RuntimeError):
    """Raised on registration after the registry froze (INV-7: compute once)."""


class ModuleRegistry:
    """Collects module declarations and derives structural weights.

    Lifecycle: register(...) during startup, then the first call to
    `centrality_weights()` (or an explicit `freeze()`) freezes the registry
    permanently. Weights are cached; there is no code path by which they
    can ever be recomputed or updated afterwards.
    """

    def __init__(self) -> None:
        self._modules: dict[str, ModuleDecl] = {}
        self._frozen: bool = False
        self._weights: Mapping[HypothesisId, float] | None = None

    def register(
        self, name: str, reads: Iterable[str], writes: Iterable[str]
    ) -> ModuleDecl:
        """Declare a module. Allowed only before the registry freezes."""
        if self._frozen:
            raise RegistryFrozenError(
                f"registry is frozen; cannot register {name!r} (INV-7)"
            )
        if not name:
            raise ValueError("module name must be non-empty")
        if name in self._modules:
            raise ValueError(f"module {name!r} already registered")
        decl = ModuleDecl(name=name, reads=frozenset(reads), writes=frozenset(writes))
        self._modules[name] = decl
        return decl

    def freeze(self) -> None:
        """Permanently close the registry to new declarations."""
        self._frozen = True

    @property
    def frozen(self) -> bool:
        return self._frozen

    def modules(self) -> Mapping[str, ModuleDecl]:
        return MappingProxyType(dict(self._modules))

    def dependency_edges(self) -> frozenset[tuple[str, str]]:
        """Directed edges (A, B): B reads something A writes; A != B."""
        edges: set[tuple[str, str]] = set()
        for a in self._modules.values():
            for b in self._modules.values():
                if a.name != b.name and a.writes & b.reads:
                    edges.add((a.name, b.name))
        return frozenset(edges)

    def centrality_weights(self) -> Mapping[HypothesisId, float]:
        """Salience consequence-weights from declarations only (INV-7).

        Normalized out-degree over the dependency graph, computed exactly
        once: this call freezes the registry and caches the result. If the
        declared graph has no edges, weights fall back to uniform. Keys are
        the registered module names; by convention a hypothesis-owning
        module registers under its `hypothesis_id`.
        """
        if self._weights is not None:
            return self._weights
        if not self._modules:
            raise ValueError("cannot compute centrality of an empty registry")
        self.freeze()
        out_degree: dict[str, int] = {name: 0 for name in self._modules}
        for source, _target in self.dependency_edges():
            out_degree[source] += 1
        total = sum(out_degree.values())
        if total == 0:
            uniform = 1.0 / len(self._modules)
            weights = {name: uniform for name in self._modules}
        else:
            weights = {name: deg / total for name, deg in out_degree.items()}
        self._weights = MappingProxyType(weights)
        return self._weights


DEFAULT_REGISTRY_NOTE: Final[str] = (
    "Consequence-weights are structural, not empirical: computed once at "
    "startup from declared reads/writes, never from outcome statistics."
)
