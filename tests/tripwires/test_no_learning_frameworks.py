"""Tripwire 2 (INV-2): no learning frameworks reachable from agent packages.

Two independent checks: (a) dynamically import every module under topos/
except topos/metrics/ and assert sys.modules gains no forbidden framework;
(b) statically parse every agent-facing file's import statements.
"""

from __future__ import annotations

import ast
import importlib
import pkgutil
import sys
from pathlib import Path

import topos

FORBIDDEN_ROOTS = {"torch", "jax", "tensorflow", "sklearn"}
TOPOS_ROOT = Path(topos.__file__).resolve().parent


def _agent_facing_module_names() -> list[str]:
    names = ["topos"]
    for info in pkgutil.walk_packages(topos.__path__, prefix="topos."):
        if info.name == "topos.metrics" or info.name.startswith("topos.metrics."):
            continue
        names.append(info.name)
    return names


def test_importing_agent_packages_pulls_no_learning_framework() -> None:
    before = {name.split(".")[0] for name in sys.modules}
    preloaded = before & FORBIDDEN_ROOTS
    assert not preloaded, (
        f"test environment already had {sorted(preloaded)} imported; "
        "the dynamic check would be blind — fix the test environment"
    )
    for name in _agent_facing_module_names():
        importlib.import_module(name)
    after = {name.split(".")[0] for name in sys.modules}
    gained = (after - before) & FORBIDDEN_ROOTS
    assert not gained, (
        f"INV-2 tripwire: importing agent packages pulled in {sorted(gained)}"
    )


def _static_import_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                roots.add(node.module.split(".")[0])
    return roots


def test_no_static_imports_of_learning_frameworks() -> None:
    offenders: list[str] = []
    for path in sorted(TOPOS_ROOT.rglob("*.py")):
        if path.relative_to(TOPOS_ROOT).parts[0] == "metrics":
            continue
        bad = _static_import_roots(path) & FORBIDDEN_ROOTS
        if bad:
            offenders.append(f"{path}: imports {sorted(bad)}")
    assert not offenders, "INV-2 tripwire:\n" + "\n".join(offenders)
