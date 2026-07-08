"""Tripwire 3: agent code must never import topos.metrics.

Static import scan over topos/{beliefs,selfmodel,drives,proposer,workspace,
motor,agent,env}. Handles plain imports, from-imports, relative imports,
and `from topos import metrics`.
"""

from __future__ import annotations

import ast
from pathlib import Path

import topos

TOPOS_ROOT = Path(topos.__file__).resolve().parent
AGENT_PACKAGES = (
    "beliefs",
    "selfmodel",
    "drives",
    "proposer",
    "workspace",
    "motor",
    "agent",
    "env",
)


def _module_name(path: Path) -> str:
    relative = path.relative_to(TOPOS_ROOT.parent).with_suffix("")
    parts = list(relative.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _imported_module_candidates(path: Path) -> set[str]:
    """Every module name an import statement in this file could bind."""
    module_name = _module_name(path)
    if path.name == "__init__.py":
        package_parts = module_name.split(".")
    else:
        package_parts = module_name.split(".")[:-1]

    candidates: set[str] = set()
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            candidates.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                base = node.module or ""
            else:
                anchor = package_parts[: len(package_parts) - (node.level - 1)]
                base = ".".join(anchor)
                if node.module:
                    base = f"{base}.{node.module}" if base else node.module
            if base:
                candidates.add(base)
            # `from topos import metrics` binds topos.metrics via the alias.
            for alias in node.names:
                if base:
                    candidates.add(f"{base}.{alias.name}")
    return candidates


def _touches_metrics(candidates: set[str]) -> bool:
    return any(
        name == "topos.metrics" or name.startswith("topos.metrics.")
        for name in candidates
    )


def test_agent_packages_never_import_metrics() -> None:
    offenders: list[str] = []
    for package in AGENT_PACKAGES:
        package_root = TOPOS_ROOT / package
        assert package_root.is_dir(), f"expected package missing: {package_root}"
        for path in sorted(package_root.rglob("*.py")):
            if _touches_metrics(_imported_module_candidates(path)):
                offenders.append(str(path))
    assert not offenders, (
        "metrics-isolation tripwire: agent code imports topos.metrics:\n"
        + "\n".join(offenders)
    )
