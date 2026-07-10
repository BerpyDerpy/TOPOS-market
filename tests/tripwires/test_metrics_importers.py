"""Tripwire: nothing outside the measurement side may import topos.metrics.

Repo-wide static scan (the companion of test_metrics_isolation, which
covers the agent packages): every Python file in the repository is
checked, and only files under tests/, experiments/, or topos/metrics/
itself may import ``topos.metrics`` by any absolute form. The metrics
package may import anything; nothing agent-facing may know it exists.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

ALLOWED_PREFIXES = (
    REPO_ROOT / "tests",
    REPO_ROOT / "experiments",
    REPO_ROOT / "topos" / "metrics",
)

EXCLUDED_DIR_NAMES = {".venv", "__pycache__", ".git", "results", "build"}


def _scanned_files() -> list[Path]:
    files: list[Path] = []
    for path in sorted(REPO_ROOT.rglob("*.py")):
        if any(part in EXCLUDED_DIR_NAMES for part in path.parts):
            continue
        files.append(path)
    return files


def _imports_metrics(path: Path) -> bool:
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "topos.metrics" or alias.name.startswith(
                    "topos.metrics."
                ):
                    return True
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                continue  # relative imports cannot reach topos.metrics from
                # outside topos/, and inside topos/ only metrics/ is allowed
                # anyway (checked by path below before this is consulted)
            base = node.module or ""
            if base == "topos.metrics" or base.startswith("topos.metrics."):
                return True
            if base == "topos" and any(
                alias.name == "metrics" for alias in node.names
            ):
                return True
    return False


def _is_allowed(path: Path) -> bool:
    return any(path.is_relative_to(prefix) for prefix in ALLOWED_PREFIXES)


def test_scan_covers_the_repo() -> None:
    files = _scanned_files()
    assert len(files) >= 40, "scan found suspiciously few files; layout changed?"
    # The scan must actually see the agent and experiments trees.
    assert any("topos" in f.parts and "agent" in f.parts for f in files)
    assert any("experiments" in f.parts for f in files)


def test_only_tests_and_experiments_import_metrics() -> None:
    offenders = [
        str(path.relative_to(REPO_ROOT))
        for path in _scanned_files()
        if not _is_allowed(path) and _imports_metrics(path)
    ]
    assert not offenders, (
        "metrics-importer tripwire: only tests/, experiments/ and "
        "topos/metrics/ may import topos.metrics; offenders:\n"
        + "\n".join(offenders)
    )


def test_relative_import_note_holds() -> None:
    """The relative-import shortcut in the scanner is sound only while no
    package outside metrics/ lives UNDER topos/metrics/'s parent with a
    sibling relative path to it — i.e. while `topos/metrics` has no
    subpackages that agent code could sit inside. Pin that assumption."""
    metrics_dir = REPO_ROOT / "topos" / "metrics"
    subpackages = [
        p for p in metrics_dir.iterdir() if p.is_dir() and p.name != "__pycache__"
    ]
    assert not subpackages, (
        "topos/metrics grew subdirectories; revisit the relative-import "
        "shortcut in this tripwire"
    )
