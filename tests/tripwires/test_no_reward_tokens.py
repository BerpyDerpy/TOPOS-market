"""Tripwire 1 (INV-1): no scalar-feedback vocabulary in agent-facing source.

Scans every .py file under topos/ except topos/metrics/, comments and
strings included — the point is that the concept must not exist in
agent-facing code, not merely that no variable is named after it.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator

TOPOS_ROOT = Path(__file__).resolve().parents[2] / "topos"
FORBIDDEN = re.compile(r"\breward\b|\breturn_\b|\bq_value\b")


def _scanned_files() -> Iterator[Path]:
    for path in sorted(TOPOS_ROOT.rglob("*.py")):
        relative = path.relative_to(TOPOS_ROOT)
        if relative.parts[0] == "metrics":
            continue
        yield path


def test_scan_actually_covers_the_tree() -> None:
    files = list(_scanned_files())
    assert len(files) >= 10, "scan found suspiciously few files; layout changed?"


def test_no_reward_tokens() -> None:
    offenders: list[str] = []
    for path in _scanned_files():
        for lineno, line in enumerate(path.read_text().splitlines(), start=1):
            if FORBIDDEN.search(line):
                offenders.append(f"{path}:{lineno}: {line.strip()}")
    assert not offenders, (
        "INV-1 tripwire: forbidden feedback-signal tokens in agent-facing "
        "source:\n" + "\n".join(offenders)
    )
