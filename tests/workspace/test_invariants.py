"""Workspace-local invariant pins.

INV-5 at package scale: arbitration consumes ``SelfStateCognitive`` only,
and the homeostat's byproducts (drives, vetoes, corrective intent,
projector) arrive as exported values. Mechanically: no account-state
vocabulary and no drives/metrics import anywhere under
``topos/workspace/`` — the same boundary the proposer pins for itself.

INV-7 at package scale: nothing under ``topos/workspace/`` may reach for
an outcome statistic to weight attention with. The behavioral half lives
in test_weights.py; here the source is scanned for the drives/metrics
boundary that would be the easiest smuggling route.
"""

from __future__ import annotations

import re
from pathlib import Path

import topos.workspace

WORKSPACE_ROOT = Path(topos.workspace.__file__).resolve().parent

ACCOUNT_TOKENS = re.compile(
    r"\bpnl\b|\bprofit\b|\bwealth\b|\bdrawdown\b|SelfStateFull",
    re.IGNORECASE,
)
FORBIDDEN_IMPORTS = re.compile(
    r"^\s*(from|import)\s+topos\.(drives|metrics)\b", re.MULTILINE
)


def test_no_account_state_vocabulary_in_workspace_source() -> None:
    offenders: list[str] = []
    for path in sorted(WORKSPACE_ROOT.rglob("*.py")):
        for lineno, line in enumerate(path.read_text().splitlines(), start=1):
            if ACCOUNT_TOKENS.search(line):
                offenders.append(f"{path}:{lineno}: {line.strip()}")
    assert not offenders, (
        "INV-5: account-state vocabulary in workspace source:\n"
        + "\n".join(offenders)
    )


def test_workspace_imports_neither_drives_nor_metrics() -> None:
    offenders: list[str] = []
    for path in sorted(WORKSPACE_ROOT.rglob("*.py")):
        if FORBIDDEN_IMPORTS.search(path.read_text()):
            offenders.append(str(path))
    assert not offenders, (
        "the workspace must receive homeostat byproducts as exported "
        "values, never import drives/ or metrics/: " + ", ".join(offenders)
    )
