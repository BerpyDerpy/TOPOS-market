"""Acceptance fixtures: one small-scale ablation per test session.

The falsification suite needs the real cross-condition dataset, so the
session fixture runs the CI-scale ablation (experiments.configs.SMALL:
fixed seeds, fixed schedule, all five conditions) exactly once and every
acceptance test reads from it. This is deliberately the same entry point
``python -m experiments.run_ablation --scale small`` uses — the tests
certify the shipped instrument, not a private variant.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import pytest

from experiments.configs import SMALL
from experiments.run_ablation import DEFAULT_WORKERS, run_ablation
from topos.metrics import AblationRows, CheckResult, MetricResult


@dataclass(frozen=True)
class AblationArtifacts:
    rows: AblationRows
    results: Mapping[str, MetricResult]
    checks: Sequence[CheckResult]
    meta: Mapping[str, Any]

    def check(self, check_id: str) -> CheckResult:
        for check in self.checks:
            if check.check_id == check_id:
                return check
        raise KeyError(check_id)


@pytest.fixture(scope="session")
def ablation() -> AblationArtifacts:
    rows, results, checks, meta = run_ablation(
        SMALL, max_workers=DEFAULT_WORKERS
    )
    return AblationArtifacts(rows=rows, results=results, checks=checks, meta=meta)
