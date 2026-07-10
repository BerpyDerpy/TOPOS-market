"""Acceptance: the ablation report and artifacts, and the fairness frame.

Certifies the deliverable surface: ``write_outputs`` produces the single
markdown report, the JSON summary and the CSV tables; the report keeps
the load-bearing framing (PnL as measured outcome, falsification suite
first, paired deltas front and center, outcome last); and the dataset
itself honors the fairness rules (every condition ran the identical
fixed seed list).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.run_ablation import write_outputs
from tests.acceptance.conftest import AblationArtifacts


@pytest.fixture(scope="session")
def out_dir(
    ablation: AblationArtifacts, tmp_path_factory: pytest.TempPathFactory
) -> Path:
    target = tmp_path_factory.mktemp("ablation-out")
    write_outputs(
        target, ablation.rows, ablation.results, ablation.checks, ablation.meta
    )
    return target


def test_paired_seed_fairness(ablation: AblationArtifacts) -> None:
    """Identical fixed seed lists per condition — no cherry-picking."""
    seeds_by_condition: dict[str, set[int]] = {}
    for row in ablation.rows.behavior_runs:
        seeds_by_condition.setdefault(row["condition"], set()).add(row["seed"])
    expected = set(ablation.meta["seeds"])
    assert set(seeds_by_condition) == set(ablation.meta["conditions"])
    for condition, seeds in seeds_by_condition.items():
        assert seeds == expected, f"{condition} ran a different seed set"


def test_artifacts_written(out_dir: Path) -> None:
    assert (out_dir / "report.md").is_file()
    summary = json.loads((out_dir / "summary.json").read_text())
    assert {"meta", "checks", "metrics"} <= set(summary)
    assert len(summary["checks"]) == 7
    tables = {p.name for p in (out_dir / "tables").glob("*.csv")}
    assert {
        "belief_calibration.csv",
        "eig_calibration.csv",
        "scientific_efficiency.csv",
        "impact_validation.csv",
        "behavior_segments.csv",
        "behavior_switches.csv",
        "behavior_runs.csv",
        "outcome_pnl.csv",
    } <= tables


def test_report_framing(out_dir: Path) -> None:
    report = (out_dir / "report.md").read_text()
    # PnL is outcome, never objective — the framing is part of the
    # instrument and must survive rendering.
    assert "measured outcome" in report
    assert "never an objective" in report
    # Section order: falsification first, paired deltas next, outcome last.
    falsification = report.index("## Falsification suite")
    paired = report.index("## FULL vs ablations")
    outcome = report.index("## Metric 6")
    assert falsification < paired < outcome
    for check_id in ("F1", "F2", "F3", "F4", "F5", "F6", "F7"):
        assert check_id in report
    # The F5 scope note (world hypotheses cannot be ledgered under FULL)
    # must reach readers, not just DESIGN.md.
    assert "F5 scope note" in report


def test_all_metric_summaries_cover_all_conditions(
    ablation: AblationArtifacts,
) -> None:
    conditions = set(ablation.meta["conditions"])
    for name in (
        "scientific_efficiency",
        "behavior_signatures",
        "outcome_pnl",
    ):
        summary = ablation.results[name].summary
        covered = {k for k, v in summary.items() if isinstance(v, dict)}
        assert conditions <= covered, f"{name} summary missing conditions"
