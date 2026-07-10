"""Per-run reduction and cross-run assembly for the ablation harness.

``reduce_run`` turns one RunData into compact tidy rows for every metric
family — the only thing workers return to the parent process, so full
RunLogs (which hold per-step book snapshots and cumulative account
views) never accumulate across the ablation. ``merge_rows`` concatenates
per-run reductions; ``summarize_all`` produces the six MetricResults the
report renders.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from topos.metrics.behavior import (
    behavior_run_rows,
    behavior_segment_rows,
    behavior_switch_rows,
    summarize_behavior,
)
from topos.metrics.calibration import calibration_rows, summarize_calibration
from topos.metrics.collect import RunData
from topos.metrics.efficiency import efficiency_rows, summarize_efficiency
from topos.metrics.eig import eig_rows, summarize_eig
from topos.metrics.impact import impact_rows, summarize_impact
from topos.metrics.outcome import outcome_rows, summarize_outcome
from topos.metrics.tables import MetricResult

Row = dict[str, Any]


@dataclass
class AblationRows:
    """All tidy rows of an ablation (or of one run), keyed by family."""

    calibration: list[Row] = field(default_factory=list)
    eig: list[Row] = field(default_factory=list)
    efficiency: list[Row] = field(default_factory=list)
    impact: list[Row] = field(default_factory=list)
    behavior_segments: list[Row] = field(default_factory=list)
    behavior_switches: list[Row] = field(default_factory=list)
    behavior_runs: list[Row] = field(default_factory=list)
    outcome: list[Row] = field(default_factory=list)

    def extend(self, other: "AblationRows") -> None:
        self.calibration.extend(other.calibration)
        self.eig.extend(other.eig)
        self.efficiency.extend(other.efficiency)
        self.impact.extend(other.impact)
        self.behavior_segments.extend(other.behavior_segments)
        self.behavior_switches.extend(other.behavior_switches)
        self.behavior_runs.extend(other.behavior_runs)
        self.outcome.extend(other.outcome)


def reduce_run(run: RunData) -> AblationRows:
    """All metric-family rows of one run (RunLog-free, cheaply picklable)."""
    return AblationRows(
        calibration=calibration_rows(run),
        eig=eig_rows(run),
        efficiency=efficiency_rows(run),
        impact=impact_rows(run),
        behavior_segments=behavior_segment_rows(run),
        behavior_switches=behavior_switch_rows(run),
        behavior_runs=behavior_run_rows(run),
        outcome=outcome_rows(run),
    )


def merge_rows(parts: Iterable[AblationRows]) -> AblationRows:
    merged = AblationRows()
    for part in parts:
        merged.extend(part)
    return merged


def summarize_all(rows: AblationRows) -> dict[str, MetricResult]:
    """The six MetricResults, in the report's deliberate order: science
    first, outcome last."""
    return {
        "belief_calibration": summarize_calibration(rows.calibration),
        "eig_calibration": summarize_eig(rows.eig),
        "scientific_efficiency": summarize_efficiency(rows.efficiency),
        "impact_validation": summarize_impact(rows.impact),
        "behavior_signatures": summarize_behavior(
            rows.behavior_segments, rows.behavior_switches, rows.behavior_runs
        ),
        "outcome_pnl": summarize_outcome(rows.outcome),
    }
