"""Fast end-to-end smoke of the metrics pipeline on one tiny episode.

One 120-step FULL run through collect -> reduce -> summarize -> checks ->
report. This is the cheap CI guard for the measurement instrument itself;
the statistical content of the falsification suite runs at ablation scale
in tests/acceptance/.
"""

from __future__ import annotations

import math

import pytest

from experiments.configs import ablation_background
from topos.metrics import (
    RunData,
    collect_run,
    reduce_run,
    render_report,
    run_all_checks,
    summarize_all,
    to_json_compat,
)
from topos.metrics.ablation import AblationRows

N_STEPS = 330
"""Small but large enough that both scheduled switches (at thirds) admit
full reawakening windows on each side (REAWAKENING_WINDOW_DEFAULT)."""
SEED = 101


@pytest.fixture(scope="module")
def run() -> RunData:
    return collect_run(
        "FULL", SEED, n_steps=N_STEPS, background=ablation_background(N_STEPS)
    )


@pytest.fixture(scope="module")
def rows(run: RunData) -> AblationRows:
    return reduce_run(run)


def test_paired_steps_alignment(run: RunData) -> None:
    pairs = list(run.paired_steps())
    assert len(pairs) == N_STEPS
    for step, record in pairs:
        assert record.step == step.step


def test_every_row_family_is_populated(rows: AblationRows) -> None:
    assert rows.calibration, "no calibration rows"
    assert rows.eig, "no ledger entries — the agent never probed"
    assert rows.efficiency, "no efficiency rows"
    assert rows.impact, "no impact rows — the agent never placed"
    assert rows.behavior_segments, "no segment rows"
    assert rows.behavior_switches, "no switch rows despite the schedule"
    assert len(rows.behavior_runs) == 1
    assert rows.outcome, "no outcome rows"


def test_calibration_channels_present_and_sane(rows: AblationRows) -> None:
    channels = {r["channel"] for r in rows.calibration}
    assert {"fair_value", "flow_intensity", "queue_position", "regime"} <= channels
    fair = [r for r in rows.calibration if r["channel"] == "fair_value"]
    assert sum(r["n"] for r in fair) > N_STEPS // 2
    for r in fair:
        if r["n"]:
            assert math.isfinite(r["rms_z"]) and r["rms_z"] > 0.0
    flow = [r for r in rows.calibration if r["channel"] == "flow_intensity"]
    assert flow, "flow rows require the twin log; collect_run defaults with_twin"
    for r in flow:
        assert r["truth"] > 0.0


def test_eig_rows_shape(rows: AblationRows) -> None:
    for r in rows.eig:
        assert r["condition"] == "FULL"
        assert r["promised"] >= 0.0
        assert math.isfinite(r["realized"])
        assert r["step_resolved"] > r["step_issued"]


def test_impact_rows_channels(rows: AblationRows) -> None:
    assert {r["channel"] for r in rows.impact} <= {
        "aggression",
        "resting",
        "placebo",
    }
    for r in rows.impact:
        assert math.isfinite(r["realized"])
        assert r["predicted_sd"] >= 0.0


def test_behavior_run_totals_match_message_log(
    run: RunData, rows: AblationRows
) -> None:
    run_row = rows.behavior_runs[0]
    assert run_row["total_messages"] == len(run.message_log)
    assert 0.0 <= run_row["excursion_frac"] <= 1.0


def test_summaries_checks_and_report_render(rows: AblationRows) -> None:
    results = summarize_all(rows)
    assert set(results) == {
        "belief_calibration",
        "eig_calibration",
        "scientific_efficiency",
        "impact_validation",
        "behavior_signatures",
        "outcome_pnl",
    }
    # Everything the runner persists must be JSON-serializable.
    to_json_compat({k: r.summary for k, r in results.items()})

    checks = run_all_checks(rows)
    assert [c.check_id for c in checks] == [f"F{i}" for i in range(1, 8)]
    # Single-condition data: the paired hard checks must report NOT
    # EVALUABLE rather than a spurious verdict.
    by_id = {c.check_id: c for c in checks}
    for check_id in ("F1", "F2", "F3", "F4"):
        assert by_id[check_id].passed is None

    report = render_report(
        rows, results, checks, {"scale": "unit", "n_steps": N_STEPS, "seeds": [SEED]}
    )
    for needle in (
        "# TOPOS-Market ablation report",
        "## Falsification suite",
        "## FULL vs ablations",
        "measured outcome",
        "## Metric 6",
    ):
        assert needle in report
    # Outcome stays last among the metric sections.
    assert report.rindex("## Metric 6") > report.rindex("## Metric 5")
