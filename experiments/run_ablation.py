"""Run the P13 ablation and produce the single markdown report.

    python -m experiments.run_ablation --scale small
    python -m experiments.run_ablation --scale full --out results/full

Runs every (condition, seed) episode of the chosen scale — conditions and
seed lists are FIXED in ``experiments.configs`` (fairness: no
cherry-picking) — reduces each run to compact tidy rows inside worker
processes (full RunLogs never accumulate in the parent), then writes:

* ``report.md``     — the single human-readable ablation report,
* ``summary.json``  — every metric summary and falsification check,
* ``tables/*.csv``  — the full tidy tables per metric family.

Each worker holds at most one episode (run + twin) in memory; at the
full scale those logs are a few hundred MB each, so the default worker
count is conservative — raise ``--workers`` on a large machine.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from topos.metrics import (
    AblationRows,
    CheckResult,
    CONDITIONS,
    MetricResult,
    checks_to_json,
    collect_run,
    merge_rows,
    reduce_run,
    render_report,
    run_all_checks,
    summaries_to_json,
    summarize_all,
    to_json_compat,
)
from topos.metrics.tables import TidyTable

from experiments.configs import SCALES, AblationScale

DEFAULT_WORKERS = min(6, os.cpu_count() or 1)


def _reduce_one(condition: str, seed: int, n_steps: int) -> AblationRows:
    """Worker: one (condition, seed) episode -> compact metric rows.

    The background config is rebuilt from ``n_steps`` inside the worker
    (cheap, deterministic) so the task payload stays tiny.
    """
    from experiments.configs import ablation_background

    run = collect_run(
        condition,
        seed,
        n_steps=n_steps,
        background=ablation_background(n_steps),
        with_twin=True,
    )
    return reduce_run(run)


def run_ablation(
    scale: AblationScale,
    *,
    max_workers: int = DEFAULT_WORKERS,
    progress: Callable[[str], None] = lambda _line: None,
) -> tuple[AblationRows, dict[str, MetricResult], list[CheckResult], dict[str, Any]]:
    """Run the whole ablation at one scale; return rows, results, checks, meta."""
    tasks = [
        (condition, seed)
        for condition in CONDITIONS
        for seed in scale.seeds
    ]
    started = time.time()
    parts: dict[tuple[str, int], AblationRows] = {}
    if max_workers <= 1:
        for condition, seed in tasks:
            parts[(condition, seed)] = _reduce_one(condition, seed, scale.n_steps)
            progress(f"done {condition} seed={seed}")
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(_reduce_one, condition, seed, scale.n_steps): (
                    condition,
                    seed,
                )
                for condition, seed in tasks
            }
            for future, key in futures.items():
                parts[key] = future.result()
                progress(f"done {key[0]} seed={key[1]}")
    # Merge in the fixed (condition, seed) order, independent of completion.
    rows = merge_rows(parts[key] for key in tasks)
    results = summarize_all(rows)
    checks = run_all_checks(rows)
    meta: dict[str, Any] = {
        "scale": scale.name,
        "n_steps": scale.n_steps,
        "seeds": list(scale.seeds),
        "conditions": list(CONDITIONS),
        "schedule": list(scale.schedule),
        "workers": max_workers,
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "wall_seconds": time.time() - started,
    }
    return rows, results, checks, meta


def _rows_tables(rows: AblationRows) -> dict[str, TidyTable]:
    return {
        name: TidyTable.from_records(name, records)
        for name, records in (
            ("belief_calibration", rows.calibration),
            ("eig_calibration", rows.eig),
            ("scientific_efficiency", rows.efficiency),
            ("impact_validation", rows.impact),
            ("behavior_segments", rows.behavior_segments),
            ("behavior_switches", rows.behavior_switches),
            ("behavior_runs", rows.behavior_runs),
            ("outcome_pnl", rows.outcome),
        )
    }


def write_outputs(
    out_dir: Path,
    rows: AblationRows,
    results: Mapping[str, MetricResult],
    checks: Sequence[CheckResult],
    meta: Mapping[str, Any],
) -> Path:
    """Write report.md, summary.json and tables/*.csv; return report path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = out_dir / "tables"
    tables_dir.mkdir(exist_ok=True)
    for name, table in _rows_tables(rows).items():
        (tables_dir / f"{name}.csv").write_text(table.to_csv())
    summary = {
        "meta": to_json_compat(dict(meta)),
        "checks": to_json_compat(checks_to_json(checks)),
        "metrics": to_json_compat(summaries_to_json(results)),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    report_path = out_dir / "report.md"
    report_path.write_text(render_report(rows, results, checks, meta))
    return report_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m experiments.run_ablation",
        description="Run the TOPOS-Market P13 ablation and write the report.",
    )
    parser.add_argument(
        "--scale",
        choices=sorted(SCALES),
        default="small",
        help="experiment size (seed lists and step counts are fixed per scale)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output directory (default: results/<scale>)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"parallel worker processes (default {DEFAULT_WORKERS})",
    )
    args = parser.parse_args(argv)
    scale = SCALES[args.scale]
    out_dir = args.out if args.out is not None else Path("results") / scale.name

    total = len(CONDITIONS) * len(scale.seeds)
    done = {"n": 0}

    def progress(line: str) -> None:
        done["n"] += 1
        print(f"[{done['n']:>3}/{total}] {line}", flush=True)

    print(
        f"ablation scale={scale.name}: {scale.n_steps} steps x "
        f"{len(scale.seeds)} seeds x {len(CONDITIONS)} conditions, "
        f"{args.workers} workers",
        flush=True,
    )
    rows, results, checks, meta = run_ablation(
        scale, max_workers=args.workers, progress=progress
    )
    report_path = write_outputs(out_dir, rows, results, checks, meta)
    print(f"report: {report_path}")
    for check in checks:
        print(f"  {check.check_id}: {check.status} (value={check.value:.4g})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
