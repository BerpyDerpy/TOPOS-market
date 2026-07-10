"""Metric 6 — outcome: PnL curves, per-regime PnL, max drawdown.

PnL here is a MEASURED OUTCOME of curiosity-driven behavior, never an
objective: no agent-facing code can read it (INV-5), no constant anywhere
was set against it, and this module exists precisely to observe what
profit-shaped behavior does or does not emerge when none is asked for.
Any report text generated from this metric must keep that framing.

Equity is engine ground truth (INV-11 channel): cash + inventory * mid,
in tick-lots, with the mid carried forward through one-sided books. The
per-run equity series is reduced to per-regime-segment deltas, final PnL,
and maximum drawdown.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from topos.metrics.collect import RunData
from topos.metrics.segments import regime_segments
from topos.metrics.stats import iqr, max_drawdown, median
from topos.metrics.tables import MetricResult, TidyTable


def _equity_series(run: RunData) -> list[float]:
    agent_id = run.run_log.agent_actor_id
    mid: float | None = None
    series: list[float] = []
    for step in run.run_log.steps:
        book_mid = step.book.mid
        if book_mid is not None:
            mid = book_mid
        account = step.account(agent_id)
        if mid is None:
            series.append(float(account.cash))
        else:
            series.append(account.cash + account.inventory_lots * mid)
    return series


def outcome_rows(run: RunData) -> list[dict[str, Any]]:
    """Per-segment PnL deltas plus one whole-run row (scope='run')."""
    equity = _equity_series(run)
    base = {"condition": run.condition, "seed": run.root_seed}
    rows: list[dict[str, Any]] = []
    for seg in regime_segments(run.run_log):
        start_equity = equity[seg.start_step - 1] if seg.start_step > 0 else 0.0
        end_equity = equity[min(seg.end_step, len(equity)) - 1]
        rows.append(
            {
                **base,
                "scope": "segment",
                "segment": seg.index,
                "regime_id": seg.regime_id,
                "length": seg.length,
                "pnl_ticklots": end_equity - start_equity,
                "max_drawdown_ticklots": max_drawdown(
                    equity[seg.start_step : seg.end_step]
                ),
            }
        )
    rows.append(
        {
            **base,
            "scope": "run",
            "segment": -1,
            "regime_id": "ALL",
            "length": len(equity),
            "pnl_ticklots": equity[-1] if equity else 0.0,
            "max_drawdown_ticklots": max_drawdown(equity),
        }
    )
    return rows


def summarize_outcome(rows: Sequence[Mapping[str, Any]]) -> MetricResult:
    table = TidyTable.from_records("outcome_pnl", list(rows))
    summary: dict[str, Any] = {
        "framing": (
            "PnL is a measured outcome of curiosity-driven behavior, "
            "never an objective; no agent constant was set against it."
        )
    }
    run_rows = [r for r in rows if r["scope"] == "run"]
    for condition in sorted({r["condition"] for r in run_rows}):
        of_condition = [r for r in run_rows if r["condition"] == condition]
        pnl = [float(r["pnl_ticklots"]) for r in of_condition]
        drawdown = [float(r["max_drawdown_ticklots"]) for r in of_condition]
        summary[condition] = {
            "n_seeds": len(of_condition),
            "median_final_pnl_ticklots": median(pnl),
            "iqr_final_pnl_ticklots": iqr(pnl),
            "median_max_drawdown_ticklots": median(drawdown),
            "frac_seeds_pnl_positive": (
                sum(1 for v in pnl if v > 0) / len(pnl) if pnl else math.nan
            ),
        }
    return MetricResult(name="outcome_pnl", table=table, summary=summary)


def outcome(runs: Sequence[RunData]) -> MetricResult:
    """Pure function RunData(s) -> tidy table + summary (metric 6 — last,
    deliberately: the outcome is read only after the science is scored)."""
    rows: list[dict[str, Any]] = []
    for run in runs:
        rows.extend(outcome_rows(run))
    return summarize_outcome(rows)
