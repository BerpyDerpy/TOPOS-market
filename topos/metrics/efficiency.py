"""Metric 3 — scientific efficiency: information gained per unit of action.

Information has two ledgers here, reported side by side:

* **Bought information** — realized IG (nats) from resolved experiment-
  ledger entries (INV-10 snapshots): the information the agent paid for
  with actions.
* **Total epistemic movement** — the drop in summed parameter-posterior
  entropy across all headline modules since the first cycle (free, passive
  information included). This can move without any action at all (world
  modules learn from public data), which is exactly the comparison the
  metric exists to expose.

Efficiency ratios divide by exchange messages sent and by fills received.
Cumulative series are emitted at a fixed bin cadence so the tidy table
stays CI-sized.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from topos.metrics.collect import RunData
from topos.metrics.stats import median
from topos.metrics.tables import MetricResult, TidyTable

SERIES_BIN_DEFAULT = 25


def _entropy_total(record: Any) -> float:
    return float(sum(h.epistemic_entropy_nats for h in record.headlines))


def efficiency_rows(
    run: RunData, *, series_bin: int = SERIES_BIN_DEFAULT
) -> list[dict[str, Any]]:
    """Binned cumulative efficiency series for one run (plus a final row)."""
    steps = run.run_log.steps
    n = len(steps)
    agent_id = run.run_log.agent_actor_id

    messages_by_step: dict[int, int] = {}
    for message_step, _message in run.message_log:
        messages_by_step[message_step] = messages_by_step.get(message_step, 0) + 1

    ig_by_step: dict[int, float] = {}
    for entry in run.experiments:
        ig_by_step[entry.step_resolved] = (
            ig_by_step.get(entry.step_resolved, 0.0) + entry.realized_ig_nats
        )

    first_entropy: float | None = None
    rows: list[dict[str, Any]] = []
    cum_messages = 0
    cum_ig = 0.0
    for i, step in enumerate(steps):
        cum_messages += messages_by_step.get(step.step, 0)
        cum_ig += ig_by_step.get(step.step, 0.0)
        record = run.records[i + 1] if i + 1 < len(run.records) else None
        entropy = _entropy_total(record) if record is not None else math.nan
        if first_entropy is None and math.isfinite(entropy):
            first_entropy = entropy
        is_last = i == n - 1
        if not is_last and (i % series_bin) != series_bin - 1:
            continue
        cum_fills = len(step.account(agent_id).fills)
        entropy_drop = (
            first_entropy - entropy
            if first_entropy is not None and math.isfinite(entropy)
            else math.nan
        )
        rows.append(
            {
                "condition": run.condition,
                "seed": run.root_seed,
                "step": step.step,
                "final": is_last,
                "cum_messages": cum_messages,
                "cum_fills": cum_fills,
                "cum_realized_ig_nats": cum_ig,
                "entropy_drop_nats": entropy_drop,
                "ig_per_message": cum_ig / cum_messages if cum_messages else math.nan,
                "ig_per_fill": cum_ig / cum_fills if cum_fills else math.nan,
            }
        )
    return rows


def summarize_efficiency(rows: Sequence[Mapping[str, Any]]) -> MetricResult:
    table = TidyTable.from_records("scientific_efficiency", list(rows))
    summary: dict[str, Any] = {}
    finals = [r for r in rows if r.get("final")]
    for condition in sorted({r["condition"] for r in finals}):
        of_condition = [r for r in finals if r["condition"] == condition]

        def med(key: str) -> float:
            values = [
                float(r[key])
                for r in of_condition
                if isinstance(r[key], (int, float)) and math.isfinite(float(r[key]))
            ]
            return median(values)

        summary[condition] = {
            "n_seeds": len(of_condition),
            "median_total_messages": med("cum_messages"),
            "median_total_fills": med("cum_fills"),
            "median_total_realized_ig_nats": med("cum_realized_ig_nats"),
            "median_entropy_drop_nats": med("entropy_drop_nats"),
            "median_ig_per_message": med("ig_per_message"),
            "median_ig_per_fill": med("ig_per_fill"),
        }
    return MetricResult(name="scientific_efficiency", table=table, summary=summary)


def scientific_efficiency(
    runs: Sequence[RunData], *, series_bin: int = SERIES_BIN_DEFAULT
) -> MetricResult:
    """Pure function RunData(s) -> tidy table + summary (metric 3)."""
    rows: list[dict[str, Any]] = []
    for run in runs:
        rows.extend(efficiency_rows(run, series_bin=series_bin))
    return summarize_efficiency(rows)
