"""Metric 2 — EIG calibration: promised vs realized information gain.

One row per experiment-ledger entry: the EIG the arbiter acted on
(``eig_promised_nats``) against the realized information gain from the
INV-10 entropy snapshots (``realized_ig_nats``). Per hypothesis we report
the scatter's OLS regression slope: slope ~ 1 means the agent's curiosity
is calibrated (it gets the information it pays for); slope ~ 0 with
realized ~ 0 while promises are materially positive is the signature of
the INV-10 snapshot-ordering bug (both snapshots taken after the update),
and the summary flags it as such rather than as a design failure.

Under SURPRISE_CURIOSITY the promised column is a z-scored surprise, not
nats — the miscalibration against realized nats is the point of that
ablation, not a unit bug.

Windowed realized IG (the ledger-timing correction)
---------------------------------------------------
The P12 ledger resolves a probe on the NEXT cycle's observation, but the
harness answers an action two observations later (the two-slot shift,
DESIGN item 33), and a fill trial then needs the fill horizon on top —
so for fill_rate probes the ledger's realized IG is measured before the
probe's own outcome can have reached the posterior. ``realized_windowed``
is the metrics-side corrected estimator: the target module's headline
entropy at issuance minus its headline entropy ``ack shift (2) + probe
horizon`` cycles later. It is contaminated by neighboring experiments
resolving inside the window and by slow-tick forgetting (which injects
negative IG), so it is reported NEXT TO the ledger figure, never instead
of it. See DESIGN.md, Open questions (P13).
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from topos.metrics.collect import RunData
from topos.metrics.stats import ols
from topos.metrics.tables import MetricResult, TidyTable

INV10_SLOPE_SUSPECT = 0.1
"""|slope| below this, with realized ~ 0, triggers the INV-10 flag."""

INV10_REALIZED_FRACTION = 0.05
"""'Realized ~ 0' means |mean realized| below this fraction of the mean
promise (with promises materially positive)."""


ACK_SHIFT_STEPS = 2
"""Observations between deciding an action and seeing its acks (item 33)."""


def eig_rows(run: RunData) -> list[dict[str, Any]]:
    """One tidy row per resolved ledger entry of one run."""
    # Headline entropy per (cycle step -> hypothesis -> entropy). The
    # reset record shares stamp 0 with the first step record; the LAST
    # record per stamp is the cycle that saw that engine step's
    # observation, which is the one the ledger snapshots align with.
    entropy_at: dict[int, dict[str, float]] = {}
    for record in run.records:
        entropy_at[record.step] = {
            h.hypothesis_id: h.epistemic_entropy_nats for h in record.headlines
        }
    horizon = run.agent_config.fill_horizon_steps
    rows: list[dict[str, Any]] = []
    for entry in run.experiments:
        target_step = entry.step_issued + ACK_SHIFT_STEPS + horizon
        h_issue = entropy_at.get(entry.step_issued, {}).get(entry.target_id)
        h_later = entropy_at.get(target_step, {}).get(entry.target_id)
        rows.append(
            {
                "condition": run.condition,
                "seed": run.root_seed,
                "hypothesis": entry.target_id,
                "step_issued": entry.step_issued,
                "step_resolved": entry.step_resolved,
                "promised": entry.eig_promised_nats,
                "realized": entry.realized_ig_nats,
                "realized_windowed": (
                    h_issue - h_later
                    if h_issue is not None and h_later is not None
                    else math.nan
                ),
            }
        )
    return rows


def _slope_summary(entries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    promised = [float(e["promised"]) for e in entries]
    realized = [float(e["realized"]) for e in entries]
    n = len(entries)
    mean_promised = sum(promised) / n if n else math.nan
    mean_realized = sum(realized) / n if n else math.nan
    fit = ols(promised, realized)
    windowed_pairs = [
        (float(e["promised"]), float(e["realized_windowed"]))
        for e in entries
        if math.isfinite(float(e.get("realized_windowed", math.nan)))
    ]
    fit_windowed = ols(
        [p for p, _ in windowed_pairs], [r for _, r in windowed_pairs]
    )
    suspect_inv10 = (
        n >= 5
        and math.isfinite(fit.slope)
        and abs(fit.slope) < INV10_SLOPE_SUSPECT
        and mean_promised > 1e-4
        and abs(mean_realized) < INV10_REALIZED_FRACTION * mean_promised
    )
    return {
        "n": n,
        "slope": fit.slope,
        "slope_se": fit.se_slope,
        "slope_ci95": list(fit.slope_ci95),
        "r2": fit.r2,
        "mean_promised": mean_promised,
        "mean_realized": mean_realized,
        "ratio_realized_over_promised": (
            mean_realized / mean_promised if mean_promised else math.nan
        ),
        "slope_windowed": fit_windowed.slope,
        "slope_windowed_se": fit_windowed.se_slope,
        "n_windowed": fit_windowed.n,
        "mean_realized_windowed": (
            sum(r for _, r in windowed_pairs) / len(windowed_pairs)
            if windowed_pairs
            else math.nan
        ),
        "suspect_inv10_snapshot_bug": suspect_inv10,
    }


def summarize_eig(rows: Sequence[Mapping[str, Any]]) -> MetricResult:
    table = TidyTable.from_records("eig_calibration", list(rows))
    summary: dict[str, Any] = {}
    conditions = sorted({r["condition"] for r in rows})
    for condition in conditions:
        of_condition = [r for r in rows if r["condition"] == condition]
        hypotheses = sorted({r["hypothesis"] for r in of_condition})
        summary[condition] = {
            hypothesis: _slope_summary(
                [r for r in of_condition if r["hypothesis"] == hypothesis]
            )
            for hypothesis in hypotheses
        }
    return MetricResult(name="eig_calibration", table=table, summary=summary)


def eig_calibration(runs: Sequence[RunData]) -> MetricResult:
    """Pure function RunData(s) -> tidy table + summary (metric 2)."""
    rows: list[dict[str, Any]] = []
    for run in runs:
        rows.extend(eig_rows(run))
    return summarize_eig(rows)
