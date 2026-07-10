"""Metric 5 — behavior signatures: the shape of curiosity over time.

Three tidy row families per run, all sliced by the TRUE regime segments
(harness ground truth, identical across paired conditions):

* segment rows — per (run, regime segment): probe rate (committed non-null
  non-flatten intents per step — the "experiments per step" the arbiter
  chose), message and fill rates, null-action fractions (whole segment and
  its last third: quiescence depth), a fitted exponential decay of the
  probe rate within the segment (babbling decay), inventory statistics,
  and the fraction of steps spent beyond any homeostat soft bound
  (recomputed metrics-side from engine ground truth, so the NO_HOMEOSTAT
  condition is measured by the same instrument it ablated).
* switch rows — per (run, regime switch): probe-rate ratio in the W steps
  after vs before the switch (reawakening), and the soft-bound excursion
  fraction in the post-switch window (the F4 window).
* run rows — one per run: totals for the paired-seed falsification checks
  (total messages for F1, mean |inventory| for F3, excursion time for F4,
  minimum message-budget headroom).

Soft-bound excursions replicate the homeostat's definitions exactly
(normalized excursion per variable; drawdown measured against the running
peak of cash + inventory * mid) but from harness-side account truth —
INV-11's channel — never from the agent's own bookkeeping.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from topos.metrics.collect import RunData
from topos.metrics.segments import regime_segments
from topos.metrics.stats import fit_decay, median
from topos.metrics.tables import MetricResult, TidyTable

DECAY_MIN_STEPS = 90
"""Minimum post-trim segment length for a decay fit (three windows with
enough exposure for the Poisson ratio estimator)."""
REAWAKENING_WINDOW_DEFAULT = 100
"""Steps on each side of a switch for the reawakening ratios. Must cover
the response latency: BOCPD needs a few slow ticks to detect the shift
and forgetting is applied per slow tick, so the epistemic reopening lands
~2-5 ticks (40-100 steps) after the true switch; a shorter window reads
a real reawakening as absent."""
LATE_FRACTION = 1.0 / 3.0
"""The trailing fraction of a segment used for quiescence depth."""
RATIO_EPS_EIG = 1e-3
"""Continuity floor for the epistemic reawakening ratio."""


def _excursion(value: float, soft: float, hard: float) -> float:
    return max(0.0, (abs(value) - soft) / (hard - soft))


class _StepSeries:
    """Per-step behavioral series extracted once per run."""

    def __init__(self, run: RunData) -> None:
        steps = run.run_log.steps
        agent_id = run.run_log.agent_actor_id
        cfg = run.agent_config.homeostat
        n = len(steps)

        self.step_ids = [s.step for s in steps]
        self.probe = [0.0] * n
        self.null = [0.0] * n
        self.message = [0.0] * n
        self.fill = [0.0] * n
        self.inventory = [0] * n
        self.excursion_any = [0.0] * n
        self.headroom_hard = [math.nan] * n
        self.marginal_eig_total = [0.0] * n
        """Sum of headline marginal EIGs — the cycle's total curiosity on
        offer, whether or not the workspace acted on it (the epistemic
        counterpart of the behavioral probe series)."""

        mid: float | None = None
        peak_equity: float | None = None
        window = cfg.message_window_steps
        rolling: list[int] = []
        prev_fill_count = 0
        for i, step in enumerate(steps):
            record = run.records[i + 1] if i + 1 < len(run.records) else None
            if record is not None and record.intent is not None:
                intent = record.intent
                self.null[i] = 1.0 if intent.is_null else 0.0
                if not intent.is_null and not intent.is_flatten:
                    self.probe[i] = 1.0
            if record is not None:
                self.marginal_eig_total[i] = sum(
                    h.best_marginal_eig_nats for h in record.headlines
                )
            self.message[i] = float(len(step.agent_messages))
            account = step.account(agent_id)
            fill_count = len(account.fills)
            self.fill[i] = float(fill_count - prev_fill_count)
            prev_fill_count = fill_count
            inventory = account.inventory_lots
            self.inventory[i] = inventory

            book_mid = step.book.mid
            if book_mid is not None:
                mid = book_mid
            rolling.append(int(self.message[i]))
            if len(rolling) > window:
                rolling.pop(0)
            rolling_count = sum(rolling)
            self.headroom_hard[i] = cfg.message_budget.hard - rolling_count

            if mid is not None:
                equity = account.cash + inventory * mid
                peak_equity = (
                    equity if peak_equity is None else max(peak_equity, equity)
                )
                drawdown = max(0.0, peak_equity - equity)
                excursions = (
                    _excursion(
                        float(inventory), cfg.inventory.soft, cfg.inventory.hard
                    ),
                    _excursion(
                        inventory * mid,
                        cfg.gross_exposure.soft,
                        cfg.gross_exposure.hard,
                    ),
                    _excursion(
                        float(rolling_count),
                        cfg.message_budget.soft,
                        cfg.message_budget.hard,
                    ),
                    _excursion(drawdown, cfg.drawdown.soft, cfg.drawdown.hard),
                )
                self.excursion_any[i] = 1.0 if max(excursions) > 0.0 else 0.0


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def behavior_segment_rows(run: RunData) -> list[dict[str, Any]]:
    series = _StepSeries(run)
    # The decay fit runs on the RAW probe series: behavioral decay in
    # homeostat-bearing conditions conflates curiosity satiation with
    # message-budget and drive throttling, and no timing window can
    # separate them (the budget binds within the same first window the
    # babbling burst occupies). The suite therefore measures raw decay
    # uniformly and adjudicates cause elsewhere: NO_HOMEOSTAT's decay is
    # throttle-free satiation, and the epistemic reawakening series
    # (marginal EIG on offer) separates "curiosity closed" from
    # "curiosity open but outbid" — see DESIGN.md, Open questions (P13).
    rows: list[dict[str, Any]] = []
    for seg in regime_segments(run.run_log):
        lo, hi = seg.start_step, seg.end_step
        probe = series.probe[lo:hi]
        late_start = hi - max(1, int(round(seg.length * LATE_FRACTION)))
        decay_k = decay_se = decay_r2 = decay_a0 = math.nan
        if seg.length >= DECAY_MIN_STEPS:
            fit = fit_decay(probe)
            decay_k, decay_se = fit.k, fit.se
            decay_r2, decay_a0 = fit.r2, fit.a0
        abs_inventory = [abs(v) for v in series.inventory[lo:hi]]
        rows.append(
            {
                "condition": run.condition,
                "seed": run.root_seed,
                "segment": seg.index,
                "regime_id": seg.regime_id,
                "start_step": lo,
                "end_step": hi,
                "length": seg.length,
                "probe_rate": _mean(probe),
                "message_rate": _mean(series.message[lo:hi]),
                "fill_rate": _mean(series.fill[lo:hi]),
                "null_frac": _mean(series.null[lo:hi]),
                "null_frac_late": _mean(series.null[late_start:hi]),
                "decay_k": decay_k,
                "decay_se": decay_se,
                "decay_r2": decay_r2,
                "decay_a0": decay_a0,
                "mean_abs_inventory": _mean(abs_inventory),
                "max_abs_inventory": max(abs_inventory) if abs_inventory else 0,
                "excursion_frac": _mean(series.excursion_any[lo:hi]),
            }
        )
    return rows


def behavior_switch_rows(
    run: RunData, *, window: int = REAWAKENING_WINDOW_DEFAULT
) -> list[dict[str, Any]]:
    series = _StepSeries(run)
    segments = regime_segments(run.run_log)
    n = len(series.probe)
    rows: list[dict[str, Any]] = []
    correction = 1.0 / window
    for seg in segments[1:]:
        s = seg.start_step
        if s < window or s + window > n:
            continue  # a truncated window is not comparable
        before = _mean(series.probe[s - window : s])
        after = _mean(series.probe[s : s + window])
        if after == 0.0 and before == 0.0:
            ratio = math.nan
        elif before == 0.0:
            ratio = math.inf
        else:
            ratio = after / before
        # Continuity-corrected ratio: a 0/0 window — the agent stayed
        # fully quiescent THROUGH a regime switch — reads as 1.0 (no
        # reawakening), which is evidence, not missing data. Dropping
        # those windows would bias any reawakening claim upward.
        ratio_corrected = (after + correction) / (before + correction)
        eig_before = _mean(series.marginal_eig_total[s - window : s])
        eig_after = _mean(series.marginal_eig_total[s : s + window])
        rows.append(
            {
                "condition": run.condition,
                "seed": run.root_seed,
                "switch_step": s,
                "into_regime": seg.regime_id,
                "source": seg.source,
                "probe_rate_before": before,
                "probe_rate_after": after,
                "reawakening_ratio": ratio,
                "reawakening_ratio_corrected": ratio_corrected,
                "eig_offer_before": eig_before,
                "eig_offer_after": eig_after,
                "reawakening_eig_ratio": (
                    (eig_after + RATIO_EPS_EIG) / (eig_before + RATIO_EPS_EIG)
                ),
                "excursion_frac_after": _mean(
                    series.excursion_any[s : s + window]
                ),
            }
        )
    return rows


def behavior_run_rows(run: RunData) -> list[dict[str, Any]]:
    series = _StepSeries(run)
    abs_inventory = [abs(v) for v in series.inventory]
    return [
        {
            "condition": run.condition,
            "seed": run.root_seed,
            "n_steps": len(series.probe),
            "total_messages": len(run.message_log),
            "total_probes": int(sum(series.probe)),
            "total_fills": int(sum(series.fill)),
            "mean_abs_inventory": _mean(abs_inventory),
            "max_abs_inventory": max(abs_inventory) if abs_inventory else 0,
            "excursion_frac": _mean(series.excursion_any),
            "min_headroom_hard": min(series.headroom_hard),
        }
    ]


def summarize_behavior(
    segment_rows: Sequence[Mapping[str, Any]],
    switch_rows: Sequence[Mapping[str, Any]],
    run_rows: Sequence[Mapping[str, Any]],
) -> MetricResult:
    table = TidyTable.from_records("behavior_segments", list(segment_rows))
    summary: dict[str, Any] = {}
    conditions = sorted({r["condition"] for r in run_rows})
    for condition in conditions:
        seg = [r for r in segment_rows if r["condition"] == condition]
        sw = [r for r in switch_rows if r["condition"] == condition]
        runs = [r for r in run_rows if r["condition"] == condition]

        def med(rows: Sequence[Mapping[str, Any]], key: str) -> float:
            values = [
                float(r[key])
                for r in rows
                if isinstance(r[key], (int, float))
                and math.isfinite(float(r[key]))
            ]
            return median(values)

        summary[condition] = {
            "n_runs": len(runs),
            "median_total_messages": med(runs, "total_messages"),
            "median_total_probes": med(runs, "total_probes"),
            "median_mean_abs_inventory": med(runs, "mean_abs_inventory"),
            "median_excursion_frac": med(runs, "excursion_frac"),
            "median_min_headroom_hard": med(runs, "min_headroom_hard"),
            "median_probe_rate": med(seg, "probe_rate"),
            "median_null_frac_late": med(seg, "null_frac_late"),
            "median_decay_k": med(seg, "decay_k"),
            "median_reawakening_ratio": med(sw, "reawakening_ratio"),
            "median_reawakening_ratio_corrected": med(
                sw, "reawakening_ratio_corrected"
            ),
            "median_reawakening_eig_ratio": med(sw, "reawakening_eig_ratio"),
            "median_excursion_frac_after_switch": med(
                sw, "excursion_frac_after"
            ),
            "run_rows": [dict(r) for r in run_rows if r["condition"] == condition],
            "switch_rows": [dict(r) for r in sw],
        }
    return MetricResult(name="behavior_signatures", table=table, summary=summary)


def behavior_signatures(
    runs: Sequence[RunData],
    *,
    window: int = REAWAKENING_WINDOW_DEFAULT,
) -> MetricResult:
    """Pure function RunData(s) -> tidy table + summary (metric 5)."""
    segment_rows: list[dict[str, Any]] = []
    switch_rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    for run in runs:
        segment_rows.extend(behavior_segment_rows(run))
        switch_rows.extend(behavior_switch_rows(run, window=window))
        run_rows.extend(behavior_run_rows(run))
    return summarize_behavior(segment_rows, switch_rows, run_rows)
