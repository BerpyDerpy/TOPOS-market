"""Metric 1 — belief calibration against harness ground truth.

Per BeliefModule, z-scores / coverage of ground-truth (or realized)
quantities under the agent's published posteriors, over time, reported
per regime segment. Four channels:

* ``fair_value``   — z of the REALIZED next-step microprice under the
  module's one-step Student-t predictive (headline mean/variance). A
  calibrated filter puts ~95% of realizations inside +/-1.96 sd.
* ``flow_intensity`` — z of the segment's TRUE background-flow rate under
  the module's PARAMETER posterior over the total rate. The truth is
  empirical: lots/step of background arrivals + cancels + prints in the
  NULL-AGENT TWIN (identical background draws, zero agent footprint) over
  the same segment — the committed P4 reading that the flow model sees
  background-caused flow only. The parameter variance is recovered
  exactly from the headline's negative-binomial moments:
  Var[rate] = forecast_var - forecast_mean (the predictive minus its
  Poisson floor), which holds cell-by-cell for independent Gamma cells.
  Known bias sources, deliberately not corrected: the agent extracts from
  the 10-level visible window while the truth counts full depth, and
  level entry/exit at the window edge reads as arrival/cancel (DESIGN
  item 14) — a calibration gap here is a finding about the instrument,
  not noise to be tuned away.
* ``queue_position`` — z / coverage / MAE of the TRUE lots-ahead of each
  resting agent order (harness ``QueueTruth``, INV-11) under the working-
  order rank posterior broadcast in the cycle record.
* ``regime`` — detection calibration: the tracker's published
  P(changepoint within R_RECENT) in the first ``detect_window`` steps of
  each post-switch segment vs in the remainder ("stable"). A working
  tracker shows post-switch >> stable.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from topos.beliefs.fair_value import microprice_from_observation
from topos.contracts.intent import FAIR_VALUE, FLOW_INTENSITY
from topos.env.harness import BookSnapshot, RunLog

from topos.metrics.collect import RunData
from topos.metrics.segments import RegimeSegment, regime_segments
from topos.metrics.stats import median
from topos.metrics.tables import MetricResult, TidyTable

Z_95 = 1.959963984540054

DETECT_WINDOW_DEFAULT = 100
"""Steps after a true switch counted as the detection window: five slow
ticks at the default cadence. BOCPD needs a few post-switch summary
vectors before the run-length posterior collapses, so a window shorter
than ~3 ticks files the detection itself under 'stable' and inverts the
contrast this channel measures."""


def _headline(record: Any, hypothesis_id: str) -> Any | None:
    for headline in record.headlines:
        if headline.hypothesis_id == hypothesis_id:
            return headline
    return None


def _is_premarket(record: Any) -> bool:
    """Pre-market cycle records carry the sentinel mid of 0.0 (DESIGN item
    37); real tick prices are positive."""
    return bool(record.world_summary.mid_ticks == 0.0)


def _level_map(levels: tuple[tuple[int, int], ...]) -> dict[int, int]:
    return {price: size for price, size in levels}


def _background_flow_lots(twin_log: RunLog) -> list[float]:
    """Per-step background flow in lots (arrivals + cancels + prints) from
    the twin's full-depth book snapshots and public trade prints."""
    series: list[float] = []
    prev_bids: dict[int, int] = {}
    prev_asks: dict[int, int] = {}
    for step in twin_log.steps:
        book: BookSnapshot = step.book
        cur_bids = _level_map(book.bids)
        cur_asks = _level_map(book.asks)
        traded: dict[int, int] = {}
        market_lots = 0
        for trade in step.observation.trades:
            traded[trade.price_ticks] = (
                traded.get(trade.price_ticks, 0) + trade.size_lots
            )
            market_lots += trade.size_lots
        arrivals = 0
        cancels = 0
        for prev, cur in ((prev_bids, cur_bids), (prev_asks, cur_asks)):
            for price in set(prev) | set(cur):
                delta = cur.get(price, 0) - prev.get(price, 0)
                if delta > 0:
                    arrivals += delta
                elif delta < 0:
                    unexplained = -delta - traded.get(price, 0)
                    if unexplained > 0:
                        cancels += unexplained
        series.append(float(arrivals + cancels + market_lots))
        prev_bids, prev_asks = cur_bids, cur_asks
    return series


def _z_stats(zs: Sequence[float]) -> dict[str, float]:
    finite = [z for z in zs if math.isfinite(z)]
    if not finite:
        return {"n": 0, "mean_z": math.nan, "rms_z": math.nan, "coverage95": math.nan}
    n = len(finite)
    return {
        "n": n,
        "mean_z": sum(finite) / n,
        "rms_z": math.sqrt(sum(z * z for z in finite) / n),
        "coverage95": sum(1 for z in finite if abs(z) <= Z_95) / n,
    }


def calibration_rows(
    run: RunData, *, detect_window: int = DETECT_WINDOW_DEFAULT
) -> list[dict[str, Any]]:
    """Per (channel, regime segment) calibration aggregates for one run."""
    segments = regime_segments(run.run_log)
    steps = run.run_log.steps
    base = {"condition": run.condition, "seed": run.root_seed}

    fair_z: dict[int, list[float]] = {s.index: [] for s in segments}
    queue_z: dict[int, list[float]] = {s.index: [] for s in segments}
    queue_err: dict[int, list[float]] = {s.index: [] for s in segments}
    flow_moments: dict[int, list[tuple[float, float]]] = {
        s.index: [] for s in segments
    }
    p_recent: dict[int, list[tuple[int, float]]] = {s.index: [] for s in segments}

    seg_iter = iter(segments)
    segment: RegimeSegment | None = next(seg_iter, None)
    for i, (step, record) in enumerate(run.paired_steps()):
        while segment is not None and step.step >= segment.end_step:
            segment = next(seg_iter, None)
        if segment is None or not segment.contains(step.step):
            continue
        if _is_premarket(record):
            continue
        idx = segment.index

        # fair_value: realized next-step microprice vs one-step predictive.
        fair = _headline(record, FAIR_VALUE)
        if fair is not None and i + 1 < len(steps):
            realized = microprice_from_observation(steps[i + 1].observation)
            if (
                realized is not None
                and math.isfinite(fair.forecast_var)
                and fair.forecast_var > 0.0
            ):
                fair_z[idx].append(
                    (realized - fair.forecast_mean) / math.sqrt(fair.forecast_var)
                )

        # flow_intensity: store parameter-posterior moments; the segment
        # truth is only known once the twin series is aggregated below.
        flow = _headline(record, FLOW_INTENSITY)
        if flow is not None and math.isfinite(flow.forecast_var):
            param_var = flow.forecast_var - flow.forecast_mean
            if param_var > 0.0:
                flow_moments[idx].append((flow.forecast_mean, param_var))

        # queue_position: true lots-ahead vs the rank posterior, matched
        # by order id (both views describe end-of-step book state).
        truth_by_id = {qt.order_id: qt for qt in step.agent_queue_truth}
        for view in record.self_state.working_orders:
            truth = truth_by_id.get(view.order_id)
            if truth is None:
                continue
            error = truth.lots_ahead - view.queue_rank_mean
            queue_err[idx].append(abs(error))
            if view.queue_rank_var > 1e-12:
                queue_z[idx].append(error / math.sqrt(view.queue_rank_var))
            else:
                # Point-mass posterior: correct iff the error is zero.
                queue_z[idx].append(0.0 if error == 0 else math.inf)

        # regime: published P(recent changepoint).
        posterior = record.world_summary.regime_posterior
        if posterior:
            p_recent[idx].append((step.step, float(posterior[0])))

    flow_truth_by_segment: dict[int, float] = {}
    if run.twin_log is not None:
        flow_series = _background_flow_lots(run.twin_log)
        for seg in segments:
            window = flow_series[seg.start_step : seg.end_step]
            # Drop the first step of the run itself: the step-0 "diff" is
            # book seeding from an empty book, not stationary flow.
            if seg.start_step == 0:
                window = window[1:]
            if window:
                flow_truth_by_segment[seg.index] = sum(window) / len(window)

    rows: list[dict[str, Any]] = []
    for seg in segments:
        seg_base = {
            **base,
            "segment": seg.index,
            "regime_id": seg.regime_id,
            "start_step": seg.start_step,
            "end_step": seg.end_step,
        }
        rows.append(
            {
                **seg_base,
                "channel": "fair_value",
                **_z_stats(fair_z[seg.index]),
                "mae": math.nan,
                "truth": math.nan,
            }
        )
        flow_truth = flow_truth_by_segment.get(seg.index)
        if flow_truth is not None:
            zs = [
                (flow_truth - mean) / math.sqrt(var)
                for mean, var in flow_moments[seg.index]
            ]
            rows.append(
                {
                    **seg_base,
                    "channel": "flow_intensity",
                    **_z_stats(zs),
                    "mae": (
                        median(
                            [
                                abs(flow_truth - m)
                                for m, _ in flow_moments[seg.index]
                            ]
                        )
                        if flow_moments[seg.index]
                        else math.nan
                    ),
                    "truth": flow_truth,
                }
            )
        errs = queue_err[seg.index]
        rows.append(
            {
                **seg_base,
                "channel": "queue_position",
                **_z_stats(queue_z[seg.index]),
                "mae": (sum(errs) / len(errs)) if errs else math.nan,
                "truth": math.nan,
            }
        )
        pr = p_recent[seg.index]
        if pr:
            cutoff = seg.start_step + detect_window
            post = [p for s, p in pr if s < cutoff]
            rest = [p for s, p in pr if s >= cutoff]
            rows.append(
                {
                    **seg_base,
                    "channel": "regime",
                    "n": len(pr),
                    "mean_z": math.nan,
                    "rms_z": math.nan,
                    "coverage95": math.nan,
                    "mae": math.nan,
                    "truth": math.nan,
                    "p_recent_postswitch": (
                        sum(post) / len(post)
                        if post and seg.index > 0
                        else math.nan
                    ),
                    "p_recent_stable": sum(rest) / len(rest) if rest else math.nan,
                }
            )
    return rows


def summarize_calibration(rows: Sequence[Mapping[str, Any]]) -> MetricResult:
    """Aggregate per-(run, segment, channel) rows into the metric result."""
    table = TidyTable.from_records("belief_calibration", list(rows))
    summary: dict[str, Any] = {}
    conditions = sorted({r["condition"] for r in rows})
    for condition in conditions:
        by_channel: dict[str, Any] = {}
        for channel in ("fair_value", "flow_intensity", "queue_position"):
            selected = [
                r
                for r in rows
                if r["condition"] == condition
                and r["channel"] == channel
                and r.get("n", 0)
            ]
            if not selected:
                continue
            total_n = sum(r["n"] for r in selected)
            coverages = [
                r["coverage95"]
                for r in selected
                if isinstance(r["coverage95"], float)
                and math.isfinite(r["coverage95"])
            ]
            rms = [
                r["rms_z"]
                for r in selected
                if isinstance(r["rms_z"], float) and math.isfinite(r["rms_z"])
            ]
            by_channel[channel] = {
                "n_obs": total_n,
                "median_segment_coverage95": median(coverages),
                "median_segment_rms_z": median(rms),
            }
        detect_post = [
            r["p_recent_postswitch"]
            for r in rows
            if r["condition"] == condition
            and r["channel"] == "regime"
            and isinstance(r.get("p_recent_postswitch"), float)
            and math.isfinite(r["p_recent_postswitch"])
        ]
        detect_rest = [
            r["p_recent_stable"]
            for r in rows
            if r["condition"] == condition
            and r["channel"] == "regime"
            and isinstance(r.get("p_recent_stable"), float)
            and math.isfinite(r["p_recent_stable"])
        ]
        by_channel["regime"] = {
            "median_p_recent_postswitch": median(detect_post),
            "median_p_recent_stable": median(detect_rest),
        }
        summary[condition] = by_channel
    return MetricResult(name="belief_calibration", table=table, summary=summary)


def belief_calibration(
    runs: Sequence[RunData], *, detect_window: int = DETECT_WINDOW_DEFAULT
) -> MetricResult:
    """Pure function RunData(s) -> tidy table + summary (metric 1)."""
    rows: list[dict[str, Any]] = []
    for run in runs:
        rows.extend(calibration_rows(run, detect_window=detect_window))
    return summarize_calibration(rows)
