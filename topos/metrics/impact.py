"""Metric 4 — impact-model validation against counterfactual ground truth.

The harness's twin replay (P3 ``impact()`` machinery) gives the only true
measurement of the agent's price impact: the run-vs-twin mid divergence,
which by INV-9 can differ from zero only through the agent's causal
footprint. For every agent action we compare

    realized  = mid_divergence(step + h) - mid_divergence(step - 1)
    predicted = the agent's end-of-run impact posterior's own-effect
                (mean, variance) for the channels the action exercised

where h is the impact model's fixed horizon. Actions are classified with
the same book-context helpers the model itself uses (``offset_band_of``
against the pre-action observation): a marketable placement exercises the
aggression channel, a touch placement the resting channel, deeper
placements neither — those form a PLACEBO group whose predicted own
effect is exactly 0, reported separately as a specification check.

The posterior is the end-of-run snapshot (the model's best final answer),
so this validates what was LEARNED, not the learning trajectory; the
trajectory's health is metric 2's business. Predictive z uses coefficient
variance plus the posterior-mean noise scale — the divergence measurement
inherits chaotic amplification noise that the coefficient variance alone
does not cover.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from topos.contracts.market import PlaceLimit
from topos.env.harness import divergence_series, impact
from topos.selfmodel.common import context_from_observation, offset_band_of

from topos.metrics.collect import RunData
from topos.metrics.stats import ols
from topos.metrics.tables import MetricResult, TidyTable
from topos.metrics.calibration import Z_95


def impact_rows(run: RunData) -> list[dict[str, Any]]:
    """One row per agent action with a measurable divergence window."""
    if run.twin_log is None or run.impact_posterior is None:
        return []
    posterior = run.impact_posterior
    horizon = posterior.horizon_steps
    series = divergence_series(run.run_log, run.twin_log)
    mid_delta_by_step = {d.step: d.mid_delta for d in series}
    # ``impact()`` validates the twin and enumerates the action windows;
    # the divergence values are read from the full series so the pre-action
    # baseline (step - 1) is available.
    action_records = impact(run.run_log, run.twin_log, horizon)

    obs_by_step = {s.step: s.observation for s in run.run_log.steps}
    rows: list[dict[str, Any]] = []
    for record in action_records:
        message = record.messages[0]
        if not isinstance(message, PlaceLimit):
            continue  # cancels have no modeled own-effect channel
        obs = obs_by_step.get(record.step)
        if obs is None:
            continue
        ctx = context_from_observation(obs)
        band = offset_band_of(
            message.side, message.price_ticks, ctx.best_bid, ctx.best_ask
        )
        signed = message.side.value * message.size_lots
        aggression = float(signed) if band == "cross" else 0.0
        resting = float(signed) if band == "touch" else 0.0
        channel = (
            "aggression"
            if aggression
            else ("resting" if resting else "placebo")
        )

        target = mid_delta_by_step.get(record.step + horizon)
        baseline = (
            mid_delta_by_step.get(record.step - 1, None)
            if record.step > 0
            else 0.0
        )
        if target is None or baseline is None:
            continue
        realized = target - baseline
        pred_mean, pred_var = posterior.own_effect(aggression, resting)
        z_var = pred_var + posterior.noise_scale_mean
        rows.append(
            {
                "condition": run.condition,
                "seed": run.root_seed,
                "step": record.step,
                "channel": channel,
                "aggression_lots": aggression,
                "resting_touch_lots": resting,
                "predicted_mean": pred_mean,
                "predicted_sd": math.sqrt(pred_var),
                "realized": realized,
                "z": (realized - pred_mean) / math.sqrt(z_var)
                if z_var > 0
                else math.nan,
            }
        )
    return rows


def summarize_impact(rows: Sequence[Mapping[str, Any]]) -> MetricResult:
    table = TidyTable.from_records("impact_validation", list(rows))
    summary: dict[str, Any] = {}
    for condition in sorted({r["condition"] for r in rows}):
        of_condition = [r for r in rows if r["condition"] == condition]
        modeled = [r for r in of_condition if r["channel"] != "placebo"]
        placebo = [r for r in of_condition if r["channel"] == "placebo"]
        fit = ols(
            [float(r["predicted_mean"]) for r in modeled],
            [float(r["realized"]) for r in modeled],
        )
        zs = [
            float(r["z"])
            for r in modeled
            if isinstance(r["z"], float) and math.isfinite(r["z"])
        ]
        summary[condition] = {
            "n_actions": len(of_condition),
            "n_modeled": len(modeled),
            "n_placebo": len(placebo),
            "slope_realized_on_predicted": fit.slope,
            "slope_se": fit.se_slope,
            "slope_ci95": list(fit.slope_ci95),
            "r2": fit.r2,
            "z_coverage95": (
                sum(1 for z in zs if abs(z) <= Z_95) / len(zs) if zs else math.nan
            ),
            "placebo_median_abs_realized": _median_abs(
                [float(r["realized"]) for r in placebo]
            ),
            "modeled_median_abs_realized": _median_abs(
                [float(r["realized"]) for r in modeled]
            ),
        }
    return MetricResult(name="impact_validation", table=table, summary=summary)


def _median_abs(values: Sequence[float]) -> float:
    if not values:
        return math.nan
    ordered = sorted(abs(v) for v in values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def impact_validation(runs: Sequence[RunData]) -> MetricResult:
    """Pure function RunData(s) -> tidy table + summary (metric 4)."""
    rows: list[dict[str, Any]] = []
    for run in runs:
        rows.extend(impact_rows(run))
    return summarize_impact(rows)
