"""Ablation report renderer: one markdown document per ablation run.

Section order is deliberate and load-bearing:

1. the falsification suite (what the design predicted, and whether the
   data killed it),
2. FULL-vs-ablation paired-seed deltas (the ablation's headline),
3. the science metrics (calibration, EIG, efficiency, impact, behavior),
4. the OUTCOME — PnL — last, framed as a measured outcome and never as an
   objective. That framing is part of the instrument: the research
   question is whether competent behavior emerges from curiosity alone,
   so profit is evidence, not target.
"""

from __future__ import annotations

import json
import math
from typing import Any, Mapping, Sequence

from topos.metrics.ablation import AblationRows
from topos.metrics.collect import FULL
from topos.metrics.falsification import CheckResult
from topos.metrics.stats import median, paired_deltas
from topos.metrics.tables import MetricResult, TidyTable


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return f"{value:.4g}"
    return str(value)


def _md_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(" --- " for _ in headers) + "|",
    ]
    lines.extend("| " + " | ".join(_fmt(v) for v in row) + " |" for row in rows)
    return "\n".join(lines)


def _per_seed(
    rows: Sequence[Mapping[str, Any]], condition: str, key: str
) -> dict[int, float]:
    out: dict[int, float] = {}
    for r in rows:
        if r.get("condition") != condition:
            continue
        value = r.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            out[int(r["seed"])] = float(value)
    return out


def _conditions(rows: AblationRows) -> list[str]:
    seen: list[str] = []
    for r in rows.behavior_runs:
        if r["condition"] not in seen:
            seen.append(r["condition"])
    return seen


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------

def _header(meta: Mapping[str, Any]) -> str:
    seeds = meta.get("seeds", ())
    lines = [
        "# TOPOS-Market ablation report",
        "",
        "**Research question.** Does competent market behavior emerge from "
        "curiosity alone? The agent maximizes nothing: it runs closed-form "
        "Bayesian updates and acts on expected information gain over its "
        "parameter posteriors. PnL appears in this report exactly once, in "
        "the final section, as a **measured outcome** — it is never an "
        "objective, never a score, and no constant anywhere in the agent "
        "was set against it (INV-5; fairness rules in the falsification "
        "section).",
        "",
        "**Design.** Five conditions differing ONLY through the P12 "
        "ablation flags, run on identical configs, identical fixed seed "
        "lists, and identical regime schedules (INV-9 makes background "
        "event streams bit-identical per seed). All comparisons are "
        "seed-paired; medians over means; dispersion reported.",
        "",
        f"* scale: **{meta.get('scale', '?')}** — "
        f"{meta.get('n_steps', '?')} steps x {len(seeds)} seeds x "
        f"{len(meta.get('conditions', ()))} conditions",
        f"* seeds (fixed in config): {list(seeds)}",
        f"* regime schedule: {meta.get('schedule', ())}",
        f"* generated: {meta.get('generated', '?')}"
        + (
            f" — wall time {meta['wall_seconds']:.0f}s"
            if isinstance(meta.get("wall_seconds"), (int, float))
            else ""
        ),
    ]
    return "\n".join(lines)


def _falsification_section(checks: Sequence[CheckResult]) -> str:
    rows = [
        (
            c.check_id,
            "hard" if c.hard else "report",
            c.status,
            c.value,
            c.description,
        )
        for c in checks
    ]
    lines = [
        "## Falsification suite",
        "",
        "The design PREDICTS each of these; a hard FAIL falsifies the "
        "corresponding prediction (or implicates the named wiring "
        "suspect). Per the fairness rules, a failure here is a **result** "
        "to record in DESIGN.md — never a license to adjust an agent "
        "constant.",
        "",
        _md_table(("check", "kind", "status", "value", "prediction"), rows),
    ]
    for c in checks:
        note = c.details.get("inv10_note")
        if note:
            lines += ["", f"> **INV-10 note ({c.check_id}).** {note}"]
    lines += [
        "",
        "> **F5 scope note.** The spec names fair_value and "
        "flow_intensity, but under FULL those hypotheses structurally "
        "never acquire ledger entries: world-probe marginals are exactly "
        "0 (DESIGN items 13/28), so their information rides the untracked "
        "null action. F5 therefore binds on impact — the one ledgerable "
        "hypothesis whose evidence arrives inside the ledger's resolution "
        "window; fill_rate is reported but not bound (its acks arrive "
        "after the ledger has already resolved — DESIGN item 42), and "
        "world-hypothesis slopes are reported for any condition that "
        "ledgers them (SURPRISE_CURIOSITY does).",
    ]
    return "\n".join(lines)


_PAIRED_KEYS: tuple[tuple[str, str, str], ...] = (
    ("behavior_runs", "total_messages", "total messages"),
    ("behavior_runs", "total_probes", "total probes"),
    ("behavior_runs", "mean_abs_inventory", "mean |inventory| (lots)"),
    ("behavior_runs", "excursion_frac", "soft-bound excursion time frac"),
    ("efficiency_final", "cum_realized_ig_nats", "realized IG (nats)"),
    ("efficiency_final", "ig_per_message", "IG per message (nats)"),
    ("outcome_run", "pnl_ticklots", "final PnL (tick-lots; outcome)"),
    ("outcome_run", "max_drawdown_ticklots", "max drawdown (tick-lots)"),
)


def _paired_rows_for(rows: AblationRows, family: str) -> Sequence[Mapping[str, Any]]:
    if family == "behavior_runs":
        return rows.behavior_runs
    if family == "efficiency_final":
        return [r for r in rows.efficiency if r.get("final")]
    if family == "outcome_run":
        return [r for r in rows.outcome if r.get("scope") == "run"]
    raise ValueError(family)


def _paired_deltas_section(rows: AblationRows) -> str:
    conditions = [c for c in _conditions(rows) if c != FULL]
    header = ["metric (per-seed median)", "FULL"] + [
        f"{c} - FULL" for c in conditions
    ]
    body: list[list[Any]] = []
    for family, key, label in _PAIRED_KEYS:
        family_rows = _paired_rows_for(rows, family)
        full = _per_seed(family_rows, FULL, key)
        row: list[Any] = [label, median(list(full.values()))]
        for condition in conditions:
            other = _per_seed(family_rows, condition, key)
            deltas = paired_deltas(other, full)
            row.append(median(deltas) if deltas else None)
        body.append(row)
    return "\n".join(
        [
            "## FULL vs ablations — paired-seed deltas",
            "",
            "Median over seeds of the per-seed difference (ablation - "
            "FULL); identical seeds and background streams per pair. These "
            "deltas are the ablation's headline: what each removed "
            "mechanism was doing.",
            "",
            _md_table(header, body),
        ]
    )


def _summary_table(
    result: MetricResult, columns: Sequence[tuple[str, str]]
) -> str:
    """Render a {condition: {key: value}} summary as one markdown table."""
    conditions = [k for k in result.summary if isinstance(result.summary[k], dict)]
    body = [
        [condition] + [result.summary[condition].get(key) for key, _ in columns]
        for condition in conditions
    ]
    return _md_table(
        ["condition"] + [label for _, label in columns],
        body,
    )


def _calibration_section(result: MetricResult) -> str:
    lines = [
        "## Metric 1 — belief calibration vs harness ground truth",
        "",
        "Per-channel z-scores/coverage of ground-truth (or realized) "
        "quantities under the agent's published posteriors, per regime "
        "segment (full per-segment table in the CSV appendix). Known "
        "instrument biases are documented in the metric module and NOT "
        "corrected: a calibration gap is a finding.",
        "",
    ]
    conditions = [k for k in result.summary if isinstance(result.summary[k], dict)]
    for condition in conditions:
        per_channel = result.summary[condition]
        body = []
        for channel, stats in per_channel.items():
            if channel == "regime":
                body.append(
                    [
                        channel,
                        None,
                        None,
                        None,
                        stats.get("median_p_recent_postswitch"),
                        stats.get("median_p_recent_stable"),
                    ]
                )
            else:
                body.append(
                    [
                        channel,
                        stats.get("n_obs"),
                        stats.get("median_segment_coverage95"),
                        stats.get("median_segment_rms_z"),
                        None,
                        None,
                    ]
                )
        lines += [
            f"### {condition}",
            "",
            _md_table(
                (
                    "channel",
                    "n",
                    "coverage@95 (median seg)",
                    "rms z (median seg)",
                    "P(recent) post-switch",
                    "P(recent) stable",
                ),
                body,
            ),
            "",
        ]
    return "\n".join(lines)


def _eig_section(result: MetricResult) -> str:
    lines = [
        "## Metric 2 — EIG calibration (promised vs realized)",
        "",
        "One point per experiment-ledger entry; slope ~ 1 means the "
        "curiosity signal is calibrated (the agent gets the information "
        "it pays for). Slope ~ 0 with realized ~ 0 while promises are "
        "positive is the INV-10 snapshot-bug signature and is flagged as "
        "wiring suspicion, not design failure. Under SURPRISE_CURIOSITY "
        "the 'promised' column is a retrospective z-score, not nats — "
        "its miscalibration against realized nats is that ablation's "
        "point.",
        "",
    ]
    body: list[list[Any]] = []
    for condition, per_hypothesis in result.summary.items():
        if not isinstance(per_hypothesis, dict):
            continue
        for hypothesis, stats in per_hypothesis.items():
            if not isinstance(stats, dict):
                continue
            body.append(
                [
                    condition,
                    hypothesis,
                    stats.get("n"),
                    stats.get("slope"),
                    stats.get("slope_se"),
                    stats.get("r2"),
                    stats.get("mean_promised"),
                    stats.get("mean_realized"),
                    stats.get("suspect_inv10_snapshot_bug"),
                ]
            )
    lines.append(
        _md_table(
            (
                "condition",
                "hypothesis",
                "n",
                "slope",
                "se",
                "r2",
                "mean promised",
                "mean realized",
                "INV-10 suspect",
            ),
            body,
        )
    )
    return "\n".join(lines)


def _efficiency_section(result: MetricResult) -> str:
    return "\n".join(
        [
            "## Metric 3 — scientific efficiency",
            "",
            "Information gained per unit of action: realized IG from the "
            "experiment ledger (bought information) and the total "
            "parameter-entropy drop (passive information included), per "
            "message and per fill. Medians over seeds.",
            "",
            _summary_table(
                result,
                (
                    ("median_total_messages", "messages"),
                    ("median_total_fills", "fills"),
                    ("median_total_realized_ig_nats", "realized IG (nats)"),
                    ("median_entropy_drop_nats", "entropy drop (nats)"),
                    ("median_ig_per_message", "IG/message"),
                    ("median_ig_per_fill", "IG/fill"),
                ),
            ),
        ]
    )


def _impact_section(result: MetricResult) -> str:
    return "\n".join(
        [
            "## Metric 4 — impact-model validation vs counterfactual truth",
            "",
            "Ground truth is the run-vs-twin mid divergence (P3 "
            "counterfactual replay): realized per-action divergence "
            "change over the impact horizon vs the end-of-run impact "
            "posterior's predicted own-effect. 'placebo' actions (deep "
            "placements, no modeled channel) should show ~ 0 realized "
            "divergence.",
            "",
            _summary_table(
                result,
                (
                    ("n_modeled", "modeled actions"),
                    ("n_placebo", "placebo actions"),
                    ("slope_realized_on_predicted", "slope"),
                    ("slope_se", "se"),
                    ("z_coverage95", "z coverage@95"),
                    ("modeled_median_abs_realized", "|realized| modeled"),
                    ("placebo_median_abs_realized", "|realized| placebo"),
                ),
            ),
        ]
    )


def _behavior_section(result: MetricResult) -> str:
    return "\n".join(
        [
            "## Metric 5 — behavior signatures",
            "",
            "Probe rate = committed non-null, non-flatten intents per "
            "step (the agent's experiments). Babbling decay is the "
            "fitted exponential decay of the probe rate within regime "
            "segments; quiescence is the null-action fraction late in a "
            "segment; reawakening is the probe-rate ratio around true "
            "regime switches. Soft-bound excursions are recomputed from "
            "engine ground truth with the homeostat's own band "
            "definitions.",
            "",
            _summary_table(
                result,
                (
                    ("median_total_messages", "messages"),
                    ("median_probe_rate", "probe rate"),
                    ("median_decay_k", "decay k /step"),
                    ("median_null_frac_late", "late null frac"),
                    ("median_reawakening_ratio_corrected", "reawakening"),
                    ("median_reawakening_eig_ratio", "EIG-offer reawakening"),
                    ("median_excursion_frac", "excursion frac"),
                    ("median_min_headroom_hard", "min msg headroom"),
                ),
            ),
        ]
    )


def _outcome_section(result: MetricResult) -> str:
    return "\n".join(
        [
            "## Metric 6 — outcome: PnL (measured outcome, never objective)",
            "",
            "This section is deliberately last. " + str(
                result.summary.get("framing", "")
            ),
            "",
            _summary_table(
                result,
                (
                    ("median_final_pnl_ticklots", "median final PnL"),
                    ("iqr_final_pnl_ticklots", "IQR"),
                    ("median_max_drawdown_ticklots", "median max drawdown"),
                    ("frac_seeds_pnl_positive", "frac seeds > 0"),
                ),
            ),
            "",
            "If PnL is positive under FULL, the claim this supports is "
            "'boring, self-predictable trading fell out of curiosity "
            "satiation' — not 'the agent is a good trader'. If it is "
            "negative, the design made no promise it broke.",
        ]
    )


def render_report(
    rows: AblationRows,
    results: Mapping[str, MetricResult],
    checks: Sequence[CheckResult],
    meta: Mapping[str, Any],
) -> str:
    """Assemble the single markdown ablation report."""
    sections = [
        _header(meta),
        _falsification_section(checks),
        _paired_deltas_section(rows),
        _calibration_section(results["belief_calibration"]),
        _eig_section(results["eig_calibration"]),
        _efficiency_section(results["scientific_efficiency"]),
        _impact_section(results["impact_validation"]),
        _behavior_section(results["behavior_signatures"]),
        _outcome_section(results["outcome_pnl"]),
        "\n".join(
            [
                "## Appendix — artifacts",
                "",
                "Full tidy tables are written as CSVs next to this report; "
                "`summary.json` holds every metric summary and check "
                "verbatim.",
            ]
        ),
    ]
    return "\n\n".join(sections) + "\n"


def checks_to_json(checks: Sequence[CheckResult]) -> list[dict[str, Any]]:
    return [
        {
            "check_id": c.check_id,
            "hard": c.hard,
            "status": c.status,
            "passed": c.passed,
            "value": c.value,
            "description": c.description,
            "details": c.details,
        }
        for c in checks
    ]


def summaries_to_json(
    results: Mapping[str, MetricResult]
) -> dict[str, Any]:
    return {name: result.summary for name, result in results.items()}


def tables_of(results: Mapping[str, MetricResult]) -> dict[str, TidyTable]:
    return {name: result.table for name, result in results.items()}


def to_json_compat(obj: Any) -> Any:
    """Round-trip helper: make summaries strictly JSON-serializable."""
    return json.loads(json.dumps(obj, default=str))
