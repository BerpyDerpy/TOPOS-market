"""The falsification suite: what the design PREDICTS, checked as data.

Each check returns a ``CheckResult``. ``hard`` checks are asserted by
tests/acceptance/ — a hard failure falsifies a design prediction (or
implicates the named wiring suspect). Soft checks are generated report
checks: computed, displayed, never asserted.

Fairness: every comparison is seed-paired (identical configs, root seeds
and regime schedules across conditions — INV-9 makes the pairing exact),
uses medians over means, and reports dispersion. NO number computed here
may be used to set any agent constant; if a check fails, that failure is
a RESULT to be recorded in DESIGN.md, not a bug to be tuned away.

F5 deviation, recorded rather than papered over: the spec names
fair_value and flow_intensity, but under FULL those hypotheses never
acquire ledger entries — world-probe marginals are exactly 0 (DESIGN
items 13/28), so the arbiter never opens a world experiment and their
information rides the untracked null action. The slope check therefore
binds on the hypotheses that structurally CAN be ledgered under FULL
(fill_rate, impact); fair_value/flow_intensity slopes are still reported
whenever a condition produces entries for them (SURPRISE_CURIOSITY does).
See DESIGN.md, Open questions (P13).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

from topos.contracts.intent import FILL_RATE, IMPACT

from topos.metrics.ablation import AblationRows, Row
from topos.metrics.collect import (
    FULL,
    NO_HOMEOSTAT,
    NO_REFLEXIVE,
    NO_SELF_MODEL,
    SURPRISE_CURIOSITY,
)
from topos.metrics.eig import _slope_summary
from topos.metrics.stats import median, paired_deltas, paired_ratios, t_interval

F1_MEDIAN_RATIO_BOUND = 0.5
F5_SLOPE_BAND = (0.5, 1.5)
F5_HYPOTHESES: tuple[tuple[str, str], ...] = ((IMPACT, "slope"),)
"""(hypothesis, which slope binds). Only impact binds: its evidence (the
next mid move) arrives within the ledger's one-cycle resolution window,
so promised and realized describe the same experiment. fill_rate's
outcome arrives ack-shift + fill-horizon cycles after issuance (item 33)
— after the ledger has already resolved — so its ledger slope is
structurally lagged (the INV-10-wiring suspicion F5 names, found and
recorded in DESIGN.md), and the windowed records-based estimator is too
contaminated by neighboring experiments to bind (its window spans ~2
probes at the observed cadence). Both fill_rate figures are REPORTED in
the details, never asserted."""


@dataclass(frozen=True)
class CheckResult:
    check_id: str
    hard: bool
    passed: bool | None
    """None = not evaluable on this dataset (insufficient data)."""
    value: float
    """The check's headline number (see description for its meaning)."""
    description: str
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def status(self) -> str:
        if self.passed is None:
            return "NOT EVALUABLE"
        return "PASS" if self.passed else "FAIL"


def _per_seed(
    rows: Sequence[Row], condition: str, key: str
) -> dict[int, float]:
    return {
        int(r["seed"]): float(r[key])
        for r in rows
        if r["condition"] == condition
    }


def f1_message_churn(rows: AblationRows) -> CheckResult:
    """F1 (hard): FULL total messages << SURPRISE_CURIOSITY's.

    EIG satiates (conjugate posteriors concentrate, marginals fall into
    the boredom band); retrospective surprise does not (noise keeps
    scoring). Median over per-seed ratios FULL/SURPRISE must be < 0.5.
    """
    full = _per_seed(rows.behavior_runs, FULL, "total_messages")
    surprise = _per_seed(rows.behavior_runs, SURPRISE_CURIOSITY, "total_messages")
    ratios = paired_ratios(full, surprise)
    value = median(ratios)
    passed = value < F1_MEDIAN_RATIO_BOUND if ratios else None
    return CheckResult(
        check_id="F1",
        hard=True,
        passed=passed,
        value=value,
        description=(
            "median over seeds of (FULL msgs / SURPRISE_CURIOSITY msgs) "
            f"< {F1_MEDIAN_RATIO_BOUND}"
        ),
        details={
            "per_seed_ratio": {
                s: full[s] / surprise[s] if surprise[s] else math.inf
                for s in sorted(set(full) & set(surprise))
            },
            "median_full_messages": median(list(full.values())),
            "median_surprise_messages": median(list(surprise.values())),
        },
    )


def _per_seed_decay(rows: Sequence[Row], condition: str) -> dict[int, float]:
    """Per-seed mean fitted decay rate over fit-eligible segments."""
    by_seed: dict[int, list[float]] = {}
    for r in rows:
        if r["condition"] != condition:
            continue
        k = r.get("decay_k")
        if isinstance(k, float) and math.isfinite(k):
            by_seed.setdefault(int(r["seed"]), []).append(k)
    return {s: sum(ks) / len(ks) for s, ks in by_seed.items()}


def f2_decay(rows: AblationRows) -> CheckResult:
    """F2 (hard): FULL probe rate decays within regimes; NO_SELF_MODEL's
    does not (its fills stay informative forever — frozen posteriors never
    satiate). Across-seed t-CI on the per-seed mean decay rate k: FULL's
    CI must exclude 0 from above; NO_SELF_MODEL's must include 0."""
    full = list(_per_seed_decay(rows.behavior_segments, FULL).values())
    nsm = list(_per_seed_decay(rows.behavior_segments, NO_SELF_MODEL).values())
    full_ci = t_interval(full)
    nsm_ci = t_interval(nsm)
    evaluable = full_ci.n >= 3 and nsm_ci.n >= 3
    passed = (full_ci.lo > 0.0 and nsm_ci.includes_zero()) if evaluable else None
    return CheckResult(
        check_id="F2",
        hard=True,
        passed=passed,
        value=full_ci.mean,
        description=(
            "FULL decay-rate 95% CI excludes 0 (decays) AND NO_SELF_MODEL "
            "decay-rate CI includes 0 (does not decay)"
        ),
        details={
            "full_ci": [full_ci.mean, full_ci.lo, full_ci.hi, full_ci.n],
            "no_self_model_ci": [nsm_ci.mean, nsm_ci.lo, nsm_ci.hi, nsm_ci.n],
        },
    )


def f3_inventory(rows: AblationRows) -> CheckResult:
    """F3: NO_REFLEXIVE mean |inventory| > FULL's (report the magnitude;
    hard-assert the sign of the paired-seed median delta)."""
    ablated = _per_seed(rows.behavior_runs, NO_REFLEXIVE, "mean_abs_inventory")
    full = _per_seed(rows.behavior_runs, FULL, "mean_abs_inventory")
    deltas = paired_deltas(ablated, full)
    value = median(deltas)
    return CheckResult(
        check_id="F3",
        hard=True,
        passed=(value > 0.0) if deltas else None,
        value=value,
        description=(
            "median over seeds of (NO_REFLEXIVE - FULL) mean |inventory| > 0"
        ),
        details={
            "median_no_reflexive": median(list(ablated.values())),
            "median_full": median(list(full.values())),
            "n_pairs": len(deltas),
        },
    )


def f4_homeostat(rows: AblationRows) -> CheckResult:
    """F4: hard vetoes remain in the motor, so no hard-bound breach is
    expected even ablated; instead NO_HOMEOSTAT must spend (much) more
    time beyond the soft bounds than FULL, especially in the post-switch
    windows. Hard-assert the sign of the paired median delta of overall
    soft-bound excursion time; report the post-switch comparison."""
    ablated = _per_seed(rows.behavior_runs, NO_HOMEOSTAT, "excursion_frac")
    full = _per_seed(rows.behavior_runs, FULL, "excursion_frac")
    deltas = paired_deltas(ablated, full)
    value = median(deltas)

    def post_switch(condition: str) -> float:
        values = [
            float(r["excursion_frac_after"])
            for r in rows.behavior_switches
            if r["condition"] == condition
            and math.isfinite(float(r["excursion_frac_after"]))
        ]
        return median(values)

    return CheckResult(
        check_id="F4",
        hard=True,
        passed=(value > 0.0) if deltas else None,
        value=value,
        description=(
            "median over seeds of (NO_HOMEOSTAT - FULL) soft-bound "
            "excursion time fraction > 0 (magnitude and post-switch "
            "windows reported)"
        ),
        details={
            "median_no_homeostat": median(list(ablated.values())),
            "median_full": median(list(full.values())),
            "post_switch_no_homeostat": post_switch(NO_HOMEOSTAT),
            "post_switch_full": post_switch(FULL),
            "n_pairs": len(deltas),
        },
    )


def f5_eig_slope(rows: AblationRows) -> CheckResult:
    """F5 (hard): FULL promised-vs-realized slope in [0.5, 1.5], pooled
    across seeds, for every hypothesis that acquires ledger entries under
    FULL (fill_rate, impact — see the module docstring for why the spec's
    fair_value/flow_intensity cannot be ledgered under FULL, and
    ``F5_HYPOTHESES`` for which slope binds per hypothesis). A failure
    with slope ~ 0 and realized ~ 0 implicates INV-10 snapshot wiring
    before the design."""
    lo, hi = F5_SLOPE_BAND
    details: dict[str, Any] = {}
    passed: bool | None = True
    worst = math.nan
    for hypothesis, slope_key in F5_HYPOTHESES:
        entries = [
            r
            for r in rows.eig
            if r["condition"] == FULL and r["hypothesis"] == hypothesis
        ]
        summary: dict[str, Any] = (
            _slope_summary(entries) if entries else {"n": 0}
        )
        summary["binding_slope"] = slope_key
        details[hypothesis] = summary
        if summary["n"] < 5:
            passed = None
            continue
        slope = summary[slope_key]
        if not math.isfinite(slope):
            # Degenerate design matrix (all promises identical): the slope
            # is not estimable — not evaluable rather than falsified.
            details[f"{hypothesis}_note"] = (
                "slope not estimable (no variance in promised EIG)"
            )
            if passed is True:
                passed = None
            continue
        if math.isnan(worst) or abs(slope - 1.0) > abs(worst - 1.0):
            worst = slope
        if not (lo <= slope <= hi):
            passed = False
        if summary.get("suspect_inv10_snapshot_bug"):
            details["inv10_note"] = (
                f"{hypothesis}: slope ~ 0 with realized ~ 0 — suspect the "
                "INV-10 snapshot ordering (both snapshots after the update) "
                "before suspecting the design."
            )
    # Reported, never asserted: FULL's fill_rate (ledger slope structurally
    # lagged — see F5_HYPOTHESES) and the world hypotheses when a condition
    # ledgers them.
    for condition, hypothesis in (
        (FULL, FILL_RATE),
        (SURPRISE_CURIOSITY, "fair_value"),
        (SURPRISE_CURIOSITY, "flow_intensity"),
    ):
        entries = [
            r
            for r in rows.eig
            if r["condition"] == condition and r["hypothesis"] == hypothesis
        ]
        if entries:
            details[f"{condition}:{hypothesis}"] = _slope_summary(entries)
    binding = ", ".join(f"{h} ({k})" for h, k in F5_HYPOTHESES)
    return CheckResult(
        check_id="F5",
        hard=True,
        passed=passed,
        value=worst,
        description=(
            f"FULL EIG-calibration slope in [{lo}, {hi}] for {binding}, "
            "pooled across seeds"
        ),
        details=details,
    )


def f6_babbling(rows: AblationRows) -> CheckResult:
    """F6 (report): FULL probe rate in the first regime segment decays;
    the fitted curve (a0, k) is reported per seed."""
    first_segment = [
        r
        for r in rows.behavior_segments
        if r["condition"] == FULL
        and r["segment"] == 0
        and isinstance(r.get("decay_k"), float)
        and math.isfinite(r["decay_k"])
    ]
    ks = [float(r["decay_k"]) for r in first_segment]
    value = median(ks)
    return CheckResult(
        check_id="F6",
        hard=False,
        passed=(value > 0.0) if ks else None,
        value=value,
        description=(
            "FULL first-segment probe rate decays (median fitted decay "
            "rate k > 0; fitted curves reported)"
        ),
        details={
            "per_seed_fit": [
                {
                    "seed": r["seed"],
                    "decay_k": r["decay_k"],
                    "decay_se": r["decay_se"],
                    "a0": r["decay_a0"],
                    "r2": r["decay_r2"],
                }
                for r in first_segment
            ]
        },
    )


def f7_reawakening(rows: AblationRows) -> CheckResult:
    """F7 (report): FULL probe-rate ratio around true regime switches > 1
    — curiosity reawakens when the world changes.

    Binds on the CONTINUITY-CORRECTED ratio, under which a fully
    quiescent window pair (0 probes before AND after — the agent slept
    through the switch) reads as 1.0: that is evidence of no behavioral
    reawakening, not missing data, and dropping it would bias this check
    toward passing. The EPISTEMIC ratio (total marginal EIG on offer,
    after/before) is reported beside it: it separates 'curiosity never
    reopened' from 'curiosity reopened but never won the workspace'."""
    full_switches = [r for r in rows.behavior_switches if r["condition"] == FULL]
    ratios = [
        float(r["reawakening_ratio_corrected"])
        for r in full_switches
        if math.isfinite(float(r["reawakening_ratio_corrected"]))
    ]
    eig_ratios = [
        float(r["reawakening_eig_ratio"])
        for r in full_switches
        if math.isfinite(float(r["reawakening_eig_ratio"]))
    ]
    value = median(ratios)
    return CheckResult(
        check_id="F7",
        hard=False,
        passed=(value > 1.0) if ratios else None,
        value=value,
        description=(
            "FULL median corrected probe-rate ratio (after / before) "
            "across regime switches > 1"
        ),
        details={
            "n_switches": len(ratios),
            "median_eig_offer_ratio": median(eig_ratios),
            "n_quiescent_windows": sum(
                1
                for r in full_switches
                if r["probe_rate_before"] == 0.0 and r["probe_rate_after"] == 0.0
            ),
        },
    )


def run_all_checks(rows: AblationRows) -> list[CheckResult]:
    return [
        f1_message_churn(rows),
        f2_decay(rows),
        f3_inventory(rows),
        f4_homeostat(rows),
        f5_eig_slope(rows),
        f6_babbling(rows),
        f7_reawakening(rows),
    ]


def hard_failures(checks: Sequence[CheckResult]) -> list[CheckResult]:
    return [c for c in checks if c.hard and c.passed is False]
