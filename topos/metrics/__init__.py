"""Evaluation, ablations, falsification suite (P13).

Agent code MUST NOT import this package; that boundary is enforced
mechanically by tests/tripwires/test_metrics_isolation.py (agent side)
and tests/tripwires/test_metrics_importers.py (repo side: only tests/
and experiments/ may import it). Profit, if it appears, is a measured
outcome computed here and only here — never an input to any decision.
Harness-only channels (ground-truth regimes and queue positions,
engine-side account state, counterfactual twins) terminate in this
package (INV-11).

The measurement instrument for the research question "does competent
behavior emerge from curiosity alone?":

* metrics 1-6 — pure functions ``Sequence[RunData] -> MetricResult``
  (tidy table + serializable summary),
* ``collect_run`` / ``reduce_run`` — episode collection and per-run
  reduction (what the parallel ablation runner distributes),
* ``run_all_checks`` — the F1-F7 falsification suite,
* ``render_report`` — the single markdown ablation report.
"""

from topos.metrics.ablation import (
    AblationRows,
    merge_rows,
    reduce_run,
    summarize_all,
)
from topos.metrics.behavior import behavior_signatures
from topos.metrics.calibration import belief_calibration
from topos.metrics.collect import (
    CONDITION_FLAGS,
    CONDITIONS,
    FULL,
    NO_HOMEOSTAT,
    NO_REFLEXIVE,
    NO_SELF_MODEL,
    SURPRISE_CURIOSITY,
    ImpactPosterior,
    RunData,
    collect_run,
)
from topos.metrics.efficiency import scientific_efficiency
from topos.metrics.eig import eig_calibration
from topos.metrics.falsification import (
    CheckResult,
    hard_failures,
    run_all_checks,
)
from topos.metrics.impact import impact_validation
from topos.metrics.outcome import outcome
from topos.metrics.report import (
    checks_to_json,
    render_report,
    summaries_to_json,
    to_json_compat,
)
from topos.metrics.segments import RegimeSegment, regime_segments
from topos.metrics.tables import MetricResult, TidyTable

__all__ = [
    "AblationRows",
    "CheckResult",
    "CONDITION_FLAGS",
    "CONDITIONS",
    "FULL",
    "ImpactPosterior",
    "MetricResult",
    "NO_HOMEOSTAT",
    "NO_REFLEXIVE",
    "NO_SELF_MODEL",
    "RegimeSegment",
    "RunData",
    "SURPRISE_CURIOSITY",
    "TidyTable",
    "behavior_signatures",
    "belief_calibration",
    "checks_to_json",
    "collect_run",
    "eig_calibration",
    "hard_failures",
    "impact_validation",
    "merge_rows",
    "outcome",
    "reduce_run",
    "regime_segments",
    "render_report",
    "run_all_checks",
    "scientific_efficiency",
    "summaries_to_json",
    "summarize_all",
    "to_json_compat",
]
