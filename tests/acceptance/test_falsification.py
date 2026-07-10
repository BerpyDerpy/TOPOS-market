"""The falsification suite as executable acceptance tests.

Every check is asserted against its RECORDED adjudication (the constants
below, argued in DESIGN.md, Open questions — P13): checks the data
supports assert PASS; predictions the data killed assert FALSIFIED. A
falsified prediction is a scientific result of this project — pinning it
keeps the result mechanically visible, and any architecture change that
flips a status fails here and forces a conscious DESIGN.md update. Per
the fairness rules, flipping a FALSIFIED pin by adjusting an agent
constant is forbidden; the pin may only move with a recorded design
change.

The dataset is deterministic: fixed seed list, fixed schedule, INV-9
event streams, and a deterministic agent — so these are exact
assertions, not statistical hopes.

Summary of the recorded adjudication (DESIGN.md item 43, "the drive
lock"): drawdown is measured against an all-time peak and has no
corrective action, so its soft-band excursion is absorbing; the drive
then outbids curiosity every cycle. This locks every homeostat-bearing
condition into quiescence within a few hundred steps and is what
falsifies F1 (FULL's traffic is corrective churn; z-scored surprise
self-normalizes), F2's second leg (NO_SELF_MODEL's probing decays too —
throttled, not satiated), F3 (reversed), and F7's behavioral ratio
(while the epistemic EIG-offer ratio shows curiosity reopening).
"""

from __future__ import annotations

import math

from tests.acceptance.conftest import AblationArtifacts

RECORDED_ADJUDICATION: dict[str, bool] = {
    "F1": False,  # inverted by corrective churn + self-normalizing surprise
    "F2": False,  # FULL leg holds; NO_SELF_MODEL decays too (drive lock)
    "F3": False,  # reversed: NO_REFLEXIVE carries LESS inventory
    "F4": True,  # NO_HOMEOSTAT lives beyond the soft bounds
    "F5": True,  # impact promised-vs-realized slope in [0.5, 1.5]
    "F6": True,  # babbling decay in the first segment
    "F7": False,  # behavioral reawakening absent (epistemic present)
}
"""check id -> the adjudicated outcome pinned by this suite.
See DESIGN.md, Open questions (P13) before changing ANY entry."""


def _assert_recorded(artifacts: AblationArtifacts, check_id: str) -> None:
    check = artifacts.check(check_id)
    expected = RECORDED_ADJUDICATION[check_id]
    assert check.passed is not None, (
        f"{check_id} was not evaluable — the instrument is broken or "
        f"underpowered: {check.description}; details={check.details}"
    )
    if expected:
        assert check.passed is True, (
            f"{check_id} FALSIFIED (hard): {check.description} — "
            f"value={check.value:.4g}, details={check.details}. Record the "
            "falsification in DESIGN.md (P13) and update the adjudication "
            "table; do not tune agent constants."
        )
    else:
        assert check.passed is False, (
            f"{check_id} now PASSES (value={check.value:.4g}) but is "
            "recorded as falsified. If a deliberate design change fixed "
            "it, update DESIGN.md (P13) and the adjudication table; if "
            "nothing changed on purpose, something drifted — investigate."
        )


def test_f1_message_churn_adjudication(ablation: AblationArtifacts) -> None:
    """Predicted: FULL messages << SURPRISE (ratio < 0.5). Falsified:
    FULL sends ~2x MORE — its traffic is homeostat corrective churn, and
    the z-scored surprise signal self-normalizes instead of churning."""
    _assert_recorded(ablation, "F1")
    check = ablation.check("F1")
    # The falsification is decisive, not marginal: the ratio is inverted.
    assert check.value > 1.0


def test_f2_decay_adjudication(ablation: AblationArtifacts) -> None:
    """Predicted: FULL probe rate decays, NO_SELF_MODEL's does not.
    Half-confirmed: FULL decays (CI > 0) — but NO_SELF_MODEL decays too
    (the drive lock throttles it; frozen posteriors never satiate but the
    workspace stops listening). The check as specified is falsified."""
    _assert_recorded(ablation, "F2")
    full_lo = ablation.check("F2").details["full_ci"][1]
    assert full_lo > 0.0, "the FULL leg (decay exists) should still hold"


def test_f3_inventory_adjudication(ablation: AblationArtifacts) -> None:
    """Predicted: NO_REFLEXIVE mean |inventory| > FULL. Falsified —
    reversed at the fixed seeds."""
    _assert_recorded(ablation, "F3")


def test_f4_homeostat_excursion(ablation: AblationArtifacts) -> None:
    """NO_HOMEOSTAT spends decisively more time beyond the soft bounds
    than FULL (hard vetoes remain; the drive is what was ablated)."""
    _assert_recorded(ablation, "F4")


def test_f5_eig_slope(ablation: AblationArtifacts) -> None:
    """FULL promised-vs-realized slope in [0.5, 1.5] for impact — the one
    ledgerable hypothesis whose evidence arrives inside the resolution
    window. fill_rate is reported but not bound (ledger resolves before
    its acks arrive — DESIGN.md item 42); world hypotheses cannot be
    ledgered under FULL at all (item 41)."""
    _assert_recorded(ablation, "F5")
    details = ablation.check("F5").details
    assert "FULL:fill_rate" in details, "fill_rate must still be reported"
    assert details["impact"]["n"] >= 5


def test_f6_babbling_reported(ablation: AblationArtifacts) -> None:
    """Report check: FULL's first-segment probe rate decays; the fitted
    curve is reported per seed."""
    _assert_recorded(ablation, "F6")
    fits = ablation.check("F6").details["per_seed_fit"]
    assert len(fits) >= len(ablation.meta["seeds"]) // 2


def test_f7_reawakening_split(ablation: AblationArtifacts) -> None:
    """Report check, falsified behaviorally: the corrected probe-rate
    ratio around switches is ~1 (the agent sleeps through switches) while
    the EPISTEMIC EIG-offer ratio exceeds 1 — curiosity reopens but never
    wins the workspace back from the drives (DESIGN.md item 43)."""
    _assert_recorded(ablation, "F7")
    details = ablation.check("F7").details
    assert math.isfinite(details["median_eig_offer_ratio"])
    assert details["median_eig_offer_ratio"] > 1.0, (
        "the epistemic reawakening should be visible even while the "
        "behavioral one is locked out"
    )
    assert details["n_quiescent_windows"] > 0
