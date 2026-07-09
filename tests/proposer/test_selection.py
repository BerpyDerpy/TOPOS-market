"""The exported selection rule in isolation: lexicographic order, hard
gates, and the null/flatten fallback — no scalarized trade-off anywhere."""

from __future__ import annotations

import pytest

from tests.proposer.conftest import mk_candidate
from topos.proposer import select_candidate


def test_lexicographic() -> None:
    """Within EPSILON_EIG the lower self-entropy candidate is selected;
    outside it, higher marginal EIG wins regardless of self-entropy."""
    null = mk_candidate(kind="null", null=True, self_entropy=0.0)
    high_eig_noisy = mk_candidate(kind="a", marginal=0.50, self_entropy=5.0)
    near_tie_calm = mk_candidate(kind="b", marginal=0.49, self_entropy=1.0)
    picked = select_candidate((null, high_eig_noisy, near_tie_calm), {})
    assert picked is near_tie_calm

    far_behind_calm = mk_candidate(kind="c", marginal=0.40, self_entropy=0.1)
    picked = select_candidate((null, high_eig_noisy, far_behind_calm), {})
    assert picked is high_eig_noisy


def test_null_joins_the_tie_break_inside_the_boredom_band() -> None:
    """A saturated top marginal (within epsilon of 0) pulls the null into
    the pool, where minimum self-entropy hands the cycle to watching —
    the churn-extinction mechanism."""
    null = mk_candidate(kind="null", null=True, self_entropy=0.0)
    saturated = mk_candidate(kind="tiny", marginal=0.001, self_entropy=0.7)
    assert select_candidate((null, saturated), {}) is null
    # But a genuinely informative probe stays out of the null's reach.
    informative = mk_candidate(kind="big", marginal=0.3, self_entropy=0.7)
    assert select_candidate((null, informative), {}) is informative


def test_gates() -> None:
    """Vetoed and bound-violating candidates are never selected, no
    matter how large their EIG."""
    null = mk_candidate(kind="null", null=True, self_entropy=0.0)
    vetoed = mk_candidate(
        kind="vetoed", marginal=10.0, self_entropy=0.0, gates=False, vetoed=True
    )
    unsafe = mk_candidate(
        kind="unsafe", marginal=10.0, self_entropy=0.0, gates=False, confidence=0.5
    )
    modest = mk_candidate(kind="modest", marginal=0.05, self_entropy=2.0)
    assert select_candidate((null, vetoed, unsafe, modest), {}) is modest
    # With every probe gated out, the fallback pair takes over.
    assert select_candidate((null, vetoed, unsafe), {}) is null


def test_fallback_pair_null_by_default_flatten_on_drive() -> None:
    null = mk_candidate(kind="null", null=True, self_entropy=0.0)
    flat = mk_candidate(kind="flatten", flatten=True, marginal=0.0)
    blocked = mk_candidate(kind="vetoed", marginal=9.0, gates=False, vetoed=True)
    # No positive gated marginal + a nonzero drive distance => flatten.
    assert (
        select_candidate((null, blocked, flat), {"inventory": 0.4}) is flat
    )
    # All drives at zero => null by default.
    assert select_candidate((null, blocked, flat), {"inventory": 0.0}) is null
    # Without a flatten candidate (nothing to flatten), null even under drive.
    assert select_candidate((null, blocked), {"inventory": 0.4}) is null


def test_flatten_never_competes_on_eig() -> None:
    """Flatten carries no experiment bookkeeping: even a fabricated
    positive marginal must not let it win rules (b)/(c)."""
    null = mk_candidate(kind="null", null=True, self_entropy=0.0)
    flat = mk_candidate(
        kind="flatten", flatten=True, marginal=5.0, self_entropy=0.0
    )
    modest = mk_candidate(kind="modest", marginal=0.05, self_entropy=2.0)
    assert select_candidate((null, flat, modest), {}) is modest


def test_missing_null_candidate_is_an_error() -> None:
    probe = mk_candidate(marginal=0.3)
    with pytest.raises(ValueError, match="null candidate"):
        select_candidate((probe,), {})
