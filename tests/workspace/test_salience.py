"""The salience competition: surprise attends, EIG acts; drives preempt;
quiet minds watch.

The three required behaviors:

* ``test_surprise_gates_through_eig`` — surprise multiplies EIG, so zero
  reducible uncertainty means zero salience no matter how loud the
  errors (the noisy-TV immunity, restated at the attention level);
* ``test_homeostat_wins_near_bounds`` — drive bids diverge as u -> 1 and
  beat ANY finite EIG (INV-6's preemption);
* ``test_ignition_threshold`` — below S_MIN nothing ignites, the focus
  is None, and the record says so.
"""

from __future__ import annotations

from tests.workspace.conftest import (
    FakeModule,
    make_registry,
    make_workspace,
    run_cycle,
)
from topos.contracts.intent import SELF_TRAJECTORY, flatten_intent
from topos.drives.homeostat import D_NATS
from topos.workspace import S_MIN, SalienceBid, compete, hypothesis_salience


def _drive(u: float) -> float:
    """P7's drive law, D * u^2 / (1 - u) — the exchange rate under test."""
    if u >= 1.0:
        return float("inf")
    return D_NATS * u * u / (1.0 - u)


def test_surprise_gates_through_eig() -> None:
    """A high-surprise, zero-EIG hypothesis never wins focus."""
    modules = {
        "hyp_noisy_tv": FakeModule(
            hypothesis_id="hyp_noisy_tv", probe_gain=0.0, surprise=1e9
        ),
        "hyp_learnable": FakeModule(
            hypothesis_id="hyp_learnable", probe_gain=0.2, surprise=0.0
        ),
    }
    workspace = make_workspace(modules, registry=make_registry(tuple(modules)))
    record = run_cycle(workspace)
    assert record.focus is not None
    assert record.focus.hypothesis_id == "hyp_learnable"

    # Alone, the surprising-but-saturated hypothesis leaves the mind
    # quiet: its salience is exactly zero, whatever the surprise.
    alone = {
        "hyp_noisy_tv": FakeModule(
            hypothesis_id="hyp_noisy_tv", probe_gain=0.0, surprise=1e12
        )
    }
    quiet = make_workspace(alone, registry=make_registry(tuple(alone)))
    quiet_record = run_cycle(quiet)
    assert quiet_record.focus is None
    assert quiet_record.intent is not None and quiet_record.intent.is_null

    # The formula itself: surprise multiplies EIG, never adds to it.
    assert hypothesis_salience(0.5, 0.0, 1e12) == 0.0
    assert hypothesis_salience(0.5, 0.1, 0.0) == 0.5 * 0.1
    assert hypothesis_salience(0.5, 0.1, 2.0) > hypothesis_salience(0.5, 0.1, 0.0)
    # Negative surprise (better than expected) never dims an answerable
    # question below its EIG bid.
    assert hypothesis_salience(0.5, 0.1, -3.0) == 0.5 * 0.1


def test_homeostat_wins_near_bounds() -> None:
    """As u -> 1 the drive bid diverges and dominates any finite EIG."""

    def build(gain: float):  # type: ignore[no-untyped-def]
        modules = {"hyp_rich": FakeModule(hypothesis_id="hyp_rich", probe_gain=gain)}
        return make_workspace(modules, registry=make_registry(("hyp_rich",)))

    # Mid-band excursion: a strong question still out-bids the drive.
    record = run_cycle(
        build(100.0),
        inventory=5,
        drives={"inventory": _drive(0.5)},
        corrective_intent=flatten_intent(5),
    )
    assert record.focus is not None and not record.focus.is_homeostatic

    # Approaching the hard bound the drive preempts — even a hypothesis
    # promising an absurd finite EIG.
    for gain, u in ((100.0, 0.99), (1e6, 1.0 - 1e-9), (1e12, 1.0)):
        record = run_cycle(
            build(gain),
            inventory=5,
            drives={"inventory": _drive(u)},
            corrective_intent=flatten_intent(5),
        )
        assert record.focus is not None
        assert record.focus.is_homeostatic
        assert record.focus.hypothesis_id == "inventory"
        # Arbitration: the homeostat's corrective intent, verbatim, aimed
        # at the self-trajectory; it promises action, not information.
        assert record.intent == flatten_intent(5)
        assert record.intent.target_id == SELF_TRAJECTORY
        assert record.eig_promised_nats == 0.0

    # Monotone dominance in u: the bid diverges.
    assert _drive(0.999) > _drive(0.99) > _drive(0.9)
    assert _drive(1.0) == float("inf")


def test_ignition_threshold() -> None:
    """Below S_MIN nothing ignites: focus None, intent null, record says so."""
    modules = {
        "hyp_a": FakeModule(hypothesis_id="hyp_a", probe_gain=0.001),
        "hyp_b": FakeModule(hypothesis_id="hyp_b", probe_gain=0.0005),
    }
    workspace = make_workspace(modules, registry=make_registry(tuple(modules)))
    # Max possible bid: weight <= 1, gain 0.001, no surprise -> below S_MIN.
    assert max(workspace.weights.values()) * 0.001 < S_MIN

    record = run_cycle(workspace, drives={"inventory": S_MIN * 0.5})
    assert record.focus is None
    assert record.intent is not None
    assert record.intent.is_null
    assert record.eig_promised_nats is None
    assert record.compiled_messages == ()
    # A quiet cycle still broadcasts its headlines: watching is data.
    assert len(record.headlines) == 2

    # The threshold is strict: salience exactly at S_MIN does not ignite.
    at = [SalienceBid(bid_id="hyp_a", salience=S_MIN, is_homeostatic=False)]
    assert compete(at, s_min=S_MIN) is None
    above = [
        SalienceBid(bid_id="hyp_a", salience=S_MIN * 1.01, is_homeostatic=False)
    ]
    focus = compete(above, s_min=S_MIN)
    assert focus is not None and focus.hypothesis_id == "hyp_a"
    assert compete([], s_min=S_MIN) is None
