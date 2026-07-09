"""SelfTrajectory: ordinal properties of the reflexive self-entropy.

Every assertion here is ORDINAL (no tuned thresholds): the quantity under
test is an entropy compiled from posteriors, so what must hold are its
qualitative orderings —

    flat book, no orders, null intent      => minimal self-entropy,
    held inventory in a volatile market    => higher self-entropy,
    aggressive intent in an illiquid book  => high self-entropy through
                                              fill/impact uncertainty,

plus the spec's required chain

    H(null, flat) < H(null, large inventory) < H(aggressive, large inventory)

on fixed synthetic posteriors. A mechanical source scan pins the INV-5
half of the invariant: the compiler must never read the account-bearing
self-state or any account field.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pytest

import topos.selfmodel.self_trajectory as self_trajectory_module
from tests.beliefs.conftest import empty_events, obs_for_mid
from topos.beliefs import FairValueKF
from topos.contracts.intent import FILL_RATE, Intent, SELF_TRAJECTORY
from topos.contracts.market import Side
from topos.contracts.workspace import SelfStateCognitive, WorkingOrderView, WorldSummary
from topos.selfmodel import FillModel, ImpactModel, SelfTrajectory, UNIT_CELL_VAR

BUDGET_LOTS = 10

WORLD = WorldSummary(
    mid_ticks=1000.0,
    spread_ticks=2,
    imbalance=0.0,
    depth_profile=(5.0,) * 10,
    trade_tempo=3.0,
    realized_vol=1.0,
    regime_posterior=(1.0,),
)

NULL_INTENT = Intent(
    side=0.0, offset_ticks=0.0, size_frac=0.0, patience=1.0,
    target_id=FILL_RATE, commitment=0.0,
)
AGGRESSIVE_BUY = Intent(
    side=1.0, offset_ticks=-1.0, size_frac=1.0, patience=0.0,
    target_id=FILL_RATE, commitment=1.0,
)
FLATTEN_SELL = Intent(
    side=-1.0, offset_ticks=-1.0, size_frac=1.0, patience=0.0,
    target_id=SELF_TRAJECTORY, commitment=1.0,
)

FLAT = SelfStateCognitive(inventory_lots=0, working_orders=(), drive_distances={})
INV_50 = SelfStateCognitive(inventory_lots=50, working_orders=(), drive_distances={})


def make_kf(scale_b: float) -> FairValueKF:
    """A settled filter with a FIXED synthetic noise-scale posterior:
    IG(12, scale_b), mean noise scale scale_b / 11."""
    kf = FairValueKF()
    for step in range(5):
        kf.update(obs_for_mid(step, 1000.0 + 0.1 * step), empty_events(step))
    kf.noise_scale_posterior.a = 12.0
    kf.noise_scale_posterior.b = scale_b
    return kf


def make_compiler(
    *,
    scale_b: float = 11.0,
    fill_model: FillModel | None = None,
    impact_model: ImpactModel | None = None,
) -> SelfTrajectory:
    return SelfTrajectory(
        fill_model or FillModel(horizon_steps=3, size_budget_lots=BUDGET_LOTS),
        impact_model or ImpactModel(n_context=1, size_budget_lots=BUDGET_LOTS),
        make_kf(scale_b),
        size_budget_lots=BUDGET_LOTS,
    )


def entropy(
    compiler: SelfTrajectory, state: SelfStateCognitive, intent: Intent
) -> float:
    compiler.begin_cycle(state, WORLD)
    return compiler.self_entropy_nats(intent)


# =====================================================================
# The spec's required ordinal chain
# =====================================================================


def test_required_ordinal_chain() -> None:
    """H(null, flat) < H(null, large inventory) < H(aggressive, large
    inventory), on fixed synthetic posteriors."""
    compiler = make_compiler()
    h_flat = entropy(compiler, FLAT, NULL_INTENT)
    h_inventory = entropy(compiler, INV_50, NULL_INTENT)
    h_aggressive = entropy(compiler, INV_50, AGGRESSIVE_BUY)
    assert h_flat < h_inventory < h_aggressive


def test_flat_book_null_intent_is_minimal_and_deterministic() -> None:
    """Nothing can happen to a flat, orderless, non-acting agent: the
    inventory forecast is a point mass and the value forecast carries
    only the unit-cell term."""
    compiler = make_compiler()
    compiler.begin_cycle(FLAT, WORLD)
    forecast = compiler.forecast(NULL_INTENT)
    assert forecast.inventory_pmf == ((0, 1.0),)
    assert forecast.inventory_entropy_nats == 0.0
    assert forecast.value_change_variance == pytest.approx(0.0)
    assert forecast.entropy_nats == pytest.approx(
        0.5 * math.log(2.0 * math.pi * math.e * UNIT_CELL_VAR)
    )
    # ...and it is the minimum over a grid of states and intents.
    working = WorkingOrderView(
        order_id=0, side=Side.BUY, price_ticks=999, size_lots_remaining=5,
        age_steps=1, queue_rank_mean=3.0, queue_rank_var=2.0,
    )
    grid = [
        (FLAT, AGGRESSIVE_BUY),
        (INV_50, NULL_INTENT),
        (INV_50, AGGRESSIVE_BUY),
        (
            SelfStateCognitive(
                inventory_lots=0, working_orders=(working,), drive_distances={}
            ),
            NULL_INTENT,
        ),
    ]
    for state, intent in grid:
        assert forecast.entropy_nats < entropy(compiler, state, intent)


def test_inventory_in_a_volatile_market_raises_self_entropy() -> None:
    """Same held inventory, same everything, wider fair-value noise-scale
    posterior => strictly higher self-entropy (and monotone in scale)."""
    calm = make_compiler(scale_b=11.0)
    volatile = make_compiler(scale_b=110.0)
    h_calm = entropy(calm, INV_50, NULL_INTENT)
    h_volatile = entropy(volatile, INV_50, NULL_INTENT)
    assert h_calm < h_volatile
    # But a FLAT agent does not care how volatile the market is.
    assert entropy(calm, FLAT, NULL_INTENT) == pytest.approx(
        entropy(volatile, FLAT, NULL_INTENT)
    )


def test_working_orders_add_self_uncertainty() -> None:
    working = WorkingOrderView(
        order_id=0, side=Side.BUY, price_ticks=999, size_lots_remaining=5,
        age_steps=1, queue_rank_mean=3.0, queue_rank_var=2.0,
    )
    with_order = SelfStateCognitive(
        inventory_lots=0, working_orders=(working,), drive_distances={}
    )
    compiler = make_compiler()
    assert entropy(compiler, FLAT, NULL_INTENT) < entropy(
        compiler, with_order, NULL_INTENT
    )


def _certain_fill_model() -> FillModel:
    model = FillModel(horizon_steps=3, size_budget_lots=BUDGET_LOTS)
    for cell in model.cells.values():
        cell.a, cell.b = 199.0, 2.0  # fills near-certain, parameter settled
    return model


def _learned_impact_model() -> ImpactModel:
    model = ImpactModel(n_context=1, size_budget_lots=BUDGET_LOTS)
    rng = np.random.default_rng(3)
    for _ in range(2000):
        aggression = float(rng.integers(-10, 11))
        resting = float(rng.integers(-10, 11))
        context = float(rng.normal())
        y = 0.05 * aggression + 0.02 * resting + rng.normal(0.0, 0.3)
        model.observe_point(aggression, resting, (context,), float(y))
    return model


def test_aggressive_intent_in_an_illiquid_book_is_high_entropy() -> None:
    """The aggressive intent's extra self-entropy flows from the fill and
    impact POSTERIORS, channel by channel — not from a size penalty.

    Fill channel: with an unprobed (illiquid) fill bucket, the outcome is
    maximally uncertain — the inventory forecast carries a full bit; a
    settled bucket forecloses it. Impact channel: holding the fill
    posterior FIXED (so the exposure profile is identical), an ignorant
    impact posterior strictly raises the self-entropy over a learned one.
    """
    # -- fill channel -------------------------------------------------------
    ignorant = make_compiler()
    ignorant.begin_cycle(FLAT, WORLD)
    ignorant_forecast = ignorant.forecast(AGGRESSIVE_BUY)
    assert ignorant_forecast.inventory_entropy_nats == pytest.approx(math.log(2.0))
    settled = make_compiler(fill_model=_certain_fill_model())
    settled.begin_cycle(FLAT, WORLD)
    settled_forecast = settled.forecast(AGGRESSIVE_BUY)
    assert settled_forecast.inventory_entropy_nats < 0.1 * math.log(2.0)
    # ...and the aggressive intent towers over the null in the same
    # ignorant book: this IS "high self-entropy through fill/impact
    # uncertainty".
    assert ignorant_forecast.entropy_nats > entropy(ignorant, FLAT, NULL_INTENT)

    # -- impact channel, fill posterior held fixed --------------------------
    ignorant_impact = make_compiler(fill_model=_certain_fill_model())
    learned_impact = make_compiler(
        fill_model=_certain_fill_model(), impact_model=_learned_impact_model()
    )
    h_ignorant_impact = entropy(ignorant_impact, FLAT, AGGRESSIVE_BUY)
    h_learned_impact = entropy(learned_impact, FLAT, AGGRESSIVE_BUY)
    assert h_learned_impact < h_ignorant_impact


def test_flattening_with_learned_fills_reduces_self_entropy() -> None:
    """The emergent inventory-aversion story at unit scale: once the
    flatten path is predictable (fill posterior learned), shedding a
    large inventory FORECLOSES more uncertainty than holding it."""
    fill = FillModel(horizon_steps=3, size_budget_lots=50)
    for cell in fill.cells.values():
        cell.a, cell.b = 199.0, 2.0
    compiler = SelfTrajectory(
        fill,
        ImpactModel(n_context=1, size_budget_lots=50),
        make_kf(11.0),
        size_budget_lots=50,
    )
    h_hold = entropy(compiler, INV_50, NULL_INTENT)
    h_flatten = entropy(compiler, INV_50, FLATTEN_SELL)
    assert h_flatten < h_hold


def test_forecast_uses_the_horizon() -> None:
    """More steps of fair-value diffusion => wider value forecast for a
    held inventory (the compiler really consumes the horizon)."""
    compiler = make_compiler()
    compiler.begin_cycle(INV_50, WORLD)
    short = compiler.forecast(NULL_INTENT, horizon_steps=1)
    long = compiler.forecast(NULL_INTENT, horizon_steps=10)
    assert short.value_change_variance < long.value_change_variance
    assert short.entropy_nats < long.entropy_nats


def test_forecast_requires_cycle_context_and_valid_horizon() -> None:
    compiler = make_compiler()
    with pytest.raises(RuntimeError):
        compiler.forecast(NULL_INTENT)
    compiler.begin_cycle(FLAT, WORLD)
    with pytest.raises(ValueError):
        compiler.forecast(NULL_INTENT, horizon_steps=0)


# =====================================================================
# INV-5, mechanically: the compiler cannot see account state
# =====================================================================

FORBIDDEN_ACCOUNT_TOKEN = re.compile(r"pnl|profit|drawdown|wealth", re.IGNORECASE)


def test_compiler_source_has_no_account_access() -> None:
    """The tripwire pattern applied to the compiler's whole source file
    (comments and strings included), plus the account-bearing type's
    name: the reflexive quantity must be UNABLE to read account state,
    not merely observed not to."""
    source = Path(self_trajectory_module.__file__).read_text()
    assert "SelfStateFull" not in source
    offenders = [
        f"{lineno}: {line.strip()}"
        for lineno, line in enumerate(source.splitlines(), start=1)
        if FORBIDDEN_ACCOUNT_TOKEN.search(line)
    ]
    assert not offenders, (
        "INV-5: account vocabulary in the trajectory compiler:\n"
        + "\n".join(offenders)
    )


def test_self_entropy_is_not_scaled_variance() -> None:
    """Ordinal separation from the forbidden implementation: for a pure
    lambda * variance rule, doubling inventory would quadruple the score;
    the entropy grows logarithmically instead. This pins the FUNCTIONAL
    FORM, not a calibration."""
    compiler = make_compiler()
    inv_25 = SelfStateCognitive(
        inventory_lots=25, working_orders=(), drive_distances={}
    )
    h_25 = entropy(compiler, inv_25, NULL_INTENT)
    h_50 = entropy(compiler, INV_50, NULL_INTENT)
    compiler.begin_cycle(inv_25, WORLD)
    var_25 = compiler.forecast(NULL_INTENT).value_change_variance
    compiler.begin_cycle(INV_50, WORLD)
    var_50 = compiler.forecast(NULL_INTENT).value_change_variance
    # The variance itself quadruples...
    assert var_50 == pytest.approx(4.0 * var_25, rel=1e-6)
    # ...but the self-entropy moves by ~ln(2) per doubling, nowhere near
    # a factor of 4 (or even 2).
    assert h_50 - h_25 == pytest.approx(math.log(2.0), abs=0.05)
    assert h_50 < 2.0 * h_25
