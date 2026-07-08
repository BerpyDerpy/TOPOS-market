"""Background market tests (P2).

- test_draw_invariance: the INV-9 consequence — an extra order-injecting
  actor leaves every background actor's raw draw log bit-identical.
- test_stylized_facts: seeded statistical sanity — heavy tails vs Gaussian,
  positive |return| autocorrelation, stable mean book depth.
- test_regime_switch: ground-truth log matches the configured schedule;
  hazard switches are reproducible under the seed.
- test_determinism: full event log bit-identical across repeated runs.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

import numpy as np
import numpy.typing as npt
from scipy import stats

from topos.contracts.market import PlaceLimit, Side
from topos.env.background import (
    BackgroundConfig,
    BackgroundMarket,
    DrawRecord,
    RegimeParams,
    ZI_ACTOR_ID,
)
from topos.env.engine import EngineEvent, MatchingEngine

ROOT_SEED = 20260708

Intruder = Callable[[int, MatchingEngine], PlaceLimit | None]


@dataclass
class SimResult:
    market: BackgroundMarket
    engine: MatchingEngine
    events: list[EngineEvent]
    mids: list[float | None]
    depths: list[int]


def run_sim(
    n_steps: int,
    root_seed: int,
    config: BackgroundConfig | None = None,
    intruder: Intruder | None = None,
) -> SimResult:
    """Drive engine + background market for n_steps.

    `intruder` plays the role of the agent: it may inject one extra order
    per step through the engine's step wrapper.
    """
    cfg = config if config is not None else BackgroundConfig()
    engine = MatchingEngine()
    market = BackgroundMarket(cfg, root_seed=root_seed)

    events: list[EngineEvent] = []
    mids: list[float | None] = []
    depths: list[int] = []

    for _ in range(n_steps):
        step = engine.current_step
        bg_events = market.events_for_step(engine)
        action = intruder(step, engine) if intruder is not None else None
        _, step_events = engine.step(bg_events, agent_id="intruder", agent_action=action)
        events.extend(step_events)

        book = engine.book
        bb, ba = book.best_bid, book.best_ask
        mids.append((bb + ba) / 2.0 if bb is not None and ba is not None else None)
        depths.append(
            sum(total for _, total in book.bid_levels())
            + sum(total for _, total in book.ask_levels())
        )

    return SimResult(market=market, engine=engine, events=events, mids=mids, depths=depths)


# =====================================================================
# INV-9: draw invariance under an extra actor
# =====================================================================

def _aggressive_intruder(step: int, engine: MatchingEngine) -> PlaceLimit | None:
    """Every third step, lift the ask hard enough to move the book."""
    if step % 3 != 0:
        return None
    best_ask = engine.book.best_ask
    if best_ask is None:
        return None
    return PlaceLimit(side=Side.BUY, price_ticks=best_ask + 4, size_lots=15, tif_steps=1)


def test_draw_invariance() -> None:
    n_steps = 150
    base = run_sim(n_steps, ROOT_SEED)
    perturbed = run_sim(n_steps, ROOT_SEED, intruder=_aggressive_intruder)

    # The intruder really did perturb the market (otherwise the assertion
    # below would be vacuous)...
    assert base.mids != perturbed.mids
    assert base.events != perturbed.events

    # ...yet every background actor's raw draw log is bit-identical, and so
    # is the regime chain: all divergence is mediated through the book.
    assert base.market.draw_log == perturbed.market.draw_log
    assert base.market.regime_log == perturbed.market.regime_log


def test_draw_log_is_complete_and_purpose_keyed() -> None:
    """Every (actor, step, purpose) appears at most once — each purpose is a
    fresh single-draw stream — and regime draws happen every step."""
    n_steps = 50
    result = run_sim(n_steps, ROOT_SEED)
    keys = [(r.actor_id, r.step, r.purpose) for r in result.market.draw_log]
    assert len(keys) == len(set(keys))

    regime_draws = {
        (r.step, r.purpose)
        for r in result.market.draw_log
        if r.actor_id == "background:regime"
    }
    for step in range(n_steps):
        assert (step, "regime_hazard") in regime_draws
        assert (step, "regime_choice") in regime_draws

    # MM innovations are drawn unconditionally every step too.
    mm_draws = {
        (r.actor_id, r.step)
        for r in result.market.draw_log
        if r.purpose == "ref_price_noise"
    }
    for step in range(n_steps):
        assert ("background:mm0", step) in mm_draws
        assert ("background:mm1", step) in mm_draws


# =====================================================================
# Determinism
# =====================================================================

def test_determinism_full_event_log() -> None:
    n_steps = 150
    a = run_sim(n_steps, ROOT_SEED)
    b = run_sim(n_steps, ROOT_SEED)

    assert a.events == b.events  # bit-identical engine event logs
    assert a.market.draw_log == b.market.draw_log
    assert a.market.regime_log == b.market.regime_log
    assert a.mids == b.mids
    assert a.depths == b.depths

    c = run_sim(n_steps, ROOT_SEED + 1)
    assert a.events != c.events


# =====================================================================
# Regime switching
# =====================================================================

def _no_hazard_regimes() -> tuple[RegimeParams, ...]:
    return tuple(replace(r, hazard=0.0) for r in BackgroundConfig().regimes)


def test_regime_switch_schedule() -> None:
    config = BackgroundConfig(
        regimes=_no_hazard_regimes(),
        initial_regime_id="calm",
        schedule=((40, "stressed"), (80, "calm")),
    )
    n_steps = 120
    result = run_sim(n_steps, ROOT_SEED, config=config)
    log = result.market.regime_log
    assert len(log) == n_steps

    by_id = {r.regime_id: r for r in config.regimes}
    for record in log:
        expected = "calm" if record.step < 40 or record.step >= 80 else "stressed"
        assert record.regime_id == expected, record
        # True parameters recorded, not just the id.
        assert record.params == by_id[expected]

    assert log[40].source == "schedule"
    assert log[80].source == "schedule"
    assert all(r.source == "carry" for r in log if r.step not in (40, 80))


def test_regime_switch_hazard_reproducible() -> None:
    regimes = tuple(replace(r, hazard=0.15) for r in BackgroundConfig().regimes)
    config = BackgroundConfig(regimes=regimes)
    n_steps = 200

    a = run_sim(n_steps, ROOT_SEED, config=config)
    b = run_sim(n_steps, ROOT_SEED, config=config)
    assert a.market.regime_log == b.market.regime_log

    # With hazard 0.15 over 200 steps, switches certainly occurred.
    sources = [r.source for r in a.market.regime_log]
    assert sources.count("hazard") >= 1

    # A different seed yields a different switch pattern.
    c = run_sim(n_steps, ROOT_SEED + 7, config=config)
    assert [r.regime_id for r in a.market.regime_log] != [
        r.regime_id for r in c.market.regime_log
    ]


def test_regime_hazard_unperturbed_by_intruder() -> None:
    """The regime chain specifically (not just draws) is agent-proof."""
    regimes = tuple(replace(r, hazard=0.15) for r in BackgroundConfig().regimes)
    config = BackgroundConfig(regimes=regimes)
    a = run_sim(150, ROOT_SEED, config=config)
    b = run_sim(150, ROOT_SEED, config=config, intruder=_aggressive_intruder)
    assert a.market.regime_log == b.market.regime_log


# =====================================================================
# Stylized facts (statistical, seeded — loose sanity bounds)
# =====================================================================

def _autocorr(x: npt.NDArray[np.float64], lag: int) -> float:
    centered = x - x.mean()
    denom = float(np.dot(centered, centered))
    if denom == 0.0:
        return 0.0
    return float(np.dot(centered[:-lag], centered[lag:]) / denom)


def test_stylized_facts() -> None:
    n_steps = 2000
    burn_in = 200
    result = run_sim(n_steps, ROOT_SEED)

    # A stressed market may momentarily go one-sided; that must stay rare,
    # and the mid series is forward-filled across those gaps.
    mids_raw = result.mids[burn_in:]
    n_one_sided = sum(1 for m in mids_raw if m is None)
    assert n_one_sided <= len(mids_raw) // 100, f"{n_one_sided} one-sided steps"
    filled: list[float] = []
    for m in mids_raw:
        if m is not None:
            filled.append(m)
        else:
            assert filled, "book one-sided at burn-in boundary"
            filled.append(filled[-1])
    mids = np.array(filled, dtype=np.float64)
    diffs = np.diff(mids)
    assert diffs.std() > 0.0

    # Heavy tails relative to Gaussian: positive excess (Fisher) kurtosis.
    excess_kurtosis = float(stats.kurtosis(diffs, fisher=True))
    assert excess_kurtosis > 0.5, f"excess kurtosis {excess_kurtosis:.2f} not heavy-tailed"

    # Volatility clustering: positive autocorrelation of |mid changes|.
    abs_diffs = np.abs(diffs)
    acs = [_autocorr(abs_diffs, lag) for lag in range(1, 6)]
    assert float(np.mean(acs)) > 0.02, f"|diff| autocorr too weak: {acs}"

    # Mean book depth stable: non-trivial and no secular drift between the
    # second and final quarter of the run.
    depths = np.array(result.depths[burn_in:], dtype=np.float64)
    assert depths.min() > 0
    quarter = len(depths) // 4
    early = float(depths[quarter : 2 * quarter].mean())
    late = float(depths[3 * quarter :].mean())
    assert early > 20.0
    assert 0.4 < late / early < 2.5, f"depth drifted: early {early:.1f}, late {late:.1f}"


# =====================================================================
# Behavioral sanity of the individual actors
# =====================================================================

def test_zi_flow_places_and_cancels() -> None:
    result = run_sim(200, ROOT_SEED)
    log = result.market.draw_log
    purposes = {r.purpose.split(":")[0] for r in log if r.actor_id == ZI_ACTOR_ID}
    assert {
        "n_limit_arrivals",
        "limit_side",
        "placement_depth",
        "limit_size",
        "n_market_orders",
        "n_cancels",
        "cancel_choice",
    } <= purposes
    # The ZI actor really traded: it holds resting orders and has fills.
    gt = result.engine.ground_truth_view(ZI_ACTOR_ID)
    assert gt.fills, "ZI flow never traded in 200 steps"


def test_mm_quotes_both_sides_and_respects_cap() -> None:
    config = BackgroundConfig()
    result = run_sim(300, ROOT_SEED, config=config)
    cap = config.mm.inventory_cap_lots
    for i in range(config.n_market_makers):
        actor = f"background:mm{i}"
        gt = result.engine.ground_truth_view(actor)
        # Inventory can overshoot the cap by at most one quote's worth
        # (the cap gates NEW quotes; a resting quote may still fill).
        assert abs(gt.inventory_lots) <= cap + config.mm.size_lots, actor
        assert gt.fills, f"{actor} never traded in 300 steps"


def test_draw_records_are_frozen() -> None:
    record = DrawRecord(actor_id="background:zi", step=0, purpose="n_cancels", value=1.0)
    try:
        record.value = 2.0  # type: ignore[misc]
    except AttributeError:
        return
    raise AssertionError("DrawRecord must be frozen")
