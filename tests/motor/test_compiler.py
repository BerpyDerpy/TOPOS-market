"""Tests for topos.motor — the deterministic intent→message compiler (INV-8).

Required tests:
1. Determinism (hypothesis): same inputs => bit-identical output tuples.
2. Legality: prices on tick, sizes positive lots, never both sides crossed,
   STP holds for random intent/book combinations.
3. Veto property: under each veto flag, no emitted message increases that
   variable's excursion.
4. Null semantics: commitment below threshold emits no PlaceLimit but may
   emit cancels per patience rule.
5. Log format: (intent, messages) pairs serialize losslessly side by side.
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import asdict

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from topos.contracts.intent import (
    FAIR_VALUE,
    FILL_RATE,
    NULL_THRESHOLD,
    SELF_TRAJECTORY,
    Intent,
    flatten_intent,
)
from topos.contracts.market import (
    N_LEVELS,
    BookLevel,
    Cancel,
    ExchangeMessage,
    PlaceLimit,
    Side,
)
from topos.contracts.workspace import WorkingOrderView
from topos.motor.compiler import (
    _best_ask_tick,
    _best_bid_tick,
    _live_levels,
    _mid_ticks,
    _stp_cancels,
    _veto_suppresses,
    compile,
)
from topos.motor.config import MotorConfig


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

def _pad(levels: list[BookLevel], n: int = N_LEVELS) -> tuple[BookLevel, ...]:
    """Pad a list of BookLevels to exactly n entries with size_lots=0."""
    while len(levels) < n:
        price = levels[-1].price_ticks + 1 if levels else 1000
        levels.append(BookLevel(price_ticks=price, size_lots=0))
    return tuple(levels[:n])


def _default_cfg() -> MotorConfig:
    return MotorConfig(size_budget_lots=10, max_patience_steps=50)


def _simple_book(
    best_bid: int = 999, best_ask: int = 1001, depth: int = 3
) -> tuple[tuple[BookLevel, ...], tuple[BookLevel, ...]]:
    bids = [BookLevel(price_ticks=best_bid - i, size_lots=5) for i in range(depth)]
    asks = [BookLevel(price_ticks=best_ask + i, size_lots=5) for i in range(depth)]
    return _pad(bids), _pad(asks)


@st.composite
def st_book(draw: st.DrawFn):
    mid = draw(st.integers(500, 1500))
    spread = draw(st.integers(1, 10))
    best_bid = mid - spread // 2
    best_ask = best_bid + spread
    n_live = draw(st.integers(1, N_LEVELS))
    bids = [BookLevel(price_ticks=best_bid - i, size_lots=draw(st.integers(1, 100)))
            for i in range(n_live)]
    asks = [BookLevel(price_ticks=best_ask + i, size_lots=draw(st.integers(1, 100)))
            for i in range(n_live)]
    return _pad(bids), _pad(asks)


@st.composite
def st_intent(draw: st.DrawFn):
    return Intent(
        side=draw(st.sampled_from([-1.0, 1.0])),
        offset_ticks=draw(st.floats(min_value=-5, max_value=10)),
        size_frac=draw(st.floats(min_value=0.01, max_value=1.0)),
        patience=draw(st.floats(min_value=0.0, max_value=0.99)),
        target_id=FILL_RATE,
        commitment=draw(st.floats(min_value=0.5, max_value=1.0)),
    )


@st.composite
def st_working(draw: st.DrawFn):
    n = draw(st.integers(0, 4))
    orders = []
    for i in range(n):
        orders.append(WorkingOrderView(
            order_id=100 + i,
            side=draw(st.sampled_from([Side.BUY, Side.SELL])),
            price_ticks=draw(st.integers(990, 1010)),
            size_lots_remaining=draw(st.integers(1, 10)),
            age_steps=draw(st.integers(0, 100)),
            queue_rank_mean=1.0,
            queue_rank_var=0.5,
        ))
    return tuple(orders)


# ===================================================================
# Test 1: Determinism (INV-8)
# ===================================================================

class TestDeterminism:
    """Same inputs => bit-identical output tuples."""

    @given(st_intent(), st_book(), st_working())
    @settings(max_examples=200)
    def test_determinism_property(
        self,
        intent: Intent,
        book: tuple[tuple[BookLevel, ...], tuple[BookLevel, ...]],
        working: tuple[WorkingOrderView, ...],
    ) -> None:
        bids, asks = book
        cfg = _default_cfg()
        vetoes: dict[str, bool] = {}
        r1 = compile(intent, bids, asks, working, vetoes, cfg)
        r2 = compile(intent, bids, asks, working, vetoes, cfg)
        assert r1 == r2

    def test_determinism_fixed(self) -> None:
        bids, asks = _simple_book()
        intent = Intent(side=1.0, offset_ticks=2.0, size_frac=0.5,
                        patience=0.3, target_id=FILL_RATE, commitment=0.9)
        cfg = _default_cfg()
        results = [compile(intent, bids, asks, (), {}, cfg) for _ in range(50)]
        assert all(r == results[0] for r in results)


# ===================================================================
# Test 2: Legality
# ===================================================================

class TestLegality:
    """Prices on tick, sizes positive, STP, never both sides crossed."""

    @given(st_intent(), st_book(), st_working())
    @settings(max_examples=200)
    def test_prices_integer_and_sizes_positive(
        self,
        intent: Intent,
        book: tuple[tuple[BookLevel, ...], tuple[BookLevel, ...]],
        working: tuple[WorkingOrderView, ...],
    ) -> None:
        bids, asks = book
        msgs = compile(intent, bids, asks, working, {}, _default_cfg())
        for m in msgs:
            if isinstance(m, PlaceLimit):
                assert isinstance(m.price_ticks, int)
                assert m.size_lots > 0

    @given(st_intent(), st_book(), st_working())
    @settings(max_examples=200)
    def test_stp_no_self_cross(
        self,
        intent: Intent,
        book: tuple[tuple[BookLevel, ...], tuple[BookLevel, ...]],
        working: tuple[WorkingOrderView, ...],
    ) -> None:
        """After compile, no emitted PlaceLimit crosses a surviving working order."""
        bids, asks = book
        msgs = compile(intent, bids, asks, working, {}, _default_cfg())
        canceled_ids = {m.order_id for m in msgs if isinstance(m, Cancel)}
        surviving = [w for w in working if w.order_id not in canceled_ids]
        places = [m for m in msgs if isinstance(m, PlaceLimit)]
        for pl in places:
            for wo in surviving:
                if wo.side == pl.side:
                    continue
                if pl.side == Side.BUY:
                    assert pl.price_ticks < wo.price_ticks, (
                        f"BUY@{pl.price_ticks} crosses own SELL@{wo.price_ticks}"
                    )
                else:
                    assert pl.price_ticks > wo.price_ticks, (
                        f"SELL@{pl.price_ticks} crosses own BUY@{wo.price_ticks}"
                    )

    def test_never_both_sides_placed(self) -> None:
        """A single compile never places orders on both sides."""
        bids, asks = _simple_book()
        intent = Intent(side=1.0, offset_ticks=1.0, size_frac=0.5,
                        patience=0.5, target_id=FILL_RATE, commitment=0.9)
        msgs = compile(intent, bids, asks, (), {}, _default_cfg())
        sides = {m.side for m in msgs if isinstance(m, PlaceLimit)}
        assert len(sides) <= 1


# ===================================================================
# Test 3: Veto property (INV-6)
# ===================================================================

class TestVetoProperty:
    """Under each veto, no emitted message increases that variable's excursion."""

    @given(st_intent(), st_book(), st.integers(-20, 20))
    @settings(max_examples=200)
    def test_inventory_veto(
        self,
        intent: Intent,
        book: tuple[tuple[BookLevel, ...], tuple[BookLevel, ...]],
        inv: int,
    ) -> None:
        bids, asks = book
        vetoes = {"inventory": True}
        msgs = compile(intent, bids, asks, (), vetoes, _default_cfg(),
                       inventory_lots=inv)
        for m in msgs:
            if isinstance(m, PlaceLimit):
                delta = m.size_lots if m.side == Side.BUY else -m.size_lots
                assert abs(inv + delta) <= abs(inv), (
                    f"inventory veto violated: |{inv}+{delta}| > |{inv}|"
                )

    @given(st_intent(), st_book(), st.integers(-20, 20))
    @settings(max_examples=200)
    def test_message_budget_veto_suppresses_places(
        self,
        intent: Intent,
        book: tuple[tuple[BookLevel, ...], tuple[BookLevel, ...]],
        inv: int,
    ) -> None:
        bids, asks = book
        vetoes = {"message_budget": True}
        msgs = compile(intent, bids, asks, (), vetoes, _default_cfg(),
                       inventory_lots=inv)
        for m in msgs:
            assert isinstance(m, Cancel), "message_budget veto must suppress PlaceLimit"

    @given(st_intent(), st_book(), st.integers(-20, 20))
    @settings(max_examples=200)
    def test_drawdown_veto(
        self,
        intent: Intent,
        book: tuple[tuple[BookLevel, ...], tuple[BookLevel, ...]],
        inv: int,
    ) -> None:
        bids, asks = book
        vetoes = {"drawdown": True}
        msgs = compile(intent, bids, asks, (), vetoes, _default_cfg(),
                       inventory_lots=inv)
        for m in msgs:
            if isinstance(m, PlaceLimit):
                delta = m.size_lots if m.side == Side.BUY else -m.size_lots
                assert abs(inv + delta) < abs(inv), (
                    f"drawdown veto violated: |{inv}+{delta}| >= |{inv}|"
                )


# ===================================================================
# Test 4: Null semantics
# ===================================================================

class TestNullSemantics:
    """commitment below threshold => no PlaceLimit; may emit cancels."""

    def test_null_intent_no_place(self) -> None:
        bids, asks = _simple_book()
        intent = Intent(side=1.0, offset_ticks=1.0, size_frac=0.5,
                        patience=0.0, target_id=FILL_RATE,
                        commitment=NULL_THRESHOLD - 0.01)
        assert intent.is_null
        msgs = compile(intent, bids, asks, (), {}, _default_cfg())
        assert not any(isinstance(m, PlaceLimit) for m in msgs)

    def test_null_intent_can_cancel_stale(self) -> None:
        """Null action can still cancel stale orders."""
        bids, asks = _simple_book()
        intent = Intent(side=1.0, offset_ticks=1.0, size_frac=0.5,
                        patience=0.0, target_id=FILL_RATE,
                        commitment=0.0)
        stale_order = WorkingOrderView(
            order_id=42, side=Side.BUY, price_ticks=990,
            size_lots_remaining=3, age_steps=100,
            queue_rank_mean=2.0, queue_rank_var=1.0,
        )
        cfg = MotorConfig(size_budget_lots=10, max_patience_steps=50)
        msgs = compile(intent, bids, asks, (stale_order,), {}, cfg)
        assert any(isinstance(m, Cancel) and m.order_id == 42 for m in msgs)

    def test_zero_size_frac_no_place(self) -> None:
        bids, asks = _simple_book()
        intent = Intent(side=1.0, offset_ticks=1.0, size_frac=0.0,
                        patience=0.5, target_id=FILL_RATE, commitment=0.9)
        msgs = compile(intent, bids, asks, (), {}, _default_cfg())
        assert not any(isinstance(m, PlaceLimit) for m in msgs)


# ===================================================================
# Test 5: Log format (INV-8)
# ===================================================================

class TestLogFormat:
    """(intent, messages) pairs serialize losslessly side by side."""

    def _serialize_pair(self, intent: Intent, msgs: tuple[ExchangeMessage, ...]) -> str:
        """Serialize an (intent, messages) pair to JSON."""
        record: dict = {"intent": asdict(intent), "messages": []}
        for m in msgs:
            entry: dict = {"type": type(m).__name__}
            entry.update(asdict(m))
            # Convert Side enum to its value for serialization.
            if isinstance(m, PlaceLimit):
                entry["side"] = m.side.value
            record["messages"].append(entry)
        return json.dumps(record, sort_keys=True)

    def _deserialize_pair(self, s: str) -> tuple[dict, list[dict]]:
        record = json.loads(s)
        return record["intent"], record["messages"]

    def test_roundtrip(self) -> None:
        bids, asks = _simple_book()
        intent = Intent(side=1.0, offset_ticks=2.0, size_frac=0.5,
                        patience=0.3, target_id=FILL_RATE, commitment=0.9)
        msgs = compile(intent, bids, asks, (), {}, _default_cfg())
        s = self._serialize_pair(intent, msgs)
        intent_d, msgs_d = self._deserialize_pair(s)
        s2 = json.dumps({"intent": intent_d, "messages": msgs_d}, sort_keys=True)
        assert s == s2, "Roundtrip serialization is not lossless"

    @given(st_intent(), st_book())
    @settings(max_examples=50)
    def test_roundtrip_property(
        self,
        intent: Intent,
        book: tuple[tuple[BookLevel, ...], tuple[BookLevel, ...]],
    ) -> None:
        bids, asks = book
        msgs = compile(intent, bids, asks, (), {}, _default_cfg())
        s = self._serialize_pair(intent, msgs)
        intent_d, msgs_d = self._deserialize_pair(s)
        s2 = json.dumps({"intent": intent_d, "messages": msgs_d}, sort_keys=True)
        assert s == s2


# ===================================================================
# Additional: flatten path
# ===================================================================

class TestFlattenPath:
    """Flatten intents: passive-first, never increase |inventory|."""

    def test_flatten_positive_inventory_sells(self) -> None:
        bids, asks = _simple_book()
        intent = flatten_intent(10)
        assert intent.is_flatten
        msgs = compile(intent, bids, asks, (), {}, _default_cfg(), inventory_lots=10)
        places = [m for m in msgs if isinstance(m, PlaceLimit)]
        assert len(places) == 1
        assert places[0].side == Side.SELL

    def test_flatten_negative_inventory_buys(self) -> None:
        bids, asks = _simple_book()
        intent = flatten_intent(-10)
        msgs = compile(intent, bids, asks, (), {}, _default_cfg(), inventory_lots=-10)
        places = [m for m in msgs if isinstance(m, PlaceLimit)]
        assert len(places) == 1
        assert places[0].side == Side.BUY

    def test_flatten_zero_inventory_null(self) -> None:
        bids, asks = _simple_book()
        intent = flatten_intent(0)
        assert intent.is_null
        msgs = compile(intent, bids, asks, (), {}, _default_cfg())
        assert not any(isinstance(m, PlaceLimit) for m in msgs)

    def test_flatten_passive_at_touch(self) -> None:
        """Non-urgent flatten quotes at touch, not crossing."""
        bids, asks = _simple_book(best_bid=999, best_ask=1001)
        intent = flatten_intent(10)
        cfg = MotorConfig(size_budget_lots=10, flatten_urgent=False)
        msgs = compile(intent, bids, asks, (), {}, cfg, inventory_lots=10)
        places = [m for m in msgs if isinstance(m, PlaceLimit)]
        assert len(places) == 1
        # Selling to reduce positive inv: quote at the ask (passive)
        assert places[0].price_ticks == 1001

    def test_flatten_urgent_crosses(self) -> None:
        """Urgent flatten uses a crossing price."""
        bids, asks = _simple_book(best_bid=999, best_ask=1001)
        intent = flatten_intent(10)
        cfg = MotorConfig(size_budget_lots=10, flatten_urgent=True)
        msgs = compile(intent, bids, asks, (), {}, cfg, inventory_lots=10)
        places = [m for m in msgs if isinstance(m, PlaceLimit)]
        assert len(places) == 1
        # Selling urgently: hit the bid
        assert places[0].price_ticks == 999


# ===================================================================
# Tripwire re-runs
# ===================================================================

class TestTripwires:
    """Motor must not break any architectural tripwire."""

    def test_no_reward_tokens(self) -> None:
        from tests.tripwires.test_no_reward_tokens import test_no_reward_tokens
        test_no_reward_tokens()

    def test_no_learning_frameworks(self) -> None:
        from tests.tripwires.test_no_learning_frameworks import (
            test_no_static_imports_of_learning_frameworks,
        )
        test_no_static_imports_of_learning_frameworks()

    def test_metrics_isolation(self) -> None:
        from tests.tripwires.test_metrics_isolation import (
            _imported_module_candidates,
            _touches_metrics,
        )
        from pathlib import Path
        import topos
        motor_root = Path(topos.__file__).resolve().parent / "motor"
        for path in sorted(motor_root.rglob("*.py")):
            candidates = _imported_module_candidates(path)
            assert not _touches_metrics(candidates), (
                f"motor/ imports topos.metrics: {path}"
            )
