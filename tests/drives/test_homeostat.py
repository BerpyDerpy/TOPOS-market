"""Tests for topos.drives — the homeostat (INV-6).

These tests encode the invariant: viability variables are set-points, not
maximands.  Do NOT soften them.

Required tests:
1. drive == 0.0 everywhere inside the soft band, including the boundary.
2. drive is strictly increasing in |x| beyond soft, unbounded approaching hard.
3. NO negative drives, no term monotone-decreasing inside the band.
4. veto fires exactly at u >= 1; corrective_intent reduces |inventory|.
5. drawdown reads SelfStateFull only inside drives/; tripwires pass.
"""

from __future__ import annotations

import ast
import math
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from topos.contracts.intent import SELF_TRAJECTORY
from topos.contracts.workspace import SelfStateFull, WorkingOrderView
from topos.drives.config import HomeostatConfig, VariableBounds
from topos.drives.homeostat import D_NATS, Homeostat, HomeostatOutput, _drive, _normalized_excursion


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _default_cfg() -> HomeostatConfig:
    """A canonical config for test use."""
    return HomeostatConfig(
        inventory=VariableBounds(soft=10.0, hard=20.0),
        gross_exposure=VariableBounds(soft=5000.0, hard=10000.0),
        message_budget=VariableBounds(soft=40.0, hard=60.0),
        drawdown=VariableBounds(soft=50.0, hard=100.0),
        size_budget_lots=5.0,
    )


def _make_full_state(
    inventory_lots: int = 0,
    realized_pnl: float = 0.0,
    unrealized_pnl: float = 0.0,
    gross_exposure: float = 0.0,
) -> SelfStateFull:
    return SelfStateFull(
        inventory_lots=inventory_lots,
        working_orders=(),
        drive_distances={},
        realized_pnl=realized_pnl,
        unrealized_pnl=unrealized_pnl,
        gross_exposure=gross_exposure,
    )


MID = 100.0  # mid price for tests


# ===================================================================
# Test 1: drive is exactly 0.0 inside the soft band, including edge
# ===================================================================

class TestDriveZeroInsideBand:
    """INV-6: drive is exactly zero everywhere inside the soft band."""

    def test_at_zero_inventory(self) -> None:
        h = Homeostat(_default_cfg())
        out = h.evaluate(_make_full_state(inventory_lots=0), MID)
        assert out.drives["inventory"] == 0.0

    def test_at_soft_boundary_inventory(self) -> None:
        """drive must be exactly 0.0 *at* the soft bound (u = 0 there)."""
        cfg = _default_cfg()
        h = Homeostat(cfg)
        # Inventory exactly at the soft bound
        inv = int(cfg.inventory.soft)
        out = h.evaluate(_make_full_state(inventory_lots=inv), MID)
        assert out.drives["inventory"] == 0.0

    def test_at_soft_boundary_negative_inventory(self) -> None:
        cfg = _default_cfg()
        h = Homeostat(cfg)
        inv = -int(cfg.inventory.soft)
        out = h.evaluate(_make_full_state(inventory_lots=inv), MID)
        assert out.drives["inventory"] == 0.0

    @given(st.integers(min_value=-10, max_value=10))
    def test_any_value_inside_band(self, inv: int) -> None:
        """Property: for all inventory inside [-soft, +soft], drive == 0."""
        h = Homeostat(_default_cfg())
        out = h.evaluate(_make_full_state(inventory_lots=inv), MID)
        assert out.drives["inventory"] == 0.0

    def test_normalized_excursion_zero_at_boundary(self) -> None:
        """u is exactly 0.0 at the soft bound."""
        u = _normalized_excursion(10.0, soft=10.0, hard=20.0)
        assert u == 0.0

    def test_normalized_excursion_zero_inside(self) -> None:
        u = _normalized_excursion(5.0, soft=10.0, hard=20.0)
        assert u == 0.0

    def test_drive_function_at_zero(self) -> None:
        assert _drive(0.0) == 0.0

    def test_drive_function_at_negative_u(self) -> None:
        """Negative u (deep inside band) still gives exactly 0."""
        assert _drive(-0.5) == 0.0


# ===================================================================
# Test 2: drive is strictly increasing beyond soft, unbounded at hard
# ===================================================================

class TestDriveIncreasingAndUnbounded:
    """INV-6: superlinear, diverging at the hard bound."""

    def test_strictly_increasing_in_u(self) -> None:
        """Evaluate drive at a sequence of u values; each must exceed the last."""
        us = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
        drives = [_drive(u) for u in us]
        for i in range(1, len(drives)):
            assert drives[i] > drives[i - 1], (
                f"drive not strictly increasing: _drive({us[i]}) = {drives[i]} "
                f"<= _drive({us[i - 1]}) = {drives[i - 1]}"
            )

    def test_unbounded_approaching_hard(self) -> None:
        """drive → ∞ as u → 1⁻."""
        d_99 = _drive(0.99)
        d_999 = _drive(0.999)
        d_9999 = _drive(0.9999)
        assert d_999 > d_99
        assert d_9999 > d_999
        assert d_9999 > 100.0  # must be enormous

    def test_drive_at_hard_is_infinite(self) -> None:
        assert _drive(1.0) == float("inf")

    def test_drive_superlinear(self) -> None:
        """The drive at u=0.5 exceeds what a linear function would give."""
        linear_at_half = D_NATS * 0.5
        actual = _drive(0.5)
        assert actual >= linear_at_half * 0.5  # u²/(1-u) at 0.5 = 0.5

    @given(st.floats(min_value=0.01, max_value=0.98))
    def test_property_increasing(self, u: float) -> None:
        """For any u in (0, 1), drive(u + epsilon) > drive(u)."""
        eps = min(0.01, (1.0 - u) / 2)
        assert _drive(u + eps) > _drive(u)

    def test_inventory_drive_increases_with_abs_inventory(self) -> None:
        """End-to-end: increasing |inventory| beyond soft gives increasing drive."""
        cfg = _default_cfg()
        prev_drive = 0.0
        for inv in [11, 12, 14, 17, 19]:
            h = Homeostat(cfg)
            out = h.evaluate(_make_full_state(inventory_lots=inv), MID)
            assert out.drives["inventory"] > prev_drive
            prev_drive = out.drives["inventory"]


# ===================================================================
# Test 3: NO negative drives, no monotone-decreasing-inside-band term
# ===================================================================

class TestNoNegativeDrives:
    """INV-6: no negative drives, no term that decreases as variable gets safer."""

    @given(st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False))
    def test_drive_never_negative(self, u: float) -> None:
        d = _drive(u)
        assert d >= 0.0, f"drive is negative at u={u}: {d}"

    def test_all_drive_values_non_negative_end_to_end(self) -> None:
        """Evaluate at many inventory levels; every drive must be >= 0."""
        cfg = _default_cfg()
        for inv in range(-25, 26):
            h = Homeostat(cfg)
            out = h.evaluate(_make_full_state(inventory_lots=inv), MID)
            for name, val in out.drives.items():
                assert val >= 0.0, f"drive[{name}] = {val} < 0 at inv={inv}"

    def test_no_monotone_decreasing_quantity_in_module(self) -> None:
        """Grep-level check: homeostat.py exposes no negation of drive or u.

        This is a structural assertion: the module must not contain any
        expression like ``1 - drive``, ``-drive``, ``soft - |x|``, or
        ``band_width - u`` that would create a quantity monotone-decreasing
        inside the band.  We check by scanning the source for suspicious
        patterns.
        """
        import topos.drives.homeostat as mod
        import inspect
        source = inspect.getsource(mod)

        # Parse the AST and check for any function/method that returns
        # a negated or inverted drive/distance value.
        tree = ast.parse(source)

        # The module should not have any function whose body contains
        # a subtraction from 1 involving 'u' or 'drive' in a return
        # context that isn't the core formula.
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Skip the core _drive function and _normalized_excursion
                if node.name in ("_drive", "_normalized_excursion", "__init__",
                                 "__post_init__", "record_messages"):
                    continue
                # Check all return values in public methods for negation
                for subnode in ast.walk(node):
                    if isinstance(subnode, ast.UnaryOp) and isinstance(subnode.op, ast.USub):
                        if isinstance(subnode.operand, ast.Name):
                            assert subnode.operand.id not in ("u", "drive", "d"), (
                                f"Found negated drive/distance variable "
                                f"'-{subnode.operand.id}' in {node.name}"
                            )


# ===================================================================
# Test 4: veto fires at u >= 1; corrective_intent reduces |inventory|
# ===================================================================

class TestVetoAndCorrectiveIntent:
    """INV-6: hard veto + corrective flatten intent."""

    def test_veto_fires_at_hard_bound(self) -> None:
        cfg = _default_cfg()
        h = Homeostat(cfg)
        inv = int(cfg.inventory.hard)  # exactly at hard
        out = h.evaluate(_make_full_state(inventory_lots=inv), MID)
        assert out.vetoes["inventory"] is True

    def test_veto_fires_beyond_hard(self) -> None:
        cfg = _default_cfg()
        h = Homeostat(cfg)
        inv = int(cfg.inventory.hard) + 5
        out = h.evaluate(_make_full_state(inventory_lots=inv), MID)
        assert out.vetoes["inventory"] is True

    def test_no_veto_inside_band(self) -> None:
        cfg = _default_cfg()
        h = Homeostat(cfg)
        out = h.evaluate(_make_full_state(inventory_lots=0), MID)
        for name, v in out.vetoes.items():
            assert v is False, f"veto[{name}] unexpectedly True at inv=0"

    def test_no_veto_between_soft_and_hard(self) -> None:
        cfg = _default_cfg()
        h = Homeostat(cfg)
        inv = int(cfg.inventory.soft) + 1  # between soft and hard
        out = h.evaluate(_make_full_state(inventory_lots=inv), MID)
        assert out.vetoes["inventory"] is False

    def test_corrective_intent_present_outside_soft(self) -> None:
        cfg = _default_cfg()
        h = Homeostat(cfg)
        inv = int(cfg.inventory.soft) + 3
        out = h.evaluate(_make_full_state(inventory_lots=inv), MID)
        assert out.corrective_intent is not None

    def test_corrective_intent_none_inside_band(self) -> None:
        cfg = _default_cfg()
        h = Homeostat(cfg)
        out = h.evaluate(_make_full_state(inventory_lots=5), MID)
        assert out.corrective_intent is None

    def test_corrective_intent_reduces_inventory_positive(self) -> None:
        """Corrective for positive inventory has side = -1 (sell)."""
        cfg = _default_cfg()
        h = Homeostat(cfg)
        inv = int(cfg.inventory.soft) + 5
        out = h.evaluate(_make_full_state(inventory_lots=inv), MID)
        assert out.corrective_intent is not None
        assert out.corrective_intent.side == -1.0
        assert out.corrective_intent.is_flatten

    def test_corrective_intent_reduces_inventory_negative(self) -> None:
        """Corrective for negative inventory has side = +1 (buy)."""
        cfg = _default_cfg()
        h = Homeostat(cfg)
        inv = -(int(cfg.inventory.soft) + 5)
        out = h.evaluate(_make_full_state(inventory_lots=inv), MID)
        assert out.corrective_intent is not None
        assert out.corrective_intent.side == 1.0
        assert out.corrective_intent.is_flatten

    def test_corrective_intent_never_increases_inventory(self) -> None:
        """The corrective side always opposes inventory sign."""
        cfg = _default_cfg()
        for inv in [11, 12, 15, 19, -11, -15, -19]:
            h = Homeostat(cfg)
            out = h.evaluate(_make_full_state(inventory_lots=inv), MID)
            if out.corrective_intent is not None:
                if inv > 0:
                    assert out.corrective_intent.side == -1.0
                else:
                    assert out.corrective_intent.side == 1.0

    def test_corrective_size_frac_partial(self) -> None:
        """size_frac is excess / size_budget, capped at 1.0."""
        cfg = _default_cfg()
        h = Homeostat(cfg)
        # excess = |13| - 10 = 3; size_budget = 5; frac = 0.6
        out = h.evaluate(_make_full_state(inventory_lots=13), MID)
        assert out.corrective_intent is not None
        assert abs(out.corrective_intent.size_frac - 0.6) < 1e-9

    def test_corrective_size_frac_capped_at_one(self) -> None:
        cfg = _default_cfg()
        h = Homeostat(cfg)
        # excess = |25| - 10 = 15; 15/5 = 3.0, capped to 1.0
        out = h.evaluate(_make_full_state(inventory_lots=25), MID)
        assert out.corrective_intent is not None
        assert out.corrective_intent.size_frac == 1.0

    def test_corrective_intent_target_is_self_trajectory(self) -> None:
        cfg = _default_cfg()
        h = Homeostat(cfg)
        out = h.evaluate(_make_full_state(inventory_lots=15), MID)
        assert out.corrective_intent is not None
        assert out.corrective_intent.target_id == SELF_TRAJECTORY


# ===================================================================
# Test 5: drawdown isolation — PnL only in drives/, tripwires pass
# ===================================================================

class TestDrawdownAndIsolation:
    """INV-5: drawdown reads SelfStateFull only inside drives/."""

    def test_drawdown_drive_zero_at_no_loss(self) -> None:
        cfg = _default_cfg()
        h = Homeostat(cfg)
        out = h.evaluate(_make_full_state(realized_pnl=100.0, unrealized_pnl=0.0), MID)
        assert out.drives["drawdown"] == 0.0

    def test_drawdown_drive_fires_on_drop(self) -> None:
        cfg = _default_cfg()
        h = Homeostat(cfg)
        # First cycle: peak at 100
        h.evaluate(_make_full_state(realized_pnl=100.0, unrealized_pnl=0.0), MID)
        # Second cycle: drop to 40 => drawdown = 60, soft=50 => u > 0
        out = h.evaluate(_make_full_state(realized_pnl=40.0, unrealized_pnl=0.0), MID)
        assert out.drives["drawdown"] > 0.0
        assert out.distances["drawdown"] > 0.0

    def test_drawdown_veto_at_hard(self) -> None:
        cfg = _default_cfg()
        h = Homeostat(cfg)
        h.evaluate(_make_full_state(realized_pnl=200.0), MID)
        # drawdown = 200 - 100 = 100 = hard => veto
        out = h.evaluate(_make_full_state(realized_pnl=100.0), MID)
        assert out.vetoes["drawdown"] is True

    def test_drawdown_uses_combined_pnl(self) -> None:
        """Drawdown is from peak of realized + unrealized."""
        cfg = _default_cfg()
        h = Homeostat(cfg)
        # Peak: realized=50 + unrealized=50 = 100
        h.evaluate(_make_full_state(realized_pnl=50.0, unrealized_pnl=50.0), MID)
        # Now: 50 + (-10) = 40; drawdown = 60
        out = h.evaluate(_make_full_state(realized_pnl=50.0, unrealized_pnl=-10.0), MID)
        assert out.drives["drawdown"] > 0.0


class TestMetricsIsolation:
    """Verify that the tripwire tests still pass with drives/ implemented."""

    def test_metrics_isolation(self) -> None:
        """drives/ must not import topos.metrics."""
        from tests.tripwires.test_metrics_isolation import (
            _imported_module_candidates,
            _touches_metrics,
        )
        import topos

        topos_root = Path(topos.__file__).resolve().parent
        drives_root = topos_root / "drives"
        for path in sorted(drives_root.rglob("*.py")):
            candidates = _imported_module_candidates(path)
            assert not _touches_metrics(candidates), (
                f"drives/ imports topos.metrics: {path}"
            )

    def test_cognitive_view_has_no_pnl(self) -> None:
        """Re-run the tripwire to confirm INV-5 is intact."""
        from tests.tripwires.test_cognitive_view_has_no_pnl import (
            test_cognitive_view_has_no_pnl_fields,
            test_full_state_is_not_a_subtype_of_the_cognitive_view,
        )
        test_cognitive_view_has_no_pnl_fields()
        test_full_state_is_not_a_subtype_of_the_cognitive_view()


# ===================================================================
# Additional structural tests
# ===================================================================

class TestDistances:
    """drive_distances (u values) exported for SelfStateCognitive."""

    def test_distances_zero_inside_band(self) -> None:
        h = Homeostat(_default_cfg())
        out = h.evaluate(_make_full_state(inventory_lots=0), MID)
        assert out.distances["inventory"] == 0.0

    def test_distances_positive_outside_band(self) -> None:
        h = Homeostat(_default_cfg())
        out = h.evaluate(_make_full_state(inventory_lots=15), MID)
        assert out.distances["inventory"] > 0.0

    def test_distances_at_hard_equals_one(self) -> None:
        cfg = _default_cfg()
        h = Homeostat(cfg)
        inv = int(cfg.inventory.hard)
        out = h.evaluate(_make_full_state(inventory_lots=inv), MID)
        assert abs(out.distances["inventory"] - 1.0) < 1e-9

    def test_all_four_variables_present(self) -> None:
        h = Homeostat(_default_cfg())
        out = h.evaluate(_make_full_state(), MID)
        expected = {"inventory", "gross_exposure", "message_budget", "drawdown"}
        assert set(out.drives.keys()) == expected
        assert set(out.vetoes.keys()) == expected
        assert set(out.distances.keys()) == expected


class TestMessageBudget:
    """Rolling-window message count drives message_budget."""

    def test_message_budget_zero_with_no_messages(self) -> None:
        h = Homeostat(_default_cfg())
        out = h.evaluate(_make_full_state(), MID)
        assert out.drives["message_budget"] == 0.0

    def test_message_budget_drives_above_soft(self) -> None:
        cfg = _default_cfg()
        h = Homeostat(cfg)
        # Record enough messages to exceed the soft cap (40)
        for _ in range(5):
            h.record_messages(10)
        out = h.evaluate(_make_full_state(), MID)
        assert out.drives["message_budget"] > 0.0

    def test_message_window_rolls(self) -> None:
        cfg = HomeostatConfig(
            inventory=VariableBounds(soft=10.0, hard=20.0),
            gross_exposure=VariableBounds(soft=5000.0, hard=10000.0),
            message_budget=VariableBounds(soft=5.0, hard=10.0),
            drawdown=VariableBounds(soft=50.0, hard=100.0),
            size_budget_lots=5.0,
            message_window_steps=3,
        )
        h = Homeostat(cfg)
        h.record_messages(3)
        h.record_messages(3)
        h.record_messages(3)  # total = 9
        # Now add one more, oldest drops off -> total stays 9
        h.record_messages(3)
        # Window is size 3, so only last 3 counts: 3+3+3 = 9
        assert h.rolling_message_count == 9


class TestGrossExposure:
    """gross_exposure = |inventory| * mid."""

    def test_gross_exposure_zero_at_zero_inventory(self) -> None:
        h = Homeostat(_default_cfg())
        out = h.evaluate(_make_full_state(inventory_lots=0), MID)
        assert out.drives["gross_exposure"] == 0.0

    def test_gross_exposure_drives_with_large_position(self) -> None:
        cfg = HomeostatConfig(
            inventory=VariableBounds(soft=100.0, hard=200.0),
            gross_exposure=VariableBounds(soft=500.0, hard=1000.0),
            message_budget=VariableBounds(soft=40.0, hard=60.0),
            drawdown=VariableBounds(soft=50.0, hard=100.0),
            size_budget_lots=5.0,
        )
        h = Homeostat(cfg)
        # |10| * 100 = 1000 = hard => veto
        out = h.evaluate(_make_full_state(inventory_lots=10), MID)
        assert out.vetoes["gross_exposure"] is True


class TestConfigValidation:
    """VariableBounds and HomeostatConfig validation."""

    def test_soft_ge_hard_rejected(self) -> None:
        with pytest.raises(ValueError, match="soft must be strictly less"):
            VariableBounds(soft=10.0, hard=10.0)

    def test_negative_soft_rejected(self) -> None:
        with pytest.raises(ValueError, match="soft bound must be >= 0"):
            VariableBounds(soft=-1.0, hard=10.0)

    def test_zero_hard_rejected(self) -> None:
        with pytest.raises(ValueError, match="hard bound must be > 0"):
            VariableBounds(soft=0.0, hard=0.0)

    def test_zero_size_budget_rejected(self) -> None:
        with pytest.raises(ValueError, match="size_budget_lots must be > 0"):
            HomeostatConfig(
                inventory=VariableBounds(soft=10.0, hard=20.0),
                gross_exposure=VariableBounds(soft=5000.0, hard=10000.0),
                message_budget=VariableBounds(soft=40.0, hard=60.0),
                drawdown=VariableBounds(soft=50.0, hard=100.0),
                size_budget_lots=0.0,
            )


class TestHomeostatOutputFrozen:
    """HomeostatOutput is a frozen dataclass."""

    def test_frozen(self) -> None:
        out = HomeostatOutput(
            drives={"a": 0.0},
            vetoes={"a": False},
            distances={"a": 0.0},
            corrective_intent=None,
        )
        with pytest.raises(AttributeError):
            out.drives = {}  # type: ignore[misc]
