"""Agent wiring configuration (P12).

Every value here is a WIRING constant — a horizon, a window, a band — fixed
before the market opens and never tuned against run outcomes. The bands and
budgets are the homeostat's viability constraints (INV-6: set-points, not
maximands); the horizons are the hypotheses' own definitions ("filled
within H steps" IS what the fill posterior answers), threaded through so
that every consumer of a posterior asks it the one question it answers
without extrapolation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from topos.beliefs.regime import RegimeConfig
from topos.drives.config import HomeostatConfig, VariableBounds
from topos.motor.config import MotorConfig

_DEFAULT_MOTOR = MotorConfig(size_budget_lots=4)

_DEFAULT_HOMEOSTAT = HomeostatConfig(
    # Inventory band: soft at 2.5x / hard at 5x the per-step size budget —
    # a handful of unshed probe fills is tolerable, a persistent pile is not.
    inventory=VariableBounds(soft=10.0, hard=20.0),
    # Gross exposure: the inventory band expressed in tick-lots at the
    # background market's price scale (mid ~1000 ticks).
    gross_exposure=VariableBounds(soft=15_000.0, hard=30_000.0),
    # Message budget: at one message per engine step (the exchange
    # interface's cap), soft at half the rolling window, hard at the full
    # window — sustained max-rate messaging is exactly what the budget
    # exists to extinguish.
    message_budget=VariableBounds(soft=10.0, hard=20.0),
    # Drawdown buffer in tick-lots, on the same price scale as gross
    # exposure: soft at ~3 ticks of adverse move on a soft-band inventory.
    drawdown=VariableBounds(soft=30.0, hard=60.0),
    size_budget_lots=float(_DEFAULT_MOTOR.size_budget_lots),
    message_window_steps=20,
)

_DEFAULT_REGIME = RegimeConfig(hazard=0.02)
"""Constant-hazard prior on regime segment length: mean 50 slow ticks."""


@dataclass(frozen=True)
class AgentConfig:
    """Construction-time wiring for the integrated agent.

    ``fill_horizon_steps`` is BOTH the fill model's trial horizon and the
    single probe horizon every ProbeSpec carries (DESIGN item 22: the only
    horizon the fill posteriors answer without extrapolation; marginals
    compare candidate vs null at the same horizon, so it cancels out of
    every comparison). ``slow_tick_every_steps`` is M: the regime tracker
    consumes one summary vector and regime-gated forgetting is applied
    every M engine steps. ``vol_window_steps`` sizes the public
    realized-vol window feeding WorldSummary and the regime tracker.
    """

    motor: MotorConfig = _DEFAULT_MOTOR
    homeostat: HomeostatConfig = _DEFAULT_HOMEOSTAT
    regime: RegimeConfig = field(default=_DEFAULT_REGIME)
    fill_horizon_steps: int = 2
    impact_horizon_steps: int = 1
    slow_tick_every_steps: int = 20
    vol_window_steps: int = 20

    def __post_init__(self) -> None:
        if self.fill_horizon_steps < 1:
            raise ValueError(
                f"fill_horizon_steps must be >= 1, got {self.fill_horizon_steps}"
            )
        if self.impact_horizon_steps < 1:
            raise ValueError(
                "impact_horizon_steps must be >= 1, "
                f"got {self.impact_horizon_steps}"
            )
        if self.slow_tick_every_steps < 1:
            raise ValueError(
                "slow_tick_every_steps must be >= 1, "
                f"got {self.slow_tick_every_steps}"
            )
        if self.vol_window_steps < 2:
            raise ValueError(
                f"vol_window_steps must be >= 2, got {self.vol_window_steps}"
            )
        if self.homeostat.size_budget_lots != float(self.motor.size_budget_lots):
            raise ValueError(
                "homeostat and motor must share one per-step size budget "
                f"(the meaning of Intent.size_frac): homeostat has "
                f"{self.homeostat.size_budget_lots}, motor has "
                f"{self.motor.size_budget_lots}"
            )
