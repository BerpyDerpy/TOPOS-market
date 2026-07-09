"""Homeostat configuration: soft and hard bounds for each viability variable.

All bounds are positive magnitudes.  The homeostat evaluates signed or
unsigned excursion depending on the variable:

- inventory_lots:  signed excursion, symmetric band [-bound, +bound]
- gross_exposure:  unsigned, [0, bound]
- message_budget:  unsigned, count in rolling window vs cap
- drawdown_buffer: unsigned, running-peak PnL drop vs limit

Nothing here is ever tuned against run outcomes (INV-6).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VariableBounds:
    """Soft and hard bounds for one viability variable.

    The soft bound defines the edge of the zero-drive zone; the hard bound
    is where the veto fires.  Both are positive magnitudes.
    """

    soft: float
    hard: float

    def __post_init__(self) -> None:
        if self.soft < 0:
            raise ValueError(f"soft bound must be >= 0, got {self.soft}")
        if self.hard <= 0:
            raise ValueError(f"hard bound must be > 0, got {self.hard}")
        if self.soft >= self.hard:
            raise ValueError(
                f"soft must be strictly less than hard, got soft={self.soft} "
                f"hard={self.hard}"
            )


@dataclass(frozen=True)
class HomeostatConfig:
    """Bounds for every viability variable, plus the size budget.

    `size_budget_lots` is the per-step size budget (lots). The corrective
    intent's `size_frac` = min(excess_over_soft / size_budget_lots, 1.0),
    producing partial corrections that shed the excess across cycles.

    `message_window_steps` is the number of trailing steps whose sent-message
    counts are summed to form the rolling-window message count.
    """

    inventory: VariableBounds
    gross_exposure: VariableBounds
    message_budget: VariableBounds
    drawdown: VariableBounds
    size_budget_lots: float
    message_window_steps: int = 20

    def __post_init__(self) -> None:
        if self.size_budget_lots <= 0:
            raise ValueError(
                f"size_budget_lots must be > 0, got {self.size_budget_lots}"
            )
        if self.message_window_steps <= 0:
            raise ValueError(
                f"message_window_steps must be > 0, got {self.message_window_steps}"
            )
