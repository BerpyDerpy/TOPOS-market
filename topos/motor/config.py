"""Motor compiler configuration.

Every parameter here is a declared input to the pure ``compile`` function
(INV-8): no hidden state, no randomness, no policy beyond what cfg exposes.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MotorConfig:
    """All knobs the motor compiler is allowed to inspect.

    Parameters
    ----------
    size_budget_lots:
        Maximum order size in lots per compile call.
    max_patience_steps:
        Ceiling on the staleness window.  ``patience == 0`` maps to 1 step
        (immediate cancel-replace); ``patience == 1`` maps to this value
        (never cancel-replace on staleness alone).
    flatten_urgent:
        If True, the flatten path is allowed to emit crossing (marketable)
        limits.  If False, flatten quotes passively at the touch only.
    """

    size_budget_lots: int
    max_patience_steps: int = 50
    flatten_urgent: bool = False

    def __post_init__(self) -> None:
        if self.size_budget_lots <= 0:
            raise ValueError(
                f"size_budget_lots must be > 0, got {self.size_budget_lots}"
            )
        if self.max_patience_steps < 1:
            raise ValueError(
                f"max_patience_steps must be >= 1, got {self.max_patience_steps}"
            )

    def patience_steps(self, patience: float) -> int:
        """Map patience ∈ [0, 1] to a staleness window in steps.

        ``patience == 0`` ⇒ 1 step (replace immediately when price differs).
        ``patience == 1`` ⇒ ``max_patience_steps`` (never stale).

        The mapping is linear: ``1 + round(patience * (max - 1))``.
        Monotone increasing as required by spec.
        """
        return 1 + round(patience * (self.max_patience_steps - 1))
