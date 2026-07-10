"""Ablation experiment configurations.

Fairness rules, mechanically honored here:

* The seed lists are FIXED literals. Nothing may add, drop, or reorder
  seeds based on results — no cherry-picking.
* Every condition of one scale runs the identical ``RunConfig`` (same
  n_steps, same background market, same regime schedule) under the
  identical seed list; conditions differ ONLY through the P12 ablation
  flags (``topos.metrics.CONDITION_FLAGS``).
* The environment knobs below (low regime hazard, scheduled switches)
  are measurement design — they buy long, comparable regime segments for
  the decay/reawakening statistics. They are environment configuration,
  never agent constants: no agent parameter is set anywhere in this
  package (see DESIGN.md, Open questions, P13).
"""

from __future__ import annotations

from dataclasses import dataclass

from topos.env.background import BackgroundConfig, RegimeParams

SCHEDULE_HAZARD = 0.002
"""Residual per-step hazard under scheduled switching: low enough that
the fixed schedule dominates segment structure (mean spontaneous dwell
500 steps), nonzero so hazard switching stays exercised; segments are
always measured from the ground-truth regime log either way."""

ABLATION_REGIMES: tuple[RegimeParams, ...] = (
    RegimeParams(
        regime_id="calm",
        limit_rate=6.0,
        market_rate=1.2,
        cancel_rate=3.0,
        imbalance=0.0,
        mm_spread_ticks=4,
        hazard=SCHEDULE_HAZARD,
    ),
    RegimeParams(
        regime_id="stressed",
        limit_rate=9.0,
        market_rate=4.0,
        cancel_rate=5.0,
        imbalance=-0.3,
        mm_spread_ticks=10,
        hazard=SCHEDULE_HAZARD,
    ),
)
"""The P2 default regime pair with the hazard lowered per above."""


def _schedule(n_steps: int) -> tuple[tuple[int, str], ...]:
    """Two scheduled switches at thirds: calm -> stressed -> calm."""
    return ((n_steps // 3, "stressed"), (2 * n_steps // 3, "calm"))


def ablation_background(n_steps: int) -> BackgroundConfig:
    return BackgroundConfig(
        regimes=ABLATION_REGIMES,
        initial_regime_id="calm",
        schedule=_schedule(n_steps),
    )


@dataclass(frozen=True)
class AblationScale:
    """One named size of the ablation experiment."""

    name: str
    n_steps: int
    seeds: tuple[int, ...]

    @property
    def background(self) -> BackgroundConfig:
        return ablation_background(self.n_steps)

    @property
    def schedule(self) -> tuple[tuple[int, str], ...]:
        return _schedule(self.n_steps)


SMALL_SEEDS: tuple[int, ...] = tuple(range(101, 113))  # 12 seeds
FULL_SEEDS: tuple[int, ...] = tuple(range(101, 121))  # 20 seeds

SMALL = AblationScale(name="small", n_steps=510, seeds=SMALL_SEEDS)
"""CI scale: 3 regime segments of 170 steps each (long enough for the
decay estimator after the budget-window trim), 12 seeds, 5 conditions.
Longer runs would add little: the P13 headline finding (see DESIGN.md)
is that behavior goes drive-locked within a few hundred steps, so the
informative dynamics live early and power comes from seeds."""

FULL_SCALE = AblationScale(name="full", n_steps=3000, seeds=FULL_SEEDS)
"""The real experiment: >= 20 seeds per the design brief."""

SCALES: dict[str, AblationScale] = {
    SMALL.name: SMALL,
    FULL_SCALE.name: FULL_SCALE,
}
