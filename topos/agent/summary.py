"""WorldSummary assembly from public observations only.

Everything here is computable by any market participant from the visible
book and tape: mid/spread from the best quotes, imbalance over the full
visible window, per-level depth, trade tempo (lots printed this step), and
realized volatility (root mean square of one-step mid changes over a fixed
trailing window). No own-order or account quantity enters (those belong to
the self-state, INV-5), and nothing harness-only is reachable (INV-11).

The same four scalars the summary carries for the workspace — trade tempo,
realized vol, imbalance, mean depth — are what the slow loop feeds the
regime tracker (its declared observation dimensions, in that order).
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

from topos.contracts.market import N_LEVELS, Observation
from topos.contracts.workspace import WorldSummary
from topos.selfmodel.common import context_from_observation


@dataclass(frozen=True)
class SlowStats:
    """The regime tracker's four public summary statistics (its STAT_NAMES
    order: trade_tempo, realized_vol, imbalance, mean_depth)."""

    trade_tempo: float
    realized_vol: float
    imbalance: float
    mean_depth: float


class WorldSummaryTracker:
    """Folds one observation per cycle; answers with this cycle's summary.

    ``build`` returns None until a two-sided book has been seen at least
    once (pre-market there is no mid to anchor a summary, a probe price,
    or a mark); after that, a momentarily one-sided book carries the last
    known mid and spread forward rather than fabricating a price.
    """

    def __init__(self, vol_window_steps: int) -> None:
        if vol_window_steps < 2:
            raise ValueError(
                f"vol_window_steps must be >= 2, got {vol_window_steps}"
            )
        self._mids: deque[float] = deque(maxlen=vol_window_steps + 1)
        self._mid: float | None = None
        self._spread: int = 1
        self._imbalance: float = 0.0
        self._depth: tuple[float, ...] = (0.0,) * N_LEVELS
        self._tempo: float = 0.0

    def fold(self, obs: Observation) -> None:
        """Absorb one observation's public state."""
        ctx = context_from_observation(obs)
        if ctx.best_bid is not None and ctx.best_ask is not None:
            self._mid = 0.5 * (ctx.best_bid + ctx.best_ask)
            self._spread = ctx.best_ask - ctx.best_bid
            self._mids.append(self._mid)
        self._imbalance = ctx.imbalance
        self._depth = tuple(
            0.5 * (bid.size_lots + ask.size_lots)
            for bid, ask in zip(obs.bids, obs.asks)
        )
        self._tempo = float(sum(trade.size_lots for trade in obs.trades))

    @property
    def mid(self) -> float | None:
        """Mid of the best quotes, carried forward while one-sided; None
        until a two-sided book has ever been seen."""
        return self._mid

    def realized_vol(self) -> float:
        """RMS of one-step mid changes over the trailing window."""
        if len(self._mids) < 2:
            return 0.0
        mids = list(self._mids)
        diffs = [b - a for a, b in zip(mids[:-1], mids[1:])]
        return math.sqrt(sum(d * d for d in diffs) / len(diffs))

    def slow_stats(self) -> SlowStats:
        """The four public statistics the regime tracker consumes."""
        return SlowStats(
            trade_tempo=self._tempo,
            realized_vol=self.realized_vol(),
            imbalance=self._imbalance,
            mean_depth=sum(self._depth) / len(self._depth),
        )

    def build(self, regime_posterior: tuple[float, ...]) -> WorldSummary | None:
        """This cycle's WorldSummary, or None while pre-market (no mid yet)."""
        if self._mid is None:
            return None
        return WorldSummary(
            mid_ticks=self._mid,
            spread_ticks=self._spread,
            imbalance=self._imbalance,
            depth_profile=self._depth,
            trade_tempo=self._tempo,
            realized_vol=self.realized_vol(),
            regime_posterior=regime_posterior,
        )
