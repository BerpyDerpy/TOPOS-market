"""Homeostat: viability set-points that generate competing drives and hard vetoes.

INV-6 is the entire point of this module.  Homeostat variables are
**constraints, not objectives**: nothing in the architecture gets "better"
by being far inside a band.

For each variable *k* with normalized excursion
``u = (|x| - soft) / (hard - soft)``:

* **drive_k = 0**  exactly, for all x inside the soft band (including the
  boundary).  There is no term that decreases as the variable gets safer —
  verified by test.
* **drive_k = D · u² / (1 − u)**  for u ∈ (0, 1): superlinear, diverging at
  the hard bound.
* **veto_k = True**  at u ≥ 1: exported as a flag; enforcement lives in the
  motor compiler.

Additionally ``drive_distances`` (the *u* values, clipped to [0, ∞)) are
exported for ``SelfStateCognitive``.

The corrective intent is built with ``contracts.intent.flatten_intent`` and
sized so that partial corrections shed the excess over the soft band across
cycles (re-evaluated every cycle).

This is the **only** agent package that reads ``SelfStateFull`` (INV-5).
PnL enters solely as drawdown distance-to-bound: realized + unrealized
drop from running peak vs limit.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from topos.contracts.intent import Intent, flatten_intent
from topos.contracts.workspace import SelfStateFull
from topos.drives.config import HomeostatConfig

# ---------------------------------------------------------------------------
# Architecture constant D
# ---------------------------------------------------------------------------
# D scales the drive magnitude relative to typical EIG magnitudes.  In an
# active market, a well-tuned belief module's best marginal EIG is on the
# order of 0.01–0.1 nats per cycle.  The homeostat should be *ignorable*
# deep inside the soft band (drive = 0 there anyway) and should dominate
# the salience competition only when excursion approaches the hard bound.
#
# At u = 0.5 (halfway between soft and hard):
#   drive = D · 0.25 / 0.5 = 0.5 · D
#
# Setting D = 1.0 nat means the drive at u = 0.5 is 0.5 nats — comfortably
# above typical EIG bids, producing a forceful but not yet pre-emptive
# homeostatic signal.  At u = 0.9 the drive is 1.0 · 0.81 / 0.1 = 8.1 nats,
# absolutely dominating the workspace.  This scaling is fixed by architecture,
# NEVER tuned against run outcomes (INV-6).
D_NATS: float = 1.0
"""Drive scale constant (nats).

Fixed architectural constant, not a tuning knob.  Justification:
typical marginal-EIG bids are O(0.01–0.1) nats; D = 1.0 makes the
drive ignorable at small excursions (u ≪ 1) and overwhelming as the
hard bound approaches (drive → ∞ as u → 1⁻).
"""


def _normalized_excursion(value: float, soft: float, hard: float) -> float:
    """Compute u = (|value| − soft) / (hard − soft), clipped to [0, ∞).

    Returns 0.0 when |value| is at or inside the soft band.
    """
    span = hard - soft
    if span <= 0:
        # Degenerate config (should be caught by VariableBounds validation).
        return 0.0
    raw = (abs(value) - soft) / span
    return max(0.0, raw)


def _drive(u: float) -> float:
    """D · u² / (1 − u)  for u ∈ [0, 1), 0 at u = 0, inf-capped at u ≥ 1.

    Returns exactly 0.0 when u <= 0.0 (inside or on the soft band).
    """
    if u <= 0.0:
        return 0.0
    if u >= 1.0:
        return float("inf")
    return D_NATS * (u * u) / (1.0 - u)


@dataclass(frozen=True)
class HomeostatOutput:
    """Result of one homeostat evaluation cycle.

    * ``drives``  — Mapping[str, float]: drive magnitude per variable,
      exactly 0.0 inside the soft band, superlinear beyond.
    * ``vetoes``  — Mapping[str, bool]: True when excursion ≥ hard bound.
    * ``distances`` — Mapping[str, float]: normalized excursion *u* per
      variable, clipped to [0, ∞).  Exported into ``SelfStateCognitive``
      so the workspace can see distance-to-bound without PnL exposure.
    * ``corrective_intent`` — Intent | None: a flatten intent sized to
      shed the excess over the soft band, or None when no correction is
      needed.
    """

    drives: Mapping[str, float]
    vetoes: Mapping[str, bool]
    distances: Mapping[str, float]
    corrective_intent: Intent | None


class Homeostat:
    """Viability regulator: band constraints, not objectives (INV-6).

    Constructed once with a ``HomeostatConfig``.  Call ``evaluate`` every
    cycle with the current ``SelfStateFull`` and the mid price; consume the
    returned ``HomeostatOutput``.

    This class also accumulates the rolling-window message count: callers
    must invoke ``record_messages(count)`` at the end of each cycle.
    """

    def __init__(self, cfg: HomeostatConfig) -> None:
        self._cfg = cfg
        self._peak_total_pnl: float | None = None
        # Rolling window of per-step message counts.
        self._message_counts: deque[int] = deque(
            maxlen=cfg.message_window_steps
        )

    # ------------------------------------------------------------------
    # External state feeds
    # ------------------------------------------------------------------

    def record_messages(self, count: int) -> None:
        """Record how many messages were sent this cycle (for message_budget)."""
        self._message_counts.append(count)

    @property
    def rolling_message_count(self) -> int:
        """Total messages in the current rolling window."""
        return sum(self._message_counts)

    # ------------------------------------------------------------------
    # Core evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self, self_state_full: SelfStateFull, mid: float
    ) -> HomeostatOutput:
        """Evaluate all viability variables and produce drives + vetoes.

        Parameters
        ----------
        self_state_full:
            The **full** self-state including PnL fields (INV-5: this is
            the ONLY place PnL is read, and only as distance-to-bound).
        mid:
            Current mid price (ticks as float).
        """
        cfg = self._cfg

        # --- inventory ---
        inv = float(self_state_full.inventory_lots)
        u_inv = _normalized_excursion(inv, cfg.inventory.soft, cfg.inventory.hard)

        # --- gross exposure ---
        gross = abs(self_state_full.inventory_lots) * mid
        u_gross = _normalized_excursion(
            gross, cfg.gross_exposure.soft, cfg.gross_exposure.hard
        )

        # --- message budget ---
        msg_count = float(self.rolling_message_count)
        u_msg = _normalized_excursion(
            msg_count, cfg.message_budget.soft, cfg.message_budget.hard
        )

        # --- drawdown buffer ---
        total_pnl = (
            self_state_full.realized_pnl + self_state_full.unrealized_pnl
        )
        if self._peak_total_pnl is None:
            self._peak_total_pnl = total_pnl
        else:
            self._peak_total_pnl = max(self._peak_total_pnl, total_pnl)
        drawdown = max(0.0, self._peak_total_pnl - total_pnl)
        u_dd = _normalized_excursion(
            drawdown, cfg.drawdown.soft, cfg.drawdown.hard
        )

        # --- assemble per-variable results ---
        names = ("inventory", "gross_exposure", "message_budget", "drawdown")
        us = (u_inv, u_gross, u_msg, u_dd)

        drives: dict[str, float] = {}
        vetoes: dict[str, bool] = {}
        distances: dict[str, float] = {}
        for name, u in zip(names, us, strict=True):
            drives[name] = _drive(u)
            vetoes[name] = u >= 1.0
            distances[name] = u

        # --- corrective intent ---
        corrective: Intent | None = None
        # Only emit a corrective intent when inventory exceeds its soft band.
        inv_excess = abs(inv) - cfg.inventory.soft
        if inv_excess > 0:
            frac = min(inv_excess / cfg.size_budget_lots, 1.0)
            corrective = flatten_intent(
                self_state_full.inventory_lots, size_frac=frac
            )

        return HomeostatOutput(
            drives=MappingProxyType(drives),
            vetoes=MappingProxyType(vetoes),
            distances=MappingProxyType(distances),
            corrective_intent=corrective,
        )
