"""The salience competition: one arena, two kinds of bidder.

Hypotheses bid ``s_h = w_h * best_marginal_eig_h * (1 + GAMMA * max(0,
surprise_z_h))``:

* ``w_h`` is the structural consequence-weight from
  ``ModuleRegistry.centrality_weights()``, computed once at startup from
  declared reads/writes (INV-7). It is a bug — the project's most
  philosophically expensive bug — for ``w_h`` to depend on fill counts,
  run outcomes, or any empirical statistic whatsoever: attention would
  become an optimization channel and a de-facto value signal would enter
  through the back door. The ``Workspace`` re-asserts weight identity
  every cycle.
* ``best_marginal_eig_h`` is the proposer's coarse-menu output, in nats —
  already cross-hypothesis commensurable, so no per-hypothesis scaling is
  ever needed (or permitted) here.
* surprise multiplies EIG: it can amplify an answerable question, never
  create salience where there is no reducible uncertainty. Zero marginal
  EIG means zero salience no matter how surprising the errors — surprise
  attends, EIG acts.

Homeostat drives bid ``s_k = drive_k`` directly: P7's architecture
constant D already sets the exchange rate between drive units and nats,
and rescaling it here would re-tune what P7 fixed. Inside the soft band a
drive is exactly 0 and bids nothing; approaching the hard bound it
diverges and preempts any finite EIG (INV-6).

Focus = argmax salience, subject to ignition: no bid strictly above
``S_MIN`` means no focus at all.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

from topos.contracts.workspace import Focus

from topos.workspace.config import GAMMA, S_MIN


def hypothesis_salience(
    weight: float,
    best_marginal_eig_nats: float,
    surprise_z: float,
    *,
    gamma: float = GAMMA,
) -> float:
    """``w_h * best_marginal_eig_h * (1 + gamma * max(0, surprise_z_h))``."""
    if weight < 0.0:
        raise ValueError(f"consequence-weight must be >= 0, got {weight}")
    if best_marginal_eig_nats < 0.0:
        raise ValueError(
            f"best marginal EIG must be >= 0, got {best_marginal_eig_nats}"
        )
    return weight * best_marginal_eig_nats * (1.0 + gamma * max(0.0, surprise_z))


@dataclass(frozen=True)
class SalienceBid:
    """One bidder in the competition: a hypothesis or a homeostat drive."""

    bid_id: str
    salience: float
    is_homeostatic: bool

    def __post_init__(self) -> None:
        if math.isnan(self.salience):
            raise ValueError(f"salience bid {self.bid_id!r} is NaN")
        if self.salience < 0.0:
            raise ValueError(
                f"salience bid {self.bid_id!r} is negative: {self.salience}"
            )


def compete(bids: Iterable[SalienceBid], *, s_min: float = S_MIN) -> Focus | None:
    """Argmax salience with an ignition threshold.

    Returns None — no focus, a quiet mind — unless some bid STRICTLY
    exceeds ``s_min``. Ties are broken deterministically: homeostatic
    bids first (viability outranks curiosity at equal urgency), then
    lexicographic id.
    """
    ranked = sorted(
        bids, key=lambda bid: (-bid.salience, not bid.is_homeostatic, bid.bid_id)
    )
    if not ranked:
        return None
    top = ranked[0]
    if not top.salience > s_min:
        return None
    return Focus(
        hypothesis_id=top.bid_id,
        salience=top.salience,
        is_homeostatic=top.is_homeostatic,
    )
