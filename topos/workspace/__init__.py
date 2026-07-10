"""Blackboard, salience competition, arbiter, broadcast (P9).

The bounded, typed workspace. Salience consequence-weights derive from the
module dependency graph (registry centrality), computed once at startup —
never from outcome statistics of any kind (INV-7). The `WorkspaceRecord`
logged each cycle IS the interpretability story. Arbitration receives
`SelfStateCognitive` only (INV-5).
"""

from topos.workspace.broadcast import (
    FocusConsumer,
    broadcast_focus,
    validate_consumers,
)
from topos.workspace.config import GAMMA, K_HEADLINES, S_MIN
from topos.workspace.core import (
    CoalitionError,
    WeightsIntegrityError,
    Workspace,
)
from topos.workspace.salience import SalienceBid, compete, hypothesis_salience

__all__ = [
    "CoalitionError",
    "FocusConsumer",
    "GAMMA",
    "K_HEADLINES",
    "S_MIN",
    "SalienceBid",
    "WeightsIntegrityError",
    "Workspace",
    "broadcast_focus",
    "compete",
    "hypothesis_salience",
    "validate_consumers",
]
