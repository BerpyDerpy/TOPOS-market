"""Self-model (P6).

Bookkeeping from acks/fills, Beta-Bernoulli fill model, Bayesian-linear
impact model, inventory-trajectory forecast, and reflexive self-uncertainty.
The agent is part of the market it models: this package makes the agent's
own trading predictable, hence boring.

Two of the design's three anti-churn layers live here: the fill and impact
models condition on the agent's OWN action and context (so own outcomes
become learnable and their EIG saturates), and the trajectory compiler
turns the same posteriors into the reflexive self-entropy the proposer
scores. Adaptation is closed-form conjugate updates plus forgetting
(INV-2); curiosity quantities are parameter-posterior mutual information
(INV-3); the cognitive view of self-state carries no PnL (INV-5); nothing
here observes engine-side account state (INV-11).
"""

from topos.selfmodel.books import BookkeepingRecord, Books
from topos.selfmodel.common import (
    IMBALANCE_BANDS,
    OFFSET_BANDS,
    BookContext,
    ImpliedOrder,
    LedgerOrder,
    OwnOrderLedger,
    context_from_observation,
    imbalance_band_of,
    implied_order,
    offset_band_of,
    paired_placements,
)
from topos.selfmodel.fill_model import FillModel
from topos.selfmodel.impact_model import ImpactModel
from topos.selfmodel.self_trajectory import (
    UNIT_CELL_VAR,
    SelfTrajectory,
    TrajectoryForecast,
)

__all__ = [
    "IMBALANCE_BANDS",
    "OFFSET_BANDS",
    "UNIT_CELL_VAR",
    "BookContext",
    "BookkeepingRecord",
    "Books",
    "FillModel",
    "ImpactModel",
    "ImpliedOrder",
    "LedgerOrder",
    "OwnOrderLedger",
    "SelfTrajectory",
    "TrajectoryForecast",
    "context_from_observation",
    "imbalance_band_of",
    "implied_order",
    "offset_band_of",
    "paired_placements",
]
