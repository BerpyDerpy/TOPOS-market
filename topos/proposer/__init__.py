"""Experiment/probe generation and EIG scoring (P8).

Generates candidate probes (including the null action, a first-class
candidate with its own EIG in an active market) and scores them by
MARGINAL expected information gain over null (INV-4). Receives
`SelfStateCognitive` only: no account-state fields exist on anything this
package sees (INV-5).

Two-stage menu: a standing coarse menu scored every cycle for every
hypothesis (headline input for P9 salience), then a refined grid around
the coarse winner for the hypothesis in focus. Selection is hard gates
plus a lexicographic order (marginal EIG, then minimum self-entropy
within an epsilon band) — never a scalarized trade-off.
"""

from topos.proposer.candidates import (
    Candidate,
    ProbeShape,
    Proposal,
    coarse_shapes,
    intent_for,
    null_intent,
    refined_shapes,
)
from topos.proposer.config import (
    DEEP_OFFSET_TICKS,
    EPSILON_EIG_NATS,
    GATE_DELTA,
    GATE_FORECAST_HORIZON_STEPS,
    REFINED_OFFSET_STEPS,
    REFINED_PATIENCE_GRID,
    REFINED_SIZE_FACTORS,
)
from topos.proposer.core import Proposer, SelectionRule
from topos.proposer.gates import (
    DistanceProjector,
    GateReport,
    book_from_summary,
    compiled_messages,
    evaluate_gates,
)
from topos.proposer.selection import select_candidate

__all__ = [
    "Candidate",
    "DEEP_OFFSET_TICKS",
    "DistanceProjector",
    "EPSILON_EIG_NATS",
    "GATE_DELTA",
    "GATE_FORECAST_HORIZON_STEPS",
    "GateReport",
    "ProbeShape",
    "Proposal",
    "Proposer",
    "REFINED_OFFSET_STEPS",
    "REFINED_PATIENCE_GRID",
    "REFINED_SIZE_FACTORS",
    "SelectionRule",
    "book_from_summary",
    "coarse_shapes",
    "compiled_messages",
    "evaluate_gates",
    "intent_for",
    "null_intent",
    "refined_shapes",
    "select_candidate",
]
