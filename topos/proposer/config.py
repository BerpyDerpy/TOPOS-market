"""Architecture constants for the proposer (P8).

Every constant here is fixed by STRUCTURE — a grid resolution the
architecture already lives on, or a convention already pinned elsewhere in
the codebase — with the justification recorded on the constant itself.
None of them is ever tuned against run outcomes: there is no coefficient
anywhere in this package that trades information against anything else
(the selection rule is hard gates plus a lexicographic order, see
``topos.proposer.selection``).
"""

from __future__ import annotations

from typing import Final

EPSILON_EIG_NATS: Final[float] = 0.02
"""Width of the "epistemically indistinguishable" band in the
lexicographic selection rule: candidates whose marginal EIGs are within
this many nats of the best are treated as informationally tied, and the
tie is broken by MINIMUM self-entropy.

Structural justification: the saturation tripwires already pin
EIG < 0.02 nats as "nothing left to learn" (tests/beliefs/
test_eig_saturation.py and tests/selfmodel/test_fill_model.py both assert
convergence below exactly this level) — it is the architecture's own
boredom scale, reused rather than invented. Two candidates separated by
less than the scale the architecture treats as exhausted cannot be
meaningfully ranked by information, so the comparison falls through to
self-predictability. Because conjugate EIGs approach 0 asymptotically
without ever reaching it, this band is also what lets the null candidate
(marginal EIG exactly 0) rejoin the comparison once every probe has
saturated: churn extinction happens through THIS constant and the
self-entropy tie-break, never through a tuned threshold on EIG itself.
"""

GATE_DELTA: Final[float] = 0.05
"""Tail mass a candidate is allowed to place outside the homeostat's soft
bands in the one-step self-forecast: the hard gate requires the predicted
post-action distances to be within the soft bands with probability at
least ``1 - GATE_DELTA``.

Structural justification: 0.05 is the per-tail mass of the 90% central
credible-interval convention fixed throughout the conjugate cells
(``interval(0.9)`` in ``topos.beliefs.core``) — the codebase's one
existing "rare enough to disregard" convention, reused rather than
calibrated.
"""

GATE_FORECAST_HORIZON_STEPS: Final[int] = 1
"""The self-forecast the gates consume is ONE step ahead: the gate asks
"does this action keep me viable through the next cycle", where it will
be re-evaluated. Fill outcomes inside that forecast still carry the fill
model's own-horizon probabilities (the only horizon those posteriors
answer without extrapolation), which over-states one-step exposure — a
conservative, documented approximation, not a knob."""

DEEP_OFFSET_TICKS: Final[float] = 4.0
"""Offset (ticks behind the own-side best) of the coarse menu's deep
quote. Structural justification: 4 is the shallowest price of the "deep"
band shared verbatim by the flow model and the fill model (band edges
0 / 1-3 / 4+); the coarse deep quote probes that band at its boundary, in
the discretization the posteriors already live on."""

REFINED_OFFSET_STEPS: Final[tuple[float, ...]] = (-1.0, 0.0, 1.0)
"""Refined-menu offsets: plus/minus one tick around the coarse winner —
the price lattice's own resolution, the smallest move that can change
anything."""

REFINED_SIZE_FACTORS: Final[tuple[float, ...]] = (0.5, 1.0, 2.0)
"""Refined-menu sizes: halve / keep / double the coarse winner's size
fraction, clipped to (0, 1]. Scale-free refinement — no calibrated size
constants."""

REFINED_PATIENCE_GRID: Final[tuple[float, ...]] = (0.0, 0.5, 1.0)
"""Refined-menu patience values: the endpoints and midpoint of the
contract's patience range [0, 1]."""
