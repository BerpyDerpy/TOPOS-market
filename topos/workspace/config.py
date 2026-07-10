"""Architecture constants for the workspace (P9).

Like the proposer's constants, every value here is fixed by STRUCTURE —
either given by the architecture spec directly or derived from scales the
codebase already owns — with the justification recorded on the constant
itself. None of them is ever tuned against run outcomes.
"""

from __future__ import annotations

from typing import Final

from topos.contracts.intent import KNOWN_HYPOTHESIS_IDS
from topos.proposer.config import EPSILON_EIG_NATS

K_HEADLINES: Final[int] = 6
"""Blackboard capacity: at most this many hypothesis headlines are
broadcast per cycle. If more hypotheses exist than slots, only the top-K
by salience appear in the `WorkspaceRecord`; the rest are genuinely
absent — not summarized, not appended to any side channel. Capacity IS
the attention mechanism: a bounded workspace is what makes winning the
salience competition mean something."""

GAMMA: Final[float] = 0.5
"""Surprise amplification in the salience formula
``s_h = w_h * best_marginal_eig_h * (1 + GAMMA * max(0, surprise_z_h))``.

Surprise ATTENDS, EIG ACTS: because the surprise term multiplies EIG, a
hypothesis with zero reducible uncertainty has zero salience no matter
how surprising its errors — surprise can only make an already-answerable
question louder, never turn an unanswerable one into a target (the
noisy-TV immunity of INV-3, restated at the attention level)."""

S_MIN: Final[float] = EPSILON_EIG_NATS / len(KNOWN_HYPOTHESIS_IDS)
"""Ignition threshold: if no salience bid strictly exceeds this, the
cycle has no focus and the intent is null — a quiet mind watches.

Structural justification: both factors are scales the architecture
already owns. ``EPSILON_EIG_NATS`` (0.02) is the boredom band — the
convergence level the saturation tripwires pin as "nothing left to
learn" — and ``1 / len(KNOWN_HYPOTHESIS_IDS)`` is the uniform centrality
weight, the registry's own no-edges fallback. Their product is the
salience of a boredom-band question carried at baseline structural
consequence: any bid at or below it is a question the architecture
already treats as exhausted, held by a hypothesis of no more than
uniform consequence — not worth waking the workspace for. A homeostat
drive crosses this threshold at a normalized excursion of about 0.05
(u^2/(1-u) = S_MIN), so a hair past the soft band does not seize the
workspace, while excursions approaching the hard bound dominate it
(INV-6's "ignorable at small excursions, preemptive near the bound")."""
