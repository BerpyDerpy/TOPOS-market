"""The selection rule, exported for the arbiter (P9): gates, then
LEXICOGRAPHIC order — never a scalarized trade-off.

    (a) hard gates: motor-legal, no homeostat veto, predicted post-action
        distances within the soft bands at confidence 1 - GATE_DELTA;
    (b) among gated candidates, rank by marginal EIG — only STRICTLY
        positive marginals are eligible to beat the null;
    (c) among candidates within EPSILON_EIG_NATS of the top, choose the
        MINIMUM self-entropy. The null (marginal exactly 0) joins this
        pool whenever the top is itself within epsilon of 0 — which is
        how a saturated question stops being probed: conjugate EIGs never
        reach exactly 0, but once they fall inside the architecture's
        boredom band the tie-break prefers the more self-predictable
        action, and nothing is more self-predictable than watching;
    (d) if no gated candidate has a strictly positive marginal, the
        null/flatten pair wins: null by default, flatten if any homeostat
        drive is nonzero (a drive is nonzero exactly when its exported
        distance is > 0, so the cognitive view suffices — INV-5).

The lexicographic structure is the point: a scalarized score like
EIG - lambda*self_entropy - mu*cost would smuggle in tuned trade-off
coefficients. There are none here, and none may ever be added.

Flatten never competes on EIG: it targets SELF_TRAJECTORY, which is a
forecast compiler, not a probeable hypothesis (adjudication A2) — if a
flatten-shaped action is genuinely the best experiment for the focus, the
refined grid contains the equivalent probe with the focus as its target,
which keeps the realized-IG bookkeeping intact. Flatten enters only as
the rule-(d) corrective fallback.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from topos.proposer.candidates import Candidate
from topos.proposer.config import EPSILON_EIG_NATS


def select_candidate(
    candidates: Sequence[Candidate],
    drive_distances: Mapping[str, float],
) -> Candidate:
    """Apply the lexicographic rule to one cycle's candidate set.

    ``candidates`` must contain the null candidate (INV-4: it is
    first-class, always on the menu). ``drive_distances`` is the current
    cognitive self-state's exported distance map, used only for the
    rule-(d) flatten trigger.
    """
    null_candidate = next(
        (c for c in candidates if c.probe.intent.is_null), None
    )
    if null_candidate is None:
        raise ValueError(
            "candidate set must contain the null candidate (INV-4)"
        )
    flatten_candidate = next(
        (c for c in candidates if c.probe.intent.is_flatten), None
    )

    gated = [c for c in candidates if c.gates_passed]
    eligible = [
        c
        for c in gated
        if c.marginal_eig_nats > 0.0
        and not c.probe.intent.is_null
        and not c.probe.intent.is_flatten
    ]
    if eligible:
        top = max(c.marginal_eig_nats for c in eligible)
        pool: list[Candidate] = []
        if null_candidate.gates_passed and 0.0 >= top - EPSILON_EIG_NATS:
            # The null's marginal is 0 by construction; it joins the pool
            # first so exact self-entropy ties resolve to watching.
            pool.append(null_candidate)
        pool.extend(
            c for c in eligible if c.marginal_eig_nats >= top - EPSILON_EIG_NATS
        )
        return min(pool, key=lambda c: c.self_entropy_nats)

    if flatten_candidate is not None and any(
        u > 0.0 for u in drive_distances.values()
    ):
        return flatten_candidate
    return null_candidate
