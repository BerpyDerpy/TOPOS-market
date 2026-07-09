"""The proposer's core behavior: the null is first-class and positive
(INV-4), watching wins once probing has saturated, and probes win exactly
when they buy intervention-specific information.

``test_null_positive`` guards the single most damaging bug this package
could contain: scoring the null as 0 (or through a different observation
model than the probes) silently reinstates constant trading.
"""

from __future__ import annotations

import pytest

from tests.proposer.conftest import (
    BandProjector,
    make_cognitive,
    make_proposer,
    make_world,
    probe_candidates,
    saturate,
    seeded_modules,
)
from topos.contracts.intent import (
    FAIR_VALUE,
    FILL_RATE,
    FLOW_INTENSITY,
    IMPACT,
    NULL_THRESHOLD,
)
from topos.contracts.market import Side
from topos.proposer import EPSILON_EIG_NATS
from topos.selfmodel import FillModel


def _propose(modules, trajectory, focus, **overrides):
    kwargs = dict(
        world=make_world(),
        cognitive=make_cognitive(),
        focus=focus,
        vetoes={},
        projector=BandProjector(),
    )
    kwargs.update(overrides)
    return make_proposer(modules, trajectory).propose(**kwargs)


def test_null_positive() -> None:
    """Under an active market, EIG_null > 0 for flow and fair value: the
    market moves and teaches without being poked."""
    modules, trajectory = seeded_modules()
    proposal = _propose(modules, trajectory, focus=FLOW_INTENSITY)
    assert proposal.null_eig_nats[FLOW_INTENSITY] > 0.0
    assert proposal.null_eig_nats[FAIR_VALUE] > 0.0
    # No order can teach a world predictor more than watching does, so
    # the null wins the flow-focused cycle outright.
    assert proposal.selected.probe.intent.is_null
    # The logged null is unambiguous: commitment exactly 0.0 (standing
    # ruling), not merely below the threshold.
    assert proposal.selected.probe.intent.commitment == 0.0


def test_world_information_rides_the_null() -> None:
    """Marginal-over-null is exactly 0 for world predictors (their EIG is
    intent-independent) and positive for self-model hypotheses (their
    information must be bought by acting) — items 13/18 at proposer scale.
    """
    modules, trajectory = seeded_modules()
    proposal = _propose(modules, trajectory, focus=FILL_RATE)
    assert proposal.best_marginal_eig_nats[FAIR_VALUE] == pytest.approx(0.0, abs=1e-12)
    assert proposal.best_marginal_eig_nats[FLOW_INTENSITY] == pytest.approx(
        0.0, abs=1e-12
    )
    assert proposal.best_marginal_eig_nats[FILL_RATE] > 0.1
    assert proposal.best_marginal_eig_nats[IMPACT] > 0.0
    # And the null's fill-rate EIG is exactly 0: no order, no trial.
    assert proposal.null_eig_nats[FILL_RATE] == 0.0


def test_refined_menu_targets_focus_and_commitments_are_unambiguous() -> None:
    modules, trajectory = seeded_modules()
    proposal = _propose(modules, trajectory, focus=FILL_RATE)
    probes = probe_candidates(proposal.candidates)
    assert probes, "a fresh fill model must generate refined probes"
    for candidate in probes:
        assert candidate.probe.intent.target_id == FILL_RATE
        assert candidate.probe.intent.commitment >= NULL_THRESHOLD
    nulls = [c for c in proposal.candidates if c.probe.intent.is_null]
    assert len(nulls) == 1
    assert nulls[0].probe.intent.commitment == 0.0
    assert nulls[0].marginal_eig_nats == 0.0


def test_watching_beats_poking_when_saturated() -> None:
    """With a converged fill model, no probe achieves a marginal EIG
    beyond the boredom band => the null wins on minimum self-entropy."""
    modules, trajectory = seeded_modules()
    fill = modules[FILL_RATE]
    assert isinstance(fill, FillModel)
    for cell in fill.cells.values():
        saturate(cell)
    proposal = _propose(modules, trajectory, focus=FILL_RATE)
    assert proposal.selected.probe.intent.is_null
    assert proposal.selected.probe.intent.commitment == 0.0
    probes = probe_candidates(proposal.candidates)
    assert probes, "saturation must not silently empty the menu"
    assert all(c.marginal_eig_nats < EPSILON_EIG_NATS for c in probes)
    # Watching is also the most self-predictable action on the menu.
    null = next(c for c in proposal.candidates if c.probe.intent.is_null)
    assert all(null.self_entropy_nats <= c.self_entropy_nats for c in probes)


def test_intervention_specific_info_wins() -> None:
    """A deliberately widened fill bucket makes the probe exercising it
    beat the null; once simulated fills settle the bucket, it stops
    beating the null. Churn extinction end-to-end at proposer scale."""
    modules, trajectory = seeded_modules()
    fill = modules[FILL_RATE]
    assert isinstance(fill, FillModel)
    widened = (Side.BUY, "deep", "balanced")
    for key, cell in fill.cells.items():
        if key != widened:
            saturate(cell)

    proposer = make_proposer(modules, trajectory)
    kwargs = dict(
        world=make_world(),
        cognitive=make_cognitive(),
        focus=FILL_RATE,
        vetoes={},
        projector=BandProjector(),
    )
    proposal = proposer.propose(**kwargs)
    selected = proposal.selected
    assert not selected.probe.intent.is_null
    assert selected.probe.intent.target_id == FILL_RATE
    assert selected.marginal_eig_nats > EPSILON_EIG_NATS
    # The winning probe exercises exactly the widened bucket.
    assert fill.bucket_for_intent(selected.probe.intent) == widened

    # Simulated fills settle the bucket: the question is answered.
    saturate(fill.cells[widened])
    proposal_after = proposer.propose(**kwargs)
    assert proposal_after.selected.probe.intent.is_null
    assert all(
        c.marginal_eig_nats < EPSILON_EIG_NATS
        for c in probe_candidates(proposal_after.candidates)
    )
