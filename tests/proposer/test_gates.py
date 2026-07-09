"""End-to-end gate behavior through the real Proposer: homeostat vetoes
and soft-band forecasts exclude probes before any EIG comparison."""

from __future__ import annotations

import math

from tests.proposer.conftest import (
    BandProjector,
    make_cognitive,
    make_proposer,
    make_world,
    probe_candidates,
    saturate,
    seeded_modules,
)
from topos.contracts.intent import FILL_RATE
from topos.contracts.market import Side
from topos.proposer import GATE_DELTA
from topos.selfmodel import FillModel


def test_vetoed_probes_never_selected_end_to_end() -> None:
    """An inventory veto suppresses every |inventory|-increasing probe at
    the motor, so the gate excludes them all regardless of a wide-open
    bucket's EIG; the fallback flatten wins because a drive is nonzero."""
    modules, trajectory = seeded_modules()
    fill = modules[FILL_RATE]
    assert isinstance(fill, FillModel)
    widened = (Side.BUY, "deep", "balanced")
    for key, cell in fill.cells.items():
        if key != widened:
            saturate(cell)
    proposal = make_proposer(modules, trajectory).propose(
        world=make_world(),
        cognitive=make_cognitive(inventory=5, distances={"inventory": 0.6}),
        focus=FILL_RATE,
        vetoes={"inventory": True},
        projector=BandProjector(),
    )
    probes = probe_candidates(proposal.candidates)
    assert probes, "the widened bucket must still generate a refined menu"
    assert all(c.vetoed and not c.gates_passed for c in probes)
    assert proposal.selected.probe.intent.is_flatten
    # The flatten reduces |inventory|, so the veto does not suppress it.
    assert proposal.selected.probe.intent.side == -1.0


def test_soft_bound_forecast_gates_out_probes() -> None:
    """When any fill would already breach the soft band, the one-step
    self-forecast fails the 1-delta confidence gate for every probe and
    the null (which forecasts no breach) is selected."""
    modules, trajectory = seeded_modules()
    tight = BandProjector(inventory_soft=0.5, inventory_hard=10.0)
    proposal = make_proposer(modules, trajectory).propose(
        world=make_world(),
        cognitive=make_cognitive(),
        focus=FILL_RATE,
        vetoes={},
        projector=tight,
    )
    probes = probe_candidates(proposal.candidates)
    assert probes
    for candidate in probes:
        assert candidate.within_soft_confidence < 1.0 - GATE_DELTA
        assert not candidate.gates_passed
        assert not candidate.vetoed  # gated by the forecast, not by vetoes
    null = next(c for c in proposal.candidates if c.probe.intent.is_null)
    assert null.gates_passed
    assert proposal.selected is null


def test_gate_report_attachments_are_populated() -> None:
    """Self-consequences are attached, not scalarized: every committed
    candidate carries its message cost and predicted distances."""
    modules, trajectory = seeded_modules()
    proposal = make_proposer(modules, trajectory).propose(
        world=make_world(),
        cognitive=make_cognitive(),
        focus=FILL_RATE,
        vetoes={},
        projector=BandProjector(),
    )
    null = next(c for c in proposal.candidates if c.probe.intent.is_null)
    assert null.message_cost == 0
    for candidate in probe_candidates(proposal.candidates):
        assert candidate.message_cost >= 1
        assert candidate.motor_legal
        assert set(candidate.predicted_distances) >= {
            "inventory",
            "gross_exposure",
            "message_budget",
        }
        assert math.isfinite(candidate.self_entropy_nats)
