"""Ablation isolation: each flag changes exactly its documented path.

Two halves. With all flags OFF, no ablation strategy object exists in the
wiring at all — the intact architecture cannot consult what was never
instantiated. With one flag ON, the corresponding strategy is consulted
(instrumented via its own counter) and only its documented surface moves;
everything the flag does not document is asserted unchanged.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from tests.agent.conftest import canned_stream, make_agent, run_canned, spy
from tests.proposer.conftest import mk_candidate
from topos.agent import (
    AblationFlags,
    FrozenFillModel,
    FrozenImpactModel,
    NoReflexiveSelection,
    NullDistanceProjector,
    SurpriseAsCuriosity,
    VetoOnlyHomeostat,
)
from topos.contracts.intent import FILL_RATE, REGIME
from topos.proposer import select_candidate
from topos.selfmodel import FillModel, ImpactModel

_ABLATION_TYPES = (
    SurpriseAsCuriosity,
    FrozenFillModel,
    FrozenImpactModel,
    NoReflexiveSelection,
    NullDistanceProjector,
    VetoOnlyHomeostat,
)


# ---------------------------------------------------------------------------
# All flags off: no flag object exists, let alone gets consulted
# ---------------------------------------------------------------------------


def test_flags_off_wires_no_ablation_objects() -> None:
    agent = make_agent()

    # The selection rule is the exported lexicographic rule itself.
    assert agent.selection is select_candidate
    # The self-model modules are the plain classes, not the frozen ones.
    assert type(agent.fill) is FillModel
    assert type(agent.impact) is ImpactModel
    # The scoring map holds the agent's own module objects, unwrapped.
    for hypothesis, module in agent.scoring_modules.items():
        assert module is agent.modules[hypothesis]
    # No homeostat filter, no null projector.
    assert agent.homeostat_filter is None
    assert agent.null_projector is None
    # Belt and braces: no ablation type is reachable anywhere in the seams.
    seams = [
        agent.selection,
        agent.fill,
        agent.impact,
        *agent.scoring_modules.values(),
        agent.homeostat_filter,
        agent.null_projector,
    ]
    for obj in seams:
        assert not isinstance(obj, _ABLATION_TYPES)


# ---------------------------------------------------------------------------
# SURPRISE_CURIOSITY
# ---------------------------------------------------------------------------


def test_surprise_curiosity_replaces_eig_and_nothing_else() -> None:
    agent = make_agent(flags=AblationFlags(surprise_curiosity=True))

    wrappers = list(agent.scoring_modules.values())
    assert all(isinstance(w, SurpriseAsCuriosity) for w in wrappers)
    # The agent's own module map (updates, snapshots, realized IG) is raw.
    for module in agent.modules.values():
        assert not isinstance(module, SurpriseAsCuriosity)

    # The raw modules' parameter-EIG machinery is never consulted.
    raw_eig_calls: list[str] = []
    for hypothesis, module in agent.modules.items():
        spy(raw_eig_calls, module, "eig_nats", f"eig:{hypothesis}")

    run_canned(agent, canned_stream(20))

    assert raw_eig_calls == [], (
        "under SURPRISE_CURIOSITY no parameter-EIG quantity may be "
        f"computed for scoring; saw {raw_eig_calls}"
    )
    assert all(w.consultations > 0 for w in wrappers)  # type: ignore[union-attr]

    # Every EIG quantity in the broadcast is the retrospective signal:
    # each probeable headline's best marginal is max(0, surprise_z).
    record = agent.records[-1]
    by_id = {h.hypothesis_id: h for h in record.headlines}
    for hypothesis in agent.scoring_modules:
        headline = by_id[hypothesis]
        assert headline.best_marginal_eig_nats == pytest.approx(
            max(0.0, headline.last_surprise_z)
        )
    # The regime headline stays at 0 (passive-only, unchanged by the flag).
    assert by_id[REGIME].best_marginal_eig_nats == 0.0

    # The null action is scored 0 through the wrapper.
    from tests.beliefs.conftest import null_probe

    wrapper = agent.scoring_modules[FILL_RATE]
    assert wrapper.eig_nats(null_probe(FILL_RATE)) == 0.0


# ---------------------------------------------------------------------------
# NO_SELF_MODEL
# ---------------------------------------------------------------------------


def test_no_self_model_freezes_exactly_the_two_posteriors() -> None:
    from topos.env.harness import RunConfig, run

    steps = 150
    baseline = make_agent()
    ablated = make_agent(flags=AblationFlags(no_self_model=True))
    run(RunConfig(n_steps=steps), baseline, 20260710)
    run(RunConfig(n_steps=steps), ablated, 20260710)

    # Premise: the intact run really traded and really learned about
    # itself, otherwise "frozen" would be vacuously true.
    assert baseline.message_log, "baseline never traded"
    assert any(
        (cell.a, cell.b) != (cell.prior_a, cell.prior_b)
        for cell in baseline.fill.cells.values()
    ), "baseline fill posterior never moved; premise broken"
    scale = baseline.impact.noise_scale_posterior
    assert (scale.a, scale.b) != (scale.prior_a, scale.prior_b)

    # The ablated run traded too, yet its self-model stayed at the prior,
    # bit for bit.
    assert ablated.message_log, "ablated agent never traded"
    for cell in ablated.fill.cells.values():
        assert (cell.a, cell.b) == (cell.prior_a, cell.prior_b)
    frozen_scale = ablated.impact.noise_scale_posterior
    assert (frozen_scale.a, frozen_scale.b) == (
        frozen_scale.prior_a,
        frozen_scale.prior_b,
    )
    cov = ablated.impact.coef_scale_free_cov
    assert float(np.abs(cov - np.eye(cov.shape[0])).max()) == 0.0

    # self_trajectory still compiles — from the frozen posteriors.
    from topos.proposer import null_intent

    forecast = ablated.trajectory.forecast(null_intent(FILL_RATE))
    assert forecast.entropy_nats == forecast.entropy_nats  # finite, not NaN

    # Everything else is untouched: the world models learned normally.
    fair_scale = ablated.fair_value.noise_scale_posterior
    assert (fair_scale.a, fair_scale.b) != (fair_scale.prior_a, fair_scale.prior_b)


# ---------------------------------------------------------------------------
# NO_REFLEXIVE
# ---------------------------------------------------------------------------


def test_no_reflexive_drops_tiebreak_c_only() -> None:
    rule = NoReflexiveSelection()

    # (c) removed: within the epsilon band, the default rule prefers the
    # LOWER self-entropy; the ablated rule takes the max marginal.
    high = mk_candidate(kind="high", marginal=0.100, self_entropy=5.0)
    close = mk_candidate(kind="close", marginal=0.095, self_entropy=0.1)
    null = mk_candidate(null=True, self_entropy=0.0)
    assert select_candidate([null, high, close], {}) is close
    assert rule([null, high, close], {}) is high

    # Boredom-band re-entry removed: with the top marginal inside epsilon,
    # the default rule lets the null win on self-predictability; the
    # ablated rule keeps probing (the churn this ablation exhibits).
    tiny = mk_candidate(kind="tiny", marginal=0.01, self_entropy=1.0)
    assert select_candidate([null, tiny], {}) is null
    assert rule([null, tiny], {}) is tiny

    # Exact EIG ties break by lowest message cost.
    cheap = mk_candidate(kind="cheap", marginal=0.08, self_entropy=9.0)
    costly = dataclasses.replace(
        mk_candidate(kind="costly", marginal=0.08, self_entropy=0.0),
        message_cost=3,
    )
    assert rule([null, costly, cheap], {}) is cheap

    # Rule (d) intact: no eligible candidate => flatten on any nonzero
    # drive distance, else null.
    flatten = mk_candidate(flatten=True)
    gated_out = mk_candidate(kind="gated", marginal=0.2, gates=False)
    assert rule([null, flatten, gated_out], {"inventory": 0.4}) is flatten
    assert rule([null, flatten, gated_out], {"inventory": 0.0}) is null
    assert rule.consultations == 5


def test_no_reflexive_is_wired_into_the_proposer() -> None:
    agent = make_agent(flags=AblationFlags(no_reflexive=True))
    assert isinstance(agent.selection, NoReflexiveSelection)
    assert agent.proposer._selection is agent.selection  # noqa: SLF001

    run_canned(agent, canned_stream(15))
    assert agent.selection.consultations > 0


# ---------------------------------------------------------------------------
# NO_HOMEOSTAT
# ---------------------------------------------------------------------------


def test_no_homeostat_silences_drive_but_keeps_vetoes() -> None:
    agent = make_agent(flags=AblationFlags(no_homeostat=True))
    assert agent.homeostat_filter is not None
    assert agent.null_projector is not None

    window = agent.config.homeostat.message_window_steps

    def breach_message_budget() -> None:
        """Saturate the rolling window so the hard veto is armed."""
        for _ in range(window):
            agent.homeostat.record_messages(1)

    seen: list[dict[str, object]] = []
    workspace_cycle = agent.workspace.cycle

    def recording_cycle(**kwargs: object) -> object:
        seen.append(
            {
                "drives": dict(kwargs["drives"]),  # type: ignore[call-overload, arg-type]
                "vetoes": dict(kwargs["vetoes"]),  # type: ignore[call-overload, arg-type]
                "corrective": kwargs["corrective_intent"],
                "projector": kwargs["projector"],
            }
        )
        return workspace_cycle(**kwargs)  # type: ignore[arg-type]

    agent.workspace.cycle = recording_cycle  # type: ignore[method-assign]

    stream = canned_stream(3)
    for obs in stream:
        # Re-arm before every cycle: each cycle's own zero-count append
        # would otherwise drain the rolling window.
        breach_message_budget()
        record, action = agent.cycle(obs)
        # Placements suppressed by the (surviving) veto; nothing sent.
        assert action is None

    assert len(seen) == 3
    for call in seen:
        assert call["drives"] == {}, "drives must never bid under NO_HOMEOSTAT"
        assert call["corrective"] is None
        assert call["projector"] is agent.null_projector
        vetoes = call["vetoes"]
        assert isinstance(vetoes, dict)
        assert vetoes["message_budget"] is True, "hard vetoes must survive"

    # Drives never seized the workspace: no homeostatic focus anywhere.
    assert all(
        record.focus is None or not record.focus.is_homeostatic
        for record in agent.records
    )
    # The strategies were consulted exactly while the flag was on.
    assert agent.homeostat_filter.consultations == 3
    assert agent.null_projector.consultations > 0
    # And the cognitive view carries no distances (silenced exports).
    assert all(dict(r.self_state.drive_distances) == {} for r in agent.records)


# ---------------------------------------------------------------------------
# All flags at once (P13 runs combinations; they must compose)
# ---------------------------------------------------------------------------


def test_all_flags_together_still_cycle() -> None:
    agent = make_agent(
        flags=AblationFlags(
            surprise_curiosity=True,
            no_self_model=True,
            no_reflexive=True,
            no_homeostat=True,
        )
    )
    run_canned(agent, canned_stream(20))
    assert len(agent.records) == 20
    assert all(record.intent is not None for record in agent.records)
    # Frozen posteriors under the combined stack, still.
    for cell in agent.fill.cells.values():
        assert (cell.a, cell.b) == (cell.prior_a, cell.prior_b)
