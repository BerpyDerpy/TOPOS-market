"""One real pass through the machinery: real belief modules, real
proposer, real selection rule, real conditioning hooks — the workspace
cycle end to end, with only the market synthetic.

Also pins the architectural consequence of the EIG split (DESIGN.md
items 13/18): world hypotheses' marginals are 0, so focus can only ever
land on a self-model hypothesis (or a drive) — the workspace buys
information where information is for sale.
"""

from __future__ import annotations

import numpy as np

from tests.beliefs.conftest import empty_events
from tests.proposer.conftest import (
    BandProjector,
    active_obs,
    make_cognitive,
    make_proposer,
    make_world,
    seeded_modules,
)
from tests.workspace.conftest import make_registry
from topos.contracts.intent import FAIR_VALUE, FILL_RATE, FLOW_INTENSITY, IMPACT
from topos.contracts.workspace import WorkspaceRecord
from topos.motor.config import MotorConfig
from topos.workspace import K_HEADLINES, Workspace


def test_full_cycle_with_real_modules() -> None:
    modules, trajectory = seeded_modules()
    workspace = Workspace(
        registry=make_registry(),
        proposer=make_proposer(modules, trajectory),
        modules=modules,
        motor_cfg=MotorConfig(size_budget_lots=4),
        consumers=(modules[FAIR_VALUE], modules[FLOW_INTENSITY]),
    )

    rng = np.random.default_rng(11)
    focused_ids: set[str] = set()
    for step in range(25, 31):
        obs = active_obs(step, rng)
        for module in modules.values():
            module.update(obs, empty_events(step))
        record = workspace.cycle(
            step=step,
            world=make_world(),
            cognitive=make_cognitive(),
            drives={},
            vetoes={},
            corrective_intent=None,
            projector=BandProjector(),
        )

        assert isinstance(record, WorkspaceRecord)
        assert record.step == step
        assert len(record.headlines) == min(len(modules), K_HEADLINES)
        by_id = {h.hypothesis_id: h for h in record.headlines}
        # The EIG split, visible in the broadcast: world hypotheses carry
        # zero marginal (their information rides the null), self-model
        # hypotheses carry the purchasable marginals.
        assert by_id[FAIR_VALUE].best_marginal_eig_nats == 0.0
        assert by_id[FLOW_INTENSITY].best_marginal_eig_nats == 0.0
        assert by_id[FILL_RATE].best_marginal_eig_nats > 0.0
        for headline in record.headlines:
            assert headline.epistemic_entropy_nats == (
                modules[headline.hypothesis_id].posterior_entropy_nats()
            )

        assert record.intent is not None
        if record.focus is not None and not record.focus.is_homeostatic:
            focused_ids.add(record.focus.hypothesis_id)
            if not record.intent.is_null:
                # A committed probe targets the focus and promises the
                # module's own EIG figure.
                assert record.intent.target_id == record.focus.hypothesis_id
                assert record.eig_promised_nats is not None
                assert record.eig_promised_nats > 0.0
                assert record.compiled_messages

    # Focus went somewhere, and only where information can be bought.
    assert focused_ids
    assert focused_ids <= {FILL_RATE, IMPACT}
