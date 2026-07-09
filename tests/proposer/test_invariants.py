"""Proposer-local invariant pins.

INV-5 at package scale: the proposer consumes ``SelfStateCognitive`` only.
Mechanically: no account-state vocabulary and no drives/metrics import
anywhere under ``topos/proposer/`` (the global tripwires already cover the
metrics boundary and feedback-signal vocabulary; this scan adds the
account-state tokens and the drives boundary specific to this package).

Also pins the menu structure the spec fixes: the standing coarse shapes,
and the refined grid's inability to retarget away from the focus.
"""

from __future__ import annotations

import re
from pathlib import Path

import topos.proposer
from tests.proposer.conftest import make_cognitive, make_world
from topos.contracts.intent import FILL_RATE
from topos.contracts.market import Side
from topos.contracts.workspace import WorkingOrderView
from topos.proposer import ProbeShape, coarse_shapes, refined_shapes

PROPOSER_ROOT = Path(topos.proposer.__file__).resolve().parent

ACCOUNT_TOKENS = re.compile(
    r"\bpnl\b|\bprofit\b|\bwealth\b|\bdrawdown\b|SelfStateFull",
    re.IGNORECASE,
)
FORBIDDEN_IMPORTS = re.compile(
    r"^\s*(from|import)\s+topos\.(drives|metrics)\b", re.MULTILINE
)


def test_no_account_state_vocabulary_in_proposer_source() -> None:
    offenders: list[str] = []
    for path in sorted(PROPOSER_ROOT.rglob("*.py")):
        for lineno, line in enumerate(path.read_text().splitlines(), start=1):
            if ACCOUNT_TOKENS.search(line):
                offenders.append(f"{path}:{lineno}: {line.strip()}")
    assert not offenders, (
        "INV-5: account-state vocabulary in proposer source:\n"
        + "\n".join(offenders)
    )


def test_proposer_imports_neither_drives_nor_metrics() -> None:
    offenders: list[str] = []
    for path in sorted(PROPOSER_ROOT.rglob("*.py")):
        if FORBIDDEN_IMPORTS.search(path.read_text()):
            offenders.append(str(path))
    assert not offenders, (
        "the proposer must receive homeostat byproducts as exported values, "
        "never import drives/ or metrics/: " + ", ".join(offenders)
    )


def test_coarse_menu_covers_the_standing_shapes() -> None:
    working = WorkingOrderView(
        order_id=7,
        side=Side.SELL,
        price_ticks=1001,
        size_lots_remaining=2,
        age_steps=3,
        queue_rank_mean=1.0,
        queue_rank_var=0.5,
    )
    shapes = coarse_shapes(
        make_world(), make_cognitive(inventory=3, working=(working,)), 4
    )
    names = {shape.name for shape in shapes}
    assert {
        "touch_bid",
        "touch_ask",
        "deep_bid",
        "deep_ask",
        "market_buy",
        "market_sell",
        "cancel_refresh_7",
        "flatten",
    } <= names
    by_name = {shape.name: shape for shape in shapes}
    # Touch quotes sit at the own-side best; marketable probes cross.
    assert by_name["touch_bid"].offset_ticks == 1.0
    assert by_name["market_buy"].offset_ticks == -1.0
    # The small marketable probe is one lot of a 4-lot budget.
    assert by_name["market_buy"].size_frac == 0.25
    # Cancel-refresh works the order at the immediate tempo.
    assert by_name["cancel_refresh_7"].patience == 0.0
    assert by_name["cancel_refresh_7"].side == -1.0


def test_refined_grid_is_a_neighborhood_of_the_winner() -> None:
    winner = ProbeShape("deep_bid", +1.0, 5.0, 1.0, 1.0)
    grid = refined_shapes(winner, size_budget_lots=4)
    assert grid
    offsets = {shape.offset_ticks for shape in grid}
    sizes = {shape.size_frac for shape in grid}
    patiences = {shape.patience for shape in grid}
    assert offsets == {4.0, 5.0, 6.0}
    assert sizes == {0.5, 1.0}  # doubling clips back onto 1.0
    assert patiences == {0.0, 0.5, 1.0}
    assert all(shape.side == winner.side for shape in grid)
    # Sizes that would compile to zero lots are dropped.
    tiny = ProbeShape("tiny", +1.0, 1.0, 0.05, 1.0)
    grid_tiny = refined_shapes(tiny, size_budget_lots=4)
    assert all(round(s.size_frac * 4) >= 1 for s in grid_tiny)
