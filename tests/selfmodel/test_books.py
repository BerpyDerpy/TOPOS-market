"""Bookkeeping vs harness ground truth, plus the accounting invariants.

The headline test drives a scripted agent through a full harness episode
(crossing buy, resting orders on both sides, a cancel) while folding a
``Books`` instance from own acks/fills only, then validates the claim
stream against engine-side truth through the P3 hook
(``assert_agent_bookkeeping``) — the external correctness check INV-11
demands (the environment never reports account state).
"""

from __future__ import annotations

import dataclasses
import re
from dataclasses import replace

import pytest

from tests.selfmodel.conftest import events, plain_obs
from topos.contracts.beliefs import SelfEvents
from topos.contracts.market import (
    GTC,
    Ack,
    AckStatus,
    BookLevel,
    Cancel,
    ExchangeMessage,
    Fill,
    Liquidity,
    Observation,
    PlaceLimit,
    Side,
)
from topos.contracts.workspace import SelfStateCognitive, SelfStateFull
from topos.env.background import BackgroundConfig
from topos.env.harness import ResetFn, RunConfig, StepFn, assert_agent_bookkeeping, run
from topos.selfmodel import Books

ROOT_SEED = 20260709


def calm_config(n_steps: int) -> RunConfig:
    """Regime switching off: deterministic scripting stays interpretable."""
    regimes = tuple(replace(r, hazard=0.0) for r in BackgroundConfig().regimes)
    return RunConfig(n_steps=n_steps, background=BackgroundConfig(regimes=regimes))


def _best(levels: tuple[BookLevel, ...]) -> int | None:
    for level in levels:
        if level.size_lots > 0:
            return level.price_ticks
    return None


class BookkeepingAgent:
    """Scripted driver folding a Books instance from its own event stream.

    Pairing convention: the acks in an observation answer the messages
    sent one observation earlier (engine timing; see conftest docstring).
    """

    def __init__(self) -> None:
        self.books = Books()

    def _fold(self, obs: Observation, messages: tuple[ExchangeMessage, ...]) -> None:
        self.books.update(
            obs,
            SelfEvents(
                step=obs.step,
                messages_sent=messages,
                acks=obs.own_acks,
                fills=obs.own_fills,
            ),
        )

    def _decide(self, obs: Observation) -> ExchangeMessage | None:
        if obs.step == 9 and _best(obs.asks) is not None:
            # Crossing buy: TAKER fills, stamped with the action step.
            return PlaceLimit(Side.BUY, _best(obs.asks) + 3, 5, tif_steps=1)
        if obs.step == 19 and _best(obs.bids) is not None:
            # Resting bid at the touch: MAKER fills, stamped with the
            # background step that hits them — the other claim path.
            return PlaceLimit(Side.BUY, _best(obs.bids), 6, tif_steps=GTC)
        if obs.step == 29 and _best(obs.asks) is not None:
            return PlaceLimit(Side.SELL, _best(obs.asks), 4, tif_steps=GTC)
        if obs.step == 44:
            return Cancel(order_id=1)
        return None

    def __call__(self, reset: ResetFn, step: StepFn) -> None:
        obs = reset()
        self._fold(obs, ())
        pending: tuple[ExchangeMessage, ...] = ()
        while True:
            action = self._decide(obs)
            sent = (action,) if action is not None else ()
            obs = step(action)
            self._fold(obs, pending)
            pending = sent


def test_books_match_engine_ground_truth() -> None:
    n_steps = 60
    config = calm_config(n_steps)
    agent = BookkeepingAgent()
    log = run(config, agent, ROOT_SEED)

    # Premises: the script really traded, on both liquidity paths (taker
    # fills stamp with the action step, maker fills with the background
    # step — the two stamp groups claims() must keep separate).
    truth = log.steps[-1].account(config.agent_actor_id)
    assert truth.fills, "script never traded; the run validates nothing"
    liquidities = {fill.liquidity for fill in truth.fills}
    assert Liquidity.TAKER in liquidities
    assert Liquidity.MAKER in liquidities

    # The P3 hook: every end-of-step inventory AND cash claim must equal
    # engine-side truth. (The agent never acts at the final step, so the
    # claim stream is complete through n_steps - 1.)
    claims = agent.books.claims(n_steps - 1)
    assert len(claims) == n_steps
    assert_agent_bookkeeping(log, claims)

    # Live views agree with the last claim, and the accounting
    # decomposition holds to float precision.
    assert agent.books.inventory_lots == truth.inventory_lots
    assert agent.books.cash_ticks == truth.cash
    assert agent.books.accounting_identity_gap() < 1e-9


# =====================================================================
# Accounting arithmetic (synthetic, engine-free)
# =====================================================================


def _accepted_placement(
    books: Books, step: int, order_id: int, place: PlaceLimit
) -> None:
    ack = Ack(order_id=order_id, status=AckStatus.ACCEPTED, step=step)
    books.update(
        plain_obs(step, own_acks=(ack,)),
        events(step, messages=(place,), acks=(ack,)),
    )


def test_average_cost_realized_and_unrealized() -> None:
    books = Books()
    books.update(plain_obs(0), events(0))  # mark = 1000

    # Buy 5 @ 1001 (cross).
    _accepted_placement(
        books, 1, 0, PlaceLimit(Side.BUY, 1001, 5, tif_steps=GTC)
    )
    fill = Fill(
        order_id=0, price_ticks=1001, size_lots=5,
        liquidity=Liquidity.TAKER, step=1,
    )
    books.update(plain_obs(2, own_fills=(fill,)), events(2, fills=(fill,)))
    assert books.inventory_lots == 5
    assert books.cash_ticks == -5005
    assert books.realized_pnl == 0.0

    # Book shifts up 4 ticks: mark 1004, open lots gain 3 each.
    up_bids = [(1003 - i, 20) for i in range(10)]
    up_asks = [(1005 + i, 20) for i in range(10)]
    books.update(plain_obs(3, bids=up_bids, asks=up_asks), events(3))
    assert books.mark == 1004.0
    assert books.unrealized_pnl == pytest.approx(15.0)
    assert books.gross_exposure == pytest.approx(5 * 1004.0)

    # Sell 8 @ 1003: closes the 5 (realizing 2 ticks each) and flips
    # short 3 @ 1003.
    _accepted_placement(
        books, 4, 1, PlaceLimit(Side.SELL, 1003, 8, tif_steps=GTC)
    )
    fill = Fill(
        order_id=1, price_ticks=1003, size_lots=8,
        liquidity=Liquidity.TAKER, step=4,
    )
    books.update(
        plain_obs(5, bids=up_bids, asks=up_asks, own_fills=(fill,)),
        events(5, fills=(fill,)),
    )
    assert books.inventory_lots == -3
    assert books.cash_ticks == -5005 + 8 * 1003
    assert books.realized_pnl == pytest.approx(10.0)
    assert books.unrealized_pnl == pytest.approx(-3.0)  # short 3, mark 1 above entry
    # Method-independent identity: realized + unrealized = cash + inv * mark.
    assert books.accounting_identity_gap() < 1e-9


def test_working_order_lifecycle_and_rank_prior() -> None:
    books = Books()
    books.update(plain_obs(0), events(0))
    # Deep bid, 5 lots, at a level showing 20 lots (ours included).
    _accepted_placement(
        books, 1, 0, PlaceLimit(Side.BUY, 995, 5, tif_steps=GTC)
    )
    (view,) = books.working_order_views()
    assert view.order_id == 0
    assert view.side is Side.BUY
    assert view.price_ticks == 995
    assert view.size_lots_remaining == 5
    assert view.age_steps == 0
    # Ignorance prior over queue rank: uniform on {0..15} (level 20 minus
    # own 5): mean 7.5, variance 15 * 17 / 12.
    assert view.queue_rank_mean == pytest.approx(7.5)
    assert view.queue_rank_var == pytest.approx(15 * 17 / 12)

    # Partial fill decrements; age advances with the step.
    fill = Fill(
        order_id=0, price_ticks=995, size_lots=2,
        liquidity=Liquidity.MAKER, step=3,
    )
    books.update(plain_obs(3, own_fills=(fill,)), events(3, fills=(fill,)))
    (view,) = books.working_order_views()
    assert view.size_lots_remaining == 3
    assert view.age_steps == 2

    # Cancel removes it; the account keeps the partial fill.
    cancel_ack = Ack(order_id=0, status=AckStatus.CANCELED, step=4)
    books.update(plain_obs(4, own_acks=(cancel_ack,)), events(4))
    assert books.working_order_views() == ()
    assert books.inventory_lots == 2


def test_injected_rank_lookup_overrides_the_prior() -> None:
    books = Books(rank_lookup=lambda order_id: (2.0, 1.5))
    books.update(plain_obs(0), events(0))
    _accepted_placement(
        books, 1, 0, PlaceLimit(Side.BUY, 995, 5, tif_steps=GTC)
    )
    (view,) = books.working_order_views()
    assert (view.queue_rank_mean, view.queue_rank_var) == (2.0, 1.5)


def test_views_are_the_contract_types() -> None:
    books = Books()
    books.update(plain_obs(0), events(0))
    full = books.full_view(drive_distances={"inventory": 0.2})
    cognitive = books.cognitive_view(drive_distances={"inventory": 0.2})
    assert isinstance(full, SelfStateFull)
    assert isinstance(cognitive, SelfStateCognitive)
    assert not isinstance(cognitive, SelfStateFull)
    assert cognitive.drive_distances["inventory"] == 0.2
    # INV-5 at the object level: no account vocabulary on the cognitive view.
    forbidden = re.compile(r"pnl|profit|drawdown|wealth", re.IGNORECASE)
    for field in dataclasses.fields(cognitive):
        assert not forbidden.search(field.name)


def test_claims_validate_input() -> None:
    books = Books()
    with pytest.raises(ValueError):
        books.claims(-1)
