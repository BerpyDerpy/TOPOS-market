"""Background market: zero-intelligence flow + stabilizing MMs + regime switching.

Produces the per-step background event sequences consumed by
``MatchingEngine.step(background_events=...)``.

RNG stream discipline (INV-9)
-----------------------------
Every random decision by every background actor draws from
``make_rng(root_seed, StreamKey(actor_id, step, purpose))`` with a distinct
purpose string per decision site; multi-draw sites use indexed purposes
("placement_depth:0", "placement_depth:1", ...). Draws are made
UNCONDITIONALLY and in a fixed order: the number of draws per
(actor, step, purpose) depends only on the regime chain and earlier draws by
the same actor at the same step — never on the book. Where behavior must
depend on the market, the raw draw is book-independent and only its
INTERPRETATION consults the book. Consequence (verified by
tests/env/test_background.py::test_draw_invariance): adding or removing any
other actor leaves every background actor's raw draw log bit-identical;
all behavioral divergence between two runs with the same root seed is then
causally mediated through the visible book, which is what makes
counterfactual replay (P3) valid.

Book-dependent interpretation sites (the complete list — these are the ONLY
channels through which book state, and hence the agent, can influence
background behavior):

1. ZI limit pricing (``ZIFlow._limit_price``): the drawn depth is mapped to
   a price relative to the same-side best (fallbacks: one tick inside the
   opposite best, else the configured initial price), then clamped to be
   non-marketable against the opposite best and floored at 1 tick.
2. ZI marketable pricing (``ZIFlow._marketable_price``): the crossing price
   is the opposite-side best plus/minus ``cross_ticks``; when the opposite
   side is empty the order is SKIPPED (its side/size draws still happen).
3. ZI cancel targeting (``ZIFlow.events``): each drawn uniform is mapped to
   an index into the current list of the flow's own resting orders (sorted
   by order_id); with no resting orders the draw is discarded (no event).
4. MM reference-price reversion (``StabilizingMM.events``): the reference
   price reverts toward the prevailing mid (no mid -> no reversion term);
   the Gaussian innovation is drawn unconditionally either way.
5. MM requote decision (``StabilizingMM.events``): whether to cancel/replace
   compares the desired quotes against the MM's currently resting quotes
   (quote missing = filled or expired; price drift >= threshold; partial
   fill). Deterministic — consumes no draws.
6. MM inventory cap (``StabilizingMM.events``): a side is suppressed (and
   its resting quote canceled) when the MM's current inventory is at the
   cap. Deterministic — consumes no draws.

The RegimeController consults NO book state: its hazard and choice draws
come from streams keyed by the reserved actor id ``background:regime``,
which no order-submitting actor uses, so the regime chain is identical
across any two runs with the same root seed regardless of who else trades —
the agent cannot perturb it (INV-9). Regime ground truth (id + true
parameters, one record per step) goes to a harness-only log for metrics and
is never observable by the agent (INV-11).

Note: the number of EVENTS submitted may legitimately depend on the book
(sites 2, 3, 5, 6 above) — only the number and values of RAW DRAWS may not.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
import numpy.typing as npt

from topos.contracts.market import GTC, Cancel, ExchangeMessage, PlaceLimit, Side
from topos.contracts.rng import StreamKey, make_rng
from topos.env.engine import MatchingEngine

BACKGROUND_ACTOR_PREFIX: Final[str] = "background:"
"""Reserved actor-id namespace; the agent and test actors must not use it."""

ZI_ACTOR_ID: Final[str] = "background:zi"
REGIME_ACTOR_ID: Final[str] = "background:regime"
"""Stream owner for regime draws; never submits orders, so nothing any
order-submitting actor does can touch its streams."""


def mm_actor_id(index: int) -> str:
    """Actor id of the index-th stabilizing market maker."""
    return f"background:mm{index}"


# ---------------------------------------------------------------------------
# Raw-draw log (harness/test-facing)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DrawRecord:
    """One raw draw: the value that came out of stream (actor, step, purpose).

    The full sequence of DrawRecords must be bit-identical across runs with
    the same root seed no matter what other actors do (INV-9); the draw-
    invariance test compares these logs with exact float equality.
    """

    actor_id: str
    step: int
    purpose: str
    value: float


class _StepDraws:
    """All draws for one (actor, step); every purpose is a fresh named stream.

    Helpers must be called UNCONDITIONALLY at each decision site — never
    inside a book-dependent branch — and each purpose is consumed exactly
    once (one scalar per stream), so the draw log shape is a pure function
    of the regime chain and the actor's own earlier draws this step.
    """

    def __init__(
        self, root_seed: int, actor_id: str, step: int, log: list[DrawRecord]
    ) -> None:
        self._root_seed = root_seed
        self._actor_id = actor_id
        self._step = step
        self._log = log

    def _rng(self, purpose: str) -> np.random.Generator:
        return make_rng(
            self._root_seed, StreamKey(self._actor_id, self._step, purpose)
        )

    def _record(self, purpose: str, value: float) -> float:
        self._log.append(
            DrawRecord(
                actor_id=self._actor_id,
                step=self._step,
                purpose=purpose,
                value=value,
            )
        )
        return value

    def uniform(self, purpose: str) -> float:
        return self._record(purpose, self._rng(purpose).random())

    def poisson(self, purpose: str, lam: float) -> int:
        value = self._rng(purpose).poisson(lam)
        self._record(purpose, float(value))
        return value

    def exponential(self, purpose: str, mean: float) -> float:
        return self._record(purpose, self._rng(purpose).exponential(mean))

    def normal(self, purpose: str, std: float) -> float:
        return self._record(purpose, self._rng(purpose).normal(0.0, std))


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RegimeParams:
    """True parameters of one regime (harness-only ground truth — INV-11)."""

    regime_id: str
    limit_rate: float
    """Poisson mean of ZI limit-order arrivals per step."""
    market_rate: float
    """Poisson mean of ZI marketable-order arrivals per step."""
    cancel_rate: float
    """Poisson mean of ZI cancel attempts per step."""
    imbalance: float
    """Buy pressure in [-1, 1]: P(buy) = (1 + imbalance) / 2 for ZI orders."""
    mm_spread_ticks: int
    """Full quoted spread of the stabilizing MMs (>= 2)."""
    hazard: float
    """Per-step chance in [0, 1] of a stochastic switch out of this regime."""

    def __post_init__(self) -> None:
        if min(self.limit_rate, self.market_rate, self.cancel_rate) < 0:
            raise ValueError("arrival/cancel rates must be non-negative")
        if not -1.0 <= self.imbalance <= 1.0:
            raise ValueError(f"imbalance must be in [-1, 1], got {self.imbalance}")
        if self.mm_spread_ticks < 2:
            raise ValueError(f"mm_spread_ticks must be >= 2, got {self.mm_spread_ticks}")
        if not 0.0 <= self.hazard <= 1.0:
            raise ValueError(f"hazard must be in [0, 1], got {self.hazard}")


@dataclass(frozen=True)
class ZIConfig:
    """Regime-independent parameters of the zero-intelligence flow."""

    depth_alpha: float = 1.5
    """Exponent of the truncated power law over placement depths."""
    max_depth_ticks: int = 20
    """Power-law truncation: depths are drawn from {1..max_depth_ticks}."""
    limit_size_mean: float = 4.0
    market_size_mean: float = 6.0
    limit_tif_steps: int = 60
    """Finite TIF keeps resting depth stationary even if cancels lag arrivals."""
    market_tif_steps: int = 1
    """Marketable remainders expire at the end of the step they arrive."""
    cross_ticks: int = 3
    """How many ticks beyond the opposite best a marketable order may walk."""

    def __post_init__(self) -> None:
        if self.depth_alpha <= 0:
            raise ValueError("depth_alpha must be positive")
        if self.max_depth_ticks < 1:
            raise ValueError("max_depth_ticks must be >= 1")
        if min(self.limit_size_mean, self.market_size_mean) <= 0:
            raise ValueError("size means must be positive")
        if min(self.limit_tif_steps, self.market_tif_steps) < 0:
            raise ValueError("tif_steps must be >= 0")
        if self.cross_ticks < 0:
            raise ValueError("cross_ticks must be >= 0")


@dataclass(frozen=True)
class MMConfig:
    """Regime-independent parameters of the stabilizing market makers."""

    size_lots: int = 8
    inventory_cap_lots: int = 80
    """Stop quoting (and pull the quote on) a side at this |inventory|."""
    requote_threshold_ticks: int = 2
    """Cancel/replace a quote when it drifts this far from the desired price.

    Keep <= mm_spread_ticks of every regime, else a stale opposite quote can
    be crossed by the MM's own fresh quote.
    """
    ref_reversion: float = 0.05
    """Per-step pull of the reference price toward the prevailing mid."""
    ref_noise_std: float = 0.4
    """Std (ticks) of the per-step Gaussian reference-price innovation."""

    def __post_init__(self) -> None:
        if self.size_lots <= 0:
            raise ValueError("size_lots must be positive")
        if self.inventory_cap_lots < self.size_lots:
            raise ValueError("inventory_cap_lots must be >= size_lots")
        if self.requote_threshold_ticks < 1:
            raise ValueError("requote_threshold_ticks must be >= 1")
        if not 0.0 <= self.ref_reversion <= 1.0:
            raise ValueError("ref_reversion must be in [0, 1]")
        if self.ref_noise_std < 0:
            raise ValueError("ref_noise_std must be >= 0")


DEFAULT_REGIMES: Final[tuple[RegimeParams, ...]] = (
    RegimeParams(
        regime_id="calm",
        limit_rate=6.0,
        market_rate=1.2,
        cancel_rate=3.0,
        imbalance=0.0,
        mm_spread_ticks=4,
        hazard=0.005,
    ),
    RegimeParams(
        regime_id="stressed",
        limit_rate=9.0,
        market_rate=4.0,
        cancel_rate=5.0,
        imbalance=-0.3,
        mm_spread_ticks=10,
        hazard=0.04,
    ),
)


@dataclass(frozen=True)
class BackgroundConfig:
    """Full configuration of the background market."""

    initial_price_ticks: int = 1000
    """Bootstrap anchor: MM reference start, and ZI fallback on an empty book."""
    regimes: tuple[RegimeParams, ...] = DEFAULT_REGIMES
    initial_regime_id: str = "calm"
    schedule: tuple[tuple[int, str], ...] = ()
    """(step, regime_id) forced switches; a scheduled switch overrides a
    hazard switch landing on the same step."""
    zi: ZIConfig = ZIConfig()
    mm: MMConfig = MMConfig()
    n_market_makers: int = 2

    def __post_init__(self) -> None:
        if self.initial_price_ticks <= 0:
            raise ValueError("initial_price_ticks must be positive")
        if not self.regimes:
            raise ValueError("at least one regime is required")
        ids = [r.regime_id for r in self.regimes]
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate regime ids: {ids}")
        if self.initial_regime_id not in ids:
            raise ValueError(f"unknown initial regime {self.initial_regime_id!r}")
        for step, regime_id in self.schedule:
            if step < 0:
                raise ValueError(f"schedule step must be >= 0, got {step}")
            if regime_id not in ids:
                raise ValueError(f"schedule names unknown regime {regime_id!r}")
        if self.n_market_makers < 0:
            raise ValueError("n_market_makers must be >= 0")


# ---------------------------------------------------------------------------
# Per-step actor view of the market (start-of-step snapshot)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _OwnOrder:
    order_id: int
    side: Side
    price_ticks: int
    remaining_lots: int


@dataclass(frozen=True)
class _ActorView:
    """What one background actor sees at the start of a step.

    Built for every actor from the SAME pre-event book snapshot, before any
    of this step's events are applied.
    """

    best_bid: int | None
    best_ask: int | None
    own_orders: tuple[_OwnOrder, ...]
    """This actor's resting orders, sorted by order_id (canonical order for
    interpreting cancel-choice draws)."""
    inventory_lots: int

    @property
    def mid(self) -> float | None:
        if self.best_bid is None or self.best_ask is None:
            return None
        return (self.best_bid + self.best_ask) / 2.0


def _view_for(engine: MatchingEngine, actor_id: str) -> _ActorView:
    book = engine.book
    own = sorted(
        (
            _OwnOrder(
                order_id=o.order_id,
                side=o.side,
                price_ticks=o.price_ticks,
                remaining_lots=o.remaining_lots,
            )
            for o in book.all_resting_orders()
            if o.actor_id == actor_id
        ),
        key=lambda o: o.order_id,
    )
    inventory = engine.ground_truth_view(actor_id).inventory_lots
    return _ActorView(
        best_bid=book.best_bid,
        best_ask=book.best_ask,
        own_orders=tuple(own),
        inventory_lots=inventory,
    )


# ---------------------------------------------------------------------------
# Actor 1: zero-intelligence order flow (Santa Fe / Farmer style)
# ---------------------------------------------------------------------------

class ZIFlow:
    """Zero-intelligence order flow.

    Per step: Poisson numbers of limit arrivals (placement depth from a
    truncated power law relative to the same-side best), marketable orders
    (exponential sizes), and cancels of uniformly chosen own resting
    orders. Buy/sell symmetric up to the regime's imbalance parameter.
    """

    def __init__(self, actor_id: str, config: ZIConfig, initial_price_ticks: int) -> None:
        self.actor_id = actor_id
        self._config = config
        self._initial_price = initial_price_ticks
        depths = np.arange(1, config.max_depth_ticks + 1, dtype=np.float64)
        weights: npt.NDArray[np.float64] = depths ** (-config.depth_alpha)
        self._depth_cdf: npt.NDArray[np.float64] = np.cumsum(weights) / weights.sum()

    def _depth_from_uniform(self, u: float) -> int:
        """Truncated power law P(d) ~ d^-alpha on {1..max_depth} via inverse CDF."""
        idx = int(np.searchsorted(self._depth_cdf, u, side="right"))
        return min(idx, self._config.max_depth_ticks - 1) + 1

    @staticmethod
    def _side_from_uniform(u: float, imbalance: float) -> Side:
        return Side.BUY if u < (1.0 + imbalance) / 2.0 else Side.SELL

    def _limit_price(self, side: Side, depth: int, view: _ActorView) -> int:
        """Book-dependent interpretation site 1 (see module docstring).

        Depth 1 joins the same-side best; deeper values step away from it.
        Clamped so the placed order is never marketable.
        """
        if side == Side.BUY:
            if view.best_bid is not None:
                anchor = view.best_bid
            elif view.best_ask is not None:
                anchor = view.best_ask - 1
            else:
                anchor = self._initial_price
            price = anchor - (depth - 1)
            if view.best_ask is not None:
                price = min(price, view.best_ask - 1)
        else:
            if view.best_ask is not None:
                anchor = view.best_ask
            elif view.best_bid is not None:
                anchor = view.best_bid + 1
            else:
                anchor = self._initial_price
            price = anchor + (depth - 1)
            if view.best_bid is not None:
                price = max(price, view.best_bid + 1)
        return max(1, price)

    def _marketable_price(self, side: Side, view: _ActorView) -> int | None:
        """Book-dependent interpretation site 2 (see module docstring).

        None means the opposite side is empty and the order is skipped.
        """
        if side == Side.BUY:
            if view.best_ask is None:
                return None
            return view.best_ask + self._config.cross_ticks
        if view.best_bid is None:
            return None
        return max(1, view.best_bid - self._config.cross_ticks)

    def events(
        self, draws: _StepDraws, regime: RegimeParams, view: _ActorView
    ) -> list[ExchangeMessage]:
        cfg = self._config
        msgs: list[ExchangeMessage] = []

        # -- Limit arrivals: count first, then per-arrival indexed streams.
        n_limit = draws.poisson("n_limit_arrivals", regime.limit_rate)
        for i in range(n_limit):
            u_side = draws.uniform(f"limit_side:{i}")
            u_depth = draws.uniform(f"placement_depth:{i}")
            raw_size = draws.exponential(f"limit_size:{i}", cfg.limit_size_mean)
            side = self._side_from_uniform(u_side, regime.imbalance)
            depth = self._depth_from_uniform(u_depth)
            size = max(1, math.ceil(raw_size))
            price = self._limit_price(side, depth, view)  # book site 1
            msgs.append(
                PlaceLimit(
                    side=side,
                    price_ticks=price,
                    size_lots=size,
                    tif_steps=cfg.limit_tif_steps,
                )
            )

        # -- Marketable orders.
        n_market = draws.poisson("n_market_orders", regime.market_rate)
        for i in range(n_market):
            u_side = draws.uniform(f"market_side:{i}")
            raw_size = draws.exponential(f"market_size:{i}", cfg.market_size_mean)
            side = self._side_from_uniform(u_side, regime.imbalance)
            size = max(1, math.ceil(raw_size))
            cross_price = self._marketable_price(side, view)  # book site 2
            if cross_price is not None:
                msgs.append(
                    PlaceLimit(
                        side=side,
                        price_ticks=cross_price,
                        size_lots=size,
                        tif_steps=cfg.market_tif_steps,
                    )
                )

        # -- Cancels: the uniform is always drawn; mapping it onto the list
        #    of own resting orders is book site 3.
        n_cancel = draws.poisson("n_cancels", regime.cancel_rate)
        for i in range(n_cancel):
            u_cancel = draws.uniform(f"cancel_choice:{i}")
            if view.own_orders:
                idx = min(int(u_cancel * len(view.own_orders)), len(view.own_orders) - 1)
                msgs.append(Cancel(order_id=view.own_orders[idx].order_id))

        return msgs


# ---------------------------------------------------------------------------
# Actor 2: stabilizing market maker
# ---------------------------------------------------------------------------

class StabilizingMM:
    """Quotes both sides at the regime spread around a slowly mean-reverting
    reference price; requotes when its orders fill or drift past a
    threshold; stops quoting a side at the inventory cap.

    Only randomness: one Gaussian reference-price innovation per step, drawn
    unconditionally. Everything else is a deterministic function of the
    reference price, the regime, and the start-of-step book.
    """

    def __init__(self, actor_id: str, config: MMConfig, initial_price_ticks: int) -> None:
        self.actor_id = actor_id
        self._config = config
        self._ref_price = float(initial_price_ticks)

    @property
    def ref_price(self) -> float:
        """Harness-only ground truth; never observable by the agent (INV-11)."""
        return self._ref_price

    def events(
        self, draws: _StepDraws, regime: RegimeParams, view: _ActorView
    ) -> list[ExchangeMessage]:
        cfg = self._config

        # The innovation is drawn UNCONDITIONALLY, before any book use.
        noise = draws.normal("ref_price_noise", cfg.ref_noise_std)
        mid = view.mid
        if mid is not None:  # book site 4: reversion target is the mid
            self._ref_price += cfg.ref_reversion * (mid - self._ref_price)
        self._ref_price += noise

        half = max(1, regime.mm_spread_ticks // 2)
        base = round(self._ref_price)
        desired: dict[Side, int | None] = {
            Side.BUY: max(1, base - half),
            Side.SELL: max(1, base + half),
        }
        # Book site 6: inventory cap suppresses a side.
        if view.inventory_lots >= cfg.inventory_cap_lots:
            desired[Side.BUY] = None
        if view.inventory_lots <= -cfg.inventory_cap_lots:
            desired[Side.SELL] = None

        cancels: list[ExchangeMessage] = []
        places: list[ExchangeMessage] = []
        for side in (Side.BUY, Side.SELL):
            resting = [o for o in view.own_orders if o.side == side]
            want = desired[side]
            if want is None:
                cancels.extend(Cancel(order_id=o.order_id) for o in resting)
                continue
            # Book site 5: requote decision against current resting quotes.
            keep = (
                len(resting) == 1
                and abs(resting[0].price_ticks - want) < cfg.requote_threshold_ticks
                and resting[0].remaining_lots == cfg.size_lots
            )
            if not keep:
                cancels.extend(Cancel(order_id=o.order_id) for o in resting)
                places.append(
                    PlaceLimit(
                        side=side,
                        price_ticks=want,
                        size_lots=cfg.size_lots,
                        tif_steps=GTC,
                    )
                )
        # Cancels precede placements so a fresh quote can never cross the
        # MM's own stale opposite quote.
        return cancels + places


# ---------------------------------------------------------------------------
# Actor 3: regime controller (book-independent by construction)
# ---------------------------------------------------------------------------

RegimeSource = Literal["carry", "hazard", "schedule"]


@dataclass(frozen=True)
class RegimeRecord:
    """Harness-only ground truth (INV-11): the regime in force at one step."""

    step: int
    regime_id: str
    params: RegimeParams
    source: RegimeSource
    """How this step's regime came about: carried over, hazard switch, or
    scheduled switch (schedule overrides hazard on the same step)."""


class RegimeController:
    """Holds the current regime; switches on schedule and via per-step hazard.

    Consumes exactly two draws per step (hazard uniform, choice uniform),
    always, from streams keyed by REGIME_ACTOR_ID. It never reads the book,
    so the regime chain is a pure function of (root_seed, config).
    """

    def __init__(self, config: BackgroundConfig) -> None:
        self._regimes = config.regimes
        self._by_id = {r.regime_id: r for r in config.regimes}
        self._current = self._by_id[config.initial_regime_id]
        self._schedule = dict(config.schedule)
        self._log: list[RegimeRecord] = []

    @property
    def log(self) -> tuple[RegimeRecord, ...]:
        """Harness-only ground-truth log, one record per elapsed step."""
        return tuple(self._log)

    def advance(self, draws: _StepDraws, step: int) -> RegimeParams:
        # Both draws happen every step, switch or no switch.
        u_hazard = draws.uniform("regime_hazard")
        u_choice = draws.uniform("regime_choice")

        source: RegimeSource = "carry"
        if len(self._regimes) > 1 and u_hazard < self._current.hazard:
            others = [
                r for r in self._regimes if r.regime_id != self._current.regime_id
            ]
            idx = min(int(u_choice * len(others)), len(others) - 1)
            self._current = others[idx]
            source = "hazard"

        scheduled = self._schedule.get(step)
        if scheduled is not None:
            self._current = self._by_id[scheduled]
            source = "schedule"

        self._log.append(
            RegimeRecord(
                step=step,
                regime_id=self._current.regime_id,
                params=self._current,
                source=source,
            )
        )
        return self._current


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class BackgroundMarket:
    """Generates one deterministic background event sequence per engine step.

    Usage (the P3 harness owns this loop):

        market = BackgroundMarket(BackgroundConfig(), root_seed=SEED)
        for _ in range(n_steps):
            events = market.events_for_step(engine)
            engine.step(events, agent_id="agent", agent_action=...)

    All actor views are snapshotted from the pre-event book, so every actor
    generates against the same state and generation order cannot leak one
    actor's pending events into another's decisions.
    """

    def __init__(self, config: BackgroundConfig, root_seed: int) -> None:
        self._config = config
        self._root_seed = root_seed
        self._draw_log: list[DrawRecord] = []
        self._controller = RegimeController(config)
        self._zi = ZIFlow(ZI_ACTOR_ID, config.zi, config.initial_price_ticks)
        self._mms = tuple(
            StabilizingMM(mm_actor_id(i), config.mm, config.initial_price_ticks)
            for i in range(config.n_market_makers)
        )

    @property
    def actor_ids(self) -> tuple[str, ...]:
        """Ids of all order-submitting background actors."""
        return (self._zi.actor_id, *(mm.actor_id for mm in self._mms))

    @property
    def draw_log(self) -> tuple[DrawRecord, ...]:
        """Every raw draw so far, in draw order (harness/test-facing)."""
        return tuple(self._draw_log)

    @property
    def regime_log(self) -> tuple[RegimeRecord, ...]:
        """Harness-only regime ground truth (INV-11), one record per step."""
        return self._controller.log

    def _draws(self, actor_id: str, step: int) -> _StepDraws:
        return _StepDraws(self._root_seed, actor_id, step, self._draw_log)

    def events_for_step(
        self, engine: MatchingEngine
    ) -> list[tuple[str, ExchangeMessage]]:
        """Generate this step's background events, in deterministic order.

        Order: regime advance (draws only, no events), then each MM
        (stabilizers first so they seed the book at step 0), then the ZI
        flow. Event COUNTS may depend on the pre-step book; draw counts and
        values cannot (see module docstring).
        """
        step = engine.current_step
        regime = self._controller.advance(self._draws(REGIME_ACTOR_ID, step), step)

        events: list[tuple[str, ExchangeMessage]] = []
        for mm in self._mms:
            view = _view_for(engine, mm.actor_id)
            for msg in mm.events(self._draws(mm.actor_id, step), regime, view):
                events.append((mm.actor_id, msg))

        zi_view = _view_for(engine, self._zi.actor_id)
        for msg in self._zi.events(self._draws(self._zi.actor_id, step), regime, zi_view):
            events.append((self._zi.actor_id, msg))

        return events
