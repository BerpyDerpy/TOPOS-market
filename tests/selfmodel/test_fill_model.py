"""FillModel: conditioning, trial resolution, EIG identity and saturation.

``test_fill_eig_saturation`` is anti-churn at unit scale (the design's
Layer-1 mechanism): repeated outcomes in ONE bucket must drive that
bucket's EIG toward 0 — the tenth fill is boring — while a never-probed
bucket keeps its prior EIG — the unprobed question is not. An
unconditional fill model cannot pass it.

``test_fill_eig_matches_mc`` is the P4-mandated INV-3 tripwire pattern:
the closed-form digamma EIG must equal a brute-force Monte Carlo estimate
of I(p; Y). Do NOT weaken the tolerances; fix the math.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tests.beliefs.conftest import make_obs, null_probe
from tests.selfmodel.conftest import committed_probe, events, plain_obs
from topos.beliefs.core import BetaPosterior
from topos.contracts.beliefs import ProbeSpec
from topos.contracts.intent import FILL_RATE
from topos.contracts.market import (
    GTC,
    Ack,
    AckStatus,
    Fill,
    Liquidity,
    PlaceLimit,
    Side,
)
from topos.selfmodel import FillModel, imbalance_band_of, offset_band_of

ATOL = 0.015
RTOL = 0.03


# =====================================================================
# Banding (the conditioning axes)
# =====================================================================


def test_offset_bands_follow_the_book() -> None:
    # BUY against best_bid=999 / best_ask=1001.
    assert offset_band_of(Side.BUY, 1001, 999, 1001) == "cross"
    assert offset_band_of(Side.BUY, 1000, 999, 1001) == "touch"  # improves
    assert offset_band_of(Side.BUY, 999, 999, 1001) == "touch"
    assert offset_band_of(Side.BUY, 997, 999, 1001) == "near"
    assert offset_band_of(Side.BUY, 995, 999, 1001) == "deep"
    # SELL mirrors.
    assert offset_band_of(Side.SELL, 999, 999, 1001) == "cross"
    assert offset_band_of(Side.SELL, 1001, 999, 1001) == "touch"
    assert offset_band_of(Side.SELL, 1004, 999, 1001) == "near"
    assert offset_band_of(Side.SELL, 1005, 999, 1001) == "deep"
    # One-sided books: no reference on the relevant sides => front.
    assert offset_band_of(Side.BUY, 1000, None, None) == "touch"


def test_imbalance_bands_tripartition() -> None:
    assert imbalance_band_of(-0.5) == "sell_heavy"
    assert imbalance_band_of(-1.0 / 3.0) == "balanced"
    assert imbalance_band_of(0.0) == "balanced"
    assert imbalance_band_of(1.0 / 3.0) == "balanced"
    assert imbalance_band_of(0.5) == "buy_heavy"


# =====================================================================
# Trial machinery
# =====================================================================


def _place_at_touch(
    model: FillModel, step: int, order_id: int, size_lots: int = 1
) -> None:
    """One ACCEPTED buy at the touch (999) into the balanced book."""
    place = PlaceLimit(Side.BUY, 999, size_lots, tif_steps=GTC)
    ack = Ack(order_id=order_id, status=AckStatus.ACCEPTED, step=step)
    model.update(
        plain_obs(step, own_acks=(ack,)),
        events(step, messages=(place,), acks=(ack,)),
    )


def _run_touch_trials(
    model: FillModel, n_rounds: int, fill_pattern, start_step: int = 1
):
    """Place-at-touch trials, resolved (fill or not) within the horizon.

    ``fill_pattern(r)`` -> filled lots for round r. Yields (round, step)
    after each round's resolution so callers can checkpoint.
    """
    step = start_step
    for r in range(n_rounds):
        _place_at_touch(model, step, order_id=r)
        step += 1
        filled = fill_pattern(r)
        fills = ()
        if filled > 0:
            fills = (
                Fill(
                    order_id=r, price_ticks=999, size_lots=filled,
                    liquidity=Liquidity.MAKER, step=step,
                ),
            )
        model.update(plain_obs(step, own_fills=fills), events(step, fills=fills))
        step += 1
        model.update(plain_obs(step), events(step))  # horizon passes
        step += 1
        yield r, step


TOUCH_PROBE = committed_probe(side=1.0, offset_ticks=1.0, horizon_steps=2)
DEEP_PROBE = committed_probe(side=-1.0, offset_ticks=6.0, horizon_steps=2)


def test_fill_eig_saturation() -> None:
    """The tenth fill is boring; the unprobed question is not."""
    model = FillModel(horizon_steps=2, size_budget_lots=10)
    model.update(plain_obs(0), events(0))
    prior_eig = model.eig_nats(TOUCH_PROBE)

    checkpoints = (5, 15, 40, 60)
    eig_at: dict[int, float] = {}
    for r, _step in _run_touch_trials(
        model, 60, fill_pattern=lambda r: 1 if r % 10 < 7 else 0
    ):
        if r + 1 in checkpoints:
            eig_at[r + 1] = model.eig_nats(TOUCH_PROBE)

    values = [eig_at[c] for c in checkpoints]
    assert all(later < earlier for earlier, later in zip(values, values[1:])), (
        f"probed-bucket EIG not decreasing along checkpoints: {eig_at}"
    )
    assert values[-1] < 0.02
    assert values[-1] < 0.05 * prior_eig
    # The never-probed bucket still offers its full prior EIG.
    untouched = model.eig_nats(DEEP_PROBE)
    assert untouched == pytest.approx(prior_eig)
    assert values[-1] < 0.15 * untouched
    # Saturation is epistemic, not aleatoric: the outcome itself stays
    # genuinely noisy (p ~ 0.7), only the PARAMETER is settled.
    terms = model.eig_breakdown(TOUCH_PROBE)
    assert terms is not None
    assert terms.expected_conditional_entropy_nats > 0.3
    assert math.isfinite(terms.predictive_entropy_nats)


def test_outcomes_stay_in_their_own_bucket() -> None:
    """Layer-1 conditioning: evidence flows ONLY to the exercised bucket."""
    model = FillModel(horizon_steps=2, size_budget_lots=10)
    model.update(plain_obs(0), events(0))
    for _ in _run_touch_trials(model, 20, fill_pattern=lambda r: 1):
        pass
    exercised = (Side.BUY, "touch", "balanced")
    for key, cell in model.cells.items():
        if key == exercised:
            assert cell.a + cell.b == pytest.approx(2.0 + 20.0)
        else:
            assert (cell.a, cell.b) == (1.0, 1.0), (
                f"bucket {key} moved without being exercised"
            )


def test_same_action_different_context_hits_different_buckets() -> None:
    """The same order in a buy-heavy vs sell-heavy book is a different
    experiment: context is part of the conditioning."""
    heavy_bids = [(999 - i, 40) for i in range(10)]
    thin_asks = [(1001 + i, 5) for i in range(10)]
    model = FillModel(horizon_steps=2, size_budget_lots=10)
    model.update(make_obs(0, heavy_bids, thin_asks), events(0))
    _place_at_touch(model, 1, order_id=0)
    fill = Fill(order_id=0, price_ticks=999, size_lots=1,
                liquidity=Liquidity.MAKER, step=2)
    model.update(plain_obs(2, own_fills=(fill,)), events(2, fills=(fill,)))
    model.update(plain_obs(3), events(3))
    buy_heavy_cell = model.cells[(Side.BUY, "touch", "buy_heavy")]
    balanced_cell = model.cells[(Side.BUY, "touch", "balanced")]
    assert buy_heavy_cell.a + buy_heavy_cell.b == pytest.approx(3.0)
    assert (balanced_cell.a, balanced_cell.b) == (1.0, 1.0)


def test_partial_fill_updates_fractionally() -> None:
    model = FillModel(horizon_steps=2, size_budget_lots=10)
    model.update(plain_obs(0), events(0))
    _place_at_touch(model, 1, order_id=0, size_lots=4)
    fill = Fill(order_id=0, price_ticks=999, size_lots=1,
                liquidity=Liquidity.MAKER, step=2)
    model.update(plain_obs(2, own_fills=(fill,)), events(2, fills=(fill,)))
    model.update(plain_obs(3), events(3))  # horizon: resolve at 1/4
    cell = model.cells[(Side.BUY, "touch", "balanced")]
    assert cell.a == pytest.approx(1.25)
    assert cell.b == pytest.approx(1.75)


def test_cancel_before_horizon_censors_the_trial() -> None:
    """Discarded, not counted as a failure: the horizon outcome was never
    observed, and folding own impatience in as no-fills would bias every
    bucket downward."""
    model = FillModel(horizon_steps=5, size_budget_lots=10)
    model.update(plain_obs(0), events(0))
    _place_at_touch(model, 1, order_id=0)
    cancel = Ack(order_id=0, status=AckStatus.CANCELED, step=2)
    model.update(plain_obs(2, own_acks=(cancel,)), events(2))
    assert model.open_trials == 0
    cell = model.cells[(Side.BUY, "touch", "balanced")]
    assert (cell.a, cell.b) == (1.0, 1.0)


def test_expiry_at_horizon_resolves_the_trial() -> None:
    model = FillModel(horizon_steps=2, size_budget_lots=10)
    model.update(plain_obs(0), events(0))
    _place_at_touch(model, 1, order_id=0)
    expired = Ack(order_id=0, status=AckStatus.EXPIRED, step=3)
    model.update(plain_obs(3, own_acks=(expired,)), events(3))
    cell = model.cells[(Side.BUY, "touch", "balanced")]
    assert (cell.a, cell.b) == (1.0, 2.0)  # resolved as no-fill


def test_null_and_directionless_probes_have_zero_eig() -> None:
    """No order, no outcome: this hypothesis's information must be bought
    by acting (the null's EIG lives in the world models, INV-4)."""
    model = FillModel(horizon_steps=2, size_budget_lots=10)
    model.update(plain_obs(0), events(0))
    assert model.eig_nats(null_probe(FILL_RATE, horizon_steps=2)) == 0.0
    directionless = committed_probe(side=0.0, offset_ticks=1.0, horizon_steps=2)
    assert model.eig_nats(directionless) == 0.0
    zero_size = committed_probe(
        side=1.0, offset_ticks=1.0, size_frac=0.0, horizon_steps=2
    )
    assert model.eig_nats(zero_size) == 0.0


def test_forgetting_reinflates_the_probed_bucket() -> None:
    model = FillModel(horizon_steps=2, size_budget_lots=10)
    model.update(plain_obs(0), events(0))
    for _ in _run_touch_trials(model, 40, fill_pattern=lambda r: 1 if r % 2 else 0):
        pass
    cell = model.cells[(Side.BUY, "touch", "balanced")]
    entropy_before = cell.entropy_nats()
    eig_before = model.eig_nats(TOUCH_PROBE)
    model.forget(0.3)
    assert cell.entropy_nats() > entropy_before
    assert model.eig_nats(TOUCH_PROBE) > eig_before
    # Forgetting moves sufficient statistics toward the prior, never past it.
    assert cell.a >= cell.prior_a
    assert cell.b >= cell.prior_b


def test_probe_rejects_nonpositive_horizon() -> None:
    model = FillModel(horizon_steps=2)
    bad = ProbeSpec(intent=TOUCH_PROBE.intent, horizon_steps=0)
    with pytest.raises(ValueError):
        model.eig_nats(bad)
    with pytest.raises(ValueError):
        FillModel(horizon_steps=0)


# =====================================================================
# INV-3: eig_nats must BE mutual information (P4's mandatory pattern)
# =====================================================================


def mc_beta_bernoulli_mi(
    a: float, b: float, rng: np.random.Generator, n: int = 400_000
) -> float:
    """Brute-force I(p; Y): empirical marginal entropy (Miller-Madow)
    minus the sampled conditional entropy E[h(p)], no digamma anywhere."""
    p = rng.beta(a, b, n)
    y = rng.random(n) < p
    counts = np.array([n - y.sum(), y.sum()], dtype=np.float64)
    freq = counts[counts > 0] / n
    h_marginal = float(-np.sum(freq * np.log(freq)) + (len(freq) - 1) / (2.0 * n))
    with np.errstate(divide="ignore", invalid="ignore"):
        h_terms = -(
            np.where(p > 0.0, p * np.log(p), 0.0)
            + np.where(p < 1.0, (1.0 - p) * np.log1p(-p), 0.0)
        )
    return h_marginal - float(np.mean(h_terms))


def assert_close(analytic: float, mc: float, label: str) -> None:
    tol = ATOL + RTOL * abs(analytic)
    assert abs(analytic - mc) <= tol, (
        f"{label}: analytic {analytic:.5f} vs MC {mc:.5f} "
        f"(|diff| {abs(analytic - mc):.5f} > tol {tol:.5f}) — eig_nats is "
        f"not mutual information"
    )


POSTERIOR_GRID = [(1.0, 1.0), (2.5, 7.5), (40.0, 12.0), (0.6, 0.6)]


@pytest.mark.parametrize("a,b", POSTERIOR_GRID)
def test_beta_cell_eig_matches_mc(a: float, b: float) -> None:
    cell = BetaPosterior(1.0, 1.0)
    cell.a, cell.b = a, b
    analytic = cell.eig_terms_bernoulli().eig_nats
    rng = np.random.default_rng(20260709)
    mc = mc_beta_bernoulli_mi(a, b, rng)
    assert_close(analytic, mc, f"Beta-Bernoulli cell a={a} b={b}")


@pytest.mark.parametrize("a,b", [(1.0, 1.0), (2.5, 7.5), (40.0, 12.0)])
def test_fill_eig_matches_mc(a: float, b: float) -> None:
    """Module-level: the probe's EIG is the MI of ITS bucket's parameter."""
    model = FillModel(horizon_steps=2, size_budget_lots=10)
    model.update(plain_obs(0), events(0))
    key = model.bucket_for_intent(TOUCH_PROBE.intent)
    assert key == (Side.BUY, "touch", "balanced")
    model.cells[key].a, model.cells[key].b = a, b
    analytic = model.eig_nats(TOUCH_PROBE)
    rng = np.random.default_rng(90210)
    mc = mc_beta_bernoulli_mi(a, b, rng)
    assert_close(analytic, mc, f"FillModel touch bucket a={a} b={b}")
