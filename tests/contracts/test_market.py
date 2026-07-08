from __future__ import annotations

import pytest

from tests.exemplars import contract_exemplars
from topos.contracts.market import (
    GTC,
    N_LEVELS,
    BookLevel,
    Observation,
    PlaceLimit,
    Side,
)


def test_side_values_are_price_direction_signs() -> None:
    assert Side.BUY == 1
    assert Side.SELL == -1
    assert int(Side.BUY) + int(Side.SELL) == 0


def test_gtc_is_tif_zero() -> None:
    order = PlaceLimit(side=Side.BUY, price_ticks=100, size_lots=1, tif_steps=GTC)
    assert order.tif_steps == 0


@pytest.mark.parametrize("size_lots", [0, -1])
def test_place_limit_rejects_nonpositive_size(size_lots: int) -> None:
    with pytest.raises(ValueError):
        PlaceLimit(side=Side.SELL, price_ticks=100, size_lots=size_lots, tif_steps=0)


def test_place_limit_rejects_negative_tif() -> None:
    with pytest.raises(ValueError):
        PlaceLimit(side=Side.SELL, price_ticks=100, size_lots=1, tif_steps=-1)


def test_observation_requires_exactly_n_levels_per_side() -> None:
    exemplar = contract_exemplars()[Observation]
    assert isinstance(exemplar, Observation)
    short_bids = exemplar.bids[: N_LEVELS - 1]
    with pytest.raises(ValueError):
        Observation(
            step=exemplar.step,
            bids=short_bids,
            asks=exemplar.asks,
            trades=exemplar.trades,
            own_acks=exemplar.own_acks,
            own_fills=exemplar.own_fills,
        )


def test_empty_levels_are_padded_with_zero_size() -> None:
    level = BookLevel(price_ticks=100, size_lots=0)
    assert level.size_lots == 0
