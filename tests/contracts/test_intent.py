from __future__ import annotations

import pytest

from topos.contracts.intent import (
    FLATTEN_INTENT,
    KNOWN_HYPOTHESIS_IDS,
    NULL_THRESHOLD,
    SELF_TRAJECTORY,
    Intent,
    flatten_intent,
)


def _intent(**overrides: float | str) -> Intent:
    base: dict[str, float | str] = {
        "side": 0.5,
        "offset_ticks": 1.0,
        "size_frac": 0.5,
        "patience": 0.5,
        "target_id": "fair_value",
        "commitment": 0.9,
    }
    base.update(overrides)
    return Intent(**base)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("side", 1.5),
        ("side", -1.5),
        ("size_frac", -0.1),
        ("size_frac", 1.1),
        ("patience", -0.1),
        ("patience", 1.1),
        ("commitment", -0.1),
        ("commitment", 1.1),
        ("target_id", ""),
    ],
)
def test_out_of_range_fields_raise(field: str, value: float | str) -> None:
    with pytest.raises(ValueError):
        _intent(**{field: value})


def test_negative_offset_means_crossing_and_is_legal() -> None:
    assert _intent(offset_ticks=-2.0).offset_ticks == -2.0


def test_commitment_below_threshold_is_the_null_action() -> None:
    assert _intent(commitment=NULL_THRESHOLD - 0.01).is_null
    assert not _intent(commitment=NULL_THRESHOLD).is_null


def test_flatten_opposes_inventory_sign() -> None:
    long_flatten = flatten_intent(5)
    short_flatten = flatten_intent(-5)
    assert long_flatten.side == -1.0
    assert short_flatten.side == 1.0
    assert not long_flatten.is_null
    assert not short_flatten.is_null


def test_flatten_is_passive_first_and_targets_self_trajectory() -> None:
    intent = flatten_intent(3)
    assert intent.patience == 1.0
    assert intent.offset_ticks >= 0.0
    assert intent.target_id == SELF_TRAJECTORY


def test_flatten_with_zero_inventory_is_null() -> None:
    assert flatten_intent(0).is_null


def test_flatten_is_flatten_and_sizable() -> None:
    assert flatten_intent(3).is_flatten
    assert not flatten_intent(0).is_flatten
    assert flatten_intent(3, size_frac=0.25).size_frac == 0.25


def test_spec_name_is_the_same_constructor() -> None:
    assert FLATTEN_INTENT is flatten_intent


def test_known_hypothesis_ids_are_unique_and_nonempty() -> None:
    assert len(set(KNOWN_HYPOTHESIS_IDS)) == len(KNOWN_HYPOTHESIS_IDS) == 7
    assert all(KNOWN_HYPOTHESIS_IDS)
