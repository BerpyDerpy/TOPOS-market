"""Tripwire 6 (INV-9): named RNG streams are mutually independent.

Draws for key (A, step, purpose) must be identical whether or not draws for
(B, step, purpose) — or any other stream — were consumed first. This is
what guarantees the agent's presence cannot perturb any other actor's
randomness except causally, through the visible book state.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from topos.contracts.rng import StreamKey, make_rng

ROOT_SEED = 20260708


def _draws(key: StreamKey, n: int = 64) -> npt.NDArray[np.float64]:
    return make_rng(ROOT_SEED, key).random(n)


def test_stream_is_unaffected_by_other_streams_being_consumed() -> None:
    key_a = StreamKey(actor_id="background_A", step=5, purpose="arrivals")
    key_b = StreamKey(actor_id="background_B", step=5, purpose="arrivals")

    baseline = _draws(key_a)

    # Consume a large amount of another actor's stream, then redraw A.
    make_rng(ROOT_SEED, key_b).random(100_000)
    again = _draws(key_a)

    np.testing.assert_array_equal(baseline, again)


def test_same_key_is_reproducible_across_generator_instances() -> None:
    key = StreamKey(actor_id="agent", step=12, purpose="latency")
    np.testing.assert_array_equal(_draws(key), _draws(key))


def test_distinct_key_components_give_distinct_streams() -> None:
    base = StreamKey(actor_id="agent", step=12, purpose="latency")
    variants = (
        StreamKey(actor_id="agent2", step=12, purpose="latency"),
        StreamKey(actor_id="agent", step=13, purpose="latency"),
        StreamKey(actor_id="agent", step=12, purpose="sizes"),
    )
    baseline = _draws(base)
    for variant in variants:
        assert not np.array_equal(baseline, _draws(variant)), variant


def test_distinct_root_seeds_give_distinct_streams() -> None:
    key = StreamKey(actor_id="agent", step=12, purpose="latency")
    a = make_rng(ROOT_SEED, key).random(64)
    b = make_rng(ROOT_SEED + 1, key).random(64)
    assert not np.array_equal(a, b)


def test_key_fields_do_not_concatenate_ambiguously() -> None:
    # ("ab", "c") and ("a", "bc") must not collide.
    a = _draws(StreamKey(actor_id="ab", step=1, purpose="c"))
    b = _draws(StreamKey(actor_id="a", step=1, purpose="bc"))
    assert not np.array_equal(a, b)
