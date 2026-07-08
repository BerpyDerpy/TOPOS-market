from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from topos.contracts.rng import StreamKey, make_rng

_ids = st.text(min_size=0, max_size=20)
_steps = st.integers(min_value=0, max_value=10**9)
_keys = st.builds(StreamKey, actor_id=_ids, step=_steps, purpose=_ids)


@settings(max_examples=50)
@given(root_seed=st.integers(min_value=0, max_value=2**63 - 1), key=_keys)
def test_any_key_is_reproducible(root_seed: int, key: StreamKey) -> None:
    a = make_rng(root_seed, key).random(8)
    b = make_rng(root_seed, key).random(8)
    np.testing.assert_array_equal(a, b)


@settings(max_examples=50)
@given(key_a=_keys, key_b=_keys)
def test_streams_never_interfere(key_a: StreamKey, key_b: StreamKey) -> None:
    root_seed = 7
    baseline = make_rng(root_seed, key_a).random(8)
    make_rng(root_seed, key_b).random(512)  # consume another stream
    again = make_rng(root_seed, key_a).random(8)
    np.testing.assert_array_equal(baseline, again)


@settings(max_examples=50)
@given(key_a=_keys, key_b=_keys)
def test_distinct_keys_give_distinct_streams(
    key_a: StreamKey, key_b: StreamKey
) -> None:
    if key_a == key_b:
        return
    a = make_rng(7, key_a).random(8)
    b = make_rng(7, key_b).random(8)
    assert not np.array_equal(a, b)
