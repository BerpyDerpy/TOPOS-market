"""Named counter-based RNG streams (INV-9).

Every random draw in the environment flows through a stream keyed by
(actor_id, step, purpose). A stream's contents are a pure function of
(root_seed, key): the same key yields the same draws no matter what any
other actor drew, in what order, or whether the agent exists at all. The
agent's presence therefore cannot perturb any other actor's randomness
except causally, through the visible book state.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StreamKey:
    """Names one independent randomness stream."""

    actor_id: str
    step: int
    purpose: str


def make_rng(root_seed: int, key: StreamKey) -> np.random.Generator:
    """Deterministic, stream-independent generator for (root_seed, key).

    The key material is a SHA-256 digest of the canonical `repr` of
    (root_seed, actor_id, step, purpose) — stable across processes and
    platforms (unlike builtin `hash`, which is salted per process), and
    collision-free across distinct field values because `repr` quotes and
    escapes strings. The digest seeds a numpy `SeedSequence` driving a
    counter-based Philox generator.
    """
    payload = repr((int(root_seed), str(key.actor_id), int(key.step), str(key.purpose)))
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    entropy = [
        int.from_bytes(digest[i : i + 4], "little") for i in range(0, len(digest), 4)
    ]
    seq = np.random.SeedSequence(entropy)
    return np.random.Generator(np.random.Philox(seq))
