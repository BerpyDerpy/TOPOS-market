"""Tripwire 7 (INV-1, INV-11): Observation has exactly the declared fields.

Guards against later "helpful" additions — a feedback scalar, account
state, queue position, or anything about other agents must never appear.
"""

from __future__ import annotations

import dataclasses

from topos.contracts.market import Observation

DECLARED_FIELDS = ("step", "bids", "asks", "trades", "own_acks", "own_fills")


def test_observation_has_exactly_the_declared_fields() -> None:
    actual = tuple(field.name for field in dataclasses.fields(Observation))
    assert actual == DECLARED_FIELDS, (
        f"Observation fields changed: {actual} != {DECLARED_FIELDS}. "
        "Adding fields to Observation is an INV-1/INV-11 violation unless "
        "DESIGN.md's invariants are amended first."
    )
