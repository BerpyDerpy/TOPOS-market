"""Tripwire 4 (INV-5): the cognitive self-state carries no PnL of any kind.

Reflects over SelfStateCognitive's fields — and, transitively, over every
dataclass type reachable from them — and fails on any field name matching
the account-vocabulary pattern. Also pins the structural half of INV-5:
SelfStateFull must not be a subclass of SelfStateCognitive, so a
PnL-bearing object can never satisfy an interface typed for the cognitive
view.
"""

from __future__ import annotations

import dataclasses
import re
import typing
from typing import Iterator

from topos.contracts.workspace import SelfStateCognitive, SelfStateFull

FORBIDDEN_FIELD = re.compile(r"pnl|profit|drawdown|wealth", re.IGNORECASE)


def _reachable_dataclasses(tp: object, seen: set[type]) -> Iterator[type]:
    if isinstance(tp, type) and dataclasses.is_dataclass(tp):
        if tp in seen:
            return
        seen.add(tp)
        yield tp
        hints = typing.get_type_hints(tp)
        for field in dataclasses.fields(tp):
            yield from _reachable_dataclasses(hints[field.name], seen)
    else:
        for arg in typing.get_args(tp):
            yield from _reachable_dataclasses(arg, seen)


def test_cognitive_view_has_no_pnl_fields() -> None:
    offenders: list[str] = []
    for cls in _reachable_dataclasses(SelfStateCognitive, set()):
        for field in dataclasses.fields(cls):
            if FORBIDDEN_FIELD.search(field.name):
                offenders.append(f"{cls.__qualname__}.{field.name}")
    assert not offenders, (
        "INV-5 tripwire: account vocabulary reachable from "
        "SelfStateCognitive: " + ", ".join(offenders)
    )


def test_full_state_is_not_a_subtype_of_the_cognitive_view() -> None:
    assert not issubclass(SelfStateFull, SelfStateCognitive), (
        "INV-5 tripwire: SelfStateFull subclasses SelfStateCognitive, so "
        "PnL-bearing objects could flow through cognitive-view interfaces"
    )


def test_full_state_does_carry_the_account_fields() -> None:
    names = {field.name for field in dataclasses.fields(SelfStateFull)}
    assert {"realized_pnl", "unrealized_pnl", "gross_exposure"} <= names
