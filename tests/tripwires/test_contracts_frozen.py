"""Tripwire 5: every contract dataclass is frozen; mutation raises.

Discovers dataclasses by walking topos.contracts, so a contract added later
is checked automatically — and must also gain an exemplar in
tests/exemplars.py, or this tripwire fails.
"""

from __future__ import annotations

import dataclasses
import importlib
import pkgutil

import pytest

import topos.contracts
from tests.exemplars import contract_exemplars


def _contract_dataclasses() -> set[type]:
    classes: set[type] = set()
    for info in pkgutil.iter_modules(
        topos.contracts.__path__, prefix="topos.contracts."
    ):
        module = importlib.import_module(info.name)
        for obj in vars(module).values():
            if (
                isinstance(obj, type)
                and dataclasses.is_dataclass(obj)
                and obj.__module__ == info.name
            ):
                classes.add(obj)
    return classes


def test_discovery_finds_the_contracts() -> None:
    assert len(_contract_dataclasses()) >= 20, (
        "suspiciously few contract dataclasses discovered; layout changed?"
    )


def test_every_contract_dataclass_is_frozen() -> None:
    not_frozen = sorted(
        cls.__qualname__
        for cls in _contract_dataclasses()
        if not cls.__dataclass_params__.frozen  # type: ignore[attr-defined]
    )
    assert not not_frozen, f"unfrozen contract dataclasses: {not_frozen}"


def test_mutating_any_contract_instance_raises() -> None:
    exemplars = contract_exemplars()
    missing = sorted(
        cls.__qualname__ for cls in _contract_dataclasses() - set(exemplars)
    )
    assert not missing, f"add exemplars in tests/exemplars.py for: {missing}"
    for cls, instance in exemplars.items():
        field_name = dataclasses.fields(cls)[0].name
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(instance, field_name, None)
