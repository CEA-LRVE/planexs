"""Fixtures shared by the tests of `planexs.core`."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import pytest
from nested_config_schemas import NESTED_COMPONENTS
from pydantic import BaseModel, ConfigDict

from planexs.core.CapabilityVocabulary import CapabilityVocabulary
from planexs.core.ComponentEntry import ComponentEntry
from planexs.core.KindTable import KindTable


class DummyConfig(BaseModel):
    """A configuration schema for test components."""

    model_config = ConfigDict(extra="forbid")


def dummyBuilder() -> None:
    """A builder for test components."""


@pytest.fixture
def makeEntry() -> Callable[..., ComponentEntry]:
    """A factory of complete test entries: `makeEntry("name", needs=(...), supports=(...))`."""

    def make(
        name: Any = "a_v1",
        *,
        needs: Iterable[str] = (),
        supports: Iterable[str] = (),
        description: str = "A component for tests.",
    ) -> ComponentEntry:
        return ComponentEntry(
            name=name,
            builder=dummyBuilder,
            config_schema=DummyConfig,
            description=description,
            needs=frozenset(needs),
            supports=frozenset(supports),
        )

    return make


@pytest.fixture
def vocabulary() -> CapabilityVocabulary:
    """An empty vocabulary of its own, sharing nothing with the process-wide one."""
    return CapabilityVocabulary()


@pytest.fixture
def isolatedKindTable(
    vocabulary: CapabilityVocabulary, monkeypatch: pytest.MonkeyPatch
) -> KindTable:
    """The process-wide kind table, swapped for an empty one for the duration of one test.

    `defineKind` records into the process-wide table, and kinds are defined at import time by
    modules that Python imports only once. Resetting the real table in a test would therefore
    lose real kinds for the rest of the session. Swapping it, and letting `monkeypatch` swap it
    back, leaves the real table exactly as it was. The swapped-in table checks tags against the
    empty `vocabulary` fixture.
    """
    table = KindTable(vocabulary)
    monkeypatch.setattr("planexs.core.KindTable._DEFAULT_TABLE", table)
    return table


@pytest.fixture
def isolatedVocabulary(monkeypatch: pytest.MonkeyPatch) -> CapabilityVocabulary:
    """The process-wide vocabulary, swapped for an empty one for the duration of one test."""
    vocabulary = CapabilityVocabulary()
    monkeypatch.setattr("planexs.core.CapabilityVocabulary._DEFAULT_VOCABULARY", vocabulary)
    return vocabulary


@pytest.fixture
def nestedKinds(isolatedKindTable: KindTable) -> KindTable:
    """An isolated kind table with the kinds and components of `nested_config_schemas`.

    Six kinds (kernel, design, surrogate, analysis, policy, stopping), each with its components
    registered under the name their schema pins as its `type`. The registries are reached with
    `nestedKinds.registry("policy")` and the like, and the fields marked `ComponentOf` in those
    schemas find them through the same table.
    """
    for kind, schemas in NESTED_COMPONENTS.items():
        registry = isolatedKindTable.define(kind)
        for schema in schemas:
            name = schema.model_fields["type"].default
            registry.register(
                ComponentEntry(
                    name=name,
                    builder=dummyBuilder,
                    config_schema=schema,
                    description=f"The {name} component.",
                )
            )
    return isolatedKindTable
