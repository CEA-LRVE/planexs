"""Fixtures shared by the tests of the parameter types."""

from __future__ import annotations

import pytest

from planexs.core.CapabilityVocabulary import CapabilityVocabulary
from planexs.core.KindTable import KindTable
from planexs.domain.parameters.parameterRegistry import parameterRegistry


@pytest.fixture
def isolatedParameterTable(monkeypatch: pytest.MonkeyPatch) -> KindTable:
    """The process-wide kind table, swapped for a copy of its `parameter` kind, for one test.

    A registered name is permanent and cannot be removed, so a type that a test registers
    must never reach the real registry: it would be there for every test that runs after it,
    and for the registry contract test. The table that this fixture swaps in holds a
    `parameter` kind with the very entries of the real one (`range`, `choice` and `derived`),
    so a test can register a toy type next to them and rebuild a `SearchSpace` that uses it.
    `monkeypatch` swaps the real table back, which is left exactly as it was. The swapped-in
    table checks capability tags against an empty vocabulary, so no tag is known to it.
    """
    real = parameterRegistry()
    table = KindTable(CapabilityVocabulary())
    monkeypatch.setattr("planexs.core.KindTable._DEFAULT_TABLE", table)
    isolated = table.define("parameter")
    for name in real.names():
        isolated.register(real.lookup(name))
    return table
