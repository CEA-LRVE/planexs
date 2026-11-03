"""Fixtures shared by the tests of the domain."""

from __future__ import annotations

import pytest

from planexs.core.CapabilityVocabulary import CapabilityVocabulary
from planexs.core.KindTable import KindTable
from planexs.domain.constraints.constraintRegistry import constraintRegistry
from planexs.domain.parameters.parameterRegistry import parameterRegistry


@pytest.fixture
def isolatedDomainKinds(monkeypatch: pytest.MonkeyPatch) -> KindTable:
    """The process-wide kind table, swapped for a copy of the domain's kinds, for one test.

    A registered name is permanent and cannot be removed, so a type that a test registers
    must never reach the real registry: it would be there for every test that runs after it,
    and for the registry contract test. The table that this fixture swaps in holds the two
    kinds of the domain, `parameter` and `constraint`, with the very entries of the real ones
    (`range`, `choice` and `derived`; `linear` and `nonlinear`), so a test can register a toy
    type next to them and rebuild a `SearchSpace` that uses it. A search space consults both
    kinds when it is written, so both must be there whichever of them the test is about.
    `monkeypatch` swaps the real table back, which is left exactly as it was. The swapped-in
    table checks capability tags against an empty vocabulary, so no tag is known to it.
    """
    real = {"parameter": parameterRegistry(), "constraint": constraintRegistry()}
    table = KindTable(CapabilityVocabulary())
    monkeypatch.setattr("planexs.core.KindTable._DEFAULT_TABLE", table)
    for kind, registry in real.items():
        isolated = table.define(kind)
        for name in registry.names():
            isolated.register(registry.lookup(name))
    return table
