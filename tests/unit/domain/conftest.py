"""Fixtures shared by the tests of the domain."""

from __future__ import annotations

import pytest

from planexs.core.CapabilityVocabulary import CapabilityVocabulary
from planexs.core.KindTable import KindTable
from planexs.domain.constraints.constraintRegistry import constraintRegistry
from planexs.domain.outcome_kinds.outcomeKindRegistry import outcomeKindRegistry
from planexs.domain.parameters.parameterRegistry import parameterRegistry


@pytest.fixture
def isolatedDomainKinds(monkeypatch: pytest.MonkeyPatch) -> KindTable:
    """The process-wide kind table, swapped for a copy of the domain's kinds, for one test.

    A registered name is permanent and cannot be removed, and neither can a capability tag, so
    what a test registers must never reach the real registries or the real vocabulary: it
    would be there for every test that runs after it, and for the registry contract test. The
    table that this fixture swaps in holds the three kinds of the domain, `parameter`,
    `constraint` and `outcome_kind`, with the very entries of the real ones (`range`, `choice`
    and `derived`; `linear` and `nonlinear`; `continuous` and `binary`), so a test can register
    a toy type next to them and rebuild a `SearchSpace` or an `Outcome` that uses it. A search
    space consults both of its kinds when it is written, so all of them must be there whichever
    one the test is about.

    The table checks capability tags against a copy of the process-wide vocabulary: the tags
    of the real kinds are known to it, and a tag that a toy kind registers stays in the copy.
    `monkeypatch` swaps the real table back, which is left exactly as it was.
    """
    real = {
        "parameter": parameterRegistry(),
        "constraint": constraintRegistry(),
        "outcome_kind": outcomeKindRegistry(),
    }
    source = CapabilityVocabulary.default()
    vocabulary = CapabilityVocabulary()
    for tag in source.tags():
        vocabulary.register(tag, source.describe(tag))
    table = KindTable(vocabulary)
    monkeypatch.setattr("planexs.core.KindTable._DEFAULT_TABLE", table)
    for kind, registry in real.items():
        isolated = table.define(kind)
        for name in registry.names():
            isolated.register(registry.lookup(name))
    return table
