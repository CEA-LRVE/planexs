"""Fixtures shared by the tests of `planexs.goals`."""

from __future__ import annotations

import pytest

from planexs.core.CapabilityVocabulary import CapabilityVocabulary
from planexs.core.KindTable import KindTable


@pytest.fixture
def isolatedGoalKind(monkeypatch: pytest.MonkeyPatch) -> KindTable:
    """The process-wide kind table, swapped for a copy of it, for one test.

    A registered name is permanent and cannot be removed, and neither can a capability tag, so
    what a test registers must never reach the real registry or the real vocabulary: it would
    be there for every test that runs after it, and for the registry contract test. The table
    that this fixture swaps in holds a copy of every kind of the real one, with the very same
    entries (the `goal` kind has none yet, the goals arrive with the next tasks), so a test can
    register a fake next to them. Every kind is copied, and not the `goal` kind alone, because
    a goal test builds a `Problem`, whose outcomes look their kind up in the `outcome_kind`
    registry of the process-wide table. Its capability tags are checked against a copy of the
    process-wide vocabulary, so a tag that a test registers stays in the copy. `monkeypatch`
    swaps the real table back, which is left exactly as it was.
    """
    real = KindTable.default()
    source = real.vocabulary
    vocabulary = CapabilityVocabulary()
    for tag in source.tags():
        vocabulary.register(tag, source.describe(tag))
    table = KindTable(vocabulary)
    for kind in real.kinds():
        copy = table.define(kind)
        original = real.registry(kind)
        for name in original.names():
            copy.register(original.lookup(name))
    monkeypatch.setattr("planexs.core.KindTable._DEFAULT_TABLE", table)
    return table
