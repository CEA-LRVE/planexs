"""Fixtures shared by the tests of the acquisition contract (`planexs.acquisition`)."""

from __future__ import annotations

import pytest

from planexs.core.CapabilityVocabulary import CapabilityVocabulary
from planexs.core.KindTable import KindTable


@pytest.fixture
def isolatedAcquisitionKind(monkeypatch: pytest.MonkeyPatch) -> KindTable:
    """The process-wide kind table, swapped for a copy of it, for one test.

    A registered name is permanent and cannot be removed, and neither can a capability tag, so
    what a test registers must never reach the real registry or the real vocabulary: it would
    be there for every test that runs after it, and for the registry contract test. The table
    that this fixture swaps in holds a copy of every kind of the real one, with the very same
    entries, except the `acquisition` kind, which starts empty: it is the kind under test, so a
    test of the registration mechanism sees exactly the acquisitions that it registers,
    whatever real ones (`log_ei_v1` and the later ones) the package has registered by then.
    Every other kind is copied with its entries, because a test builds a `Problem` and an
    `OptimizeGoal`, which look their kinds up in the registries of the process-wide table. Its
    capability tags are checked against a copy of the process-wide vocabulary, so a tag that a
    test registers stays in the copy. `monkeypatch` swaps the real table back, which is left
    exactly as it was.
    """
    real = KindTable.default()
    source = real.vocabulary
    vocabulary = CapabilityVocabulary()
    for tag in source.tags():
        vocabulary.register(tag, source.describe(tag))
    table = KindTable(vocabulary)
    for kind in real.kinds():
        copy = table.define(kind)
        if kind == "acquisition":
            continue  # the kind under test starts empty, see above
        original = real.registry(kind)
        for name in original.names():
            copy.register(original.lookup(name))
    monkeypatch.setattr("planexs.core.KindTable._DEFAULT_TABLE", table)
    return table
