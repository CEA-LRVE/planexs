"""Fixtures shared by the tests of `planexs.policies`."""

from __future__ import annotations

import pytest

from planexs.core.CapabilityVocabulary import CapabilityVocabulary
from planexs.core.KindTable import KindTable


@pytest.fixture
def isolatedPolicyKind(monkeypatch: pytest.MonkeyPatch) -> KindTable:
    """The process-wide kind table, swapped for a copy of it, for one test.

    A registered name is permanent and cannot be removed, and neither can a capability tag, so
    what a test registers must never reach the real registry or the real vocabulary: it would
    be there for every test that runs after it, and for the registry contract test. The table
    that this fixture swaps in holds a copy of every kind of the real one, with the very same
    entries, so a test can register a fake next to them. The one exception is the `policy`
    kind, which starts empty: the contract tests of the kind count and list the policies that
    they register, and the real policies (the composed policy, random search and the ones to
    come) must not change what they see. Every other kind is copied, because a policy test
    builds a `Problem`, whose outcomes look their kind up in the `outcome_kind` registry of the
    process-wide table, and a goal that checks its own tags looks itself up in the `goal`
    registry. Its capability tags are checked against a copy of the process-wide vocabulary, so
    a tag that a test registers stays in the copy. `monkeypatch` swaps the real table back,
    which is left exactly as it was.
    """
    real = KindTable.default()
    source = real.vocabulary
    vocabulary = CapabilityVocabulary()
    for tag in source.tags():
        vocabulary.register(tag, source.describe(tag))
    table = KindTable(vocabulary)
    for kind in real.kinds():
        copy = table.define(kind)
        if kind == "policy":
            continue
        original = real.registry(kind)
        for name in original.names():
            copy.register(original.lookup(name))
    monkeypatch.setattr("planexs.core.KindTable._DEFAULT_TABLE", table)
    return table
