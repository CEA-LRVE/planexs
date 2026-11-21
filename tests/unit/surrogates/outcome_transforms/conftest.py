"""Fixtures shared by the tests of `planexs.surrogates.outcome_transforms`."""

from __future__ import annotations

import pytest

from planexs.core.KindTable import KindTable


@pytest.fixture
def isolatedOutcomeTransformKind(isolatedSurrogateKind: KindTable) -> KindTable:
    """The process-wide kind table, swapped for a copy of it, for one test.

    It is the fixture of `tests/unit/surrogates/conftest.py` under the name of this kind. That
    fixture copies every kind of the real table, with the very same entries (the three real
    outcome transforms among them), so a test can register a fake transform next to the real
    ones, and neither the fake nor a capability tag it registers reaches the real registry or
    vocabulary. A registered name is permanent and cannot be removed, which is why the real
    table must not see it. The real table is put back after the test.
    """
    return isolatedSurrogateKind
