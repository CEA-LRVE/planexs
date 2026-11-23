"""Fixtures shared by the tests of `planexs.surrogates.fitters`."""

from __future__ import annotations

import logging
from collections.abc import Iterator

import pytest

from planexs.core.KindTable import KindTable


@pytest.fixture
def isolatedFitterKind(isolatedSurrogateKind: KindTable) -> KindTable:
    """The process-wide kind table, swapped for a copy of it, for one test.

    It is the fixture of `tests/unit/surrogates/conftest.py` under the name of this kind. That
    fixture copies every kind of the real table, with the very same entries (the real fitter
    among them), and not the `surrogate` kind alone, so the swap holds the `fitter` kind as
    well: a test can register a fake fitter next to the real one, and neither the fake nor a
    capability tag it registers reaches the real registry or vocabulary. A registered name is
    permanent and cannot be removed, which is why the real table must not see it. The real
    table is put back after the test.
    """
    return isolatedSurrogateKind


class _Records(logging.Handler):
    """A handler that keeps what it is given, so that a test can read the engine's log."""

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


@pytest.fixture
def engineLog() -> Iterator[list[str]]:
    """The messages that BoTorch's own logger (named `botorch`) emits during a test, at
    DEBUG level, which is where its fitting routine says what each attempt did.

    The logger does not propagate to the root logger, so `caplog` does not see it: a handler is
    attached to it directly, and both the handler and the level are put back after the test.
    """
    logger = logging.getLogger("botorch")
    handler = _Records()
    previous_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    try:
        yield handler.messages
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)
