"""Rebuilds an `Outcome` from its `toDict()` form."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from planexs.core.selectComponent import selectComponent
from planexs.core.validateConfig import validateConfig
from planexs.domain.Outcome import Outcome
from planexs.domain.outcome_kinds.outcomeKindRegistry import outcomeKindRegistry


def outcomeFromDict(data: Mapping[str, Any]) -> Outcome:
    """The inverse of `Outcome.toDict`.

    The `type` entry of `data` names an outcome kind in the registry of the `outcome_kind`
    kind, so an outcome that names a kind nobody registered (a plugin that is not loaded) is an
    error that lists the registered kinds, and not an outcome that cannot be used later
    (specification 5.11, 5.12). The whole mapping is then validated as an `Outcome`.

    Args:
        data: A mapping produced by `Outcome.toDict()`, possibly after a JSON round trip, or
            written by hand in a configuration file. It is not modified.

    Returns:
        The reconstructed `Outcome`.

    Raises:
        ConfigError: if `data` is not a mapping, has no `type` entry, has a `type` that names
            no registered outcome kind (the message lists the registered ones), or does not
            validate as an `Outcome` (an unknown field, a field of the wrong type, a missing
            `name`), with the path of the field.
    """
    selectComponent(outcomeKindRegistry(), data)
    return validateConfig(Outcome, data, label="outcome")
