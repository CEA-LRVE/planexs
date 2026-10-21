"""Defines a kind of component and creates its registry."""

from __future__ import annotations

from typing import Any

from planexs.core.KindTable import KindTable
from planexs.core.Registry import Registry


def defineKind(kind: str) -> Registry[Any]:
    """Defines a kind of component in the process-wide kind table and returns its registry.

    Called once per kind, at import time, by the module that owns the kind's registry::

        SURROGATE_REGISTRY: Registry[ComponentEntry] = defineKind("surrogate")

    The kind then shows up when the table is enumerated (`KindTable.default().kinds()`), which
    is how the contract tests and the compatibility check find every registry.

    Args:
        kind: The kind's name: lowercase snake_case, for example `acquisition_optimizer`.

    Returns:
        The new, empty registry. It is typed `Registry[Any]` so that the caller can annotate it
        with the type of entry it will hold.

    Raises:
        RegistryError: if the name is not valid or the kind is already defined.
    """
    return KindTable.default().define(kind)
