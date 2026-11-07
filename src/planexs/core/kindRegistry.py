"""Looks up the registry of a kind in the process-wide kind table."""

from __future__ import annotations

from planexs.core.ComponentEntry import ComponentEntry
from planexs.core.KindTable import KindTable
from planexs.core.Registry import Registry


def kindRegistry(kind: str) -> Registry[ComponentEntry]:
    """The registry of `kind` in the process-wide kind table, looked up at this call.

    The kind owner's own function (`parameterRegistry`, `constraintRegistry`, ...) calls this
    and does not hold the registry in a module constant, as `ComponentOf` does for the kind it
    names. A test can then swap the process-wide table for a table of its own (the registries
    that code reaches through this function follow the swap), and register a component that
    must not outlive the test: a registered name is permanent and cannot be removed
    (specification 3.7).

    Args:
        kind: The kind's name, as `defineKind` was given it.

    Returns:
        The registry, typed for the usual `ComponentEntry` records.

    Raises:
        RegistryError: if the kind is not defined in the table, which only happens in a table
            that was swapped for one that does not define it. The message lists the kinds that
            are defined.
    """
    registry: Registry[ComponentEntry] = KindTable.default().registry(kind)
    return registry
