"""The registry of the `constraint` kind: the types that a parameter constraint can have."""

from __future__ import annotations

from planexs.core.ComponentEntry import ComponentEntry
from planexs.core.defineKind import defineKind
from planexs.core.KindTable import KindTable
from planexs.core.Registry import Registry

#: The name of the kind. Like a registered name, it is a permanent identifier that other
#: packages write into their own code and error messages (a plugin registers constraint types
#: in it), so it is a constant and not a setting.
_KIND = "constraint"

# Defined once, when this module is first imported (specification 5.12): the kind then shows
# up when the kind table is enumerated, which is how the registry contract test finds it.
defineKind(_KIND)


def constraintRegistry() -> Registry[ComponentEntry]:
    """The registry of the `constraint` kind in the process-wide kind table.

    It is looked up in the table at each call and not held in a module constant, as
    `ComponentOf` does for the kind it names. A test can then swap the process-wide table
    for a table of its own, and register a type that must not outlive it.

    A registered name is a permanent identifier (specification 3.7): the entries are `linear`
    and `nonlinear` (the types of specification 5.3), and a plugin may add its own.

    Returns:
        The registry, whose entries describe the constraint types: the configuration schema of
        each, and the builder that makes the domain object from a validated configuration.

    Raises:
        RegistryError: if the process-wide table has no `constraint` kind, which only happens
            in a table that was swapped for one that does not define it.
    """
    registry: Registry[ComponentEntry] = KindTable.default().registry(_KIND)
    return registry
