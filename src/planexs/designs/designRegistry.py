"""The registry of the `design` kind: the procedures that propose points without a model."""

from __future__ import annotations

from planexs.core.ComponentEntry import ComponentEntry
from planexs.core.defineKind import defineKind
from planexs.core.kindRegistry import kindRegistry
from planexs.core.Registry import Registry

#: The name of the kind. Like a registered name, it is a permanent identifier that other
#: packages write into their own code and error messages (a plugin registers designs in it),
#: so it is a constant and not a setting.
_KIND = "design"

# Defined once, when this module is first imported (specification 5.12): the kind then shows
# up when the kind table is enumerated, which is how the registry contract test finds it.
defineKind(_KIND)


def designRegistry() -> Registry[ComponentEntry]:
    """The registry of the `design` kind in the process-wide kind table.

    It is looked up in the table at each call, through `kindRegistry`, and not held in a
    module constant: a test can then swap the table for one of its own.

    A registered name is a permanent identifier (specification 3.7). The entries are
    `sobol_v1` and, with the next task, `random_v1`, and a plugin may add its own.

    Returns:
        The registry, whose entries describe the designs: the configuration schema of each
        and the builder that makes the `AbstractDesign` from a validated configuration.

    Raises:
        RegistryError: if the process-wide table has no `design` kind, which only happens in
            a table that was swapped for one that does not define it.
    """
    return kindRegistry(_KIND)
