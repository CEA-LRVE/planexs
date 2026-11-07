"""The registry of the `outcome_kind` kind: the kinds of values that an outcome can have."""

from __future__ import annotations

from planexs.core.ComponentEntry import ComponentEntry
from planexs.core.defineKind import defineKind
from planexs.core.kindRegistry import kindRegistry
from planexs.core.Registry import Registry

#: The name of the kind. Like a registered name, it is a permanent identifier that other
#: packages write into their own code and error messages (a plugin registers outcome kinds
#: in it), so it is a constant and not a setting.
_KIND = "outcome_kind"

# Defined once, when this module is first imported (specification 5.12): the kind then shows
# up when the kind table is enumerated, which is how the registry contract test finds it.
defineKind(_KIND)


def outcomeKindRegistry() -> Registry[ComponentEntry]:
    """The registry of the `outcome_kind` kind in the process-wide kind table.

    It is looked up in the table at each call, through `kindRegistry`, and not held in a
    module constant: a test can then swap the table for one of its own.

    A registered name is a permanent identifier (specification 3.7): the entries are
    `continuous` and `binary` (the kinds of specification 5.4, decision D-7), and a plugin
    may add its own. The name of a kind is the value of the `type` of an outcome.

    Returns:
        The registry, whose entries describe the outcome kinds: the configuration schema of
        each, and the builder that makes the kind from a validated configuration.

    Raises:
        RegistryError: if the process-wide table has no `outcome_kind` kind, which only
            happens in a table that was swapped for one that does not define it.
    """
    return kindRegistry(_KIND)
