"""The registry of the `outcome_transform` kind: how outcomes are rescaled before a fit."""

from __future__ import annotations

from planexs.core.ComponentEntry import ComponentEntry
from planexs.core.defineKind import defineKind
from planexs.core.kindRegistry import kindRegistry
from planexs.core.Registry import Registry

#: The name of the kind. Like a registered name, it is a permanent identifier that other
#: packages write into their own code and error messages (a surrogate's configuration holds
#: components of this kind, and a plugin registers transforms in it), so it is a constant and
#: not a setting.
_KIND = "outcome_transform"

# Defined once, when this module is first imported (specification 5.12): the kind then shows
# up when the kind table is enumerated, which is how the registry contract test finds it.
defineKind(_KIND)


def outcomeTransformRegistry() -> Registry[ComponentEntry]:
    """The registry of the `outcome_transform` kind in the process-wide kind table.

    It is looked up in the table at each call, through `kindRegistry`, and not held in a
    module constant: a test can then swap the table for one of its own.

    A registered name is a permanent identifier (specification 3.7). The entries are
    `standardize_v1`, `log_v1` and `none_v1`, and a plugin may add its own. The name is the
    value of the `type` of an element of a surrogate's `outcome_transform` field
    (specification 5.6).

    Returns:
        The registry, whose entries describe the outcome transforms: the configuration schema
        of each and the builder that makes the `AbstractOutcomeTransformBuilder` from a
        validated configuration.

    Raises:
        RegistryError: if the process-wide table has no `outcome_transform` kind, which only
            happens in a table that was swapped for one that does not define it.
    """
    return kindRegistry(_KIND)
