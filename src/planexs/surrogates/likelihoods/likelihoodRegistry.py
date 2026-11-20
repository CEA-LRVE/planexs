"""The registry of the `likelihood` kind: how the observations of an outcome are modelled."""

from __future__ import annotations

from planexs.core.ComponentEntry import ComponentEntry
from planexs.core.defineKind import defineKind
from planexs.core.kindRegistry import kindRegistry
from planexs.core.Registry import Registry

# Imported for its effect, which is to register the capability tags of the surrogate layer
# (`noise:known`, `noise:inferred`, ...) that every likelihood declares. The registry checks
# the tags of an entry when the entry is registered, so they must exist by then, whichever
# module of the layer is imported first.
from planexs.surrogates import surrogateRegistry  # noqa: F401

#: The name of the kind. Like a registered name, it is a permanent identifier that other
#: packages write into their own code and error messages (a plugin registers likelihoods in
#: it), so it is a constant and not a setting.
_KIND = "likelihood"

# Defined once, when this module is first imported (specification 5.12): the kind then shows
# up when the kind table is enumerated, which is how the registry contract test finds it.
defineKind(_KIND)


def likelihoodRegistry() -> Registry[ComponentEntry]:
    """The registry of the `likelihood` kind in the process-wide kind table.

    It is looked up in the table at each call, through `kindRegistry`, and not held in a
    module constant: a test can then swap the table for one of its own.

    A registered name is a permanent identifier (specification 3.7). The entries are
    `gaussian_v1` and `bernoulli_v1`, and a plugin may add its own. The outcome kinds that an
    entry supports are among its `supports` tags (`outcome_kind:continuous`), which is how the
    compatibility check meets the outcomes of a problem with a likelihood that can model them.

    Returns:
        The registry, whose entries describe the likelihoods: the configuration schema of each
        and the builder that makes the `AbstractLikelihoodBuilder` from a validated
        configuration.

    Raises:
        RegistryError: if the process-wide table has no `likelihood` kind, which only happens
            in a table that was swapped for one that does not define it.
    """
    return kindRegistry(_KIND)
