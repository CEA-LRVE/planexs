"""The registry of the `policy` kind: what decides which parameterizations to evaluate next."""

from __future__ import annotations

from planexs.core.ComponentEntry import ComponentEntry
from planexs.core.defineKind import defineKind
from planexs.core.kindRegistry import kindRegistry
from planexs.core.Registry import Registry

#: The name of the kind. Like a registered name, it is a permanent identifier that other
#: packages write into their own code and error messages (a plugin registers policies in it),
#: so it is a constant and not a setting.
_KIND = "policy"


def _definePolicyKind() -> None:
    """Defines the `policy` kind.

    The kind introduces no capability tag of its own: what a policy needs and supports is
    stated with the tags of the layers whose objects it uses (a composed policy takes them from
    its design, surrogate, acquisition and optimizer) or of the engine it adapts, and those are
    registered where they are defined.
    """
    defineKind(_KIND)


# Defined once, when this module is first imported (specification 5.12): the kind then shows
# up when the kind table is enumerated, which is how the registry contract test finds it.
_definePolicyKind()


def policyRegistry() -> Registry[ComponentEntry]:
    """The registry of the `policy` kind in the process-wide kind table.

    It is looked up in the table at each call, through `kindRegistry`, and not held in a
    module constant: a test can then swap the table for one of its own.

    A registered name is permanent (specification 3.7). The entries are added by the tasks that
    write policies (`composed_v1`, `random_policy_v1`, and the Ax adapter's `ax_default_v1`),
    and a plugin may add its own: a policy of any origin is registered the same way, and a
    study treats them all alike (specification 5.9).

    Returns:
        The registry, whose entries describe the policy types: the configuration schema of
        each, the builder that makes the `AbstractPolicy` from a validated configuration and
        what it is built for, and the capability tags the type needs and supports.

    Raises:
        RegistryError: if the process-wide table has no `policy` kind, which only happens in a
            table that was swapped for one that does not define it.
    """
    return kindRegistry(_KIND)
