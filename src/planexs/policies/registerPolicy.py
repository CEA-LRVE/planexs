"""The registration decorator of the `policy` kind."""

from __future__ import annotations

from collections.abc import Callable
from operator import attrgetter
from typing import Any, TypeVar

from planexs.core.ComponentConfig import ComponentConfig
from planexs.core.registerComponentClass import registerComponentClass
from planexs.policies.AbstractPolicy import AbstractPolicy
from planexs.policies.policyRegistry import policyRegistry

PolicyT = TypeVar("PolicyT", bound=AbstractPolicy[Any])


def registerPolicy(
    name: str,
    *,
    config: type[ComponentConfig],
    description: str,
    needs: frozenset[str] = frozenset(),
    supports: frozenset[str] = frozenset(),
) -> Callable[[type[PolicyT]], type[PolicyT]]:
    """Registers an `AbstractPolicy` subclass as a policy type, under a permanent name.

    Used as a class decorator, at import time (specification 5.12)::

        @registerPolicy(
            "random_policy_v1",
            config=RandomPolicyConfig,
            description="Draw suggestions from a design and ignore the observations.",
        )
        class RandomPolicy(AbstractPolicy[RandomPolicyConfig]):
            ...

    The registry entry holds the name, the configuration schema, the one-line description, the
    capability tags and the builder, which is the `fromConfig` of the class: it takes the
    validated configuration and, by keyword, the problem, the goal and the runtime settings of
    the study. The class is returned unchanged. The module that holds the decorated class must
    be imported by the `__init__` of its package, or the policy is silently unregistered
    (specification 5.12, 10.6).

    The `type` field of the schema must be pinned to the registered name
    (`Literal["random_policy_v1"] = "random_policy_v1"`), so that a configuration that says
    which policy it is can only be read by the schema of that policy.

    The tags declared here describe the policy *type*: what any configuration of it can state.
    A policy narrows them to its own configuration through its `needs` and `supports`
    properties (see `AbstractPolicy`). Declarations must be truthful (specification 10.6): a
    policy that adapts an engine declares what the engine does and does not support, and a
    test checks each tag.

    Args:
        name: The registered name: lowercase snake_case with an optional `_vN` suffix. It is
            permanent: a change that alters results gets a new name (specification 3.7).
        config: The configuration schema of the policy, a `ComponentConfig` whose `type` field
            defaults to `name`.
        description: One line saying what the policy is.
        needs: The capability tags that some configuration of the policy needs some participant
            to support.
        supports: The capability tags that some configuration of the policy supports.

    Returns:
        The decorator.

    Raises:
        RegistryError: when the decorator is applied, if the class is not an `AbstractPolicy`
            subclass, the `type` field of the schema is not pinned to `name`, or the entry is
            refused by the registry (an invalid or already taken name, a tag that is not in
            the vocabulary). Nothing is registered then.
    """

    def decorate(cls: type[PolicyT]) -> type[PolicyT]:
        """Registers `cls` and returns it unchanged."""
        registerComponentClass(
            policyRegistry(),
            cls,
            base=AbstractPolicy,
            name=name,
            config=config,
            description=description,
            needs=needs,
            supports=supports,
            builder_of=attrgetter("fromConfig"),
        )
        return cls

    return decorate
