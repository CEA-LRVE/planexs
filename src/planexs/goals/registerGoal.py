"""The registration decorator of the `goal` kind."""

from __future__ import annotations

from collections.abc import Callable
from operator import attrgetter
from typing import Any, TypeVar

from planexs.core.ComponentConfig import ComponentConfig
from planexs.core.registerComponentClass import registerComponentClass
from planexs.goals.AbstractGoal import AbstractGoal
from planexs.goals.goalRegistry import goalRegistry

GoalT = TypeVar("GoalT", bound=AbstractGoal[Any])


def registerGoal(
    name: str,
    *,
    config: type[ComponentConfig],
    description: str,
    needs: frozenset[str] = frozenset(),
    supports: frozenset[str] = frozenset(),
) -> Callable[[type[GoalT]], type[GoalT]]:
    """Registers an `AbstractGoal` subclass as a goal type, under a permanent name.

    Used as a class decorator, at import time (specification 5.12)::

        @registerGoal(
            "optimize_v1",
            config=OptimizeGoalConfig,
            description="Optimize one or several outcomes, each with a direction.",
            supports=frozenset({"goal:optimize", "objectives:single", "objectives:multi"}),
        )
        class OptimizeGoal(AbstractGoal[OptimizeGoalConfig]):
            ...

    The registry entry holds the name, the configuration schema (the goal itself, as typed
    configuration), the one-line description, the capability tags and the builder, which is
    the `fromConfig` of the class. The class is returned unchanged. The module that holds the
    decorated class must be imported by the `__init__` of its package, or the goal is silently
    unregistered (specification 5.12, 10.6).

    The `type` field of the schema must be pinned to the registered name
    (`Literal["optimize_v1"] = "optimize_v1"`), so that a configuration that says which goal it
    is can only be read by the schema of that goal.

    The tags declared here describe the goal *type*: what any configuration of it can state.
    A goal narrows them to its own configuration through its `needs` and `supports`
    properties (see `AbstractGoal`), so a type declares the union and an instance reports what
    holds for it. Declarations must be truthful (specification 10.6).

    Args:
        name: The registered name: lowercase snake_case with an optional `_vN` suffix. It is
            permanent: a change that alters results gets a new name (specification 3.7).
        config: The configuration schema of the goal, a `ComponentConfig` whose `type` field
            defaults to `name`.
        description: One line saying what the goal is.
        needs: The capability tags that some configuration of the goal needs some participant
            to support.
        supports: The capability tags that some configuration of the goal supports.

    Returns:
        The decorator.

    Raises:
        RegistryError: when the decorator is applied, if the class is not an `AbstractGoal`
            subclass, the `type` field of the schema is not pinned to `name`, or the entry is
            refused by the registry (an invalid or already taken name, a tag that is not in
            the vocabulary). Nothing is registered then.
    """

    def decorate(cls: type[GoalT]) -> type[GoalT]:
        """Registers `cls` and returns it unchanged."""
        registerComponentClass(
            goalRegistry(),
            cls,
            base=AbstractGoal,
            name=name,
            config=config,
            description=description,
            needs=needs,
            supports=supports,
            builder_of=attrgetter("fromConfig"),
        )
        return cls

    return decorate
