"""The registration decorator of the `surrogate` kind."""

from __future__ import annotations

from collections.abc import Callable
from operator import attrgetter
from typing import Any, TypeVar

from planexs.core.ComponentConfig import ComponentConfig
from planexs.core.registerComponentClass import registerComponentClass
from planexs.surrogates.AbstractSurrogateBuilder import AbstractSurrogateBuilder
from planexs.surrogates.surrogateRegistry import surrogateRegistry

SurrogateBuilderT = TypeVar("SurrogateBuilderT", bound=AbstractSurrogateBuilder[Any])


def registerSurrogate(
    name: str,
    *,
    config: type[ComponentConfig],
    description: str,
    needs: frozenset[str] = frozenset(),
    supports: frozenset[str] = frozenset(),
) -> Callable[[type[SurrogateBuilderT]], type[SurrogateBuilderT]]:
    """Registers an `AbstractSurrogateBuilder` subclass as a surrogate, under a permanent name.

    Used as a class decorator, at import time (specification 5.12)::

        @registerSurrogate(
            "single_task_gp_v1",
            config=SingleTaskGPConfig,
            description="An exact GP per outcome, with its kernel chosen in config.",
            supports=frozenset({"posterior:gaussian", "outputs:multi", "noise:known"}),
        )
        class SingleTaskGPBuilder(AbstractSurrogateBuilder[SingleTaskGPConfig]):
            ...

    The registry entry holds the name, the configuration schema (the settings of the
    surrogate), the one-line description, the capability tags and the builder, which is the
    `fromConfig` of the class. The class is returned unchanged. The module that holds the
    decorated class must be imported by the `__init__` of its package, or the surrogate is
    silently unregistered (specification 5.12, 10.6).

    The `type` field of the schema must be pinned to the registered name
    (`Literal["single_task_gp_v1"] = "single_task_gp_v1"`), so that a configuration that says
    which surrogate it is can only be read by the schema of that surrogate.

    Needs and supports must be truthful (specification 10.6): a surrogate that supports a tag
    works with it, and one that does not declare it fails the compatibility check with a clear
    message. The outcome kinds a surrogate can model are among its `supports`
    (`outcome_kind:continuous`).

    Args:
        name: The registered name: lowercase snake_case with an optional `_vN` suffix. It is
            permanent: a change that alters results gets a new name (specification 3.7).
        config: The configuration schema of the surrogate, a `ComponentConfig` whose `type`
            field defaults to `name`.
        description: One line saying what the surrogate is.
        needs: The capability tags that the surrogate needs some participant to support.
        supports: The capability tags that the surrogate supports.

    Returns:
        The decorator.

    Raises:
        RegistryError: when the decorator is applied, if the class is not an
            `AbstractSurrogateBuilder` subclass, the `type` field of the schema is not pinned
            to `name`, or the entry is refused by the registry (an invalid or already taken
            name, a tag that is not in the vocabulary). Nothing is registered then.
    """

    def decorate(cls: type[SurrogateBuilderT]) -> type[SurrogateBuilderT]:
        """Registers `cls` and returns it unchanged."""
        registerComponentClass(
            surrogateRegistry(),
            cls,
            base=AbstractSurrogateBuilder,
            name=name,
            config=config,
            description=description,
            needs=needs,
            supports=supports,
            builder_of=attrgetter("fromConfig"),
        )
        return cls

    return decorate
