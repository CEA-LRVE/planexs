"""The registration decorator of the `parameter` kind."""

from __future__ import annotations

from collections.abc import Callable
from operator import attrgetter
from typing import TypeVar

from planexs.core.ComponentConfig import ComponentConfig
from planexs.core.registerComponentClass import registerComponentClass
from planexs.domain.parameters.Parameter import Parameter
from planexs.domain.parameters.parameterRegistry import parameterRegistry

ParameterT = TypeVar("ParameterT", bound=Parameter)


def registerParameter(
    name: str,
    *,
    config: type[ComponentConfig],
    description: str,
    needs: frozenset[str] = frozenset(),
    supports: frozenset[str] = frozenset(),
) -> Callable[[type[ParameterT]], type[ParameterT]]:
    """Registers a `Parameter` subclass as a parameter type, under a permanent name.

    Used as a class decorator, at import time (specification 5.12)::

        @registerParameter(
            "cyclic", config=CyclicConfig, description="An angle that wraps around."
        )
        class Cyclic(Parameter):
            ...

    The registry entry holds the name, the configuration schema (the stored form of the
    parameter), the one-line description, the capability tags and the builder, which is the
    `fromConfig` of the class. The class is returned unchanged. The module that holds the
    decorated class must be imported by the `__init__` of its package, or the type is
    silently unregistered (specification 5.12, 10.6).

    The name is the value of the `type` entry that `toDict` writes. The `type` field of the
    schema must be pinned to that same name (`Literal["cyclic"] = "cyclic"`), so that a
    configuration that says which type it is can only be read by the schema of that type.

    Args:
        name: The registered name: lowercase snake_case with an optional `_vN` suffix. It is
            permanent: a change that alters results gets a new name (specification 3.7).
        config: The configuration schema of the type, a `ComponentConfig` whose `type` field
            defaults to `name`.
        description: One line saying what the parameter type is.
        needs: The capability tags that the type needs some participant to support.
        supports: The capability tags that the type supports.

    Returns:
        The decorator.

    Raises:
        RegistryError: when the decorator is applied, if the class is not a `Parameter`
            subclass, the `type` field of the schema is not pinned to `name`, or the entry is
            refused by the registry (an invalid or already taken name, a tag that is not in
            the vocabulary). Nothing is registered then.
    """

    def decorate(cls: type[ParameterT]) -> type[ParameterT]:
        """Registers `cls` and returns it unchanged."""
        registerComponentClass(
            parameterRegistry(),
            cls,
            base=Parameter,
            name=name,
            config=config,
            description=description,
            needs=needs,
            supports=supports,
            builder_of=attrgetter("fromConfig"),
        )
        return cls

    return decorate
