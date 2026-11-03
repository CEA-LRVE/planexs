"""The registration decorator of the `constraint` kind."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from planexs.core.ComponentConfig import ComponentConfig
from planexs.core.ComponentEntry import ComponentEntry
from planexs.core.RegistryError import RegistryError
from planexs.domain.constraints.Constraint import Constraint
from planexs.domain.constraints.constraintRegistry import constraintRegistry

ConstraintT = TypeVar("ConstraintT", bound=Constraint)


def registerConstraint(
    name: str,
    *,
    config: type[ComponentConfig],
    description: str,
    needs: frozenset[str] = frozenset(),
    supports: frozenset[str] = frozenset(),
) -> Callable[[type[ConstraintT]], type[ConstraintT]]:
    """Registers a `Constraint` subclass as a constraint type, under a permanent name.

    Used as a class decorator, at import time (specification 5.12)::

        @registerConstraint(
            "ordering", config=OrderingConfig, description="One parameter stays below another."
        )
        class Ordering(Constraint):
            ...

    The registry entry holds the name, the configuration schema (the stored form of the
    constraint), the one-line description, the capability tags and the builder, which is the
    `fromConfig` of the class. The class is returned unchanged. The module that holds the
    decorated class must be imported by the `__init__` of its package, or the type is
    silently unregistered (specification 5.12, 10.6).

    The name is the value of the `type` entry that `toDict` writes. The `type` field of the
    schema must be pinned to that same name (`Literal["ordering"] = "ordering"`), so that a
    configuration that says which type it is can only be read by the schema of that type.

    Args:
        name: The registered name: lowercase snake_case with an optional `_vN` suffix. It is
            permanent: a change that alters results gets a new name (specification 3.7).
        config: The configuration schema of the type, a `ComponentConfig` whose `type` field
            defaults to `name`.
        description: One line saying what the constraint type is.
        needs: The capability tags that the type needs some participant to support.
        supports: The capability tags that the type supports.

    Returns:
        The decorator.

    Raises:
        RegistryError: when the decorator is applied, if the class is not a `Constraint`
            subclass, the `type` field of the schema is not pinned to `name`, or the entry is
            refused by the registry (an invalid or already taken name, a tag that is not in
            the vocabulary). Nothing is registered then.
    """

    def decorate(cls: type[ConstraintT]) -> type[ConstraintT]:
        """Registers `cls` and returns it unchanged."""
        if not (isinstance(cls, type) and issubclass(cls, Constraint)):
            raise RegistryError(
                f"Cannot register {cls!r} as the constraint type {name!r}: it is not a "
                "subclass of Constraint.",
                component=f"constraint {name!r}",
                suggestion="derive the class from Constraint, the contract of the kind",
            )
        pinned = config.model_fields["type"].default
        if pinned != name:
            raise RegistryError(
                f"The configuration schema {config.__name__} of the constraint type {name!r} "
                f"has a 'type' field that defaults to {pinned!r}: it must be pinned to the "
                "registered name.",
                component=f"constraint {name!r}",
                field_path="type",
                suggestion=f"declare it as `type: Literal[{name!r}] = {name!r}`",
            )
        constraintRegistry().register(
            ComponentEntry(
                name=name,
                builder=cls.fromConfig,
                config_schema=config,
                description=description,
                needs=needs,
                supports=supports,
            )
        )
        return cls

    return decorate
