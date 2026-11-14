"""The registration decorator of the `design` kind."""

from __future__ import annotations

from collections.abc import Callable
from operator import attrgetter
from typing import Any, TypeVar

from planexs.core.ComponentConfig import ComponentConfig
from planexs.core.registerComponentClass import registerComponentClass
from planexs.designs.AbstractDesign import AbstractDesign
from planexs.designs.designRegistry import designRegistry

DesignT = TypeVar("DesignT", bound=AbstractDesign[Any])


def registerDesign(
    name: str,
    *,
    config: type[ComponentConfig],
    description: str,
    needs: frozenset[str] = frozenset(),
    supports: frozenset[str] = frozenset(),
) -> Callable[[type[DesignT]], type[DesignT]]:
    """Registers an `AbstractDesign` subclass as a design, under a permanent name.

    Used as a class decorator, at import time (specification 5.12)::

        @registerDesign(
            "sobol_v1",
            config=SobolDesignConfig,
            description="Scrambled Sobol points in the encoded space, within the constraints.",
        )
        class SobolDesign(AbstractDesign[SobolDesignConfig]):
            ...

    The registry entry holds the name, the configuration schema (the settings of the design),
    the one-line description, the capability tags and the builder, which is the `fromConfig`
    of the class. The class is returned unchanged. The module that holds the decorated class
    must be imported by the `__init__` of its package, or the design is silently unregistered
    (specification 5.12, 10.6).

    The `type` field of the schema must be pinned to the registered name
    (`Literal["sobol_v1"] = "sobol_v1"`), so that a configuration that says which design it is
    can only be read by the schema of that design.

    A design needs nothing from a study and supports nothing for it: it proposes points that
    satisfy whatever constraints it is given. `needs` and `supports` exist for a design that
    does need something, for instance one that can only work in a space with no categorical
    parameter. They must be truthful (specification 10.6).

    Args:
        name: The registered name: lowercase snake_case with an optional `_vN` suffix. It is
            permanent: a change that alters results gets a new name (specification 3.7).
        config: The configuration schema of the design, a `ComponentConfig` whose `type` field
            defaults to `name`.
        description: One line saying what the design is.
        needs: The capability tags that the design needs some participant to support.
        supports: The capability tags that the design supports.

    Returns:
        The decorator.

    Raises:
        RegistryError: when the decorator is applied, if the class is not an `AbstractDesign`
            subclass, the `type` field of the schema is not pinned to `name`, or the entry is
            refused by the registry (an invalid or already taken name, a tag that is not in
            the vocabulary). Nothing is registered then.
    """

    def decorate(cls: type[DesignT]) -> type[DesignT]:
        """Registers `cls` and returns it unchanged."""
        registerComponentClass(
            designRegistry(),
            cls,
            base=AbstractDesign,
            name=name,
            config=config,
            description=description,
            needs=needs,
            supports=supports,
            builder_of=attrgetter("fromConfig"),
        )
        return cls

    return decorate
