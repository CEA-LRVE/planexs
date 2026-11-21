"""The registration decorator of the `outcome_transform` kind."""

from __future__ import annotations

from collections.abc import Callable
from operator import attrgetter
from typing import Any, TypeVar

from planexs.core.ComponentConfig import ComponentConfig
from planexs.core.registerComponentClass import registerComponentClass
from planexs.surrogates.outcome_transforms.AbstractOutcomeTransformBuilder import (
    AbstractOutcomeTransformBuilder,
)
from planexs.surrogates.outcome_transforms.outcomeTransformRegistry import (
    outcomeTransformRegistry,
)

OutcomeTransformBuilderT = TypeVar(
    "OutcomeTransformBuilderT", bound=AbstractOutcomeTransformBuilder[Any]
)


def registerOutcomeTransform(
    name: str,
    *,
    config: type[ComponentConfig],
    description: str,
    needs: frozenset[str] = frozenset(),
    supports: frozenset[str] = frozenset(),
) -> Callable[[type[OutcomeTransformBuilderT]], type[OutcomeTransformBuilderT]]:
    """Registers an `AbstractOutcomeTransformBuilder` subclass as an outcome transform, under
    a permanent name.

    Used as a class decorator, at import time (specification 5.12)::

        @registerOutcomeTransform(
            "standardize_v1",
            config=StandardizeOutcomeTransformConfig,
            description="Centres each outcome and divides it by its standard deviation.",
        )
        class StandardizeOutcomeTransformBuilder(
            AbstractOutcomeTransformBuilder[StandardizeOutcomeTransformConfig]
        ):
            ...

    The registry entry holds the name, the configuration schema (the settings of the
    transform), the one-line description, the capability tags and the builder, which is the
    `fromConfig` of the class. The class is returned unchanged. The module that holds the
    decorated class must be imported by the `__init__` of its package, or the transform is
    silently unregistered (specification 5.12, 10.6).

    The `type` field of the schema must be pinned to the registered name
    (`Literal["standardize_v1"] = "standardize_v1"`), so that a configuration that says which
    transform it is can only be read by the schema of that transform.

    A transform needs nothing from a study and supports nothing for it: it rescales the
    outcomes it is given and un-rescales the predictions. `needs` and `supports` exist for a
    transform that does need something. They must be truthful (specification 10.6).

    Args:
        name: The registered name: lowercase snake_case with an optional `_vN` suffix. It is
            permanent: a change that alters results gets a new name (specification 3.7).
        config: The configuration schema of the transform, a `ComponentConfig` whose `type`
            field defaults to `name`.
        description: One line saying what the transform is.
        needs: The capability tags that the transform needs some participant to support.
        supports: The capability tags that the transform supports.

    Returns:
        The decorator.

    Raises:
        RegistryError: when the decorator is applied, if the class is not an
            `AbstractOutcomeTransformBuilder` subclass, the `type` field of the schema is not
            pinned to `name`, or the entry is refused by the registry (an invalid or already
            taken name, a tag that is not in the vocabulary). Nothing is registered then.
    """

    def decorate(cls: type[OutcomeTransformBuilderT]) -> type[OutcomeTransformBuilderT]:
        """Registers `cls` and returns it unchanged."""
        registerComponentClass(
            outcomeTransformRegistry(),
            cls,
            base=AbstractOutcomeTransformBuilder,
            name=name,
            config=config,
            description=description,
            needs=needs,
            supports=supports,
            builder_of=attrgetter("fromConfig"),
        )
        return cls

    return decorate
