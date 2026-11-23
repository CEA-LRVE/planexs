"""The registration decorator of the `fitter` kind."""

from __future__ import annotations

from collections.abc import Callable
from operator import attrgetter
from typing import Any, TypeVar

from planexs.core.ComponentConfig import ComponentConfig
from planexs.core.registerComponentClass import registerComponentClass
from planexs.surrogates.fitters.AbstractFitter import AbstractFitter
from planexs.surrogates.fitters.fitterRegistry import fitterRegistry

FitterT = TypeVar("FitterT", bound=AbstractFitter[Any])


def registerFitter(
    name: str,
    *,
    config: type[ComponentConfig],
    description: str,
    needs: frozenset[str] = frozenset(),
    supports: frozenset[str] = frozenset(),
) -> Callable[[type[FitterT]], type[FitterT]]:
    """Registers an `AbstractFitter` subclass as a fitter, under a permanent name.

    Used as a class decorator, at import time (specification 5.12)::

        @registerFitter(
            "map_v1",
            config=MapFitterConfig,
            description="Fits an exact GP by maximum a posteriori.",
            needs=frozenset({"posterior:gaussian"}),
        )
        class MapFitter(AbstractFitter[MapFitterConfig]):
            ...

    The registry entry holds the name, the configuration schema (the settings of the fitter),
    the one-line description, the capability tags and the builder, which is the `fromConfig`
    of the class. The class is returned unchanged. The module that holds the decorated class
    must be imported by the `__init__` of its package, or the fitter is silently unregistered
    (specification 5.12, 10.6).

    The `type` field of the schema must be pinned to the registered name
    (`Literal["map_v1"] = "map_v1"`), so that a configuration that says which fitter it is can
    only be read by the schema of that fitter.

    A fitter needs what the models it can fit provide: a fitter of exact Gaussian processes
    needs `posterior:gaussian`, which the surrogate that builds such a model supports, so that
    the compatibility check refuses it beside a surrogate whose model it cannot fit when the
    study is built, and not in the middle of an ask (specification 3.6). Needs and supports
    must be truthful (specification 10.6).

    Args:
        name: The registered name: lowercase snake_case with an optional `_vN` suffix. It is
            permanent: a change that alters results gets a new name (specification 3.7).
        config: The configuration schema of the fitter, a `ComponentConfig` whose `type` field
            defaults to `name`.
        description: One line saying what the fitter is.
        needs: The capability tags that the fitter needs some participant to support.
        supports: The capability tags that the fitter supports.

    Returns:
        The decorator.

    Raises:
        RegistryError: when the decorator is applied, if the class is not an `AbstractFitter`
            subclass, the `type` field of the schema is not pinned to `name`, or the entry is
            refused by the registry (an invalid or already taken name, a tag that is not in
            the vocabulary). Nothing is registered then.
    """

    def decorate(cls: type[FitterT]) -> type[FitterT]:
        """Registers `cls` and returns it unchanged."""
        registerComponentClass(
            fitterRegistry(),
            cls,
            base=AbstractFitter,
            name=name,
            config=config,
            description=description,
            needs=needs,
            supports=supports,
            builder_of=attrgetter("fromConfig"),
        )
        return cls

    return decorate
