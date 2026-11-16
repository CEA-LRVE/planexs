"""The registration decorator of the `kernel` kind."""

from __future__ import annotations

from collections.abc import Callable
from operator import attrgetter
from typing import Any, TypeVar

from planexs.core.ComponentConfig import ComponentConfig
from planexs.core.registerComponentClass import registerComponentClass
from planexs.surrogates.kernels.AbstractKernelBuilder import AbstractKernelBuilder
from planexs.surrogates.kernels.kernelRegistry import kernelRegistry

KernelT = TypeVar("KernelT", bound=AbstractKernelBuilder[Any])


def registerKernel(
    name: str,
    *,
    config: type[ComponentConfig],
    description: str,
    needs: frozenset[str] = frozenset(),
    supports: frozenset[str] = frozenset(),
) -> Callable[[type[KernelT]], type[KernelT]]:
    """Registers an `AbstractKernelBuilder` subclass as a kernel, under a permanent name.

    Used as a class decorator, at import time (specification 5.12)::

        @registerKernel(
            "matern_v1",
            config=MaternKernelConfig,
            description="A Matern kernel of smoothness 0.5, 1.5 or 2.5 on the columns.",
        )
        class MaternKernelBuilder(AbstractKernelBuilder[MaternKernelConfig]):
            ...

    The registry entry holds the name, the configuration schema (the settings of the kernel),
    the one-line description, the capability tags and the builder, which is the `fromConfig`
    of the class. The class is returned unchanged. The module that holds the decorated class
    must be imported by the `__init__` of its package, or the kernel is silently unregistered
    (specification 5.12, 10.6).

    The `type` field of the schema must be pinned to the registered name
    (`Literal["matern_v1"] = "matern_v1"`), so that a configuration that says which kernel it
    is can only be read by the schema of that kernel.

    A kernel needs nothing from a study and supports nothing for it: it is a function of the
    encoded columns alone. `needs` and `supports` exist for a kernel that does need something,
    for instance one that is only defined for a space with no categorical parameter. They must
    be truthful (specification 10.6).

    Args:
        name: The registered name: lowercase snake_case with an optional `_vN` suffix. It is
            permanent: a change that alters results gets a new name (specification 3.7).
        config: The configuration schema of the kernel, a `ComponentConfig` whose `type` field
            defaults to `name`.
        description: One line saying what the kernel is.
        needs: The capability tags that the kernel needs some participant to support.
        supports: The capability tags that the kernel supports.

    Returns:
        The decorator.

    Raises:
        RegistryError: when the decorator is applied, if the class is not an
            `AbstractKernelBuilder` subclass, the `type` field of the schema is not pinned to
            `name`, or the entry is refused by the registry (an invalid or already taken name,
            a tag that is not in the vocabulary). Nothing is registered then.
    """

    def decorate(cls: type[KernelT]) -> type[KernelT]:
        """Registers `cls` and returns it unchanged."""
        registerComponentClass(
            kernelRegistry(),
            cls,
            base=AbstractKernelBuilder,
            name=name,
            config=config,
            description=description,
            needs=needs,
            supports=supports,
            builder_of=attrgetter("fromConfig"),
        )
        return cls

    return decorate
