"""The registration decorator of the `acquisition` kind."""

from __future__ import annotations

from collections.abc import Callable
from operator import attrgetter
from typing import Any, TypeVar

from planexs.acquisition.AbstractAcquisitionBuilder import AbstractAcquisitionBuilder
from planexs.acquisition.acquisitionRegistry import acquisitionRegistry
from planexs.core.ComponentConfig import ComponentConfig
from planexs.core.registerComponentClass import registerComponentClass
from planexs.core.RegistryError import RegistryError

AcquisitionBuilderT = TypeVar("AcquisitionBuilderT", bound=AbstractAcquisitionBuilder[Any])


def registerAcquisition(
    name: str,
    *,
    config: type[ComponentConfig],
    description: str,
    needs: frozenset[str] = frozenset(),
    supports: frozenset[str] = frozenset(),
) -> Callable[[type[AcquisitionBuilderT]], type[AcquisitionBuilderT]]:
    """Registers an `AbstractAcquisitionBuilder` subclass as an acquisition, under a permanent
    name.

    Used as a class decorator, at import time (specification 5.12)::

        @registerAcquisition(
            "log_ei_v1",
            config=LogEIConfig,
            description="Analytic log expected improvement for one objective.",
            needs=frozenset({"goal:optimize", "objectives:single", "posterior:gaussian"}),
            supports=frozenset({"acquisition:differentiable"}),
        )
        class LogEIBuilder(AbstractAcquisitionBuilder[LogEIConfig]):
            ...

    The registry entry holds the name, the configuration schema (the settings of the
    acquisition), the one-line description, the capability tags and the builder, which is the
    `fromConfig` of the class. The class is returned unchanged. The module that holds the
    decorated class must be imported by the `__init__` of its package, or the acquisition is
    silently unregistered (specification 5.12, 10.6).

    The `type` field of the schema must be pinned to the registered name
    (`Literal["log_ei_v1"] = "log_ei_v1"`), so that a configuration that says which acquisition
    it is can only be read by the schema of that acquisition.

    The class must also be consistent about its reference set (see
    `AbstractAcquisitionBuilder`): `needs_reference_points` is a bool, and it is `True` exactly
    when the class overrides `referencePointCount`. A builder that asks for a set and says no
    size would be built with nothing to normalize over, and one that says a size but not the
    flag would never be given the set: both are refused here, when the module is imported, and
    not found in the middle of an ask (specification 3.6).

    Needs and supports must be truthful (specification 10.6): an acquisition that supports a
    tag works with it, and one that does not declare it fails the compatibility check with a
    clear message. The tags it needs are what it reads from the goal (`goal:optimize`,
    `objectives:multi`) and the surrogate (`posterior:gaussian`); the tags it supports are
    `acquisition:differentiable`, `pending:supported` and `outcome_constraints:supported`.

    Args:
        name: The registered name: lowercase snake_case with an optional `_vN` suffix. It is
            permanent: a change that alters results gets a new name (specification 3.7).
        config: The configuration schema of the acquisition, a `ComponentConfig` whose `type`
            field defaults to `name`.
        description: One line saying what the acquisition is.
        needs: The capability tags that the acquisition needs some participant to support.
        supports: The capability tags that the acquisition supports.

    Returns:
        The decorator.

    Raises:
        RegistryError: when the decorator is applied, if the class is not an
            `AbstractAcquisitionBuilder` subclass, its reference-set declaration is not
            consistent, the `type` field of the schema is not pinned to `name`, or the entry is
            refused by the registry (an invalid or already taken name, a tag that is not in the
            vocabulary). Nothing is registered then.
    """

    def decorate(cls: type[AcquisitionBuilderT]) -> type[AcquisitionBuilderT]:
        """Registers `cls` and returns it unchanged."""
        # A class that is not a builder is left to `registerComponentClass`, which reports it
        # as such.
        if isinstance(cls, type) and issubclass(cls, AbstractAcquisitionBuilder):
            _checkReferenceDeclaration(cls, name)
        registerComponentClass(
            acquisitionRegistry(),
            cls,
            base=AbstractAcquisitionBuilder,
            name=name,
            config=config,
            description=description,
            needs=needs,
            supports=supports,
            builder_of=attrgetter("fromConfig"),
        )
        return cls

    return decorate


def _checkReferenceDeclaration(cls: type[AbstractAcquisitionBuilder[Any]], name: str) -> None:
    """Checks that a builder's flag and its `referencePointCount` agree.

    Args:
        cls: The builder class being registered.
        name: The name it is registered under, for the messages.

    Raises:
        RegistryError: if `needs_reference_points` is not a bool, or it is `True` and the class
            does not override `referencePointCount`, or it is `False` and the class does.
    """
    component = f"acquisition {name!r}"
    flag = cls.needs_reference_points
    if not isinstance(flag, bool):
        raise RegistryError(
            f"The acquisition {name!r} has a needs_reference_points that is not a bool: got "
            f"{flag!r}.",
            component=component,
            field_path="needs_reference_points",
            suggestion="declare `needs_reference_points: ClassVar[bool] = True` or leave it "
            "out",
        )
    overrides = cls.referencePointCount is not AbstractAcquisitionBuilder.referencePointCount
    if flag and not overrides:
        raise RegistryError(
            f"The acquisition {name!r} says it needs a reference set, but does not say how "
            "many points: it does not override referencePointCount.",
            component=component,
            field_path="needs_reference_points",
            suggestion="override referencePointCount to return the size that the "
            "configuration gives, a field of its schema",
        )
    if overrides and not flag:
        raise RegistryError(
            f"The acquisition {name!r} overrides referencePointCount, but does not say that "
            "it needs a reference set, so the policy would never draw one for it.",
            component=component,
            field_path="needs_reference_points",
            suggestion="declare `needs_reference_points: ClassVar[bool] = True`, or remove "
            "the override",
        )
