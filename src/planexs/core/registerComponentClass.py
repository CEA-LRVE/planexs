"""Registers a class as a component of a kind, once it honours the contract of the kind."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from planexs.core.ComponentConfig import ComponentConfig
from planexs.core.ComponentEntry import ComponentEntry
from planexs.core.Registry import Registry
from planexs.core.RegistryError import RegistryError


def registerComponentClass(
    registry: Registry[ComponentEntry],
    cls: type,
    *,
    base: type,
    name: str,
    config: type[ComponentConfig],
    description: str,
    needs: frozenset[str],
    supports: frozenset[str],
    builder_of: Callable[[Any], Callable[..., Any]],
) -> None:
    """Checks that `cls` honours the contract of its kind, and registers its entry.

    This is the part of a registration decorator that does not depend on the kind: a kind's
    own decorator (`registerParameter`, `registerConstraint`, ...) calls it with the registry
    and the contract class of that kind, and adds what is specific to the kind. It exists so
    that each kind has a thin decorator of its own (specification 10.6) and not a copy of
    these checks.

    Two things are checked here, before the registry is touched. The class must derive from
    the contract class of the kind. The `type` field of the configuration schema must default
    to the registered name (`Literal["name"] = "name"`), so that a configuration that says
    which type it is can only be read by the schema of that type. The registry then refuses
    what it refuses (an invalid or already taken name, a tag that is not in the vocabulary),
    and nothing is stored unless every check passes.

    Args:
        registry: The registry of the kind.
        cls: The class being registered.
        base: The contract class of the kind: `cls` must be a subclass of it.
        name: The registered name, a permanent identifier (specification 3.7).
        config: The configuration schema of the component.
        description: One line saying what the component is.
        needs: The capability tags that the component needs some participant to support.
        supports: The capability tags that the component supports.
        builder_of: Given the class, returns what creates the native object of the component
            from a validated configuration (its signature belongs to the kind), for example
            `operator.attrgetter("fromConfig")`. It is called after the class has been checked,
            so a class that does not honour the contract is reported as such, and not by the
            error of a missing attribute.

    Raises:
        RegistryError: if `cls` is not a subclass of `base`, the `type` field of the schema is
            not pinned to `name`, or the registry refuses the entry. Nothing is registered
            then.
    """
    kind = registry.kind
    if not (isinstance(cls, type) and issubclass(cls, base)):
        raise RegistryError(
            f"Cannot register {cls!r} as the {kind} type {name!r}: it is not a subclass of "
            f"{base.__name__}.",
            component=f"{kind} {name!r}",
            suggestion=f"derive the class from {base.__name__}, the contract of the kind",
        )
    pinned = config.model_fields["type"].default
    if pinned != name:
        raise RegistryError(
            f"The configuration schema {config.__name__} of the {kind} type {name!r} has a "
            f"'type' field that defaults to {pinned!r}: it must be pinned to the registered "
            "name.",
            component=f"{kind} {name!r}",
            field_path="type",
            suggestion=f"declare it as `type: Literal[{name!r}] = {name!r}`",
        )
    registry.register(
        ComponentEntry(
            name=name,
            builder=builder_of(cls),
            config_schema=config,
            description=description,
            needs=needs,
            supports=supports,
        )
    )
