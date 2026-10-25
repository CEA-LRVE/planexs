"""Finds the registered component that a configuration mapping names."""

from __future__ import annotations

import reprlib
from collections.abc import Mapping
from typing import Any

from planexs.core.ComponentConfig import ComponentConfig
from planexs.core.ConfigError import ConfigError
from planexs.core.Registry import Registry
from planexs.core.RegistryError import RegistryError
from planexs.core.ResolvedComponent import EntryT

#: The key that names a component in a configuration mapping: the `type` field of
#: `ComponentConfig` (specification 5.12).
_TYPE_FIELD = "type"


def selectComponent(
    registry: Registry[EntryT], data: Mapping[str, Any]
) -> tuple[EntryT, type[ComponentConfig]]:
    """Finds the entry that the `type` of a configuration mapping names, and its schema.

    This is the first half of resolving a component (the second is validating the mapping
    against the schema): it only decides which component the mapping is for. Nothing else in
    `data` is looked at, so the settings of the component are not checked here.

    Args:
        registry: The registry of the kind of component the configuration is for.
        data: The configuration: a mapping from field names to values, with a `type` entry
            that names a registered component of that kind. It is not modified.

    Returns:
        The registered entry and its configuration schema, a `ComponentConfig` subclass.

    Raises:
        ConfigError: if `data` is not a mapping or has a key that is not a string, has no
            `type` entry, or has a `type` that is not a string or names nothing in `registry`.
            The error names the kind, the field `type` where it applies and the registered
            names.
        RegistryError: if the component is registered with a configuration schema that does
            not derive from `ComponentConfig`, which is a defect of the registration and not
            of the configuration.
    """
    kind = registry.kind
    listed = ", ".join(registry.names()) or "none"

    if not isinstance(data, Mapping):
        raise ConfigError(
            f"A {kind} configuration must be a mapping with a 'type' key; got "
            f"{type(data).__name__}.",
            component=kind,
            suggestion="write the configuration as a mapping, for example {'type': 'name'}",
        )
    for key in data:
        if not isinstance(key, str):
            raise ConfigError(
                f"A {kind} configuration must have string keys; got the key "
                f"{reprlib.repr(key)} ({type(key).__name__}).",
                component=kind,
                suggestion="write every key of the configuration as a string",
            )
    if _TYPE_FIELD not in data:
        raise ConfigError(
            f"The {kind} configuration has no 'type' field, so it does not say which {kind} "
            f"to use. Registered names: {listed}.",
            component=kind,
            field_path=_TYPE_FIELD,
            suggestion="add a 'type' entry that names a registered component",
        )
    name = data[_TYPE_FIELD]
    if not isinstance(name, str):
        raise ConfigError(
            f"The 'type' of a {kind} configuration must be a string naming a registered "
            f"{kind}; got {reprlib.repr(name)} ({type(name).__name__}). Registered names: "
            f"{listed}.",
            component=kind,
            field_path=_TYPE_FIELD,
            suggestion="write the registered name as a string, such as 'single_task_gp_v1'",
        )
    if name not in registry:
        raise ConfigError(
            f"Unknown {kind} type {name!r}. Registered names: {listed}.",
            component=kind,
            field_path=_TYPE_FIELD,
            suggestion="check the spelling, or import the module or plugin that registers it",
        )

    entry = registry.lookup(name)
    schema = entry.config_schema
    if not issubclass(schema, ComponentConfig):
        raise RegistryError(
            f"The {kind} {name!r} is registered with a configuration schema, "
            f"{schema.__name__}, that does not derive from ComponentConfig, so its "
            "configuration has no 'type' field.",
            component=f"{kind} {name!r}",
            field_path="config_schema",
            suggestion="derive the configuration schema from ComponentConfig",
        )
    return entry, schema
