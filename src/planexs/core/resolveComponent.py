"""Resolves a component configuration mapping against the registry of its kind."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from planexs.core.Registry import Registry
from planexs.core.ResolvedComponent import EntryT, ResolvedComponent
from planexs.core.selectComponent import selectComponent
from planexs.core.validateConfig import validateConfig


def resolveComponent(
    registry: Registry[EntryT], data: Mapping[str, Any]
) -> ResolvedComponent[EntryT]:
    """Turns a configuration mapping into the registry entry it names and its validated
    configuration (specification 5.12).

    `data` is a mapping such as `{"type": "single_task_gp_v1", "min_observations": 3}`, read
    from Python, JSON or YAML. Its `type` entry is looked up in `registry`, and the whole
    mapping, `type` included, is then validated against the configuration schema that the
    component registered. Validation fills in the defaults of the schema and rejects unknown
    fields, so the returned configuration is complete. A component nested in a field of the
    schema (see `ComponentOf`) is resolved in turn, to any depth, so the configuration holds
    the validated configuration of each component below it.

    Every failure is a `ConfigError` that names the kind (the registry), the offending field
    and, when the problem is the `type`, the registered names. Nothing is stored or changed:
    `data` and the registry are left as they were.

    The field path of an error is dotted, with a list position in brackets, for example
    `kernel.nu` or `values[2]`; below a nested component it is the full path from `data`, for
    example `surrogate.kernel.nu`. When several fields are wrong, the message lists every one
    of them and `field_path` is the path of the first.

    Args:
        registry: The registry of the kind of component the configuration is for.
        data: The configuration: a mapping from field names to values, with a `type` entry
            that names a registered component of that kind. It is not modified.

    Returns:
        The registered entry and the validated configuration. The configuration is an instance
        of the entry's own `config_schema`.

    Raises:
        ConfigError: if `data` is not a mapping or has a key that is not a string, has no
            `type` entry, has a `type` that is not a string or names nothing in `registry`, or
            does not validate against the component's schema (an unknown field, a field of the
            wrong type, a missing required field, a value a validator rejects), including
            in a nested component.
        RegistryError: if the component is registered with a configuration schema that does
            not derive from `ComponentConfig`, or a component field names a kind that is not
            defined: defects of the registration, not of the configuration.
    """
    entry, schema = selectComponent(registry, data)
    config = validateConfig(schema, data, label=f"{registry.kind} {entry.name!r}")
    return ResolvedComponent(entry=entry, config=config)
