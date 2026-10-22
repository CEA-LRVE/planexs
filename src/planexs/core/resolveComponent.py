"""Resolves a component configuration mapping against the registry of its kind."""

from __future__ import annotations

import reprlib
from collections.abc import Mapping
from typing import Any

from pydantic import ValidationError
from pydantic_core import ErrorDetails

from planexs.core.ComponentConfig import ComponentConfig
from planexs.core.ConfigError import ConfigError
from planexs.core.Registry import Registry
from planexs.core.RegistryError import RegistryError
from planexs.core.ResolvedComponent import EntryT, ResolvedComponent

#: The key that names a component in a configuration mapping: the `type` field of
#: `ComponentConfig` (specification 5.12).
_TYPE_FIELD = "type"

#: Pydantic error types whose `input` is not the offending value: for a missing field it is the
#: whole mapping the field is missing from, and an unknown field is reported by its name.
_NO_VALUE_TO_SHOW = frozenset({"missing", "extra_forbidden"})


def resolveComponent(
    registry: Registry[EntryT], data: Mapping[str, Any]
) -> ResolvedComponent[EntryT]:
    """Turns a configuration mapping into the registry entry it names and its validated
    configuration (specification 5.12).

    `data` is a mapping such as `{"type": "single_task_gp_v1", "min_observations": 3}`, read
    from Python, JSON or YAML. Its `type` entry is looked up in `registry`, and the whole
    mapping, `type` included, is then validated against the configuration schema that the
    component registered. Validation fills in the defaults of the schema and rejects unknown
    fields, so the returned configuration is complete.

    Every failure is a `ConfigError` that names the kind (the registry), the offending field
    and, when the problem is the `type`, the registered names. Nothing is stored or changed:
    `data` and the registry are left as they were.

    The field path of an error is dotted, with a list position in brackets, for example
    `kernel.nu` or `values[2]`. When several fields are wrong, the message lists every one of
    them and `field_path` is the path of the first.

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
            wrong type, a missing required field, a value a validator rejects).
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
    try:
        config = schema.model_validate(dict(data))
    except ValidationError as error:
        raise _invalidConfiguration(kind, name, schema, error) from error
    return ResolvedComponent(entry=entry, config=config)


def _invalidConfiguration(
    kind: str, name: str, schema: type[ComponentConfig], error: ValidationError
) -> ConfigError:
    """Turns the validation errors of a configuration into one `ConfigError`.

    Args:
        kind: The kind of component, which names the registry.
        name: The registered name of the component.
        schema: The schema the configuration was validated against.
        error: The failure that pydantic raised.

    Returns:
        An error whose message lists every problem with its field path, whose `field_path`
        is the path of the first one (`None` if it is not about a field), and whose suggestion
        lists the fields of the schema when a top-level field is unknown.
    """
    problems = error.errors(include_url=False)
    label = f"{kind} {name!r}"
    lines = [_describe(problem) for problem in problems]
    if len(lines) == 1:
        message = f"Invalid configuration for {label}: {lines[0]}."
    else:
        details = "\n".join(f"  - {line}" for line in lines)
        message = f"Invalid configuration for {label}: {len(lines)} problems.\n{details}"

    suggestion = None
    if any(
        problem["type"] == "extra_forbidden" and len(problem["loc"]) == 1
        for problem in problems
    ):
        fields = ", ".join(schema.model_fields)
        suggestion = f"check the spelling; the fields of {label} are: {fields}"
    return ConfigError(
        message,
        component=label,
        field_path=_path(problems[0]["loc"]) or None,
        suggestion=suggestion,
    )


def _describe(problem: ErrorDetails) -> str:
    """Describes one validation problem on a line: its field path, what is wrong and, where
    it helps, the value that was given.

    Args:
        problem: One entry of the errors that pydantic reports.

    Returns:
        The description, without a trailing period.
    """
    path = _path(problem["loc"])
    text = "Unknown field" if problem["type"] == "extra_forbidden" else problem["msg"]
    if path and problem["type"] not in _NO_VALUE_TO_SHOW:
        text = f"{text} (got {reprlib.repr(problem['input'])})"
    return f"{path}: {text}" if path else text


def _path(location: tuple[int | str, ...]) -> str:
    """Writes a pydantic error location as a field path.

    Args:
        location: The location: field names and list positions, outermost first.

    Returns:
        The names joined by dots, a list position in brackets after the name it indexes, for
        example `kernel.nu` or `values[2]`. Empty for an error about the whole configuration.
    """
    path = ""
    for part in location:
        if isinstance(part, int):
            path += f"[{part}]"
        else:
            path += f".{part}" if path else part
    return path
