"""Rebuilds a `Parameter` from its `toDict()` form."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from planexs.core.RegistryError import RegistryError
from planexs.core.resolveComponent import resolveComponent
from planexs.domain.parameters.Parameter import Parameter
from planexs.domain.parameters.parameterRegistry import parameterRegistry


def parameterFromDict(data: Mapping[str, Any]) -> Parameter:
    """The inverse of `Parameter.toDict`.

    The `type` entry of `data` names a parameter type in the registry of the `parameter`
    kind. The whole mapping is validated against the configuration schema registered for that
    type, and the builder of the type makes the parameter from the validated configuration.
    No concrete class is named here, so a type that a plugin registers is rebuilt like a
    built-in one (specification 5.12).

    Always returns the registered type, never a convenience subclass: a `Real` serialized
    earlier comes back as an equivalent `Range`, a `Fixed` as a single-value `Choice`, and so
    on. They are the same domain object (the sugar classes add no field), only the
    constructor spelling is lost.

    Args:
        data: A mapping produced by `Parameter.toDict()`, possibly after a JSON round trip
            (so tuples are lists and so on). It is not modified.

    Returns:
        The reconstructed `Parameter`.

    Raises:
        ConfigError: if `data` is not a mapping, has no `type` entry, has a `type` that names
            no registered parameter type (the message lists the registered ones), or does not
            validate against the schema of its type (an unknown field, a field of the wrong
            type, a missing field), with the path of the field.
        RegistryError: if the builder registered for the type does not return a `Parameter`.
        ValidationError: if the constructor of the type rejects the values (bounds out of
            order, a default that is not legal, ...).
    """
    resolved = resolveComponent(parameterRegistry(), data)
    parameter = resolved.entry.builder(resolved.config)
    if not isinstance(parameter, Parameter):
        raise RegistryError(
            f"The builder of the parameter type {resolved.entry.name!r} returned "
            f"{type(parameter).__name__}, which is not a Parameter.",
            component=f"parameter {resolved.entry.name!r}",
            suggestion="make the builder return the Parameter that the configuration says",
        )
    return parameter
