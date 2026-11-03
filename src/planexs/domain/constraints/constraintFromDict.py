"""Rebuilds a `Constraint` from its `toDict()` form."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from planexs.core.RegistryError import RegistryError
from planexs.core.resolveComponent import resolveComponent
from planexs.domain.constraints.Constraint import Constraint
from planexs.domain.constraints.constraintRegistry import constraintRegistry


def constraintFromDict(data: Mapping[str, Any]) -> Constraint:
    """The inverse of `Constraint.toDict`.

    The `type` entry of `data` names a constraint type in the registry of the `constraint`
    kind (`linear` and `nonlinear` are the built-in ones, FR9 and FR17, ADR-0006). The whole
    mapping is validated against the configuration schema registered for that type, and the
    builder of the type makes the constraint from the validated configuration. No concrete
    class is named here, so a type that a plugin registers is rebuilt like a built-in one
    (specification 5.12), and a study that uses one can be saved and reloaded.

    Args:
        data: A mapping produced by `Constraint.toDict()`, possibly after a JSON round trip.
            It is not modified.

    Returns:
        The reconstructed `Constraint`.

    Raises:
        ConfigError: if `data` is not a mapping, has no `type` entry, has a `type` that names
            no registered constraint type (the message lists the registered ones), or does not
            validate against the schema of its type (an unknown field, a field of the wrong
            type, a missing field), with the path of the field.
        RegistryError: if the builder registered for the type does not return a `Constraint`.
        ValidationError: if the constructor of the type rejects the values (no coefficient, an
            expression outside the grammar, a bound that is not finite, ...).
    """
    resolved = resolveComponent(constraintRegistry(), data)
    constraint = resolved.entry.builder(resolved.config)
    if not isinstance(constraint, Constraint):
        raise RegistryError(
            f"The builder of the constraint type {resolved.entry.name!r} returned "
            f"{type(constraint).__name__}, which is not a Constraint.",
            component=f"constraint {resolved.entry.name!r}",
            suggestion="make the builder return the Constraint that the configuration says",
        )
    return constraint
