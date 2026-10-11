"""Rebuilds a `Constraint` from its `toDict()` form."""

from __future__ import annotations

from typing import Any

from planexs.domain.constraints.Constraint import Constraint
from planexs.domain.constraints.LinearConstraint import LinearConstraint
from planexs.domain.constraints.NonlinearConstraint import NonlinearConstraint


def constraintFromDict(data: dict[str, Any]) -> Constraint:
    """The inverse of `Constraint.toDict`.

    Handles `LinearConstraint` and `NonlinearConstraint` (FR17, ADR-0006). A
    new `Constraint` subclass must be added to this dispatcher in the same
    change, or a study using one could be saved but not reloaded.

    Raises:
        ValueError: if `data["kind"]` is missing or unknown.
    """
    kind = data.get("kind")
    if kind == "linear":
        return LinearConstraint(
            coefficients=dict(data["coefficients"]),
            bound=data["bound"],
            comparator=data["comparator"],
        )
    if kind == "nonlinear":
        return NonlinearConstraint(
            expression=data["expression"],
            comparator=data["comparator"],
            bound=data["bound"],
        )
    raise ValueError(
        f"Cannot rebuild a Constraint from kind={kind!r}; expected 'linear' or 'nonlinear'."
    )
