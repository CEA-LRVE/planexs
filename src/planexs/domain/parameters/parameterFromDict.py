"""Rebuilds a `Parameter` from its `toDict()` form."""

from __future__ import annotations

from typing import Any

from planexs.domain.parameters.Choice import Choice
from planexs.domain.parameters.Derived import Derived
from planexs.domain.parameters.Parameter import Parameter
from planexs.domain.parameters.Range import Range


def parameterFromDict(data: dict[str, Any]) -> Parameter:
    """The inverse of `Parameter.toDict`.

    Always returns one of the three base kinds (`Range`, `Choice`,
    `Derived`): a `Real` serialized earlier comes back as an equivalent
    `Range`, a `Fixed` as a single-value `Choice`, and so on. They are the
    same domain object (the sugar classes add no field), only the
    constructor spelling is lost.

    Args:
        data: A dict produced by `Parameter.toDict()`, possibly after a JSON
            round trip (so tuples are lists and so on).

    Returns:
        The reconstructed `Parameter`.

    Raises:
        ValueError: if `data["kind"]` is missing or unknown.
    """
    kind = data.get("kind")
    if kind == "range":
        return Range(
            name=data["name"],
            bounds=(data["bounds"][0], data["bounds"][1]),
            parameter_type=data.get("parameter_type", "float"),
            step_size=data.get("step_size"),
            scaling=data.get("scaling"),
            default=data.get("default"),
        )
    if kind == "choice":
        pairs = data.get("dependent_parameters")
        dependent = (
            {value: list(names) for value, names in pairs} if pairs is not None else None
        )
        return Choice(
            name=data["name"],
            values=list(data["values"]),
            parameter_type=data["parameter_type"],
            is_ordered=data.get("is_ordered"),
            dependent_parameters=dependent,
            default=data.get("default"),
        )
    if kind == "derived":
        return Derived(
            name=data["name"],
            expression=data["expression"],
            parameter_type=data["parameter_type"],
        )
    raise ValueError(
        f"Cannot rebuild a Parameter from kind={kind!r}; expected one of "
        "'range', 'choice', 'derived'."
    )
