"""Maps a planexs `Choice` to the keyword arguments of Ax's `ChoiceParameterConfig`."""

from __future__ import annotations

from typing import Any

from planexs.domain.parameters.Choice import Choice


def toAxChoiceKwargs(parameter: Choice) -> dict[str, Any]:
    """Field values ready to pass straight to
    `ax.api.configs.ChoiceParameterConfig(**kwargs)`.

    `Choice`'s fields, `dependent_parameters` included, carry the names of the fields of
    that Ax config: `dependent_parameters` is Ax's own mechanism for conditional
    (hierarchical) search spaces (FR14), which planexs exposes as it is and does not
    reinvent. `default` is deliberately excluded: Ax's own config has no field for it.

    Args:
        parameter: The choice, or a subclass such as `Categorical`, `Boolean` or `Fixed`.

    Returns:
        A dictionary with the keys `name`, `values`, `parameter_type`, `is_ordered` and
        `dependent_parameters`.
    """
    return {
        "name": parameter.name,
        "values": parameter.values,
        "parameter_type": parameter.parameter_type,
        "is_ordered": parameter.is_ordered,
        "dependent_parameters": parameter.dependent_parameters,
    }
