"""Maps a planexs `Derived` to the keyword arguments of Ax's `DerivedParameterConfig`."""

from __future__ import annotations

from typing import Any

from planexs.domain.parameters.Derived import Derived


def toAxDerivedKwargs(parameter: Derived) -> dict[str, Any]:
    """Field values ready to pass straight to
    `ax.api.configs.DerivedParameterConfig(**kwargs)`.

    `Derived` mirrors that Ax config directly, except that Ax names the expression field
    `expression_str` where planexs' own field is named `expression` for readability: the
    renaming is done here.

    Args:
        parameter: The derived parameter.

    Returns:
        A dictionary with the keys `name`, `expression_str` and `parameter_type`.
    """
    return {
        "name": parameter.name,
        "expression_str": parameter.expression,
        "parameter_type": parameter.parameter_type,
    }
