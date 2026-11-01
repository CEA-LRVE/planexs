"""Maps a planexs `Range` to the keyword arguments of Ax's `RangeParameterConfig`."""

from __future__ import annotations

from typing import Any

from planexs.domain.parameters.Range import Range


def toAxRangeKwargs(parameter: Range) -> dict[str, Any]:
    """Field values ready to pass straight to
    `ax.api.configs.RangeParameterConfig(**kwargs)`.

    `Range`'s `bounds`, `parameter_type`, `step_size` and `scaling` carry the names of the
    fields of that Ax config, so this is a typed pass-through and not a reinvention.
    `default` is deliberately excluded: Ax's own config has no field for it (confirmed
    against https://ax.readthedocs.io/en/stable/api.html), so it is a planexs-only addition
    that is never forwarded to Ax. The Range-to-Ax field mapping lives here, in exactly one
    place: `toAxSearchSpace` calls this function and does not repeat the mapping.

    Args:
        parameter: The range, or a subclass such as `Real` or `Integer`.

    Returns:
        A dictionary with the keys `name`, `bounds`, `parameter_type`, `step_size` and
        `scaling`.
    """
    return {
        "name": parameter.name,
        "bounds": parameter.bounds,
        "parameter_type": parameter.parameter_type,
        "step_size": parameter.step_size,
        "scaling": parameter.scaling,
    }
