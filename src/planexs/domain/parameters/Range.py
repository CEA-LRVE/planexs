"""A continuous or integer-stepped dimension of the search space."""

from __future__ import annotations

import math
from typing import Any, Literal

from planexs.domain.parameters.Parameter import Parameter


class Range(Parameter):
    """A continuous or integer-stepped dimension.

    Field names (`bounds`, `parameter_type`, `step_size`, `scaling`) match
    `ax.api.configs.RangeParameterConfig` directly: this class is a typed
    pass-through, not a reinvention. See `backends/ax/toAxSearchSpace.py`.

    Note that `ax.api.configs.RangeParameterConfig` has no `default` field
    (confirmed against https://ax.readthedocs.io/en/stable/api.html): `default`
    here is a planexs-only ergonomic addition (FR13), validated against
    `bounds`/`step_size` but never forwarded to Ax.

    Attributes:
        bounds: The `(lower, upper)` bounds of the dimension, inclusive.
        parameter_type: Whether this dimension is sampled as `"float"` or
            `"int"`.
        step_size: An optional fixed sampling step (FR13). When set, legal
            values are `bounds[0] + k * step_size` for integer `k`.
        scaling: `"linear"` (default, via Ax, when left `None`) or `"log"`,
            for a dimension that should be explored on a logarithmic scale.
    """

    bounds: tuple[float, float]
    parameter_type: Literal["float", "int"] = "float"
    step_size: float | None = None
    scaling: Literal["linear", "log"] | None = None

    def __init__(
        self,
        name: str,
        bounds: tuple[float, float],
        parameter_type: Literal["float", "int"] = "float",
        step_size: float | None = None,
        scaling: Literal["linear", "log"] | None = None,
        default: float | int | None = None,
    ) -> None:
        """Creates a range.

        Args:
            name: The parameter's name.
            bounds: The `(lower, upper)` bounds, inclusive.
            parameter_type: `"float"` or `"int"`.
            step_size: Optional spacing of the legal values.
            scaling: `"linear"` (default) or `"log"`.
            default: Optional default value.
        """
        super().__init__(
            name=name,
            bounds=bounds,
            parameter_type=parameter_type,
            step_size=step_size,
            scaling=scaling,
            default=default,
        )

    def _validate_default(self) -> None:
        """Checks the bounds, the log scaling, and the default if there is one.

        Raises:
            ValueError: if the bounds are empty or inverted, a log range is not strictly
                positive, or the default is outside the bounds or off the step grid.
        """
        lower, upper = self.bounds
        if lower >= upper:
            raise ValueError(
                f"Range {self.name!r} has an empty or inverted bounds "
                f"interval: lower={lower} must be strictly less than upper={upper}."
            )
        if self.scaling == "log" and lower <= 0.0:
            raise ValueError(
                f"Range {self.name!r} uses log scaling but its lower bound "
                f"({lower}) is not strictly positive."
            )
        if self.default is None:
            return
        value = float(self.default)
        if not (lower <= value <= upper):
            raise ValueError(
                f"Range {self.name!r} has default={self.default!r}, which "
                f"falls outside its own bounds {self.bounds!r}."
            )
        if self.step_size is not None and self.step_size > 0:
            steps_from_lower = (value - lower) / self.step_size
            if not math.isclose(steps_from_lower, round(steps_from_lower), abs_tol=1e-9):
                raise ValueError(
                    f"Range {self.name!r} has default={self.default!r}, which "
                    f"does not land on the step_size={self.step_size!r} grid "
                    f"starting at {lower!r}."
                )

    def toAxKwargs(self) -> dict[str, Any]:
        """Field values ready to pass straight to
        `ax.api.configs.RangeParameterConfig(**kwargs)`. Deliberately excludes
        `default`, which Ax's own config has no field for. Kept on this class
        rather than duplicated in `toAxSearchSpace.py` so the Range-to-Ax field
        mapping lives in exactly one place.
        """
        return {
            "name": self.name,
            "bounds": self.bounds,
            "parameter_type": self.parameter_type,
            "step_size": self.step_size,
            "scaling": self.scaling,
        }

    def toDict(self) -> dict[str, Any]:
        """Serializes the parameter.

        Returns:
            A type-tagged, JSON-compatible form, rebuilt by `parameterFromDict`.
        """
        return {
            "kind": "range",
            "name": self.name,
            "bounds": [self.bounds[0], self.bounds[1]],
            "parameter_type": self.parameter_type,
            "step_size": self.step_size,
            "scaling": self.scaling,
            "default": self.default,
        }
