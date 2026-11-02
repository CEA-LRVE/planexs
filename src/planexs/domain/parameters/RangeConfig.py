"""The configuration schema of the `range` parameter type."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from planexs.core.ComponentConfig import ComponentConfig


class RangeConfig(ComponentConfig):
    """The configuration of a `range` parameter: a continuous or integer-stepped dimension.

    It is the stored form of a `Range` (what `Range.toDict` writes and what a configuration
    file says), and `Range.fromConfig` builds the domain object from it (specification 5.3,
    5.12). Its fields are those of the `Range` constructor, and a test keeps the two in step.
    The validation that needs several fields at once (bounds in order, a default on the grid)
    stays with `Range`, which the builder constructs.

    Attributes:
        type: The registered name, always `"range"`.
        name: The parameter's name.
        bounds: The `(lower, upper)` bounds, inclusive.
        parameter_type: Whether the dimension is sampled as `"float"` or `"int"`.
        step_size: The optional spacing of the legal values.
        scaling: The scale the dimension is explored on, `"linear"` or `"log"`.
        default: The optional default value.
    """

    type: Literal["range"] = Field(
        default="range",
        description=(
            "The registered name of this parameter type, always 'range'. It selects the "
            "'range' entry of the 'parameter' registry, so that a stored parameter says "
            "which type it is (specification 5.3). It is the only value the field accepts."
        ),
    )
    name: str = Field(
        description=(
            "The parameter's name, unique within a search space (the search space checks "
            "it). It is the key of the parameter in every parameterization and in the "
            "expressions of constraints and derived parameters. It has no default: a "
            "parameter cannot be referred to without one."
        ),
    )
    bounds: tuple[float, float] = Field(
        description=(
            "The (lower, upper) bounds of the dimension, inclusive, in the unit of the "
            "quantity the parameter stands for. The lower bound must be strictly below the "
            "upper one, and strictly positive when the scaling is 'log'. It has no default: "
            "only the caller knows which range is worth searching."
        ),
    )
    parameter_type: Literal["float", "int"] = Field(
        default="float",
        description=(
            "Whether the dimension is sampled as 'float' or 'int'. The default is 'float', "
            "the general case: an 'int' range is snapped onto the integers inside its "
            "bounds, which is a deliberate choice."
        ),
    )
    step_size: float | None = Field(
        default=None,
        description=(
            "The optional spacing of the legal values, in the unit of the parameter: when "
            "set, the legal values are lower + k * step_size for an integer k. The default "
            "is None, which leaves a 'float' range continuous (and an 'int' range spaced by "
            "one): a grid is a deliberate choice."
        ),
    )
    scaling: Literal["linear", "log"] | None = Field(
        default=None,
        description=(
            "The scale the dimension is explored on: 'linear' or 'log'. The default is "
            "None, which means linear. A log scale suits a quantity that spans orders of "
            "magnitude, and needs strictly positive bounds."
        ),
    )
    default: int | float | None = Field(
        default=None,
        description=(
            "The optional default value of the parameter, in its unit. It is validated "
            "against the bounds and the step size. The default is None, which means the "
            "parameter has no default value."
        ),
    )
