"""The configuration schema of the `choice` parameter type."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from planexs.core.ComponentConfig import ComponentConfig


class ChoiceConfig(ComponentConfig):
    """The configuration of a `choice` parameter: a discrete, ordinal or categorical dimension.

    It is the stored form of a `Choice` (what `Choice.toDict` writes and what a configuration
    file says), and `Choice.fromConfig` builds the domain object from it (specification 5.3,
    5.12). Its fields are those of the `Choice` constructor, except that the conditional
    parameters are held as a list of pairs, which is how they are stored: a test keeps the
    two in step. The validation that needs several fields at once (no duplicate value, a
    default among the values) stays with `Choice`, which the builder constructs.

    Attributes:
        type: The registered name, always `"choice"`.
        name: The parameter's name.
        values: The legal values.
        parameter_type: The type of the values: `"float"`, `"int"`, `"str"` or `"bool"`.
        is_ordered: Whether the values have a meaningful order.
        dependent_parameters: The conditional parameters, as `[value, [names...]]` pairs.
        default: The optional default value.
    """

    type: Literal["choice"] = Field(
        default="choice",
        description=(
            "The registered name of this parameter type, always 'choice'. It selects the "
            "'choice' entry of the 'parameter' registry, so that a stored parameter says "
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
    values: list[bool] | list[int] | list[float] | list[str] = Field(
        description=(
            "The legal values of the dimension: a non-empty list of booleans, integers, "
            "floats or strings, without duplicates. It has no default: only the caller "
            "knows which alternatives exist."
        ),
    )
    parameter_type: Literal["float", "int", "str", "bool"] = Field(
        description=(
            "The Python type of the elements of the values: 'float', 'int', 'str' or "
            "'bool'. It has no default: it is part of what the caller declares about the "
            "values, and a wrong guess would silently change how they are read."
        ),
    )
    is_ordered: bool | None = Field(
        default=None,
        description=(
            "Whether the values have a meaningful order (ordinal, such as 'low', 'medium', "
            "'high') or not (categorical). The default is None, which leaves the decision "
            "to whatever consumes the parameter: the encoder's own rule, or a backend's."
        ),
    )
    dependent_parameters: list[tuple[bool | int | float | str, list[str]]] | None = Field(
        default=None,
        description=(
            "The conditional parameters (FR14), as [value, [names...]] pairs: each pair "
            "gives the names of the parameters that only exist when the value is chosen. "
            "The value of a pair must be one of the values of the choice. It is a list of "
            "pairs and not a JSON object because the keys of an object are always strings, "
            "which would turn an integer, boolean or float value into text. The default is "
            "None, which means no parameter is conditional on this one."
        ),
    )
    default: bool | int | float | str | None = Field(
        default=None,
        description=(
            "The optional default value of the parameter: one of the values. The default "
            "is None, which means the parameter has no default value."
        ),
    )
