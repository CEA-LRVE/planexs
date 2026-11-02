"""The configuration schema of the `derived` parameter type."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from planexs.core.ComponentConfig import ComponentConfig


class DerivedConfig(ComponentConfig):
    """The configuration of a `derived` parameter: a quantity computed from other parameters.

    It is the stored form of a `Derived` (what `Derived.toDict` writes and what a
    configuration file says), and `Derived.fromConfig` builds the domain object from it
    (specification 5.3, 5.12). It has no `default`: a derived parameter is computed, never
    chosen. A test keeps its fields in step with those of the `Derived` constructor.

    Attributes:
        type: The registered name, always `"derived"`.
        name: The parameter's name.
        expression: The expression that computes the value.
        parameter_type: The type the expression evaluates to.
    """

    type: Literal["derived"] = Field(
        default="derived",
        description=(
            "The registered name of this parameter type, always 'derived'. It selects the "
            "'derived' entry of the 'parameter' registry, so that a stored parameter says "
            "which type it is (specification 5.3). It is the only value the field accepts."
        ),
    )
    name: str = Field(
        description=(
            "The parameter's name, unique within a search space (the search space checks "
            "it). It is the key of the parameter in every parameterization and in the "
            "expressions of constraints and of other derived parameters. It has no default: "
            "a parameter cannot be referred to without one."
        ),
    )
    expression: str = Field(
        description=(
            "The expression that computes the value of the parameter from the names of "
            "other parameters, for example 'a + b'. It must not be blank. It has no "
            "default: it is what the parameter is."
        ),
    )
    parameter_type: Literal["float", "int", "str", "bool"] = Field(
        description=(
            "The Python type the expression evaluates to: 'float', 'int', 'str' or 'bool'. "
            "It has no default: only the caller knows what the expression yields, and the "
            "computed value is cast to this type."
        ),
    )
