"""The configuration schema of the `linear` constraint type."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from planexs.core.ComponentConfig import ComponentConfig


class LinearConstraintConfig(ComponentConfig):
    """The configuration of a `linear` constraint: `sum(coefficient * value) <cmp> bound`.

    It is the stored form of a `LinearConstraint` (what `LinearConstraint.toDict` writes and
    what a configuration file says), and `LinearConstraint.fromConfig` builds the domain object
    from it (specification 5.3, 5.12). Its fields are those of the `LinearConstraint`
    constructor, and a test keeps the two in step. The validation that the schema cannot
    express alone (at least one coefficient) stays with `LinearConstraint`, which the builder
    constructs.

    Attributes:
        type: The registered name, always `"linear"`.
        coefficients: The coefficient of each parameter in the sum.
        bound: The right-hand side.
        comparator: `"<="`, `">="` or `"="`.
    """

    type: Literal["linear"] = Field(
        default="linear",
        description=(
            "The registered name of this constraint type, always 'linear'. It selects the "
            "'linear' entry of the 'constraint' registry, so that a stored constraint says "
            "which type it is (specification 5.3). It is the only value the field accepts."
        ),
    )
    coefficients: dict[str, float] = Field(
        description=(
            "Maps the name of each parameter that the constraint involves to its coefficient "
            "in the sum, in the unit of the bound per unit of that parameter. At least one "
            "coefficient is required. The constraint is stated on the natural values of the "
            "parameters, what the caller sees (specification 5.3), and its names must be "
            "those of range parameters: the encoding of the search space checks it. There is "
            "no default: a constraint without a coefficient constrains nothing."
        ),
    )
    bound: float = Field(
        description=(
            "The right-hand side of the (in)equality, in the unit of the sum. There is no "
            "default: only the caller knows the limit that the constraint states."
        ),
    )
    comparator: Literal["<=", ">=", "="] = Field(
        description=(
            "How the sum compares to the bound: '<=', '>=' or '='. The '=' form is an "
            "equality, such as a mixture that sums to one: the optimization backend's own "
            "constraints take inequalities only, so planexs' own acquisition layer enforces "
            "it exactly (ADR-0008). There is no default: the direction of the limit is part "
            "of what the constraint says."
        ),
    )
