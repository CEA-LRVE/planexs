"""The configuration schema of the `nonlinear` constraint type."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from planexs.core.ComponentConfig import ComponentConfig


class NonlinearConstraintConfig(ComponentConfig):
    """The configuration of a `nonlinear` constraint: `expression <comparator> bound`.

    It is the stored form of a `NonlinearConstraint` (what `NonlinearConstraint.toDict` writes
    and what a configuration file says), and `NonlinearConstraint.fromConfig` builds the domain
    object from it (specification 5.3, 5.12). Its fields are those of the `NonlinearConstraint`
    constructor, and a test keeps the two in step. The validation that the schema cannot
    express alone (the grammar of the expression, a finite bound) stays with
    `NonlinearConstraint`, which the builder constructs.

    Attributes:
        type: The registered name, always `"nonlinear"`.
        expression: The left-hand side, over parameter names.
        comparator: `"<="` or `">="`.
        bound: The right-hand side.
    """

    type: Literal["nonlinear"] = Field(
        default="nonlinear",
        description=(
            "The registered name of this constraint type, always 'nonlinear'. It selects the "
            "'nonlinear' entry of the 'constraint' registry, so that a stored constraint says "
            "which type it is (specification 5.3). It is the only value the field accepts."
        ),
    )
    expression: str = Field(
        description=(
            "The left-hand side of the constraint: an arithmetic expression over parameter "
            "names, for example 'var * x ** z', in the restricted grammar of numbers, names, "
            "the operators + - * / **, parentheses and a short list of functions. It is "
            "parsed and evaluated on the natural values of the parameters, never with "
            "Python's eval, and where it is undefined (the square root of a negative number) "
            "the point counts as infeasible. There is no default: it is what the constraint "
            "says."
        ),
    )
    comparator: Literal["<=", ">="] = Field(
        description=(
            "How the expression compares to the bound: '<=' or '>='. There is no '=' form: "
            "an exact nonlinear equality is a set of measure zero that a numerical optimizer "
            "cannot meaningfully target. There is no default: the direction of the limit is "
            "part of what the constraint says."
        ),
    )
    bound: float = Field(
        description=(
            "The right-hand side of the inequality, in the unit of the expression. It must "
            "be finite. There is no default: only the caller knows the limit that the "
            "constraint states."
        ),
    )
