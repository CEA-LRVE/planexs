"""A constraint on an observed outcome, as a goal states it."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class OutcomeConstraint(BaseModel):
    """A limit on what an outcome may measure: `outcome <comparator> bound` (specification 4,
    5.7, FR15).

    A constraint on an outcome belongs to a goal and not to the search space: it limits what
    is observed, where a parameter constraint limits what is chosen (specification 4,
    "Constraint"). It is stated on the natural values of the outcome, in its own unit. It is
    the stored form of what the legacy layer called a `LegacyOutcomeConstraint`, without the
    `relative` flag: a bound relative to a baseline trial is out of scope (roadmap task P4-03).

    It is a plain description and judges nothing: whether the outcome exists and is of a kind
    that can be compared with a bound is checked by the goal that holds the constraint, against
    its problem (`AbstractGoal.validateAgainst`), and whether the surrogate's prediction meets
    the bound is the business of the acquisition that the goal parameterizes. Any goal that
    can be limited by what is observed can hold these, which is why the class has a file of its
    own and is not nested in the schema of one goal.

    The outcome that a constraint names need not be one the goal optimizes: a throughput that
    must stay above a floor while a yield is maximized is the typical case. Two constraints on
    one outcome, one for each comparator, state a band.

    It is built with keywords, `OutcomeConstraint(outcome="qps", comparator=">=",
    bound=100.0)`, and is frozen. Unknown fields are rejected, so a misspelled setting is an
    error and never a silent no-op.

    Attributes:
        outcome: The name of the constrained outcome.
        comparator: `"<="` or `">="`.
        bound: The right-hand side, in the unit of the outcome.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    outcome: str = Field(
        min_length=1,
        description=(
            "The name of the outcome that the constraint limits, as the problem names it. It "
            "is checked against the problem when the goal is validated against it, and need "
            "not be an outcome that the goal optimizes. It has no default: only the caller "
            "knows which measurement has a limit."
        ),
    )
    comparator: Literal["<=", ">="] = Field(
        description=(
            "How the outcome compares to the bound: '<=' (at most) or '>=' (at least). There "
            "is no '=' form: an observed continuous value that is exactly equal to a number "
            "is a set of measure zero, which no acquisition can meaningfully target. There is "
            "no default: the direction of the limit is part of what the constraint says."
        ),
    )
    bound: float = Field(
        allow_inf_nan=False,
        description=(
            "The right-hand side of the inequality, in the unit of the outcome. It must be "
            "finite: an infinite bound constrains nothing, and nan compares with nothing. It "
            "has no default: only the caller knows the limit that the constraint states."
        ),
    )
