"""A pure-data summary of what a problem contains."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from planexs.domain.constraints.LinearConstraint import LinearConstraint
from planexs.domain.parameters.Choice import Choice

if TYPE_CHECKING:
    from planexs.domain.Problem import Problem


class ProblemFacts(BaseModel):
    """What a problem contains, as plain data (specification 5.3, 5.12).

    The facts are what the compatibility check needs to know about a problem: whether it has
    several outcomes, a binary one, categorical parameters, equality constraints and so on.
    Turning them into capability tags (`space:categorical`, `outcome_kind:binary`, ...) needs
    the vocabulary and the registries, so it is done by the policies layer (roadmap task
    P6-05), not here: the domain only says what is there.

    The facts that name types (`outcome_types`, `parameter_types`, `constraint_types`) hold the
    registered names, as the serialized form of the problem writes them, so a type that a
    plugin registers shows up in them like the built-in ones.

    About `has_categorical_parameters`: a choice is categorical if its values have no order,
    and when `is_ordered` is not declared it is the encoding configuration that decides
    (specification 5.3), not the domain. A rule of the domain would be a policy value that the
    caller cannot change (specification 3.4). So the facts say what is declared:
    `has_categorical_parameters` is `True` if a searched choice is declared unordered, which is
    what `Categorical` and `Boolean` do, and `has_choices_with_undeclared_order` is `True` if
    a searched choice leaves it open, for the encoding configuration to settle.

    A facts object is frozen, like the problem it describes.

    Attributes:
        n_outcomes: The number of outcomes. At least one.
        outcome_types: The distinct outcome kinds that occur, as registered names, sorted.
        parameter_types: The distinct parameter types that occur (`range`, `choice`,
            `derived`, ...), as registered names, sorted. A `Real` is a `range`, a `Fixed` a
            `choice`: the sugar constructors are not types.
        constraint_types: The distinct constraint types that occur (`linear`, `nonlinear`,
            ...), as registered names, sorted. Empty if the space has no constraint.
        has_conditional_parameters: Whether some value of some parameter switches other
            parameters on (FR14).
        has_equality_constraints: Whether there is a linear constraint with the `"="`
            comparator (a mixture that sums to one, for example).
        has_categorical_parameters: Whether a searched choice is declared unordered. A choice
            with a single value is a constant, not searched, and does not count.
        has_choices_with_undeclared_order: Whether a searched choice leaves `is_ordered` unset.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    n_outcomes: int
    outcome_types: tuple[str, ...]
    parameter_types: tuple[str, ...]
    constraint_types: tuple[str, ...]
    has_conditional_parameters: bool
    has_equality_constraints: bool
    has_categorical_parameters: bool
    has_choices_with_undeclared_order: bool

    @classmethod
    def of(cls, problem: Problem) -> ProblemFacts:
        """The facts of a problem.

        Args:
            problem: The problem.

        Returns:
            Its facts. The same for the same problem, whatever the order of its parameters,
            constraints and outcomes.
        """
        parameters = problem.search_space.parameters
        constraints = problem.search_space.constraints
        # A choice with one value is a constant (a `Fixed`): it occupies no column and is not
        # searched, so it is neither ordered nor categorical.
        searched_choices = [
            parameter
            for parameter in parameters
            if isinstance(parameter, Choice) and len(parameter.values) > 1
        ]
        return cls(
            n_outcomes=len(problem.outcomes),
            outcome_types=tuple(sorted({outcome.type for outcome in problem.outcomes})),
            parameter_types=tuple(
                sorted({parameter.toDict()["type"] for parameter in parameters})
            ),
            constraint_types=tuple(
                sorted({constraint.toDict()["type"] for constraint in constraints})
            ),
            has_conditional_parameters=any(
                names for parameter in parameters for names in parameter.activations().values()
            ),
            has_equality_constraints=any(
                isinstance(constraint, LinearConstraint) and constraint.comparator == "="
                for constraint in constraints
            ),
            has_categorical_parameters=any(
                choice.is_ordered is False for choice in searched_choices
            ),
            has_choices_with_undeclared_order=any(
                choice.is_ordered is None for choice in searched_choices
            ),
        )
