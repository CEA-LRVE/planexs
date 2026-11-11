"""A feasibility constraint across parameters given as an arbitrary expression."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Literal

from pydantic import model_validator

from planexs.domain.constraints.Constraint import Constraint
from planexs.domain.constraints.evaluateExpression import evaluateExpression
from planexs.domain.constraints.NonlinearConstraintConfig import NonlinearConstraintConfig
from planexs.domain.constraints.registerConstraint import registerConstraint
from planexs.domain.constraints.validateExpression import validateExpression


@registerConstraint(
    "nonlinear",
    config=NonlinearConstraintConfig,
    description="A feasibility constraint given as an arithmetic expression over parameters.",
)
class NonlinearConstraint(Constraint):
    """`expression <comparator> bound`, e.g. `NonlinearConstraint("var * x ** z", "<=", 50.0)`.
    FR17.

    A string, not a Python callable, specifically so a study using one still
    round-trips through `JsonStudyRepository` (FR11, FR12) like every other
    domain object. It is parsed and evaluated by a small restricted evaluator
    (arithmetic and the short allowlist in
    `validateExpression.ALLOWED_EXPRESSION_FUNCTIONS`), never Python's own
    `eval`. Two evaluators exist: `isSatisfiedBy` checks one parameterization with
    plain numbers and imports no machine-learning library, and the acquisition
    layer compiles the same expression to tensors (`encoding/compileExpression.py`).

    No `"="` comparator: an exact nonlinear equality is a measure-zero set a
    numerical optimizer cannot meaningfully target, unlike `LinearConstraint`'s
    `"="`, which BoTorch handles as a true linear equality constraint.

    Not expressible as a linear constraint, which is all the optimization
    backend enforces natively, so the backend ignores it. It is enforced only
    by `planexs`'s own acquisition layer
    (`encoding/toBotorchNonlinearConstraints.py`), which is why
    `BayesianOptimizer.ask()` switches to that layer when one is present
    (ADR-0006).

    The expression is evaluated on the parameters' natural values (not their
    normalized encoding), so `"x ** 2"` means the square of the parameter `x`
    as the caller sees it. Names must refer to numeric parameters; behavior
    where the expression is undefined (e.g. `sqrt` of a negative number, or a
    negative base with a fractional power) is that the point counts as
    infeasible.

    Attributes:
        expression: The left-hand side, over parameter names.
        comparator: `"<="` or `">="`.
        bound: The right-hand side. Must be finite.
    """

    expression: str
    comparator: Literal["<=", ">="]
    bound: float

    def __init__(
        self,
        expression: str,
        comparator: Literal["<=", ">="],
        bound: float,
    ) -> None:
        """Creates a nonlinear constraint from an expression string.

        Args:
            expression: The left-hand side, over parameter names.
            comparator: `"<="` or `">="`.
            bound: The right-hand side. Must be finite.

        Raises:
            ValueError: if the bound is not finite or the expression is invalid.
        """
        super().__init__(expression=expression, comparator=comparator, bound=bound)

    @model_validator(mode="after")
    def _validateExpressionAndBound(self) -> NonlinearConstraint:
        """Checks the bound is finite and the expression uses only the allowed grammar.

        Returns:
            The constraint itself.

        Raises:
            ValueError: if either check fails.
        """
        if not math.isfinite(self.bound):
            raise ValueError(f"NonlinearConstraint bound must be finite; got {self.bound!r}.")
        validateExpression(self.expression)
        return self

    @property
    def requires_custom_acquisition_layer(self) -> bool:
        """Always `True`: the backend's native parameter constraints are linear only."""
        return True

    def referencedNames(self) -> frozenset[str]:
        """The parameter names this constraint's expression uses. Parses the
        expression each call, so it is a method rather than a property.

        Returns:
            The set of variable names in `expression`.
        """
        return validateExpression(self.expression)

    def isSatisfiedBy(self, parameters: Mapping[str, object]) -> bool:
        """Whether a parameterization of natural values satisfies this constraint.

        Evaluates `expression` with `evaluateExpression`, on plain numbers and without any
        machine-learning library, and compares it with `bound`. The comparison is exact: a
        value on the bound satisfies it, and there is no tolerance (the acquisition layer
        applies its own to encoded points, see `EncodedConstraints`). Where the expression is
        undefined or has no finite value (`sqrt` of a negative number, a division by zero, an
        overflow) the point is infeasible.

        Parses the expression each call, so it is a method rather than a property.

        Args:
            parameters: Parameter name to value, on natural values. Only the names this
                constraint uses are read, so a whole parameterization can be passed, with
                categorical and boolean parameters in it.

        Returns:
            `True` if the constraint holds.

        Raises:
            ValueError: if a name the expression uses is missing from `parameters`, or its
                value is not an `int` or `float` (a `bool` or a string is not a number).
                These are errors of the call, not infeasible points: a point whose parameter
                is inactive, for example, cannot be judged by this constraint.
        """
        value = evaluateExpression(self.expression, parameters)
        if math.isnan(value):
            return False
        return value <= self.bound if self.comparator == "<=" else value >= self.bound

    def violation(self, parameters: Mapping[str, object]) -> float:
        """How far a parameterization is from satisfying `expression <comparator> bound`.

        Evaluates `expression` with `evaluateExpression`. The violation is how far the value is
        on the wrong side of the bound, divided by the scale `max(1, |bound|)`, which is the
        scale that `EncodedConstraints` uses for a nonlinear constraint, so that one tolerance
        suits every constraint.

        Args:
            parameters: Parameter name to value. Only the names the expression uses are read.

        Returns:
            `0.0` where the constraint holds, otherwise the relative violation. `inf` where the
            expression is undefined or has no finite value (see `isSatisfiedBy`).

        Raises:
            ValueError: if a name the expression uses is missing from `parameters`, or its
                value is not an `int` or a `float` (a `bool` or a string is not a number).
        """
        value = evaluateExpression(self.expression, parameters)
        if math.isnan(value):
            return math.inf
        excess = value - self.bound if self.comparator == "<=" else self.bound - value
        return max(0.0, excess) / max(1.0, abs(self.bound))

    def toDict(self) -> dict[str, Any]:
        """Serializes the constraint.

        Returns:
            A type-tagged, JSON-compatible form, rebuilt by `constraintFromDict`.
        """
        return {
            "type": "nonlinear",
            "expression": self.expression,
            "comparator": self.comparator,
            "bound": self.bound,
        }
