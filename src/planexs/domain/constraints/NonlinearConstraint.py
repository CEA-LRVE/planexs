"""A feasibility constraint across parameters given as an arbitrary expression."""

from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import model_validator

from planexs.domain.constraints.Constraint import Constraint
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
    `eval`.

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
