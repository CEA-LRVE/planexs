"""A linear (in)equality constraint across parameters."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Literal

from pydantic import model_validator

from planexs.domain.constraints.Constraint import Constraint
from planexs.domain.constraints.LinearConstraintConfig import LinearConstraintConfig
from planexs.domain.constraints.registerConstraint import registerConstraint


@registerConstraint(
    "linear",
    config=LinearConstraintConfig,
    description="A linear (in)equality across parameters, such as a mixture that sums to one.",
)
class LinearConstraint(Constraint):
    """`sum(coefficients[name] * value[name]) <comparator> bound`.

    A typed, validated object, so that the caller never hand-writes an
    engine's string mini-language: an optimization backend renders it in its
    own syntax, and the domain knows no engine's syntax (specification 3.9).

    Example: a 3-component mixture summing to 1 is
    `LinearConstraint({"a": 1.0, "b": 1.0, "c": 1.0}, bound=1.0, comparator="=")`.

    Attributes:
        coefficients: Maps a parameter name to its coefficient in the sum.
        bound: The right-hand side of the (in)equality.
        comparator: `"<="`, `">="`, or `"="`.
    """

    coefficients: dict[str, float]
    bound: float
    comparator: Literal["<=", ">=", "="]

    def __init__(
        self,
        coefficients: dict[str, float],
        bound: float,
        comparator: Literal["<=", ">=", "="],
    ) -> None:
        """Creates a linear constraint.

        Args:
            coefficients: Parameter name to its coefficient in the sum.
            bound: The right-hand side.
            comparator: `"<="`, `">="` or `"="`.
        """
        super().__init__(coefficients=coefficients, bound=bound, comparator=comparator)

    @model_validator(mode="after")
    def _validateNotEmpty(self) -> LinearConstraint:
        """Checks the constraint involves at least one parameter.

        Returns:
            The constraint itself.

        Raises:
            ValueError: if there are no coefficients.
        """
        if not self.coefficients:
            raise ValueError("LinearConstraint requires at least one coefficient.")
        return self

    @property
    def requires_custom_acquisition_layer(self) -> bool:
        """`True` for the `"="` comparator: the optimization backend's native
        parameter constraints accept inequalities only, and encoding an equality
        as two inequalities leaves a zero-volume region its candidate generator
        can never sample. `planexs`'s own acquisition layer enforces equalities
        exactly (ADR-0008).
        """
        return self.comparator == "="

    def referencedNames(self) -> frozenset[str]:
        """The names of the parameters in the sum.

        Returns:
            The keys of `coefficients`.
        """
        return frozenset(self.coefficients)

    def violation(self, parameters: Mapping[str, object]) -> float:
        """How far a parameterization is from satisfying `sum(c * value) <comparator> bound`.

        The sum is exactly rounded (`math.fsum`), so that a mixture of ten parameters of `0.1`
        sums to `1.0` as it should, and not to the `0.9999999999999999` of a left-to-right
        float addition. The violation is how far the sum is on the wrong side of the bound (any
        distance from it for `"="`), divided by the scale `max(1, |bound|, sum(|c * value|))`,
        so that it measures rounding error, whatever the units of the parameters.

        Args:
            parameters: Parameter name to value. Only the names in `coefficients` are read.

        Returns:
            `0.0` where the constraint holds, otherwise the relative violation. `inf` if the
            sum has no finite value (an overflow).

        Raises:
            ValueError: if a name in `coefficients` is missing from `parameters`, or its
                value is not an `int` or a `float` (a `bool` or a string is not a number).
        """
        terms = [
            coefficient * _numberOf(parameters, name)
            for name, coefficient in self.coefficients.items()
        ]
        try:
            total = math.fsum(terms)
            magnitude = math.fsum(abs(term) for term in terms)
        except (OverflowError, ValueError):  # a sum too large for a float, or inf - inf
            return math.inf
        if not math.isfinite(total):
            return math.inf
        if self.comparator == "=":
            excess = abs(total - self.bound)
        elif self.comparator == "<=":
            excess = total - self.bound
        else:
            excess = self.bound - total
        return max(0.0, excess) / max(1.0, abs(self.bound), magnitude)

    def toDict(self) -> dict[str, Any]:
        """Serializes the constraint.

        Returns:
            A type-tagged, JSON-compatible form, rebuilt by `constraintFromDict`.
        """
        return {
            "type": "linear",
            "coefficients": dict(self.coefficients),
            "bound": self.bound,
            "comparator": self.comparator,
        }


def _numberOf(parameters: Mapping[str, object], name: str) -> float:
    """The value of one parameter of the sum, as a float.

    Args:
        parameters: Parameter name to value.
        name: The parameter.

    Returns:
        The value as a `float`.

    Raises:
        ValueError: if there is no value for `name`, or it is not an `int` or a `float` that a
            float can hold. An error of the call, which is never mistaken for a violation.
    """
    if name not in parameters:
        raise ValueError(
            f"The linear constraint uses {name!r}, which has no value among the given "
            f"parameters {sorted(parameters)!r}."
        )
    value = parameters[name]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"The linear constraint needs a number for {name!r}; got {value!r} "
            f"({type(value).__name__}). Only int and float values are numbers."
        )
    try:
        return float(value)
    except OverflowError:
        raise ValueError(
            f"The linear constraint: the value of {name!r} is too large for a float."
        ) from None
