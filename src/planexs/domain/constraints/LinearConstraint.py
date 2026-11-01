"""A linear (in)equality constraint across parameters."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import model_validator

from planexs.domain.constraints.Constraint import Constraint


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

    def toDict(self) -> dict[str, Any]:
        """Serializes the constraint.

        Returns:
            A type-tagged, JSON-compatible form, rebuilt by `constraintFromDict`.
        """
        return {
            "kind": "linear",
            "coefficients": dict(self.coefficients),
            "bound": self.bound,
            "comparator": self.comparator,
        }
