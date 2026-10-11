"""A linear (in)equality constraint across parameters."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import model_validator

from planexs.domain.constraints.Constraint import Constraint


class LinearConstraint(Constraint):
    """`sum(coefficients[name] * value[name]) <comparator> bound`.

    Rendered to the string expression `Client.configure_experiment`'s
    `parameter_constraints` expects (e.g. `"a + b + c = 1.0"`) by
    `backends/ax/toAxSearchSpace.py`; kept as a typed, validated object here
    rather than asking the caller to hand-write Ax's string mini-language
    directly.

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
        """`True` for the `"="` comparator: Ax's `parameter_constraints` accept
        inequalities only (it rejects `"a + b = 1"`, and encoding an equality as
        two inequalities leaves a zero-volume region its candidate generator can
        never sample). `planexs`'s own acquisition layer enforces equalities
        exactly (ADR-0008).
        """
        return self.comparator == "="

    def toAxParameterConstraintString(self) -> str:
        """Renders this constraint as the string
        `Client.configure_experiment`'s `parameter_constraints` argument
        expects. A coefficient of exactly `1.0` is rendered as a bare
        parameter name (`"a"`, not `"1.0 * a"`) to match the mixture-style
        example in this class's own docstring; every other coefficient is
        rendered explicitly (`"2.0 * b"`).
        """
        terms: list[str] = []
        for name, coefficient in self.coefficients.items():
            if coefficient == 1.0:
                terms.append(name)
            else:
                terms.append(f"{coefficient} * {name}")
        left_hand_side = " + ".join(terms)
        return f"{left_hand_side} {self.comparator} {self.bound}"

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
