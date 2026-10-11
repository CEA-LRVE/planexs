"""Unit tests for `LinearConstraint`."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from planexs import LinearConstraint


class TestLinearConstraint:
    def test_requires_at_least_one_coefficient(self) -> None:
        with pytest.raises(ValidationError, match="at least one coefficient"):
            LinearConstraint({}, bound=1.0, comparator="<=")

    def test_unit_coefficients_render_as_bare_names(self) -> None:
        constraint = LinearConstraint(
            {"a": 1.0, "b": 1.0, "c": 1.0}, bound=1.0, comparator="="
        )
        assert constraint.toAxParameterConstraintString() == "a + b + c = 1.0"

    def test_non_unit_coefficients_render_explicitly(self) -> None:
        constraint = LinearConstraint({"a": 2.0, "b": -1.0}, bound=5.0, comparator="<=")
        assert constraint.toAxParameterConstraintString() == "2.0 * a + -1.0 * b <= 5.0"

    def test_mixed_unit_and_non_unit_coefficients(self) -> None:
        constraint = LinearConstraint({"a": 1.0, "b": 3.0}, bound=10.0, comparator=">=")
        assert constraint.toAxParameterConstraintString() == "a + 3.0 * b >= 10.0"
