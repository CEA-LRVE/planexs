"""Unit tests for `toAxParameterConstraintString`.

Pure string rendering over planexs' own `LinearConstraint`: does not import `ax.*`, so these
tests run without a real Ax install.
"""

from __future__ import annotations

import json

from planexs import LinearConstraint
from planexs.backends.ax.toAxParameterConstraintString import toAxParameterConstraintString
from planexs.domain.constraints.constraintFromDict import constraintFromDict


class TestToAxParameterConstraintString:
    def test_unit_coefficients_render_as_bare_names(self) -> None:
        constraint = LinearConstraint(
            {"a": 1.0, "b": 1.0, "c": 1.0}, bound=1.0, comparator="="
        )
        assert toAxParameterConstraintString(constraint) == "a + b + c = 1.0"

    def test_non_unit_coefficients_render_explicitly(self) -> None:
        constraint = LinearConstraint({"a": 2.0, "b": -1.0}, bound=5.0, comparator="<=")
        assert toAxParameterConstraintString(constraint) == "2.0 * a + -1.0 * b <= 5.0"

    def test_mixed_unit_and_non_unit_coefficients(self) -> None:
        constraint = LinearConstraint({"a": 1.0, "b": 3.0}, bound=10.0, comparator=">=")
        assert toAxParameterConstraintString(constraint) == "a + 3.0 * b >= 10.0"

    def test_a_reloaded_constraint_renders_the_same_string(self) -> None:
        original = LinearConstraint({"a": 1.0, "b": 2.5}, bound=3.0, comparator="<=")
        reloaded = constraintFromDict(json.loads(json.dumps(original.toDict())))

        assert isinstance(reloaded, LinearConstraint)
        assert toAxParameterConstraintString(reloaded) == toAxParameterConstraintString(
            original
        )
