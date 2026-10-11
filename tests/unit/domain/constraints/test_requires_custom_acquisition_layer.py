"""Which constraints only planexs' own acquisition layer can enforce (ADR-0006, ADR-0008)."""

from __future__ import annotations

import pytest

from planexs import LinearConstraint, NonlinearConstraint


class TestRequiresCustomAcquisitionLayer:
    @pytest.mark.parametrize("comparator", ["<=", ">="])
    def test_a_linear_inequality_is_enforced_by_ax(self, comparator: str) -> None:
        constraint = LinearConstraint({"a": 1.0, "b": 1.0}, 1.0, comparator)  # type: ignore[arg-type]
        assert constraint.requires_custom_acquisition_layer is False

    def test_a_linear_equality_is_not(self) -> None:
        # Ax's parameter_constraints accept inequalities only.
        assert LinearConstraint(
            {"a": 1.0, "b": 1.0}, 1.0, "="
        ).requires_custom_acquisition_layer

    def test_a_nonlinear_constraint_always_needs_it(self) -> None:
        for comparator in ("<=", ">="):
            constraint = NonlinearConstraint("a * b", comparator, 1.0)  # type: ignore[arg-type]
            assert constraint.requires_custom_acquisition_layer is True

    def test_the_flag_is_not_part_of_the_serialized_form(self) -> None:
        # It is derived from the comparator, so it must not leak into the saved study.
        assert (
            "requires_custom_acquisition_layer"
            not in LinearConstraint({"a": 1.0}, 1.0, "=").toDict()
        )
