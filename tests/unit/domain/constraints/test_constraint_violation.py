"""Unit tests for how far a parameterization is from a constraint (P2-10)."""

from __future__ import annotations

import functools
import math
import operator
import random
from typing import Any

import pytest

from planexs import LinearConstraint, NonlinearConstraint
from planexs.domain.constraints.Constraint import Constraint


class TestReferencedNames:
    def test_a_linear_constraint_names_the_parameters_of_its_sum(self) -> None:
        constraint = LinearConstraint({"a": 1.0, "b": 2.0}, bound=1.0, comparator="<=")

        assert constraint.referencedNames() == frozenset({"a", "b"})

    def test_a_nonlinear_constraint_names_the_parameters_of_its_expression(self) -> None:
        constraint = NonlinearConstraint("var * x ** z", "<=", 50.0)

        assert constraint.referencedNames() == frozenset({"var", "x", "z"})


class TestLinear:
    @pytest.mark.parametrize(
        ("comparator", "bound", "point"),
        [
            ("<=", 1.0, {"a": 0.4, "b": 0.4}),
            ("<=", 1.0, {"a": 0.5, "b": 0.5}),  # on the bound
            (">=", 1.0, {"a": 0.8, "b": 0.8}),
            (">=", 1.0, {"a": 0.5, "b": 0.5}),  # on the bound
            ("=", 1.0, {"a": 0.5, "b": 0.5}),
        ],
    )
    def test_zero_where_it_holds(self, comparator: Any, bound: float, point: dict) -> None:
        constraint = LinearConstraint({"a": 1.0, "b": 1.0}, bound=bound, comparator=comparator)

        assert constraint.violation(point) == 0.0

    def test_the_violation_is_relative_to_the_scale(self) -> None:
        # 0.7 + 0.8 = 1.5, which is 0.5 above the bound 1. The scale is the larger of 1, the
        # bound and the sum of the magnitudes of the terms: 1.5. So 0.5 / 1.5.
        constraint = LinearConstraint({"a": 1.0, "b": 1.0}, bound=1.0, comparator="<=")

        assert constraint.violation({"a": 0.7, "b": 0.8}) == pytest.approx(0.5 / 1.5)

    def test_a_large_bound_sets_a_large_scale(self) -> None:
        constraint = LinearConstraint({"a": 1.0}, bound=1e6, comparator=">=")

        # 1e5 below a bound of 1e6: a tenth of the scale, however large the numbers are.
        assert constraint.violation({"a": 0.9e6}) == pytest.approx(0.1)

    def test_a_small_bound_does_not_shrink_the_scale_below_one(self) -> None:
        constraint = LinearConstraint({"a": 1.0}, bound=1e-6, comparator="<=")

        assert constraint.violation({"a": 0.5}) == pytest.approx(0.5 - 1e-6)

    @pytest.mark.parametrize("a", [0.9, 1.1])
    def test_an_equality_is_violated_on_either_side(self, a: float) -> None:
        constraint = LinearConstraint({"a": 1.0}, bound=1.0, comparator="=")

        # 0.1 away from the bound on either side, over the scale max(1, |bound|, |a|).
        assert constraint.violation({"a": a}) == pytest.approx(0.1 / max(1.0, a))

    def test_a_mixture_of_floats_sums_exactly(self) -> None:
        names = [f"p{i}" for i in range(10)]
        constraint = LinearConstraint({name: 1.0 for name in names}, bound=1.0, comparator="=")
        mixture = {name: 0.1 for name in names}

        # A left-to-right addition of ten tenths is off by a rounding, on every Python. (The
        # builtin `sum` is not a good witness: it is compensated from Python 3.12 on.)
        assert functools.reduce(operator.add, mixture.values()) != 1.0
        assert constraint.violation(mixture) == 0.0  # the exactly rounded sum is not

    def test_coefficients_and_negative_terms(self) -> None:
        constraint = LinearConstraint({"a": 2.0, "b": -1.0}, bound=0.0, comparator="<=")

        assert constraint.violation({"a": 1.0, "b": 3.0}) == 0.0  # 2 - 3 = -1
        assert constraint.violation({"a": 3.0, "b": 1.0}) == pytest.approx(5.0 / 7.0)

    def test_names_the_constraint_does_not_use_are_ignored(self) -> None:
        constraint = LinearConstraint({"a": 1.0}, bound=1.0, comparator="<=")

        assert constraint.violation({"a": 0.5, "solvent": "water", "flag": True}) == 0.0

    def test_integers_are_numbers(self) -> None:
        assert (
            LinearConstraint({"a": 1.0, "b": 1.0}, 3.0, "=").violation({"a": 1, "b": 2}) == 0.0
        )

    def test_a_sum_too_large_for_a_float_is_infinitely_violating(self) -> None:
        constraint = LinearConstraint({"a": 1e308, "b": 1e308}, bound=1.0, comparator="<=")

        assert constraint.violation({"a": 10.0, "b": 10.0}) == math.inf

    def test_it_is_never_negative_and_never_nan(self) -> None:
        constraint = LinearConstraint({"a": 1.0, "b": 1.0}, bound=1.0, comparator="<=")
        generator = random.Random(20261110)

        for _ in range(200):
            point = {"a": generator.uniform(-5, 5), "b": generator.uniform(-5, 5)}
            value = constraint.violation(point)
            assert value >= 0.0 and not math.isnan(value)

    def test_a_missing_parameter_is_an_error_not_a_violation(self) -> None:
        constraint = LinearConstraint({"a": 1.0, "b": 1.0}, bound=1.0, comparator="<=")

        with pytest.raises(ValueError, match=r"'b'.*no value"):
            constraint.violation({"a": 0.5})

    @pytest.mark.parametrize("value", ["0.5", True, None, [0.5]])
    def test_a_value_that_is_not_a_number_is_an_error(self, value: object) -> None:
        constraint = LinearConstraint({"a": 1.0}, bound=1.0, comparator="<=")

        with pytest.raises(ValueError, match="needs a number for 'a'"):
            constraint.violation({"a": value})

    def test_an_integer_too_large_for_a_float_is_an_error(self) -> None:
        constraint = LinearConstraint({"a": 1.0}, bound=1.0, comparator="<=")

        with pytest.raises(ValueError, match="too large for a float"):
            constraint.violation({"a": 10**400})


class TestNonlinear:
    def test_zero_where_it_holds_and_on_the_bound(self) -> None:
        constraint = NonlinearConstraint("x ** 2", "<=", 4.0)

        assert constraint.violation({"x": 1.0}) == 0.0
        assert constraint.violation({"x": 2.0}) == 0.0
        assert constraint.violation({"x": -2.0}) == 0.0

    def test_the_violation_is_relative_to_the_bound(self) -> None:
        # 9 is 5 above the bound 4, and the scale is max(1, |bound|) = 4.
        assert NonlinearConstraint("x ** 2", "<=", 4.0).violation({"x": 3.0}) == pytest.approx(
            1.25
        )

    def test_a_small_bound_does_not_shrink_the_scale_below_one(self) -> None:
        assert NonlinearConstraint("x", "<=", 0.001).violation({"x": 0.501}) == pytest.approx(
            0.5
        )

    def test_a_lower_bound(self) -> None:
        constraint = NonlinearConstraint("sqrt(x)", ">=", 2.0)

        assert constraint.violation({"x": 9.0}) == 0.0
        assert constraint.violation({"x": 1.0}) == pytest.approx(0.5)

    @pytest.mark.parametrize("comparator", ["<=", ">="])
    @pytest.mark.parametrize(
        ("expression", "point"),
        [
            ("sqrt(x)", {"x": -1.0}),
            ("log(x)", {"x": 0.0}),
            ("x ** (1 / 3)", {"x": -8.0}),
            ("1 / x", {"x": 0.0}),
            ("exp(x)", {"x": 1000.0}),
        ],
    )
    def test_an_undefined_point_is_infinitely_violating(
        self, comparator: Any, expression: str, point: dict[str, float]
    ) -> None:
        constraint = NonlinearConstraint(expression, comparator, 0.0)

        assert constraint.violation(point) == math.inf

    def test_a_missing_or_non_numeric_parameter_is_an_error(self) -> None:
        constraint = NonlinearConstraint("x * y", "<=", 1.0)

        with pytest.raises(ValueError, match="no value"):
            constraint.violation({"x": 1.0})
        with pytest.raises(ValueError, match="needs a number for 'x'"):
            constraint.violation({"x": "1", "y": 1.0})

    @pytest.mark.parametrize("comparator", ["<=", ">="])
    def test_it_agrees_with_the_exact_check_about_what_is_feasible(
        self, comparator: Any
    ) -> None:
        # `isSatisfiedBy` (exact, roadmap P2-09) is the tolerance-free reading of the same
        # constraint: it holds exactly where `violation` is zero.
        generator = random.Random(20261110)
        constraints = [
            NonlinearConstraint("x ** 2 + y", comparator, 1.0),
            NonlinearConstraint("sqrt(x) - y", comparator, 0.5),
            NonlinearConstraint("log(x) * y", comparator, -1.0),
        ]
        verdicts = set()
        for constraint in constraints:
            for _ in range(100):
                point = {"x": generator.uniform(-3, 3), "y": generator.uniform(-3, 3)}
                exact = constraint.isSatisfiedBy(point)
                verdicts.add(exact)
                assert (constraint.violation(point) == 0.0) is exact, (constraint, point)
        assert verdicts == {True, False}  # the sample reaches both sides


class TestTheContract:
    def test_a_type_that_does_not_say_how_to_judge_a_point_cannot_be_built(self) -> None:
        class Lazy(Constraint):
            def toDict(self) -> dict[str, Any]:
                return {"type": "lazy"}

        with pytest.raises(TypeError, match=r"referencedNames|violation"):
            Lazy()  # type: ignore[abstract]
