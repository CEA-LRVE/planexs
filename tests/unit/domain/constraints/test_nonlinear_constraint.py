"""Unit tests for `NonlinearConstraint` and its expression validator (FR17)."""

from __future__ import annotations

import json
import math
from typing import Any

import pytest

from planexs import LinearConstraint, NonlinearConstraint, Real, SearchSpace
from planexs.core.ConfigError import ConfigError
from planexs.domain.constraints.constraintFromDict import constraintFromDict
from planexs.domain.constraints.ExpressionLimitsConfig import ExpressionLimitsConfig
from planexs.domain.constraints.validateExpression import (
    ALLOWED_EXPRESSION_FUNCTIONS,
    validateExpression,
)


class TestConstruction:
    def test_accepts_the_spec_example(self) -> None:
        constraint = NonlinearConstraint("var * x ** z", "<=", 50.0)

        assert constraint.expression == "var * x ** z"
        assert constraint.comparator == "<="
        assert constraint.bound == 50.0
        assert constraint.referencedNames() == frozenset({"var", "x", "z"})

    def test_accepts_every_allowed_function(self) -> None:
        for name, arity in ALLOWED_EXPRESSION_FUNCTIONS.items():
            arguments = ", ".join(["x"] * arity)
            assert validateExpression(f"{name}({arguments}) + 1") == {"x"}

    def test_accepts_unary_minus_parentheses_and_float_literals(self) -> None:
        assert validateExpression("-(x + 2.5e-1) / (y - .5)") == {"x", "y"}

    def test_is_frozen(self) -> None:
        constraint = NonlinearConstraint("x * y", ">=", 1.0)
        with pytest.raises(ValueError, match="frozen"):
            constraint.bound = 2.0  # type: ignore[misc]

    @pytest.mark.parametrize("bound", [float("inf"), float("-inf"), float("nan")])
    def test_rejects_non_finite_bound(self, bound: float) -> None:
        with pytest.raises(ValueError, match="finite"):
            NonlinearConstraint("x * y", "<=", bound)

    def test_rejects_equality_comparator(self) -> None:
        with pytest.raises(ValueError):
            NonlinearConstraint("x * y", "=", 1.0)


class TestExpressionGrammar:
    @pytest.mark.parametrize(
        "expression",
        [
            "__import__('os').system('true')",
            "open('/etc/passwd')",
            "eval('1')",
            "x.__class__",
            "x.real",
            "x[0]",
            "[x, 1]",
            "(x, 1)",
            "lambda: x",
            "x if y else 2",
            "x < 2",
            "x and y",
            "not x",
            "'text'",
            "True + x",
            "None",
            "1j * x",
            "x // 2",
            "x % 2",
            "x ^ 2",
            "x @ y",
            "-x if x else y",
            "sqrt(x, y)",
            "sqrt()",
            "max(x)",
            "sqrt(x=1)",
            "f(x)",
            "x.sqrt(2)",
            "(lambda z: z)(x)",
            "*x",
            "sqrt(*x)",
        ],
    )
    def test_rejects_anything_outside_the_grammar(self, expression: str) -> None:
        with pytest.raises(ValueError):
            validateExpression(expression)

    def test_error_names_the_offending_operator_and_hints_at_power(self) -> None:
        with pytest.raises(ValueError, match=r"\*\*"):
            validateExpression("x ^ 2")

    @pytest.mark.parametrize("expression", ["", "   ", "\n"])
    def test_rejects_empty(self, expression: str) -> None:
        with pytest.raises(ValueError, match="empty"):
            validateExpression(expression)

    def test_rejects_a_constant_expression(self) -> None:
        with pytest.raises(ValueError, match="no parameter"):
            validateExpression("2 * 3 + sqrt(4)")

    def test_rejects_syntax_errors(self) -> None:
        with pytest.raises(ValueError, match="Invalid"):
            validateExpression("x +* 2")

    def test_rejects_overlong_expressions(self) -> None:
        with pytest.raises(ValueError, match="limited"):
            validateExpression("x + " * ExpressionLimitsConfig().max_length + "x")

    def test_redundant_parentheses_are_not_nesting(self) -> None:
        assert validateExpression("(" * 200 + "x" + ")" * 200) == {"x"}

    def test_accepts_a_long_sum(self) -> None:
        assert len(validateExpression(" + ".join(f"p{i}" for i in range(100)))) == 100

    def test_rejects_excessive_nesting(self) -> None:
        with pytest.raises(ValueError, match="deeply"):
            validateExpression("-" * 250 + "x")
        with pytest.raises(ValueError, match="deeply"):
            validateExpression("+".join(["x"] * 400))

    def test_does_not_evaluate_the_expression(self) -> None:
        # A huge power would hang or exhaust memory if it were ever evaluated.
        assert validateExpression("x ** 9 ** 9 ** 9") == {"x"}

    def test_a_parameter_may_share_a_name_with_an_allowed_function(self) -> None:
        assert validateExpression("max + sqrt(min)") == {"max", "min"}


class TestSerialization:
    def test_round_trips_through_dict(self) -> None:
        constraint = NonlinearConstraint("var * x ** z", "<=", 50.0)

        rebuilt = constraintFromDict(constraint.toDict())

        assert isinstance(rebuilt, NonlinearConstraint)
        assert rebuilt == constraint

    def test_dict_is_type_tagged(self) -> None:
        assert NonlinearConstraint("x * y", ">=", 2.0).toDict()["type"] == "nonlinear"

    def test_search_space_round_trips_mixed_constraints_through_json(self) -> None:
        space = SearchSpace(
            [Real("x", 0.1, 5.0), Real("y", 0.1, 5.0)],
            constraints=[
                LinearConstraint({"x": 1.0, "y": 1.0}, 8.0, "<="),
                NonlinearConstraint("x * y ** 2", "<=", 20.0),
            ],
        )

        rebuilt = SearchSpace.model_validate_json(space.model_dump_json())

        assert [type(c) for c in rebuilt.constraints] == [
            LinearConstraint,
            NonlinearConstraint,
        ]
        assert rebuilt.constraints == space.constraints

    def test_missing_type_is_rejected(self) -> None:
        with pytest.raises(ConfigError, match="no 'type' field"):
            constraintFromDict({"expression": "x * y", "comparator": "<=", "bound": 1.0})

    def test_unknown_type_message_lists_both_supported_types(self) -> None:
        with pytest.raises(ConfigError, match=r"linear.*nonlinear"):
            constraintFromDict({"type": "quadratic"})


class TestIsSatisfiedBy:
    """The feasibility check on natural values, with no machine-learning library (P2-09)."""

    def test_the_spec_example(self) -> None:
        constraint = NonlinearConstraint("var * x ** z", "<=", 50.0)

        assert constraint.isSatisfiedBy({"var": 2.0, "x": 3.0, "z": 2.0})  # 18 <= 50
        assert not constraint.isSatisfiedBy({"var": 10.0, "x": 3.0, "z": 2.0})  # 90 > 50

    @pytest.mark.parametrize(
        ("comparator", "x", "expected"),
        [
            ("<=", 1.9, True),
            ("<=", 2.0, True),  # on the bound: satisfied
            ("<=", 2.1, False),
            (">=", 1.9, False),
            (">=", 2.0, True),  # on the bound: satisfied
            (">=", 2.1, True),
        ],
    )
    def test_each_comparator_inside_on_and_outside_the_bound(
        self, comparator: Any, x: float, expected: bool
    ) -> None:
        constraint = NonlinearConstraint("x ** 2", comparator, 4.0)

        assert constraint.isSatisfiedBy({"x": x}) is expected

    @pytest.mark.parametrize(
        ("comparator", "x"),
        [("<=", math.nextafter(2.0, 3.0)), (">=", math.nextafter(2.0, 1.0))],
    )
    def test_the_comparison_is_exact_there_is_no_tolerance(
        self, comparator: Any, x: float
    ) -> None:
        # One representable number past the bound is already infeasible.
        constraint = NonlinearConstraint("x", comparator, 2.0)
        assert constraint.isSatisfiedBy({"x": 2.0})

        assert not constraint.isSatisfiedBy({"x": x})

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
    def test_an_undefined_point_is_infeasible_for_both_comparators(
        self, comparator: Any, expression: str, point: dict[str, float]
    ) -> None:
        constraint = NonlinearConstraint(expression, comparator, 0.0)

        assert constraint.isSatisfiedBy(point) is False

    def test_a_whole_parameterization_can_be_passed(self) -> None:
        constraint = NonlinearConstraint("x ** 2 + y", "<=", 4.0)
        parameterization = {"x": 2, "y": 0, "solvent": "water", "dry": True}

        assert constraint.isSatisfiedBy(parameterization)

    def test_integer_values_are_numbers(self) -> None:
        assert NonlinearConstraint("x * y", ">=", 6.0).isSatisfiedBy({"x": 2, "y": 3})

    def test_a_missing_parameter_is_an_error_not_an_infeasible_point(self) -> None:
        constraint = NonlinearConstraint("x * y", "<=", 1.0)

        with pytest.raises(ValueError, match=r"\['y'\].*no value"):
            constraint.isSatisfiedBy({"x": 1.0})

    @pytest.mark.parametrize("value", ["2.0", True])
    def test_a_parameter_that_is_not_a_number_is_an_error(self, value: object) -> None:
        constraint = NonlinearConstraint("x * y", "<=", 1.0)

        with pytest.raises(ValueError, match=r"needs a number for 'x'"):
            constraint.isSatisfiedBy({"x": value, "y": 1.0})

    def test_a_rebuilt_constraint_checks_the_same(self) -> None:
        constraint = NonlinearConstraint("x ** 2 + y", "<=", 4.0)
        rebuilt = constraintFromDict(constraint.toDict())

        assert isinstance(rebuilt, NonlinearConstraint)
        for point in ({"x": 1.0, "y": 1.0}, {"x": 2.0, "y": 1.0}, {"x": -3.0, "y": 0.0}):
            assert rebuilt.isSatisfiedBy(point) == constraint.isSatisfiedBy(point)

    def test_it_does_not_change_the_constraint(self) -> None:
        constraint = NonlinearConstraint("x ** 2", "<=", 4.0)
        before = constraint.toDict()

        constraint.isSatisfiedBy({"x": 1.0})

        assert constraint.toDict() == before


# Expressions that the default limits (1000 characters, 200 levels) refuse, each with limits
# that allow it. The first is long and shallow (a single name), the second deep and short (a
# chain of unary minus signs, which an even count leaves as `x`), the third both (a sum).
LONG_NAME = "v" * 1500
DEEP_CHAIN = "-" * 250 + "x"
LONG_SUM = " + ".join(f"p{i}" for i in range(250))
BEYOND_THE_DEFAULTS = [
    pytest.param(LONG_NAME, ExpressionLimitsConfig(max_length=1500), id="too-long"),
    pytest.param(DEEP_CHAIN, ExpressionLimitsConfig(max_depth=250), id="too-deep"),
    pytest.param(
        LONG_SUM,
        ExpressionLimitsConfig(max_length=2000, max_depth=300),
        id="too-long-and-deep",
    ),
]


class TestExpressionLimits:
    """The limits an expression is accepted under belong to the constraint (roadmap P3-06)."""

    @pytest.mark.parametrize(("expression", "limits"), BEYOND_THE_DEFAULTS)
    def test_an_expression_the_defaults_refuse_is_accepted_under_larger_limits(
        self, expression: str, limits: ExpressionLimitsConfig
    ) -> None:
        with pytest.raises(ValueError):
            NonlinearConstraint(expression, "<=", 5.0)

        constraint = NonlinearConstraint(expression, "<=", 5.0, limits=limits)

        assert constraint.limits == limits
        assert constraint.referencedNames() == validateExpression(expression, limits)

    @pytest.mark.parametrize(("expression", "limits"), BEYOND_THE_DEFAULTS)
    def test_it_survives_a_round_trip_through_json(
        self, expression: str, limits: ExpressionLimitsConfig
    ) -> None:
        constraint = NonlinearConstraint(expression, "<=", 5.0, limits=limits)

        stored = json.loads(json.dumps(constraint.toDict()))
        rebuilt = constraintFromDict(stored)

        assert isinstance(rebuilt, NonlinearConstraint)
        assert rebuilt == constraint
        assert rebuilt.limits == limits
        assert rebuilt.toDict() == constraint.toDict()

    def test_the_limits_are_written_in_full_even_when_only_one_differs(self) -> None:
        # In full, so that a later change of a default cannot alter how this one reads back.
        constraint = NonlinearConstraint(
            LONG_NAME, "<=", 5.0, limits=ExpressionLimitsConfig(max_length=1500)
        )

        assert constraint.toDict()["limits"] == {"max_length": 1500, "max_depth": 200}

    def test_the_default_round_trip_is_unchanged(self) -> None:
        constraint = NonlinearConstraint("var * x ** z", "<=", 50.0)

        stored = constraint.toDict()

        assert stored == {
            "type": "nonlinear",
            "expression": "var * x ** z",
            "comparator": "<=",
            "bound": 50.0,
        }
        rebuilt = constraintFromDict(json.loads(json.dumps(stored)))
        assert rebuilt == constraint
        assert rebuilt.limits == ExpressionLimitsConfig()

    def test_limits_that_are_the_defaults_are_the_same_constraint_and_are_not_written(
        self,
    ) -> None:
        plain = NonlinearConstraint("x * y", "<=", 1.0)
        explicit = NonlinearConstraint("x * y", "<=", 1.0, limits=ExpressionLimitsConfig())
        written = NonlinearConstraint(
            "x * y", "<=", 1.0, limits=ExpressionLimitsConfig(max_length=1000, max_depth=200)
        )

        assert plain == explicit == written
        assert "limits" not in explicit.toDict()
        assert "limits" not in written.toDict()

    def test_the_limits_of_a_stored_form_may_be_given_in_part(self) -> None:
        stored = {
            "type": "nonlinear",
            "expression": LONG_NAME,
            "comparator": "<=",
            "bound": 5.0,
            "limits": {"max_length": 1500},
        }

        rebuilt = constraintFromDict(stored)

        assert isinstance(rebuilt, NonlinearConstraint)
        assert rebuilt.limits == ExpressionLimitsConfig(max_length=1500, max_depth=200)

    def test_a_stored_expression_beyond_its_own_limits_is_refused(self) -> None:
        stored = {
            "type": "nonlinear",
            "expression": LONG_NAME,
            "comparator": "<=",
            "bound": 5.0,
            "limits": {"max_length": 100},
        }

        with pytest.raises(ValueError, match="limited to 100 characters"):
            constraintFromDict(stored)

    @pytest.mark.parametrize(
        ("limits", "path"),
        [
            ({"max_length": 0}, "limits.max_length"),
            ({"max_depth": -1}, "limits.max_depth"),
            ({"max_widht": 5}, "limits.max_widht"),
        ],
    )
    def test_a_bad_stored_limit_is_refused_with_its_path(
        self, limits: dict[str, int], path: str
    ) -> None:
        stored = {
            "type": "nonlinear",
            "expression": "x * y",
            "comparator": "<=",
            "bound": 1.0,
            "limits": limits,
        }

        with pytest.raises(ConfigError) as caught:
            constraintFromDict(stored)

        assert caught.value.field_path == path

    def test_tighter_limits_refuse_an_expression_the_defaults_accept(self) -> None:
        with pytest.raises(ValueError, match="deeply"):
            NonlinearConstraint(
                "x + y + z", "<=", 1.0, limits=ExpressionLimitsConfig(max_depth=1)
            )

    def test_a_search_space_keeps_the_limits_of_each_constraint_through_json(self) -> None:
        space = SearchSpace(
            [Real("x", 0.1, 5.0), Real("y", 0.1, 5.0)],
            constraints=[
                NonlinearConstraint("x * y ** 2", "<=", 20.0),
                NonlinearConstraint(
                    DEEP_CHAIN, "<=", 3.0, limits=ExpressionLimitsConfig(max_depth=250)
                ),
            ],
        )

        document = space.model_dump_json()
        rebuilt = SearchSpace.model_validate_json(document)

        stored = json.loads(document)["constraints"]
        assert "limits" not in stored[0]
        assert stored[1]["limits"] == {"max_length": 1000, "max_depth": 250}
        assert rebuilt.constraints == space.constraints
        assert [
            c.limits for c in rebuilt.constraints if isinstance(c, NonlinearConstraint)
        ] == [
            ExpressionLimitsConfig(),
            ExpressionLimitsConfig(max_depth=250),
        ]


class TestEveryReaderOfTheExpressionUsesTheLimitsOfTheConstraint:
    """A constraint that was accepted must also be checkable: if any method forgot to pass its
    limits on, it would refuse the very expression the constructor accepted.
    """

    DEEP = NonlinearConstraint(
        DEEP_CHAIN, "<=", 5.0, limits=ExpressionLimitsConfig(max_depth=250)
    )
    LONG = NonlinearConstraint(
        "v" * 1500, "<=", 5.0, limits=ExpressionLimitsConfig(max_length=1500)
    )

    @pytest.mark.parametrize("constraint", [DEEP, LONG], ids=["deep", "long"])
    def test_is_satisfied_by_judges_a_point(self, constraint: NonlinearConstraint) -> None:
        name = next(iter(constraint.referencedNames()))

        assert constraint.isSatisfiedBy({name: 2.0}) is True
        assert constraint.isSatisfiedBy({name: 6.0}) is False

    @pytest.mark.parametrize("constraint", [DEEP, LONG], ids=["deep", "long"])
    def test_violation_measures_a_point(self, constraint: NonlinearConstraint) -> None:
        name = next(iter(constraint.referencedNames()))

        assert constraint.violation({name: 2.0}) == 0.0
        assert constraint.violation({name: 6.0}) == pytest.approx(1.0 / 5.0)

    @pytest.mark.parametrize("constraint", [DEEP, LONG], ids=["deep", "long"])
    def test_a_rebuilt_constraint_judges_the_same_points(
        self, constraint: NonlinearConstraint
    ) -> None:
        rebuilt = constraintFromDict(json.loads(json.dumps(constraint.toDict())))
        name = next(iter(constraint.referencedNames()))

        assert isinstance(rebuilt, NonlinearConstraint)
        for value in (2.0, 5.0, 6.0):
            assert rebuilt.isSatisfiedBy({name: value}) == constraint.isSatisfiedBy(
                {name: value}
            )
            assert rebuilt.violation({name: value}) == constraint.violation({name: value})
