"""Unit tests for `NonlinearConstraint` and its expression validator (FR17)."""

from __future__ import annotations

import pytest

from planexs import LinearConstraint, NonlinearConstraint, Real, SearchSpace
from planexs.domain.constraints.constraintFromDict import constraintFromDict
from planexs.domain.constraints.validateExpression import (
    ALLOWED_EXPRESSION_FUNCTIONS,
    MAX_EXPRESSION_LENGTH,
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
            NonlinearConstraint("x * y", "=", 1.0)  # type: ignore[arg-type]


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
            validateExpression("x + " * MAX_EXPRESSION_LENGTH + "x")

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
        assert NonlinearConstraint("x * y", ">=", 2.0).toDict()["kind"] == "nonlinear"

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

    def test_missing_kind_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="kind=None"):
            constraintFromDict({"expression": "x * y", "comparator": "<=", "bound": 1.0})

    def test_unknown_kind_message_lists_both_supported_kinds(self) -> None:
        with pytest.raises(ValueError, match=r"linear.*nonlinear"):
            constraintFromDict({"kind": "quadratic"})
