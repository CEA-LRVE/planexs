"""Unit tests for `compileExpression`."""

from __future__ import annotations

import importlib
import math

import pytest
import torch

from planexs.domain.constraints.validateExpression import ALLOWED_EXPRESSION_FUNCTIONS
from planexs.models.compileExpression import compileExpression


def _evaluate(expression: str, **values: float) -> float:
    function = compileExpression(expression, list(values))
    tensors = {name: torch.tensor(value, dtype=torch.double) for name, value in values.items()}
    return float(function(tensors))


class TestEvaluation:
    def test_operator_precedence_and_unary_minus(self) -> None:
        assert _evaluate("2 + 3 * x ** 2", x=2.0) == 14.0
        assert _evaluate("-x ** 2", x=3.0) == -9.0
        assert _evaluate("(2 + 3) * x", x=2.0) == 10.0
        assert _evaluate("+x - -x", x=2.0) == 4.0

    def test_power_is_right_associative(self) -> None:
        assert _evaluate("2 ** 3 ** 2 + 0 * x", x=1.0) == 512.0

    def test_spec_example_var_x_pow_z(self) -> None:
        assert _evaluate("var * x ** z", var=2.0, x=3.0, z=2.0) == 18.0

    @pytest.mark.parametrize(
        ("expression", "expected"),
        [
            ("abs(x)", 3.0),
            ("sqrt(x * x)", 3.0),
            ("exp(0 * x)", 1.0),
            ("log(exp(x))", -3.0),
            ("log10(1000 + 0 * x)", 3.0),
            ("log2(8 + 0 * x)", 3.0),
            ("sin(0 * x)", 0.0),
            ("cos(0 * x)", 1.0),
            ("tan(0 * x)", 0.0),
            ("tanh(0 * x)", 0.0),
            ("min(x, 2)", -3.0),
            ("max(x, 2)", 2.0),
        ],
    )
    def test_functions(self, expression: str, expected: float) -> None:
        assert _evaluate(expression, x=-3.0) == pytest.approx(expected)

    def test_is_elementwise_over_batches(self) -> None:
        function = compileExpression("x * y + 1", ["x", "y"])
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.double)
        y = torch.tensor([[10.0, 10.0], [10.0, 10.0]], dtype=torch.double)

        result = function({"x": x, "y": y})

        assert torch.equal(
            result, torch.tensor([[11.0, 21.0], [31.0, 41.0]], dtype=torch.double)
        )

    def test_is_differentiable(self) -> None:
        function = compileExpression("x ** 2 * y", ["x", "y"])
        x = torch.tensor(3.0, dtype=torch.double, requires_grad=True)
        y = torch.tensor(2.0, dtype=torch.double, requires_grad=True)

        function({"x": x, "y": y}).backward()

        assert float(x.grad) == pytest.approx(12.0)  # d/dx x^2 y = 2xy
        assert float(y.grad) == pytest.approx(9.0)  # d/dy x^2 y = x^2

    def test_undefined_points_are_nan_not_errors(self) -> None:
        assert math.isnan(_evaluate("sqrt(x)", x=-1.0))
        assert math.isnan(_evaluate("x ** 0.5", x=-1.0))

    def test_preserves_dtype(self) -> None:
        function = compileExpression("x * 2.5", ["x"])
        assert function({"x": torch.tensor(1.0, dtype=torch.float32)}).dtype == torch.float32


class TestErrors:
    def test_unknown_variable_is_reported_at_compile_time(self) -> None:
        with pytest.raises(ValueError, match=r"\['w'\].*not among the available"):
            compileExpression("x * w", ["x", "y"])

    def test_invalid_expression_is_rejected_at_compile_time(self) -> None:
        with pytest.raises(ValueError):
            compileExpression("__import__('os')", ["x"])


def test_function_table_matches_the_domain_allowlist() -> None:
    module = importlib.import_module("planexs.models.compileExpression")
    assert set(module._FUNCTIONS) == set(ALLOWED_EXPRESSION_FUNCTIONS)
