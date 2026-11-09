"""Unit tests for `evaluateExpression`, the pure counterpart of `compileExpression` (FR17)."""

from __future__ import annotations

import ast
import builtins
import importlib
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from planexs.domain.constraints.evaluateExpression import evaluateExpression
from planexs.domain.constraints.validateExpression import ALLOWED_EXPRESSION_FUNCTIONS

SRC = Path(__file__).resolve().parents[4] / "src"
EVALUATOR_FILE = SRC / "planexs" / "domain" / "constraints" / "evaluateExpression.py"

#: The libraries that `core` and `domain` must never import (specification 5.1).
MACHINE_LEARNING_LIBRARIES = {"torch", "botorch", "gpytorch", "ax"}


class TestValues:
    @pytest.mark.parametrize(
        ("expression", "variables", "expected"),
        [
            ("x + y", {"x": 1.5, "y": 2.0}, 3.5),
            ("x - y", {"x": 1.5, "y": 2.0}, -0.5),
            ("x * y", {"x": 1.5, "y": 2.0}, 3.0),
            ("x / y", {"x": 1.0, "y": 4.0}, 0.25),
            ("x ** y", {"x": 2.0, "y": 10.0}, 1024.0),
            ("2 + 3 * x", {"x": 4.0}, 14.0),
            ("(2 + 3) * x", {"x": 4.0}, 20.0),
            ("-x ** 2", {"x": 3.0}, -9.0),
            ("x - -y", {"x": 1.0, "y": 2.0}, 3.0),
            ("+x", {"x": 7.0}, 7.0),
            ("x / 2", {"x": 1}, 0.5),
            ("var * x ** z", {"var": 2.0, "x": 3.0, "z": 2.0}, 18.0),
        ],
    )
    def test_arithmetic(
        self, expression: str, variables: dict[str, float], expected: float
    ) -> None:
        assert evaluateExpression(expression, variables) == pytest.approx(expected)

    def test_power_is_right_associative(self) -> None:
        assert evaluateExpression("x ** 3 ** 2", {"x": 2.0}) == pytest.approx(512.0)

    @pytest.mark.parametrize(
        ("expression", "variables", "expected"),
        [
            ("abs(x)", {"x": -3.0}, 3.0),
            ("sqrt(x)", {"x": 9.0}, 3.0),
            ("exp(x)", {"x": 1.0}, math.e),
            ("log(x)", {"x": math.e}, 1.0),
            ("log10(x)", {"x": 1000.0}, 3.0),
            ("log2(x)", {"x": 8.0}, 3.0),
            ("sin(x)", {"x": math.pi / 2}, 1.0),
            ("cos(x)", {"x": 0.0}, 1.0),
            ("tan(x)", {"x": math.pi / 4}, 1.0),
            ("tanh(x)", {"x": 0.0}, 0.0),
            ("min(x, y)", {"x": 2.0, "y": -1.0}, -1.0),
            ("max(x, y)", {"x": 2.0, "y": -1.0}, 2.0),
        ],
    )
    def test_every_function(
        self, expression: str, variables: dict[str, float], expected: float
    ) -> None:
        assert evaluateExpression(expression, variables) == pytest.approx(expected)

    def test_every_allowed_function_has_a_case_here(self) -> None:
        covered = {"abs", "sqrt", "exp", "log", "log10", "log2"}
        covered |= {"sin", "cos", "tan", "tanh", "min", "max"}
        assert covered == set(ALLOWED_EXPRESSION_FUNCTIONS)

    def test_the_result_is_a_float_for_integer_values(self) -> None:
        result = evaluateExpression("x + y", {"x": 1, "y": 2})

        assert result == 3.0
        assert isinstance(result, float)

    def test_names_the_expression_does_not_use_are_ignored(self) -> None:
        assert evaluateExpression("x + 1", {"x": 1, "solvent": "water", "flag": True}) == 2.0

    def test_a_name_may_share_its_name_with_an_allowed_function(self) -> None:
        assert evaluateExpression("max * 2", {"max": 4.0}) == 8.0

    def test_a_negative_base_with_an_integer_exponent_has_a_value(self) -> None:
        assert evaluateExpression("x ** y", {"x": -2.0, "y": 3.0}) == pytest.approx(-8.0)

    def test_zero_to_the_zero_is_one(self) -> None:
        assert evaluateExpression("x ** y", {"x": 0.0, "y": 0.0}) == 1.0


class TestUndefinedPoints:
    """Where the expression has no finite value the result is `nan`, never an error."""

    @pytest.mark.parametrize(
        ("expression", "variables"),
        [
            ("sqrt(x)", {"x": -1.0}),
            ("log(x)", {"x": -1.0}),
            ("log(x)", {"x": 0.0}),
            ("log10(x)", {"x": 0.0}),
            ("log2(x)", {"x": -4.0}),
            ("x ** y", {"x": -8.0, "y": 1 / 3}),
            ("x ** 0.5", {"x": -4.0}),
            ("x ** y", {"x": 0.0, "y": -1.0}),
            ("1 / x", {"x": 0.0}),
            ("x / y", {"x": 0.0, "y": 0.0}),
            ("exp(x)", {"x": 1000.0}),
            ("x ** y", {"x": 10.0, "y": 400.0}),
            ("x * y", {"x": 1e308, "y": 10.0}),
            ("x - y", {"x": 1e308, "y": -1e308}),
            ("tan(x)", {"x": math.inf}),
            ("x + 1", {"x": math.nan}),
            ("x + 1e999", {"x": 1.0}),
            ("x + " + "9" * 400, {"x": 1.0}),
        ],
    )
    def test_is_nan(self, expression: str, variables: dict[str, float]) -> None:
        assert math.isnan(evaluateExpression(expression, variables))

    def test_a_complex_number_never_comes_back(self) -> None:
        # Python's own `**` gives a complex number here: the evaluator must not.
        assert isinstance((-8.0) ** (1 / 3), complex)
        assert math.isnan(evaluateExpression("x ** (1 / 3)", {"x": -8.0}))

    @pytest.mark.parametrize("function", ["min", "max"])
    @pytest.mark.parametrize("order", ["{f}(x, 1)", "{f}(1, x)"])
    def test_min_and_max_propagate_a_nan_value_whichever_side_it_is_on(
        self, function: str, order: str
    ) -> None:
        # A nan *value* (not an undefined step, which raises before `min` is reached): Python's
        # own `min(nan, 1.0)` is nan but `min(1.0, nan)` is 1.0, and the same for `max`.
        assert math.isnan(evaluateExpression(order.format(f=function), {"x": math.nan}))

    @pytest.mark.parametrize("function", ["min", "max"])
    def test_min_and_max_propagate_a_nan_that_arithmetic_produces(self, function: str) -> None:
        # inf - inf is nan without any step raising.
        expression = f"{function}(x - y, 1)"

        assert math.isnan(evaluateExpression(expression, {"x": math.inf, "y": math.inf}))

    @pytest.mark.parametrize("expression", ["min(sqrt(x), 1)", "max(1, log(x))"])
    def test_an_undefined_step_inside_min_or_max_is_undefined(self, expression: str) -> None:
        assert math.isnan(evaluateExpression(expression, {"x": -1.0}))

    def test_an_infinity_in_the_middle_is_not_undefined_if_the_result_is_finite(self) -> None:
        # Same as the compiled form: min(inf, 5) is 5. The overflow is only a problem if it
        # reaches the result.
        assert evaluateExpression("min(x * y, 5)", {"x": 1e308, "y": 10.0}) == 5.0


class TestCallerErrors:
    """A mistake of the call is an error, and is never mistaken for an undefined point."""

    def test_a_missing_name_is_reported_with_its_name(self) -> None:
        with pytest.raises(ValueError, match=r"\['y'\].*no value"):
            evaluateExpression("x + y", {"x": 1.0})

    def test_several_missing_names_are_all_reported(self) -> None:
        with pytest.raises(ValueError, match=r"\['y', 'z'\].*have no value"):
            evaluateExpression("x + y * z", {"x": 1.0})

    @pytest.mark.parametrize("value", ["2", "abc", None, [1.0], True, False])
    def test_a_value_that_is_not_a_number_is_rejected(self, value: object) -> None:
        # "abc" and "2" are strings that `float()` would read or reject on its own: neither
        # may turn into a number, or into a `nan`.
        with pytest.raises(ValueError, match=r"needs a number for 'x'"):
            evaluateExpression("x + 1", {"x": value})

    def test_an_integer_too_large_for_a_float_is_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"too large for a float"):
            evaluateExpression("x + 1", {"x": 10**400})

    @pytest.mark.parametrize(
        "expression",
        [
            "",
            "   ",
            "1 + 2",
            "__import__('os').system('true')",
            "x.real",
            "x if y else z",
            "lambda: x",
            "x // 2",
            "x % 2",
            "foo(x)",
            "abs(x, y)",
            "x[0]",
            "'a' + x",
            "True + x",
        ],
    )
    def test_an_invalid_expression_is_rejected_before_it_is_evaluated(
        self, expression: str
    ) -> None:
        with pytest.raises(ValueError):
            evaluateExpression(expression, {"x": 1.0, "y": 1.0, "z": 1.0})

    def test_python_eval_and_exec_are_never_called(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def forbidden(*arguments: Any, **keywords: Any) -> Any:
            raise AssertionError("eval or exec was called")

        monkeypatch.setattr(builtins, "eval", forbidden)
        monkeypatch.setattr(builtins, "exec", forbidden)

        assert evaluateExpression("x * 2 + sqrt(y)", {"x": 1.0, "y": 4.0}) == 4.0


class TestGrammarDrift:
    """If `validateExpression` allows a node that this module does not know, it must fail."""

    def test_an_unknown_node_is_an_error_not_a_nan(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        module = importlib.import_module("planexs.domain.constraints.evaluateExpression")
        # Stand in for a grammar change that was not mirrored here: the validator lets a
        # list through. The evaluator must refuse it loudly, not call it an undefined point.
        monkeypatch.setattr(module, "validateExpression", lambda expression: frozenset())

        with pytest.raises(ValueError, match=r"Unsupported expression node List"):
            evaluateExpression("[x]", {})


class TestFunctionTable:
    def test_it_holds_the_names_of_the_domain_allowlist(self) -> None:
        # The three-way comparison with the compiled form is in
        # tests/integration/test_expression_evaluators_agree.py.
        module = importlib.import_module("planexs.domain.constraints.evaluateExpression")
        assert set(module._FUNCTIONS) == set(ALLOWED_EXPRESSION_FUNCTIONS)


class TestNoMachineLearningLibrary:
    """The evaluator exists so that a constraint can be checked without torch (3.9)."""

    def test_the_file_imports_only_the_standard_library_and_the_domain(self) -> None:
        tree = ast.parse(EVALUATOR_FILE.read_text(encoding="utf-8"))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported |= {alias.name for alias in node.names}
            elif isinstance(node, ast.ImportFrom):
                assert node.level == 0, "a relative import"
                imported.add(node.module or "")

        assert imported, "the scan found no import at all"
        for module in imported:
            root = module.split(".")[0]
            assert root not in MACHINE_LEARNING_LIBRARIES, module
            assert root in sys.stdlib_module_names or module.startswith("planexs.domain."), (
                module
            )

    def test_it_runs_in_a_process_where_the_libraries_cannot_be_imported(self) -> None:
        # A fresh interpreter that refuses to import torch, botorch, gpytorch and ax. The
        # `__init__` files of the planexs packages are not run: the top-level one imports
        # every layer eagerly (roadmap task P2-11), which would fail for a reason that has
        # nothing to do with this module. They hold only docstrings and re-exports.
        script = """
import importlib.abc, os, sys, types

blocked = set(sys.argv[2].split(","))

class Refuse(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in blocked:
            raise ImportError("refused by the test: " + name)

sys.meta_path.insert(0, Refuse())
source = sys.argv[1]
for name in ("planexs", "planexs.core", "planexs.domain", "planexs.domain.constraints"):
    package = types.ModuleType(name)
    package.__path__ = [os.path.join(source, *name.split("."))]
    sys.modules[name] = package

from planexs.domain.constraints.NonlinearConstraint import NonlinearConstraint

constraint = NonlinearConstraint("var * x ** z", "<=", 50.0)
assert constraint.isSatisfiedBy({"var": 2.0, "x": 3.0, "z": 2.0}) is True
assert constraint.isSatisfiedBy({"var": 10.0, "x": 3.0, "z": 2.0}) is False
assert NonlinearConstraint("sqrt(x)", ">=", 0.0).isSatisfiedBy({"x": -1.0}) is False
loaded = sorted(m for m in sys.modules if m.split(".")[0] in blocked)
assert not loaded, loaded
print("ok")
"""

        blocked = ",".join(sorted(MACHINE_LEARNING_LIBRARIES))
        result = subprocess.run(
            [sys.executable, "-c", script, str(SRC), blocked],
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "ok"
