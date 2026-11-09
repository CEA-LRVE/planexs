"""The pure evaluator, `validateExpression` and `compileExpression` read an expression alike.

A nonlinear constraint is read in two places: by the acquisition layer, which compiles its
expression to tensors, and by `NonlinearConstraint.isSatisfiedBy`, which evaluates it on plain
numbers so that a parameterization can be checked without torch (roadmap task P2-09, FR17).
These tests keep the two from drifting apart: they accept the same function names, compute the
same values at sample points, and judge the same points feasible.

They live here, not under `tests/unit/domain`, because the domain imports no torch.
"""

from __future__ import annotations

import ast
import importlib
import math
import random
from collections.abc import Callable
from typing import Any

import pytest
import torch

from planexs.domain.constraints.evaluateExpression import evaluateExpression
from planexs.domain.constraints.NonlinearConstraint import NonlinearConstraint
from planexs.domain.constraints.validateExpression import (
    ALLOWED_EXPRESSION_FUNCTIONS,
    validateExpression,
)
from planexs.encoding.compileExpression import compileExpression
from planexs.encoding.toBotorchNonlinearConstraints import toBotorchNonlinearConstraints

#: Seed of the sample points, so that the test is the same on every run.
SEED = 20261108

#: Number of random points per expression, drawn uniformly from `[-3, 3]` in every variable.
#: Wide enough to reach both sides of every domain boundary (the logarithm and square root
#: need a positive argument, a power needs a positive base), small enough that nothing
#: overflows.
POINTS = 60
LOWER, UPPER = -3.0, 3.0

#: Tolerance on the values of the two forms. Both compute in double precision, but `math` and
#: torch do not share every implementation (`tan`, `tanh`, `pow`), so the last bits can differ;
#: the relative tolerance is far above that and far below any real disagreement. The absolute
#: one is for results that cancel to nearly zero.
REL_TOLERANCE = 1e-9
ABS_TOLERANCE = 1e-12

#: Expressions that between them use every operator and every allowed function. The flag says
#: whether the expression is undefined somewhere in the sampled box (a negative argument of
#: `sqrt` or `log`, a negative base with a fractional power), so the test can insist that the
#: sample really reaches such points.
EXPRESSIONS = [
    ("x + y * z", False),
    ("x - y / z", False),
    ("-x ** 2 + y", False),
    ("(x + y) * (y - z)", False),
    ("x ** y", True),
    ("sqrt(x) + y", True),
    ("log(x) + log10(y) + log2(z)", True),
    ("exp(x) * sin(y)", False),
    ("cos(x) - tan(y)", False),
    ("tanh(x * y) + abs(z)", False),
    ("min(x, y) + max(y, z)", False),
    ("min(sqrt(x), y)", True),
    ("max(y, log(x))", True),
    ("sqrt(x) * log(y) / z", True),
]

#: Constraints for the feasibility comparison: both comparators, and expressions undefined on
#: part of the box.
CONSTRAINTS = [
    NonlinearConstraint("x ** 2 + y", "<=", 4.0),
    NonlinearConstraint("sqrt(x) - y", ">=", 0.5),
    NonlinearConstraint("log(x) + y", "<=", 1.0),
    NonlinearConstraint("x ** y", ">=", 1.0),
    NonlinearConstraint("min(x, y) * abs(y)", ">=", -1.0),
    NonlinearConstraint("tanh(x) * exp(y)", "<=", 0.5),
]


def samplePoints(count: int, names: list[str]) -> list[dict[str, float]]:
    """Seeded random points of the box, one value per name."""
    generator = random.Random(SEED)
    return [{name: generator.uniform(LOWER, UPPER) for name in names} for _ in range(count)]


def compiledMargin(constraint: NonlinearConstraint, point: dict[str, float]) -> float:
    """The signed margin of the constraint's compiled form at `point`: `>= 0` if it holds."""
    ((function, _),) = toBotorchNonlinearConstraints([constraint], ["x", "y"])
    return float(function(torch.tensor([point["x"], point["y"]], dtype=torch.double)))


def compiledFeasible(constraint: NonlinearConstraint, point: dict[str, float]) -> bool:
    """Whether the compiled form holds at `point`, as `EncodedConstraints` reads it.

    `nan` is not `>= 0`, so a point where the expression is undefined is infeasible, which is
    what `EncodedConstraints.violation` does too (it maps `nan` to an infinite violation).
    """
    return compiledMargin(constraint, point) >= 0.0


class TestFunctionNames:
    def test_the_three_function_tables_hold_the_same_names(self) -> None:
        pure = importlib.import_module("planexs.domain.constraints.evaluateExpression")
        compiled = importlib.import_module("planexs.encoding.compileExpression")

        assert (
            set(ALLOWED_EXPRESSION_FUNCTIONS)
            == set(pure._FUNCTIONS)
            == set(compiled._FUNCTIONS)
        )

    @pytest.mark.parametrize(
        "name",
        [
            *ALLOWED_EXPRESSION_FUNCTIONS,
            # Names that look as if they belonged, and one that must never be callable.
            "floor",
            "ceil",
            "pow",
            "atan",
            "sinh",
            "cosh",
            "gamma",
            "round",
            "float",
            "print",
            "eval",
        ],
    )
    def test_the_three_accept_the_same_names(self, name: str) -> None:
        arities = (
            [ALLOWED_EXPRESSION_FUNCTIONS[name]]
            if name in ALLOWED_EXPRESSION_FUNCTIONS
            else [1, 2]
        )
        for arity in arities:
            expression = f"{name}({', '.join(['x', 'y'][:arity])})"

            accepted = [
                accepts(validateExpression, expression),
                accepts(evaluateExpression, expression, {"x": 0.5, "y": 0.25}),
                accepts(compileExpression, expression, ["x", "y"]),
            ]

            assert accepted == [name in ALLOWED_EXPRESSION_FUNCTIONS] * 3, (
                expression,
                accepted,
            )

    def test_the_sample_expressions_use_every_allowed_function(self) -> None:
        called = {
            node.func.id
            for expression, _ in EXPRESSIONS
            for node in ast.walk(ast.parse(expression, mode="eval"))
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }

        assert called == set(ALLOWED_EXPRESSION_FUNCTIONS)


def accepts(function: Callable[..., object], *arguments: object) -> bool:
    """Whether a call goes through, rather than raising `ValueError` for the expression."""
    try:
        function(*arguments)
    except ValueError:
        return False
    return True


class TestValues:
    @pytest.mark.parametrize(("expression", "can_be_undefined"), EXPRESSIONS)
    def test_the_two_forms_give_the_same_value_at_sample_points(
        self, expression: str, can_be_undefined: bool
    ) -> None:
        points = samplePoints(POINTS, ["x", "y", "z"])
        compiled = compileExpression(expression, ["x", "y", "z"])(
            {
                name: torch.tensor([point[name] for point in points], dtype=torch.double)
                for name in ("x", "y", "z")
            }
        )

        n_finite = n_undefined = 0
        for index, point in enumerate(points):
            pure_value = evaluateExpression(expression, point)
            compiled_value = float(compiled[index])
            # Undefined is undefined in both: a `nan` in one and a number in the other is a
            # disagreement, however the number compares.
            assert math.isnan(pure_value) == math.isnan(compiled_value), (
                expression,
                point,
                pure_value,
                compiled_value,
            )
            if math.isnan(pure_value):
                n_undefined += 1
            else:
                n_finite += 1
                assert pure_value == pytest.approx(
                    compiled_value, rel=REL_TOLERANCE, abs=ABS_TOLERANCE
                ), (expression, point)

        # A sample that never leaves the domain, or never enters it, would agree on nothing.
        assert n_finite > 0, f"{expression}: no point had a value"
        assert (n_undefined > 0) is can_be_undefined, (
            f"{expression}: {n_undefined} undefined point(s) of {POINTS}"
        )

    @pytest.mark.parametrize("name", ALLOWED_EXPRESSION_FUNCTIONS)
    def test_each_function_gives_the_same_value_in_both_forms(self, name: str) -> None:
        arguments = ", ".join(["x", "y"][: ALLOWED_EXPRESSION_FUNCTIONS[name]])
        expression = f"{name}({arguments})"
        point = {"x": 0.5, "y": 0.25}  # inside the domain of every allowed function
        compiled = compileExpression(expression, ["x", "y"])(
            {key: torch.tensor(value, dtype=torch.double) for key, value in point.items()}
        )

        assert evaluateExpression(expression, point) == pytest.approx(
            float(compiled), rel=REL_TOLERANCE, abs=ABS_TOLERANCE
        )


class TestNanValues:
    """A `nan` that is a value, not an undefined step: `min` and `max` must pass it on."""

    @pytest.mark.parametrize(
        "expression", ["min(x, y)", "min(y, x)", "max(x, y)", "max(y, x)", "min(x - y, 1)"]
    )
    def test_the_two_forms_both_give_nan(self, expression: str) -> None:
        point = {"x": math.nan, "y": 1.0}
        if "x - y" in expression:
            point = {"x": math.inf, "y": math.inf}  # inf - inf is nan
        compiled = compileExpression(expression, ["x", "y"])(
            {key: torch.tensor(value, dtype=torch.double) for key, value in point.items()}
        )

        assert math.isnan(float(compiled))
        assert math.isnan(evaluateExpression(expression, point))


class TestFeasibility:
    @pytest.mark.parametrize(
        "constraint", CONSTRAINTS, ids=lambda c: f"{c.expression} {c.comparator} {c.bound}"
    )
    def test_the_two_forms_judge_the_same_points_feasible(
        self, constraint: NonlinearConstraint
    ) -> None:
        verdicts = [
            (constraint.isSatisfiedBy(point), compiledFeasible(constraint, point))
            for point in samplePoints(POINTS, ["x", "y"])
        ]

        assert all(pure == compiled for pure, compiled in verdicts), verdicts
        # Both outcomes occur: a constraint that nothing satisfies, or everything does,
        # would agree on nothing.
        assert any(pure for pure, _ in verdicts)
        assert not all(pure for pure, _ in verdicts)


class TestWhereTheyDiffer:
    """At a pole the compiled form gives an infinity and `isSatisfiedBy` says infeasible.

    The compiled form maps only `nan` to a violation, so an infinity satisfies the comparator
    that it points towards: `1 / 0` is `inf`, which "is `>=` anything". A pole is not a point
    that satisfies a bound, so the pure check refuses it (see `evaluateExpression`). The
    difference is deliberate, and it only ever goes one way.
    """

    @pytest.mark.parametrize(
        ("expression", "comparator", "point"),
        [
            ("1 / x", ">=", {"x": 0.0, "y": 0.0}),
            ("log(x)", "<=", {"x": 0.0, "y": 0.0}),
            ("exp(x)", ">=", {"x": 1000.0, "y": 0.0}),
        ],
    )
    def test_at_a_pole_the_pure_check_is_stricter(
        self, expression: str, comparator: Any, point: dict[str, float]
    ) -> None:
        constraint = NonlinearConstraint(expression, comparator, 0.0)

        assert compiledMargin(constraint, point) == math.inf  # the compiled form is satisfied
        assert constraint.isSatisfiedBy(point) is False  # the pure check is not

    def test_the_pure_check_is_never_more_permissive_than_the_compiled_one(self) -> None:
        # Special values: both signs, zero, the unit, and the edges of the float range, so
        # that poles, overflows and exact boundary hits are all in the grid.
        special = [-1e308, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 1e308]
        poles = [
            NonlinearConstraint("1 / x", ">=", 0.0),
            NonlinearConstraint("log(x)", "<=", 1.0),
            NonlinearConstraint("exp(y) * x", "<=", 1.0),
        ]
        checked = permitted = 0
        for constraint in [*CONSTRAINTS, *poles]:
            for x in special:
                for y in special:
                    point = {"x": x, "y": y}
                    checked += 1
                    if constraint.isSatisfiedBy(point):
                        permitted += 1
                        assert compiledMargin(constraint, point) >= 0.0, (constraint, point)

        assert permitted > 0, f"the pure check accepted none of {checked} points"
