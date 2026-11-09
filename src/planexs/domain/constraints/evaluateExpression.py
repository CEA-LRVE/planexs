"""Evaluates a validated expression string on plain numbers, without torch."""

from __future__ import annotations

import ast
import math
from collections.abc import Callable, Mapping
from typing import cast

from planexs.domain.constraints.validateExpression import validateExpression


class _UndefinedError(Exception):
    """A step of the evaluation has no value: a domain error, a pole or an overflow.

    An internal signal between the helpers and `evaluateExpression`, which turns it into `nan`.
    It never reaches a caller.
    """


def _minimum(left: float, right: float) -> float:
    """The smaller of two numbers, `nan` if either is `nan`.

    Python's `min` returns whichever operand it meets first when one is `nan`, so
    `min(nan, 1.0)` and `min(1.0, nan)` differ. The compiled form uses `torch.minimum`, which
    propagates `nan`, and so does this.
    """
    if math.isnan(left) or math.isnan(right):
        return math.nan
    return min(left, right)


def _maximum(left: float, right: float) -> float:
    """The larger of two numbers, `nan` if either is `nan` (see `_minimum`)."""
    if math.isnan(left) or math.isnan(right):
        return math.nan
    return max(left, right)


#: The same names as `validateExpression.ALLOWED_EXPRESSION_FUNCTIONS`, mapped to the `math`
#: module (a test asserts that the two, and the torch table of `encoding/compileExpression.py`,
#: stay in sync).
_FUNCTIONS: dict[str, Callable[..., float]] = {
    "abs": math.fabs,
    "sqrt": math.sqrt,
    "exp": math.exp,
    "log": math.log,
    "log10": math.log10,
    "log2": math.log2,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "tanh": math.tanh,
    "min": _minimum,
    "max": _maximum,
}


def evaluateExpression(expression: str, variables: Mapping[str, object]) -> float:
    """Evaluates `expression` on plain Python numbers and returns its value.

    This is the pure counterpart of `encoding/compileExpression.py`: the same restricted
    grammar, checked first by `validateExpression`, and the same function names, but computed
    with the `math` module on floats, so that a constraint can be checked on a parameterization
    without importing torch (specification 3.9). Python's own `eval` is never used: the
    expression is parsed with `ast` and walked node by node.

    The result is a finite number, or `nan` where the expression has no finite value: the
    square root or logarithm of a negative number, a negative base with a fractional power, a
    division by zero, a zero to a negative power, an overflow. The compiled form returns `nan`
    for the first kinds of point but an infinity for a pole or an overflow, and a constraint
    check on an infinity depends on the direction of its comparator (an infinite value
    satisfies `>=`). A pole is not a point that satisfies a bound, so here every such result
    is `nan`, and a caller has one value to test for. `min` and `max` propagate `nan`, as
    `torch.minimum` and `torch.maximum` do.

    Args:
        expression: The expression, for example `"var * x ** z"`.
        variables: Maps each name the expression uses to its value, which must be an `int` or a
            `float` (a `bool` or a string is not a number here). Names the expression does not
            use are ignored.

    Returns:
        The value as a `float`, or `nan` where it is undefined or not finite.

    Raises:
        ValueError: if `expression` is invalid (see `validateExpression`), uses a name that is
            not in `variables`, or a name whose value is not an `int` or `float` that a float
            can hold.
    """
    referenced = validateExpression(expression)
    numbers = _numbersFor(expression, referenced, variables)
    body = ast.parse(expression.strip(), mode="eval").body
    try:
        value = _evaluateNode(body, numbers)
    except _UndefinedError:
        return math.nan
    return value if math.isfinite(value) else math.nan


def _numbersFor(
    expression: str, referenced: frozenset[str], variables: Mapping[str, object]
) -> dict[str, float]:
    """The values of the names an expression uses, as floats.

    Args:
        expression: The expression, for the error messages.
        referenced: The names it uses.
        variables: The values the caller gave.

    Returns:
        Name to value as a `float`, for the referenced names only.

    Raises:
        ValueError: if a name has no value, or a value is not an `int` or `float` that a
            `float` can hold. Checked here, before any evaluation, so that the errors of a bad
            call are never mistaken for an expression that is undefined.
    """
    missing = referenced - variables.keys()
    if missing:
        raise ValueError(
            f"Expression {expression!r} uses {sorted(missing)!r}, which "
            f"{'has' if len(missing) == 1 else 'have'} no value among the given variables "
            f"{sorted(variables)!r}."
        )
    numbers: dict[str, float] = {}
    for name in sorted(referenced):
        value = variables[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(
                f"Expression {expression!r} needs a number for {name!r}; got "
                f"{value!r} ({type(value).__name__}). Only int and float values are numbers."
            )
        try:
            numbers[name] = float(value)
        except OverflowError:
            raise ValueError(
                f"Expression {expression!r}: the value of {name!r} is too large for a float."
            ) from None
    return numbers


def _evaluateNode(node: ast.expr, numbers: Mapping[str, float]) -> float:
    """Evaluates one node of a validated expression tree.

    Args:
        node: The node.
        numbers: Variable name to value.

    Returns:
        The node's value.

    Raises:
        _UndefinedError: if the node, or one of its operands, has no value.
        ValueError: for a node type that `validateExpression` should have rejected.
    """
    if isinstance(node, ast.Constant):
        try:
            return float(cast(float, node.value))
        except OverflowError:
            raise _UndefinedError from None
    if isinstance(node, ast.Name):
        return numbers[node.id]
    if isinstance(node, ast.UnaryOp):
        operand = _evaluateNode(node.operand, numbers)
        return -operand if isinstance(node.op, ast.USub) else operand
    if isinstance(node, ast.BinOp):
        left = _evaluateNode(node.left, numbers)
        right = _evaluateNode(node.right, numbers)
        return _applyOperator(node.op, left, right)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        arguments = [_evaluateNode(argument, numbers) for argument in node.args]
        return _applyFunction(node.func.id, arguments)
    # Unreachable after validateExpression: kept so a grammar change there that is not
    # mirrored here fails loudly instead of returning garbage.
    raise ValueError(f"Unsupported expression node {type(node).__name__}.")


def _applyOperator(operator: ast.operator, left: float, right: float) -> float:
    """Applies one binary operator of the grammar.

    Args:
        operator: `Add`, `Sub`, `Mult`, `Div` or `Pow`.
        left: The left operand.
        right: The right operand.

    Returns:
        The result.

    Raises:
        _UndefinedError: on a division by zero, a power that has no real value (a negative
            base with a fractional exponent, a zero to a negative power) and a result too
            large for a float.
    """
    try:
        if isinstance(operator, ast.Add):
            return left + right
        if isinstance(operator, ast.Sub):
            return left - right
        if isinstance(operator, ast.Mult):
            return left * right
        if isinstance(operator, ast.Div):
            return left / right
        # `math.pow`, not `**`: on floats `**` turns a negative base with a fractional exponent
        # into a complex number, where `math.pow` raises, as the compiled form gives `nan`.
        return math.pow(left, right)
    except (ArithmeticError, ValueError):
        raise _UndefinedError from None


def _applyFunction(name: str, arguments: list[float]) -> float:
    """Applies one allowlisted function.

    Args:
        name: A key of `_FUNCTIONS`.
        arguments: Its arguments, as many as `validateExpression` checked.

    Returns:
        The result.

    Raises:
        _UndefinedError: where the function has no value at its arguments.
    """
    try:
        return _FUNCTIONS[name](*arguments)
    except (ArithmeticError, ValueError):
        raise _UndefinedError from None
