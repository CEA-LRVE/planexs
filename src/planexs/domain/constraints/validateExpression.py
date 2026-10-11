"""Static validation of a `NonlinearConstraint` expression string."""

from __future__ import annotations

import ast

#: Functions an expression may call, mapped to the number of arguments each
#: takes. This is the whole allowlist: anything else is rejected before any
#: evaluation happens. `models/compileExpression.py` maps every name here to a
#: torch function (a test asserts the two stay in sync).
ALLOWED_EXPRESSION_FUNCTIONS: dict[str, int] = {
    "abs": 1,
    "sqrt": 1,
    "exp": 1,
    "log": 1,
    "log10": 1,
    "log2": 1,
    "sin": 1,
    "cos": 1,
    "tan": 1,
    "tanh": 1,
    "min": 2,
    "max": 2,
}

#: Upper bounds that keep a hostile or accidental expression from exhausting
#: the parser: this is a constraint on a handful of parameters, not a program.
MAX_EXPRESSION_LENGTH = 1000
_MAX_EXPRESSION_DEPTH = 200

_ALLOWED_BINARY_OPERATORS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)
_ALLOWED_UNARY_OPERATORS = (ast.UAdd, ast.USub)


def validateExpression(expression: str) -> frozenset[str]:
    """Checks that `expression` only uses the restricted arithmetic grammar
    a `NonlinearConstraint` allows, and returns the variable names it uses.

    The grammar is: numeric literals, variable names, `+ - * / **`, unary
    `+`/`-`, parentheses, and calls to the functions in
    `ALLOWED_EXPRESSION_FUNCTIONS`. The expression is parsed with `ast`, never
    evaluated: nothing here (or in `compileExpression`) uses Python's `eval`.

    Args:
        expression: The expression string, e.g. `"var * x ** z"`.

    Returns:
        The set of variable (parameter) names the expression references.

    Raises:
        ValueError: if the expression is empty, too long, not valid Python
            expression syntax, uses anything outside the allowed grammar, or
            references no variable at all (a constant is not a constraint).
    """
    stripped = expression.strip()
    if not stripped:
        raise ValueError("A nonlinear expression must not be empty.")
    if len(stripped) > MAX_EXPRESSION_LENGTH:
        raise ValueError(
            f"A nonlinear expression is limited to {MAX_EXPRESSION_LENGTH} characters; "
            f"got {len(stripped)}."
        )
    try:
        tree = ast.parse(stripped, mode="eval")
    except (SyntaxError, RecursionError, MemoryError) as error:
        raise ValueError(f"Invalid nonlinear expression {expression!r}: {error}") from error

    variables: set[str] = set()
    _visit(tree.body, variables, depth=0, expression=expression)
    if not variables:
        raise ValueError(
            f"Nonlinear expression {expression!r} references no parameter: a "
            "constraint must involve at least one parameter name."
        )
    return frozenset(variables)


def _visit(node: ast.expr, variables: set[str], depth: int, expression: str) -> None:
    """Checks one node of the expression tree and, recursively, its children.

    Args:
        node: The node to check.
        variables: Set that collects the variable names found.
        depth: Nesting depth of `node`, bounded to keep the parser's recursion shallow.
        expression: The full expression, for error messages.

    Raises:
        ValueError: if the node is anything but an allowed literal, name, operation or call.
    """
    if depth > _MAX_EXPRESSION_DEPTH:
        raise ValueError(f"Nonlinear expression {expression!r} is nested too deeply.")

    if isinstance(node, ast.Constant):
        # `bool` is an `int` subclass: reject it explicitly, `True + x` is a typo.
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError(
                f"Nonlinear expression {expression!r} contains the literal "
                f"{node.value!r}; only int and float literals are allowed."
            )
        return
    if isinstance(node, ast.Name):
        variables.add(node.id)
        return
    if isinstance(node, ast.BinOp):
        if not isinstance(node.op, _ALLOWED_BINARY_OPERATORS):
            hint = " (use `**` for a power)" if isinstance(node.op, ast.BitXor) else ""
            raise ValueError(
                f"Nonlinear expression {expression!r} uses the operator "
                f"{type(node.op).__name__}, which is not allowed{hint}. "
                "Allowed: + - * / **."
            )
        _visit(node.left, variables, depth + 1, expression)
        _visit(node.right, variables, depth + 1, expression)
        return
    if isinstance(node, ast.UnaryOp):
        if not isinstance(node.op, _ALLOWED_UNARY_OPERATORS):
            raise ValueError(
                f"Nonlinear expression {expression!r} uses the unary operator "
                f"{type(node.op).__name__}, which is not allowed."
            )
        _visit(node.operand, variables, depth + 1, expression)
        return
    if isinstance(node, ast.Call):
        if (
            not isinstance(node.func, ast.Name)
            or node.func.id not in ALLOWED_EXPRESSION_FUNCTIONS
        ):
            called = node.func.id if isinstance(node.func, ast.Name) else "<expression>"
            raise ValueError(
                f"Nonlinear expression {expression!r} calls {called!r}, which is not "
                f"an allowed function. Allowed: {sorted(ALLOWED_EXPRESSION_FUNCTIONS)}."
            )
        arity = ALLOWED_EXPRESSION_FUNCTIONS[node.func.id]
        if node.keywords or len(node.args) != arity:
            raise ValueError(
                f"Nonlinear expression {expression!r}: {node.func.id}() takes exactly "
                f"{arity} positional argument(s)."
            )
        for argument in node.args:
            _visit(argument, variables, depth + 1, expression)
        return
    raise ValueError(
        f"Nonlinear expression {expression!r} uses {type(node).__name__} syntax, which "
        "is not allowed: only arithmetic on parameter names and numbers is supported."
    )
