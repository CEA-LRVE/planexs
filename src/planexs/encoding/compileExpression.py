"""Compiles a validated expression string into a tensor-evaluating function."""

from __future__ import annotations

import ast
from collections.abc import Callable, Collection, Mapping
from typing import cast

import torch
from torch import Tensor

from planexs.domain.constraints.validateExpression import validateExpression

_FUNCTIONS: dict[str, Callable[..., Tensor]] = {
    "abs": torch.abs,
    "sqrt": torch.sqrt,
    "exp": torch.exp,
    "log": torch.log,
    "log10": torch.log10,
    "log2": torch.log2,
    "sin": torch.sin,
    "cos": torch.cos,
    "tan": torch.tan,
    "tanh": torch.tanh,
    "min": torch.minimum,
    "max": torch.maximum,
}


def compileExpression(
    expression: str, allowed_variables: Collection[str]
) -> Callable[[Mapping[str, Tensor]], Tensor]:
    """Compiles `expression` into a function of named tensors.

    The expression is first checked by `validateExpression` (restricted
    grammar, allowlisted functions) and then walked node by node: Python's own
    `eval` is never used. The returned function is differentiable with respect
    to its inputs wherever the underlying torch operations are, which is what
    BoTorch's constrained optimizer needs.

    Args:
        expression: The expression, e.g. `"var * x ** z"`.
        allowed_variables: The names the returned function will be given. Every
            variable in `expression` must be among them.

    Returns:
        A function taking a mapping from variable name to a tensor (all of one
        common shape, or broadcastable to it) and returning a tensor of that
        shape holding the expression's value.

    Raises:
        ValueError: if `expression` is invalid (see `validateExpression`) or
            references a name not in `allowed_variables`.
    """
    referenced = validateExpression(expression)
    unknown = referenced - set(allowed_variables)
    if unknown:
        raise ValueError(
            f"Expression {expression!r} references {sorted(unknown)!r}, which "
            f"{'is' if len(unknown) == 1 else 'are'} not among the available "
            f"variables {sorted(allowed_variables)!r}."
        )
    body = ast.parse(expression.strip(), mode="eval").body

    def evaluate(variables: Mapping[str, Tensor]) -> Tensor:
        """Evaluates the compiled expression for the given named tensors."""
        reference = next(iter(variables.values()))
        return _evaluateNode(body, variables, reference)

    return evaluate


def _evaluateNode(
    node: ast.expr, variables: Mapping[str, Tensor], reference: Tensor
) -> Tensor:
    """Evaluates one node of a validated expression tree.

    Args:
        node: The node.
        variables: Variable name to tensor.
        reference: A tensor whose dtype and device numeric literals take.

    Returns:
        The node's value.

    Raises:
        ValueError: for a node type the validator should have rejected.
    """
    if isinstance(node, ast.Constant):
        return torch.tensor(
            cast(float, node.value), dtype=reference.dtype, device=reference.device
        )
    if isinstance(node, ast.Name):
        return variables[node.id]
    if isinstance(node, ast.UnaryOp):
        operand = _evaluateNode(node.operand, variables, reference)
        return -operand if isinstance(node.op, ast.USub) else operand
    if isinstance(node, ast.BinOp):
        left = _evaluateNode(node.left, variables, reference)
        right = _evaluateNode(node.right, variables, reference)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        return torch.pow(left, right)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        arguments = [_evaluateNode(argument, variables, reference) for argument in node.args]
        return _FUNCTIONS[node.func.id](*arguments)
    # Unreachable after validateExpression: kept so a grammar change there that
    # is not mirrored here fails loudly instead of returning garbage.
    raise ValueError(f"Unsupported expression node {type(node).__name__}.")
