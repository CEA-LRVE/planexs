"""Compiles `NonlinearConstraint`s into the callables BoTorch's optimizer takes."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import torch
from torch import Tensor

from planexs.domain.constraints.NonlinearConstraint import NonlinearConstraint
from planexs.models.compileExpression import compileExpression


def toBotorchNonlinearConstraints(
    constraints: Sequence[NonlinearConstraint],
    parameter_order: Sequence[str],
    constants: Mapping[str, float] | None = None,
) -> list[tuple[Callable[[Tensor], Tensor], bool]]:
    """Compiles each `NonlinearConstraint`'s string expression into the
    tensor-taking callable BoTorch's `optimize_acqf` expects for its
    `nonlinear_inequality_constraints` argument (FR17, ADR-0006).

    BoTorch's convention is that a callable value `>= 0` means feasible, so a
    `">="` constraint `expr >= b` becomes `expr - b` and a `"<="` constraint
    `expr <= b` becomes `b - expr`. Each callable is an intra-point constraint
    (the boolean in each returned pair is `True`): it takes a tensor of shape
    `(..., len(parameter_order))` and returns a tensor of shape `(...)`. BoTorch
    calls it with a single point of shape `(d,)` and gets a scalar tensor
    (differentiable, since BoTorch takes its gradient).

    The callables see parameters' natural values. `parameter_order` fixes which
    tensor column is which parameter, since the callable only sees a tensor,
    not named values.

    Args:
        constraints: The constraints to compile.
        parameter_order: Names of the tensor's columns, in order.
        constants: Optional fixed numeric values (e.g. of `Fixed` parameters)
            an expression may also reference.

    Returns:
        One `(callable, True)` pair per constraint, in order.

    Raises:
        ValueError: if an expression references a name that is neither in
            `parameter_order` nor in `constants`.
    """
    constant_values = dict(constants) if constants is not None else {}
    available = [*parameter_order, *constant_values]
    compiled: list[tuple[Callable[[Tensor], Tensor], bool]] = []
    for constraint in constraints:
        try:
            function = compileExpression(constraint.expression, available)
        except ValueError as error:
            raise ValueError(
                f"Cannot use NonlinearConstraint {constraint.expression!r} "
                f"{constraint.comparator} {constraint.bound}: {error} Nonlinear constraints "
                "can only reference Range parameters and numeric Fixed parameters."
            ) from error
        compiled.append(
            (_constraintFunction(function, constraint, parameter_order, constant_values), True)
        )
    return compiled


def _constraintFunction(
    function: Callable[[Mapping[str, Tensor]], Tensor],
    constraint: NonlinearConstraint,
    parameter_order: Sequence[str],
    constants: Mapping[str, float],
) -> Callable[[Tensor], Tensor]:
    """Builds the BoTorch-convention callable for one constraint.

    Args:
        function: The compiled expression, taking a mapping from name to tensor.
        constraint: The constraint, for its comparator and bound.
        parameter_order: Names of the columns of the tensor the callable will receive.
        constants: Fixed numeric values the expression may also reference.

    Returns:
        A function of a tensor of shape `(..., len(parameter_order))` returning a tensor of
        shape `(...)` that is `>= 0` where the constraint holds.
    """
    columns = {name: position for position, name in enumerate(parameter_order)}
    bound = constraint.bound
    is_upper = constraint.comparator == "<="

    def constraintValue(x: Tensor) -> Tensor:
        """Signed margin of the constraint at `x`: positive if met, negative if violated."""
        variables = {name: x[..., position] for name, position in columns.items()}
        for name, constant in constants.items():
            variables[name] = torch.full_like(x[..., 0], constant)
        result = function(variables)
        return bound - result if is_upper else result - bound

    return constraintValue
