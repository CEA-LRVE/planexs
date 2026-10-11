"""A search space's constraints, expressed on the encoded (unit-cube) tensor."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor

from planexs.acquisition.toBotorchNonlinearConstraints import toBotorchNonlinearConstraints
from planexs.domain.constraints.LinearConstraint import LinearConstraint
from planexs.domain.constraints.NonlinearConstraint import NonlinearConstraint
from planexs.domain.SearchSpace import SearchSpace
from planexs.models.SearchSpaceEncoder import SearchSpaceEncoder


class EncodedConstraints:
    """The `LinearConstraint`s and `NonlinearConstraint`s of a `SearchSpace`,
    rewritten to act on `SearchSpaceEncoder`'s unit-cube columns, in the exact
    formats `optimize_acqf` takes, plus a feasibility check on encoded points.

    A linear constraint `sum(c_i * value_i) <cmp> b` on natural values becomes
    one on the unit columns `u_i` (with `value_i = lower_i + u_i * span_i`):
    `sum(c_i * span_i * u_i) <cmp> b - sum(c_i * lower_i)`. That substitution is
    only linear for a linearly scaled `Range`, so a linear constraint on a
    log-scaled or non-`Range` parameter is rejected.

    Attributes:
        inequality: BoTorch `inequality_constraints` (`sum(coeff * x[idx]) >=
            rhs`), or `None`.
        equality: BoTorch `equality_constraints`, or `None`.
        nonlinear: BoTorch `nonlinear_inequality_constraints` acting on encoded
            points, or `None`.

    Args:
        search_space: The space whose constraints to encode.
        encoder: The encoder defining the columns.

    Raises:
        ValueError: if a linear constraint references a parameter that is not a
            linearly scaled `Range`, or a nonlinear one references a name that
            is not a `Range` or numeric `Fixed` parameter.
        TypeError: if a constraint is of an unknown kind.
    """

    def __init__(self, search_space: SearchSpace, encoder: SearchSpaceEncoder) -> None:
        """Rewrites every constraint of `search_space` on the encoded columns.

        Arguments and errors are described in the class docstring.
        """
        inequality: list[tuple[Tensor, Tensor, float]] = []
        equality: list[tuple[Tensor, Tensor, float]] = []
        nonlinear_constraints: list[NonlinearConstraint] = []
        self._scales_linear: list[tuple[Tensor, Tensor, float, bool, float]] = []

        for constraint in search_space.constraints:
            if isinstance(constraint, LinearConstraint):
                indices, coefficients, rhs = _encodeLinear(constraint, encoder)
                if constraint.comparator == "=":
                    equality.append((indices, coefficients, rhs))
                elif constraint.comparator == ">=":
                    inequality.append((indices, coefficients, rhs))
                else:
                    inequality.append((indices, -coefficients, -rhs))
            elif isinstance(constraint, NonlinearConstraint):
                nonlinear_constraints.append(constraint)
            else:
                kind = type(constraint).__name__
                raise TypeError(f"No encoding is defined for constraint kind {kind!r}.")

        self.inequality: list[tuple[Tensor, Tensor, float]] | None = inequality or None
        self.equality: list[tuple[Tensor, Tensor, float]] | None = equality or None
        self._nonlinear_scales = [max(1.0, abs(c.bound)) for c in nonlinear_constraints]
        compiled = toBotorchNonlinearConstraints(
            nonlinear_constraints, encoder.range_names, encoder.numeric_fixed_values
        )
        self.nonlinear: list[tuple[Callable[[Tensor], Tensor], bool]] | None = [
            (_onEncoded(function, encoder), True) for function, _ in compiled
        ] or None

    @property
    def has_constraints(self) -> bool:
        """Whether any constraint exists (cheap, snake_case)."""
        return any(c is not None for c in (self.inequality, self.equality, self.nonlinear))

    def violation(self, x: Tensor) -> Tensor:
        """How far each encoded point is from satisfying every constraint.

        Args:
            x: A tensor of shape `(..., d)`.

        Returns:
            A tensor of shape `(...)`: `0` where all constraints hold, else the
            largest violation, relative to each constraint's own scale (so a
            single tolerance suits all of them). A point where a nonlinear
            expression is undefined (NaN) counts as infinitely violating.
        """
        worst = torch.zeros(x.shape[:-1], dtype=x.dtype, device=x.device)
        for indices, coefficients, rhs in self.inequality or []:
            value = (x[..., indices] * coefficients).sum(dim=-1)
            scale = max(1.0, abs(rhs), float(coefficients.abs().sum()))
            worst = torch.maximum(worst, (rhs - value) / scale)
        for indices, coefficients, rhs in self.equality or []:
            value = (x[..., indices] * coefficients).sum(dim=-1)
            scale = max(1.0, abs(rhs), float(coefficients.abs().sum()))
            worst = torch.maximum(worst, (value - rhs).abs() / scale)
        for (function, _), scale in zip(
            self.nonlinear or [], self._nonlinear_scales, strict=True
        ):
            margin = torch.nan_to_num(-function(x), nan=float("inf")) / scale
            worst = torch.maximum(worst, margin)
        return worst.clamp_min(0.0)

    def isFeasible(self, x: Tensor, tolerance: float = 1e-6) -> Tensor:
        """Whether each encoded point satisfies every constraint, up to
        `tolerance` (relative to each constraint's scale).

        Args:
            x: A tensor of shape `(..., d)`.
            tolerance: Largest accepted `violation`.

        Returns:
            A boolean tensor of shape `(...)`.
        """
        return self.violation(x) <= tolerance


def _encodeLinear(
    constraint: LinearConstraint, encoder: SearchSpaceEncoder
) -> tuple[Tensor, Tensor, float]:
    """Rewrites one `LinearConstraint` from natural values onto the unit-cube columns.

    With `value_i = lower_i + u_i * span_i`, the sum over `coefficient_i * value_i`
    becomes a sum over `coefficient_i * span_i * u_i`, and the bound moves by
    `sum(coefficient_i * lower_i)`.

    Args:
        constraint: The constraint to rewrite.
        encoder: The encoder defining the columns.

    Returns:
        `(indices, coefficients, rhs)` in BoTorch's constraint format.

    Raises:
        ValueError: if it references a parameter that is not a linearly scaled `Range`.
    """
    indices: list[int] = []
    coefficients: list[float] = []
    rhs = constraint.bound
    for name, coefficient in constraint.coefficients.items():
        encoding = encoder.getEncoding(name)
        if encoding is None or encoding.kind != "range":
            raise ValueError(
                f"LinearConstraint references {name!r}, which is not a searched Range "
                "parameter: linear constraints can only involve Range parameters."
            )
        if encoding.log_scale:
            raise ValueError(
                f"LinearConstraint references the log-scaled parameter {name!r}: a linear "
                "constraint on a log-scaled parameter is not linear in the model's encoding, "
                "which the custom acquisition layer cannot enforce yet."
            )
        span = encoding.upper - encoding.lower
        indices.append(encoding.column)
        coefficients.append(coefficient * span)
        rhs -= coefficient * encoding.lower
    return (
        torch.tensor(indices, dtype=torch.long),
        torch.tensor(coefficients, dtype=torch.double),
        rhs,
    )


def _onEncoded(
    function: Callable[[Tensor], Tensor], encoder: SearchSpaceEncoder
) -> Callable[[Tensor], Tensor]:
    """Adapts a function of natural `Range` values into a function of encoded points.

    Args:
        function: A function taking a tensor of natural values, one column per `Range`.
        encoder: The encoder that maps encoded points to those natural values.

    Returns:
        A function taking encoded points.
    """

    def onEncoded(x: Tensor) -> Tensor:
        """Evaluates `function` at the natural values of the encoded points `x`."""
        return function(encoder.rawRangeValues(x))

    return onEncoded
