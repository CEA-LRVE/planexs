"""The outcome constraints of a goal, as the feasibility constraints of an acquisition."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any, cast

import torch
from botorch.models.model import Model
from torch import Tensor

from planexs.acquisition.AcquisitionContextError import AcquisitionContextError
from planexs.goals.OutcomeConstraint import OutcomeConstraint


def outcomeConstraintCallables(
    model: Model,
    baseline: Tensor,
    constraints: Sequence[OutcomeConstraint],
    outcome_names: Sequence[str],
) -> list[Callable[[Tensor], Tensor]]:
    """Turns the outcome constraints of a goal into the `constraints` argument of a BoTorch
    Monte Carlo acquisition function (specification 5.8, FR15).

    BoTorch reads a constraint as a callable that maps posterior samples of shape
    `sample_shape x batch_shape x q x m` (one column for each output of the model) to a tensor
    of shape `sample_shape x batch_shape x q`, which is *negative where the constraint is met*.
    It weighs each sample by a smooth indicator of that sign, so that a candidate that is
    predicted to violate a constraint scores low and one that is predicted to meet it scores
    high, and the same indicator removes violating observed points from the incumbent set.
    `outcome <= bound` is met where `sample - bound` is negative, and `outcome >= bound` where
    `bound - sample` is.

    **The scale of the indicator.** BoTorch's indicator has a fixed smoothness, which is in the
    units of the constraint: an outcome measured in millions would be judged almost as a hard
    step and one measured in millionths as no constraint at all. Ax avoids this by
    standardizing the outcomes before it builds the acquisition. Here the model works in the
    natural units of the outcomes, which a bound is stated in, so each violation is divided by
    the spread of the constrained outcome, the standard deviation of the posterior mean over
    the observed points, and the indicator is the same whatever unit the outcome is in. An
    outcome with no spread (a single observed point, or all means equal) is left in its own
    unit.

    Args:
        model: The fitted native model, whose posterior gives the spread of each outcome.
        baseline: The observed points, shape `(r, d)`.
        constraints: The outcome constraints of the goal, possibly none.
        outcome_names: The names of the outcomes the model covers, in the order of its outputs.
            A constraint reads the output that has its outcome's name.

    Returns:
        One callable for each constraint, in the order given. Empty if there is none.

    Raises:
        AcquisitionContextError: if a constraint names an outcome that the model does not
            cover, so that there is no output to compare with the bound.
    """
    names = list(outcome_names)
    indices: list[int] = []
    for position, constraint in enumerate(constraints):
        if constraint.outcome not in names:
            raise AcquisitionContextError(
                f"The goal constrains the outcome {constraint.outcome!r}, which the surrogate "
                f"does not model. The surrogate covers: {', '.join(map(repr, names))}.",
                component="acquisition build context",
                field_path=f"goal.outcome_constraints[{position}].outcome",
                suggestion="fit the surrogate on every outcome that the goal constrains, or "
                "remove the constraint",
            )
        indices.append(names.index(constraint.outcome))
    if not constraints:
        return []
    spreads = _spreads(model, baseline)
    return [
        _violation(index, constraint, spreads[index])
        for index, constraint in zip(indices, constraints, strict=True)
    ]


def _spreads(model: Model, baseline: Tensor) -> list[float]:
    """The spread of each output of the model over the observed points.

    Args:
        model: The fitted native model.
        baseline: The observed points, shape `(r, d)`.

    Returns:
        For each output, the standard deviation of the posterior mean at the observed points,
        or 1.0 where that is zero or not finite, which leaves the outcome in its own unit.
    """
    with torch.no_grad():
        mean = cast(Tensor, cast(Any, model.posterior(baseline)).mean)
        points_axis = -2  # structural: the observed points of a mean of shape (r, m)
        spread = torch.std(mean, dim=points_axis, correction=0)
    return [
        value if math.isfinite(value) and value > 0.0 else 1.0 for value in spread.tolist()
    ]


def _violation(
    index: int, constraint: OutcomeConstraint, spread: float
) -> Callable[[Tensor], Tensor]:
    """The BoTorch constraint callable for one outcome constraint.

    Args:
        index: The output of the model that holds the constrained outcome.
        constraint: The constraint.
        spread: The scale that divides the violation, in the unit of the outcome.

    Returns:
        A callable from samples of shape `sample_shape x batch_shape x q x m` to the signed
        violation, shape `sample_shape x batch_shape x q`: negative where the constraint is
        met, positive where it is violated.
    """
    sign = 1.0 if constraint.comparator == "<=" else -1.0
    bound = constraint.bound

    def violation(samples: Tensor) -> Tensor:
        """The signed, scaled violation of the constraint at each sample."""
        return sign * (samples[..., index] - bound) / spread

    return violation
