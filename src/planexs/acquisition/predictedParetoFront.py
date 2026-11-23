"""The Pareto front that a model predicts at the observed points."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import torch
from botorch.models.model import Model
from botorch.utils.multi_objective.pareto import is_non_dominated
from torch import Tensor

from planexs.acquisition.ParetoObjectives import ParetoObjectives


def predictedParetoFront(
    model: Model,
    baseline: Tensor,
    objectives: ParetoObjectives,
    constraints: list[Callable[[Tensor], Tensor]] | None,
) -> Tensor:
    """The Pareto front of the objectives, as the model predicts them at the observed points
    (specification 5.8).

    It is what a reference point is inferred from. The values are the posterior means and not
    the observations, which are noisy: the front is the one the model believes in, the same
    one the acquisition measures its improvement against. The points that the model predicts
    to violate an outcome constraint are left out, since a point that is not feasible is not
    a result, unless every observed point is predicted to violate one: then the front is that
    of all of them, because a front of nothing has no range to read a reference point from.

    Args:
        model: The fitted native model.
        baseline: The observed points, shape `(r, d)`.
        objectives: The objectives and where they are on the outputs of the model.
        constraints: The feasibility constraints of the goal, as BoTorch callables (negative
            where met, see `outcomeConstraintCallables`), or `None` for no constraint.

    Returns:
        The non-dominated points, shape `(n, k)`, with all the objectives maximized: a
        minimized outcome is negated.
    """
    with torch.no_grad():
        mean = cast(Tensor, cast(Any, model.posterior(baseline)).mean)
        values = objectives.objective(dtype=mean.dtype, device=mean.device)(mean)
        if constraints:
            feasible = torch.stack(
                [constraint(mean) <= 0.0 for constraint in constraints]
            ).all(
                dim=0  # structural: a point is feasible when every constraint is met
            )
            if feasible.any():
                values = values[feasible]
        return values[is_non_dominated(values)]
