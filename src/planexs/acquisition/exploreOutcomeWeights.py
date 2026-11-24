"""The weight of each output of a model in the exploration of some of its outcomes."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from botorch.models.model import Model
from torch import Tensor

from planexs.acquisition.AcquisitionContextError import AcquisitionContextError
from planexs.acquisition.outcomeSpreads import outcomeSpreads


def exploreOutcomeWeights(
    explored: Sequence[str],
    outcome_names: Sequence[str],
    model: Model,
    baseline: Tensor,
) -> Tensor:
    """The weight of each output of a model when the outcomes in `explored` are to be learned
    and the others are not (specification 5.8, FR21).

    An exploration has no direction, so the weights have no sign: an explored outcome counts
    once, an output that is not explored (it may be in the model for a constraint) counts for
    nothing. The posterior of the explored outcomes is summed with these weights, so each
    outcome is divided by its spread over the observed points (`outcomeSpreads`): the variance
    that is left is then in units of the variation of the outcome and not in its own unit, and
    an outcome measured in millions does not outweigh one measured in millis. The weights are
    placed by name on the outputs of the model, so that the order of the outputs never
    matters.

    Args:
        explored: The names of the outcomes to learn.
        outcome_names: The names of the outcomes the model covers, in the order of its outputs.
        model: The fitted native model.
        baseline: The observed points, shape `(r, d)`.

    Returns:
        A tensor of shape `(m,)`, one entry for each output of the model, in the precision and
        on the device of `baseline`: the reciprocal of the spread of an explored outcome and 0
        for any other.

    Raises:
        AcquisitionContextError: if an outcome to learn is not modelled by the surrogate.
    """
    names = list(outcome_names)
    for outcome in explored:
        if outcome not in names:
            raise AcquisitionContextError(
                f"The goal explores the outcome {outcome!r}, which the surrogate does not "
                f"model. The surrogate covers: {', '.join(map(repr, names))}.",
                component="acquisition build context",
                field_path=f"goal.outcomes.{outcome}",
                suggestion="fit the surrogate on every outcome that the goal explores",
            )
    spreads = outcomeSpreads(model, baseline)
    values = [
        1.0 / spreads[index] if name in explored else 0.0 for index, name in enumerate(names)
    ]
    return torch.tensor(values, dtype=baseline.dtype, device=baseline.device)
