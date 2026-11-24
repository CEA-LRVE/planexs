"""The spread of each outcome of a fitted model over the observed points."""

from __future__ import annotations

import math
from typing import Any, cast

import torch
from botorch.models.model import Model
from torch import Tensor


def outcomeSpreads(model: Model, baseline: Tensor) -> list[float]:
    """The scale of each output of a model, in the unit of that output (specification 5.8).

    A model works in the natural units of its outcomes, so a term that adds or compares
    outcomes (a violation of a bound, a variance summed over several outcomes) would be
    dominated by the outcome that happens to be measured in the largest unit. Dividing each
    outcome by its spread makes the term the same whether an outcome is measured in
    milliseconds or in millions. The spread of an outcome is the standard deviation of the
    posterior mean over the observed points: it is what the model says the outcome varies by
    over the data, in the unit of the outcome, and it needs no assumption about the kind of
    model (Ax standardizes the outcomes before it builds an acquisition for the same reason).

    Args:
        model: The fitted native model, whose posterior gives the mean of each outcome.
        baseline: The observed points, shape `(r, d)`.

    Returns:
        For each output of the model, the standard deviation of the posterior mean at the
        observed points, or 1.0 where that is zero or not finite (a single observed point, or
        all means equal), which leaves that outcome in its own unit.
    """
    with torch.no_grad():
        mean = cast(Tensor, cast(Any, model.posterior(baseline)).mean)
        points_axis = -2  # structural: the observed points of a mean of shape (r, m)
        spread = torch.std(mean, dim=points_axis, correction=0)
    return [
        value if math.isfinite(value) and value > 0.0 else 1.0 for value in spread.tolist()
    ]
