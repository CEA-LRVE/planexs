"""The model-predicted Pareto front of a fitted multi-output surrogate."""

from __future__ import annotations

from typing import Any, cast

import torch
from botorch.models.model import Model
from botorch.utils.multi_objective.pareto import is_non_dominated
from torch import Tensor


def modelParetoFront(model: Model, objective_weights: Tensor) -> Tensor:
    """The non-dominated set of the model's posterior means at its own training
    inputs, in maximization form.

    Posterior means (denoised) rather than raw observations, matching what
    `AxBackend.getParetoFrontier` reports (`use_model_predictions=True`), so a
    noisy observation does not distort the front the acquisition improves on.

    Args:
        model: A fitted multi-output model whose training inputs are the
            evaluated points.
        objective_weights: A tensor of shape `(m,)`. Each outcome is multiplied
            by its weight, so a negative entry turns a minimized outcome into a
            maximized one.

    Returns:
        A tensor of shape `(k, m)`: the Pareto-optimal rows of the weighted
        outcomes.
    """
    train_x = cast(Tensor, cast(Any, model).train_inputs[0])
    while train_x.ndim > 2:
        # A multi-output SingleTaskGP stores one copy of the inputs per output.
        train_x = train_x[0]
    with torch.no_grad():
        posterior: Any = model.posterior(train_x)
        adjusted = cast(Tensor, posterior.mean * objective_weights)
    return adjusted[is_non_dominated(adjusted)]
