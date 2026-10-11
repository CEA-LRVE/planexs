"""Reads a fitted surrogate model's belief about one point."""

from __future__ import annotations

from typing import Any

import torch
from botorch.models.model import Model
from torch import Tensor


def predictWithModel(
    model: Model, x: Tensor, metric_names: list[str]
) -> dict[str, tuple[float, float]]:
    """The model's predicted mean and standard error of the mean at one
    encoded point, per metric.

    Matches the `(mean, sem)` shape of `OptimizationBackend.predict` and of
    `Client.predict`: `sem` is the standard deviation of the model's belief
    about the underlying function value (its epistemic uncertainty, without
    observation noise), not of a future noisy observation.

    Args:
        model: A fitted model with one output per entry of `metric_names`, in
            the same order.
        x: The encoded point, shape `(d,)`.
        metric_names: Names of the model's outputs.

    Returns:
        `{metric_name: (mean, sem)}` in the metrics' original units.

    Raises:
        ValueError: if the model's output count does not match `metric_names`.
    """
    if model.num_outputs != len(metric_names):
        raise ValueError(
            f"The model has {model.num_outputs} output(s) but {len(metric_names)} metric "
            f"name(s) were given: {metric_names!r}."
        )
    with torch.no_grad():
        posterior: Any = model.posterior(x.reshape(1, -1))
    mean = posterior.mean.reshape(-1)
    sem = posterior.variance.reshape(-1).clamp_min(0.0).sqrt()
    return {name: (float(mean[i]), float(sem[i])) for i, name in enumerate(metric_names)}
