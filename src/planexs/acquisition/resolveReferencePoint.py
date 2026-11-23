"""The reference point of the hypervolume, from its configuration."""

from __future__ import annotations

from collections.abc import Callable

import torch
from botorch.models.model import Model
from botorch.utils.multi_objective.hypervolume import infer_reference_point
from torch import Tensor

from planexs.acquisition.ExplicitReferencePointConfig import ExplicitReferencePointConfig
from planexs.acquisition.InferredReferencePointConfig import InferredReferencePointConfig
from planexs.acquisition.ParetoObjectives import ParetoObjectives
from planexs.acquisition.predictedParetoFront import predictedParetoFront
from planexs.core.ConfigError import ConfigError


def resolveReferencePoint(
    reference_point: ExplicitReferencePointConfig | InferredReferencePointConfig,
    objectives: ParetoObjectives,
    *,
    model: Model,
    baseline: Tensor,
    constraints: list[Callable[[Tensor], Tensor]] | None,
    component: str,
) -> Tensor:
    """The reference point of the hypervolume that the configuration asks for, in the space
    the acquisition works in (specification 5.8).

    The hypervolume is measured with every objective maximized, so the reference point is in
    that space too: a minimized outcome is negated. The explicit form gives the values in the
    natural units of the outcomes, as they are measured, and the point is made from them here,
    so that nobody has to know the sign convention. The inferred form applies its rule to the
    Pareto front that the model predicts at the observed points.

    Args:
        reference_point: The configured reference point, explicit or inferred.
        objectives: The objectives of the goal and where they are on the model's outputs.
        model: The fitted native model, which an inferred rule reads the front from.
        baseline: The observed points, shape `(r, d)`.
        constraints: The feasibility constraints of the goal, as BoTorch callables, or `None`.
        component: Who is asking, for the messages, in the form
            `"acquisition 'qlog_nehvi_v1'"`.

    Returns:
        The reference point, shape `(k,)`, one value for each objective in the order of
        `objectives`, in the space where all objectives are maximized.

    Raises:
        ConfigError: if an explicit reference point does not give exactly one value for each
            objective of the goal.
    """
    dtype, device = baseline.dtype, baseline.device
    if isinstance(reference_point, ExplicitReferencePointConfig):
        given = set(reference_point.values)
        wanted = set(objectives.names)
        if given != wanted:
            missing = sorted(wanted - given)
            extra = sorted(given - wanted)
            raise ConfigError(
                "The explicit reference point must give one value for each objective of the "
                f"goal, {sorted(wanted)!r}, and no other. "
                f"Missing: {missing!r}. Not objectives of the goal: {extra!r}.",
                component=component,
                field_path="reference_point.values",
                suggestion="list exactly the outcomes that the goal optimizes, each with the "
                "worst value that still counts as a result",
            )
        values = torch.tensor(
            [reference_point.values[name] for name in objectives.names],
            dtype=dtype,
            device=device,
        )
        return values * torch.tensor(objectives.signs, dtype=dtype, device=device)
    front = predictedParetoFront(model, baseline, objectives, constraints)
    return infer_reference_point(front, scale=reference_point.margin)
