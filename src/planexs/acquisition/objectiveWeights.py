"""The scalar objective of an optimize goal, as a weight for each output of a model."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor

from planexs.acquisition.AcquisitionContextError import AcquisitionContextError
from planexs.core.ConfigError import ConfigError
from planexs.goals.OptimizeGoalConfig import OptimizeGoalConfig


def objectiveWeights(
    config: OptimizeGoalConfig,
    outcome_names: Sequence[str],
    *,
    dtype: torch.dtype,
    device: torch.device,
    component: str,
) -> Tensor:
    """The weight of each output of a model in the scalar objective of an optimize goal
    (specification 4, "Acquisition").

    The objective is the weighted sum of the outcomes that the goal optimizes, each in its own
    direction: a minimized outcome gets a negative weight, so that a higher value of the sum
    always means a better point. The weights are the goal's own, keyed by outcome name, and
    one outcome with no weights counts once. They are placed by name on the outputs of the
    model, so that the order of the outputs never matters, and an output that the goal does not
    optimize (it may be there for a constraint) gets a weight of zero.

    Args:
        config: The configuration of the optimize goal.
        outcome_names: The names of the outcomes the model covers, in the order of its outputs.
        dtype: The precision of the weights, that of the model.
        device: The device of the weights, that of the model.
        component: Who is asking, for the messages, in the form `"acquisition 'ucb_v1'"`.

    Returns:
        A tensor of shape `(m,)`, one entry for each output of the model.

    Raises:
        ConfigError: if the goal has several outcomes and no weights, a Pareto problem.
        AcquisitionContextError: if an outcome that the goal optimizes is not modelled by the
            surrogate.
    """
    if len(config.outcomes) > 1 and config.weights is None:
        raise ConfigError(
            f"The goal optimizes {len(config.outcomes)} outcomes {list(config.outcomes)!r} "
            f"with no weights: that is a Pareto problem, and the {component} scores one "
            "scalar objective.",
            component=component,
            field_path="goal.weights",
            suggestion="give a weight to each outcome of the goal, or use a "
            "multi-objective acquisition such as 'qlog_nehvi_v1'",
        )
    names = list(outcome_names)
    values = [0.0] * len(names)
    for outcome, direction in config.outcomes.items():
        if outcome not in names:
            raise AcquisitionContextError(
                f"The goal optimizes the outcome {outcome!r}, which the surrogate does not "
                f"model. The surrogate covers: {', '.join(map(repr, names))}.",
                component="acquisition build context",
                field_path=f"goal.outcomes.{outcome}",
                suggestion="fit the surrogate on every outcome that the goal optimizes",
            )
        weight = 1.0 if config.weights is None else config.weights[outcome]
        values[names.index(outcome)] = -weight if direction == "minimize" else weight
    return torch.tensor(values, dtype=dtype, device=device)
