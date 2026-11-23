"""The objectives of an optimize goal that is a Pareto problem."""

from __future__ import annotations

from collections.abc import Sequence

from planexs.acquisition.AcquisitionContextError import AcquisitionContextError
from planexs.acquisition.ParetoObjectives import ParetoObjectives
from planexs.core.ConfigError import ConfigError
from planexs.goals.OptimizeGoalConfig import OptimizeGoalConfig


def selectParetoObjectives(
    config: OptimizeGoalConfig, outcome_names: Sequence[str], *, component: str
) -> ParetoObjectives:
    """Reads the objectives of an optimize goal that has several of them, and places each on
    an output of a model (specification 4, "Acquisition").

    A Pareto problem is a goal that optimizes several outcomes and does not say how to trade
    one against another: no weights. A goal with weights is one scalar objective, which the
    single-objective acquisitions score, and a goal with one outcome is not a Pareto problem
    (a hypervolume of one objective is the improvement of its best value), so neither is
    accepted here.

    Args:
        config: The configuration of the optimize goal.
        outcome_names: The names of the outcomes the model covers, in the order of its outputs.
        component: Who is asking, for the messages, in the form
            `"acquisition 'qlog_nehvi_v1'"`.

    Returns:
        The objectives, in the order of the outputs of the model.

    Raises:
        ConfigError: if the goal has weights, or optimizes fewer than two outcomes.
        AcquisitionContextError: if an outcome that the goal optimizes is not modelled by the
            surrogate.
    """
    if config.weights is not None:
        raise ConfigError(
            "The goal gives weights to its outcomes, which makes it one scalar objective, and "
            f"the {component} scores a Pareto problem, several objectives that are not "
            "traded against each other.",
            component=component,
            field_path="goal.weights",
            suggestion="remove the weights of the goal to keep the objectives apart, or use "
            "an acquisition for one objective such as 'qlog_nei_v1'",
        )
    if len(config.outcomes) < 2:  # structural: a Pareto problem has at least two objectives
        raise ConfigError(
            f"The goal optimizes {len(config.outcomes)} outcome, and the {component} scores a "
            "Pareto problem, which has at least two objectives.",
            component=component,
            field_path="goal.outcomes",
            suggestion="optimize several outcomes, or use an acquisition for one objective "
            "such as 'qlog_nei_v1'",
        )
    names = list(outcome_names)
    for outcome in config.outcomes:
        if outcome not in names:
            raise AcquisitionContextError(
                f"The goal optimizes the outcome {outcome!r}, which the surrogate does not "
                f"model. The surrogate covers: {', '.join(map(repr, names))}.",
                component="acquisition build context",
                field_path=f"goal.outcomes.{outcome}",
                suggestion="fit the surrogate on every outcome that the goal optimizes",
            )
    # The order is that of the model's outputs, whatever the order of the goal.
    indices = tuple(index for index, name in enumerate(names) if name in config.outcomes)
    return ParetoObjectives(
        names=tuple(names[index] for index in indices),
        indices=indices,
        signs=tuple(
            -1.0 if config.outcomes[names[index]] == "minimize" else 1.0 for index in indices
        ),
        num_outputs=len(names),
    )
