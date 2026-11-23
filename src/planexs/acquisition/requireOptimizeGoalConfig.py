"""The optimize goal that an acquisition scores, read from a build context."""

from __future__ import annotations

from planexs.acquisition.AcquisitionBuildContext import AcquisitionBuildContext
from planexs.core.ConfigError import ConfigError
from planexs.goals.OptimizeGoalConfig import OptimizeGoalConfig


def requireOptimizeGoalConfig(
    context: AcquisitionBuildContext, component: str
) -> OptimizeGoalConfig:
    """The configuration of the goal of a build context, which must be an optimize goal.

    The acquisitions that score the value of an objective (expected improvement, upper
    confidence bound) read the outcomes, the directions, the weights and the outcome
    constraints from the configuration of an optimize goal. The compatibility check refuses
    another kind of goal when a study is built (`goal:optimize`); this is the same refusal for
    a builder that is called directly.

    Args:
        context: The build context.
        component: Who is asking, for the message, in the form `"acquisition 'ucb_v1'"`.

    Returns:
        The configuration of the goal.

    Raises:
        ConfigError: if the goal is not configured like an optimize goal.
    """
    config = context.goal.config
    if not isinstance(config, OptimizeGoalConfig):
        raise ConfigError(
            f"The {component} scores the value of one objective, and the {context.goal.label} "
            "is not an optimize goal.",
            component=component,
            field_path="goal",
            suggestion="use an optimize goal, or an acquisition that serves this goal",
        )
    return config
