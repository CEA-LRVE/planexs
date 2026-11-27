"""The level-set goal that an acquisition scores, read from a build context."""

from __future__ import annotations

from planexs.acquisition.AcquisitionBuildContext import AcquisitionBuildContext
from planexs.core.ConfigError import ConfigError
from planexs.goals.LevelSetGoal import LevelSetGoal


def requireLevelSetGoal(context: AcquisitionBuildContext, component: str) -> LevelSetGoal:
    """The goal of a build context, which must be a level-set goal.

    The acquisitions that look for a boundary read the outcome and the level from a level-set
    goal. The compatibility check refuses another kind of goal when a study is built
    (`goal:level_set`); this is the same refusal for a builder that is called directly.

    Args:
        context: The build context.
        component: Who is asking, for the message, in the form `"acquisition 'straddle_v1'"`.

    Returns:
        The goal.

    Raises:
        ConfigError: if the goal is not a level-set goal.
    """
    goal = context.goal
    if not isinstance(goal, LevelSetGoal):
        raise ConfigError(
            f"The {component} looks for the boundary of a level, and the {goal.label} is not "
            "a level-set goal.",
            component=component,
            field_path="goal",
            suggestion="use a level-set goal, or an acquisition that serves this goal",
        )
    return goal
