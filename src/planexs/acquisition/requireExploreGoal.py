"""The explore goal that an acquisition scores, read from a build context."""

from __future__ import annotations

from planexs.acquisition.AcquisitionBuildContext import AcquisitionBuildContext
from planexs.core.ConfigError import ConfigError
from planexs.goals.ExploreGoal import ExploreGoal


def requireExploreGoal(context: AcquisitionBuildContext, component: str) -> ExploreGoal:
    """The goal of a build context, which must be an explore goal.

    The acquisitions that learn the model, with no direction, read which outcomes to learn
    from an explore goal. The compatibility check refuses another kind of goal when a study is
    built (`goal:explore`); this is the same refusal for a builder that is called directly.

    Args:
        context: The build context.
        component: Who is asking, for the message, in the form
            `"acquisition 'neg_integrated_variance_v1'"`.

    Returns:
        The goal.

    Raises:
        ConfigError: if the goal is not an explore goal.
    """
    goal = context.goal
    if not isinstance(goal, ExploreGoal):
        raise ConfigError(
            f"The {component} learns the outcomes with no direction, and the {goal.label} is "
            "not an explore goal.",
            component=component,
            field_path="goal",
            suggestion="use an explore goal, or an acquisition that serves this goal",
        )
    return goal
