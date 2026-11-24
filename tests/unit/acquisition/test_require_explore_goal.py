"""Unit tests for `requireExploreGoal`, the explore goal of a build context (roadmap task
P5-06, specification 5.8)."""

from __future__ import annotations

import pytest
from toy_surrogates import OBJECTIVE, makeExploreContext, makeToyContext, makeToySurrogate

from planexs.acquisition.AcquisitionBuildContext import AcquisitionBuildContext
from planexs.acquisition.requireExploreGoal import requireExploreGoal
from planexs.core.ConfigError import ConfigError
from planexs.core.RuntimeConfig import RuntimeConfig
from planexs.encoding.torch_runtime import torchGenerator
from planexs.goals.ExploreGoal import ExploreGoal
from planexs.goals.LevelSetGoal import LevelSetGoal
from planexs.goals.LevelSetGoalConfig import LevelSetGoalConfig

COMPONENT = "acquisition 'test_v1'"


class TestRequireExploreGoal:
    def test_it_gives_the_goal_of_the_context(self) -> None:
        context = makeExploreContext(makeToySurrogate())

        goal = requireExploreGoal(context, COMPONENT)

        assert goal is context.goal
        assert isinstance(goal, ExploreGoal)

    def test_an_optimize_goal_is_refused(self) -> None:
        context = makeToyContext(makeToySurrogate(), {"outcomes": {OBJECTIVE: "maximize"}})

        with pytest.raises(ConfigError, match="not an explore goal") as caught:
            requireExploreGoal(context, COMPONENT)

        assert caught.value.field_path == "goal"
        assert caught.value.component == COMPONENT

    def test_a_level_set_goal_is_refused(self) -> None:
        context = AcquisitionBuildContext(
            surrogate=makeToySurrogate(),
            goal=LevelSetGoal.fromConfig(LevelSetGoalConfig(outcome=OBJECTIVE)),
            generator=torchGenerator(7, RuntimeConfig()),
        )

        with pytest.raises(ConfigError, match="not an explore goal"):
            requireExploreGoal(context, COMPONENT)
