"""Unit tests for `requireLevelSetGoal`, the level-set goal of a build context (roadmap task
P5-09, specification 5.8)."""

from __future__ import annotations

import pytest
from toy_surrogates import OBJECTIVE, makeExploreContext, makeToyContext, makeToySurrogate

from planexs.acquisition.AcquisitionBuildContext import AcquisitionBuildContext
from planexs.acquisition.requireLevelSetGoal import requireLevelSetGoal
from planexs.core.ConfigError import ConfigError
from planexs.core.RuntimeConfig import RuntimeConfig
from planexs.encoding.torch_runtime import torchGenerator
from planexs.goals.LevelSetGoal import LevelSetGoal
from planexs.goals.LevelSetGoalConfig import LevelSetGoalConfig

COMPONENT = "acquisition 'test_v1'"


class TestRequireLevelSetGoal:
    def test_it_gives_the_goal_of_the_context(self) -> None:
        context = AcquisitionBuildContext(
            surrogate=makeToySurrogate(),
            goal=LevelSetGoal.fromConfig(LevelSetGoalConfig(outcome=OBJECTIVE, level=0.5)),
            generator=torchGenerator(7, RuntimeConfig()),
        )

        goal = requireLevelSetGoal(context, COMPONENT)

        assert goal is context.goal
        assert isinstance(goal, LevelSetGoal)

    def test_an_optimize_goal_is_refused(self) -> None:
        context = makeToyContext(makeToySurrogate(), {"outcomes": {OBJECTIVE: "maximize"}})

        with pytest.raises(ConfigError, match="not a level-set goal") as caught:
            requireLevelSetGoal(context, COMPONENT)

        assert caught.value.field_path == "goal"
        assert caught.value.component == COMPONENT

    def test_an_explore_goal_is_refused(self) -> None:
        context = makeExploreContext(makeToySurrogate())

        with pytest.raises(ConfigError, match="not a level-set goal"):
            requireLevelSetGoal(context, COMPONENT)
