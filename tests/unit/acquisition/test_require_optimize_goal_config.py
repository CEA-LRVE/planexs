"""Unit tests for `requireOptimizeGoalConfig`, the optimize goal that an acquisition scores
(roadmap tasks P5-03 and P5-04)."""

from __future__ import annotations

import pytest
from toy_surrogates import OBJECTIVE, makeToyContext, makeToySurrogate

from planexs.acquisition.AcquisitionBuildContext import AcquisitionBuildContext
from planexs.acquisition.requireOptimizeGoalConfig import requireOptimizeGoalConfig
from planexs.core.ConfigError import ConfigError
from planexs.core.RuntimeConfig import RuntimeConfig
from planexs.encoding.torch_runtime import torchGenerator
from planexs.goals.LevelSetGoal import LevelSetGoal
from planexs.goals.LevelSetGoalConfig import LevelSetGoalConfig
from planexs.goals.OptimizeGoalConfig import OptimizeGoalConfig

COMPONENT = "acquisition 'toy_v1'"


class TestRequireOptimizeGoalConfig:
    def test_an_optimize_goal_gives_its_configuration(self) -> None:
        context = makeToyContext(makeToySurrogate(), {"outcomes": {OBJECTIVE: "maximize"}})

        config = requireOptimizeGoalConfig(context, COMPONENT)

        assert isinstance(config, OptimizeGoalConfig)
        assert config.outcomes == {OBJECTIVE: "maximize"}
        assert config is context.goal.config

    def test_another_kind_of_goal_is_refused_by_name(self) -> None:
        goal = LevelSetGoal.fromConfig(LevelSetGoalConfig(outcome=OBJECTIVE))
        context = AcquisitionBuildContext(
            surrogate=makeToySurrogate(),
            goal=goal,
            generator=torchGenerator(7, RuntimeConfig()),
        )

        with pytest.raises(ConfigError, match="not an optimize goal") as caught:
            requireOptimizeGoalConfig(context, COMPONENT)

        assert caught.value.field_path == "goal"
        assert COMPONENT in str(caught.value)
        assert goal.label in str(caught.value)
        assert caught.value.suggestion
