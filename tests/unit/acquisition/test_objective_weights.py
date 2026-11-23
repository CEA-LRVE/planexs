"""Unit tests for `objectiveWeights`, the scalar objective of an optimize goal as a weight for
each output of a model (roadmap tasks P5-03 and P5-04, specification 4, "Acquisition")."""

from __future__ import annotations

import pytest
import torch

from planexs.acquisition.AcquisitionContextError import AcquisitionContextError
from planexs.acquisition.objectiveWeights import objectiveWeights
from planexs.core.ConfigError import ConfigError
from planexs.goals.OptimizeGoalConfig import OptimizeGoalConfig

COMPONENT = "acquisition 'toy_v1'"


def weights(config: OptimizeGoalConfig, *names: str) -> list[float]:
    """The weights for a model whose outputs have the given names, as a list."""
    return objectiveWeights(
        config,
        names,
        dtype=torch.float64,
        device=torch.device("cpu"),
        component=COMPONENT,
    ).tolist()


class TestTheDirection:
    def test_a_maximized_outcome_counts_positively(self) -> None:
        assert weights(OptimizeGoalConfig(outcomes={"y": "maximize"}), "y") == [1.0]

    def test_a_minimized_outcome_counts_negatively(self) -> None:
        assert weights(OptimizeGoalConfig(outcomes={"y": "minimize"}), "y") == [-1.0]


class TestTheWeights:
    def test_the_weights_of_the_goal_are_applied_with_the_direction(self) -> None:
        config = OptimizeGoalConfig(
            outcomes={"a": "maximize", "b": "minimize"}, weights={"a": 2.0, "b": 0.5}
        )

        assert weights(config, "a", "b") == [2.0, -0.5]

    def test_the_weight_of_a_single_outcome_is_applied(self) -> None:
        config = OptimizeGoalConfig(outcomes={"y": "minimize"}, weights={"y": 4.0})

        assert weights(config, "y") == [-4.0]

    def test_the_weights_are_placed_by_name_on_the_outputs_of_the_model(self) -> None:
        config = OptimizeGoalConfig(
            outcomes={"b": "minimize", "a": "maximize"}, weights={"b": 3.0, "a": 2.0}
        )

        assert weights(config, "a", "b") == [2.0, -3.0]
        assert weights(config, "b", "a") == [-3.0, 2.0]

    def test_an_output_that_the_goal_does_not_optimize_weighs_nothing(self) -> None:
        config = OptimizeGoalConfig(outcomes={"y": "maximize"})

        assert weights(config, "c", "y", "d") == [0.0, 1.0, 0.0]

    def test_the_tensor_takes_the_precision_and_the_device_that_it_is_given(self) -> None:
        config = OptimizeGoalConfig(outcomes={"y": "maximize"})

        result = objectiveWeights(
            config,
            ["y"],
            dtype=torch.float32,
            device=torch.device("cpu"),
            component=COMPONENT,
        )

        assert result.dtype == torch.float32
        assert result.device == torch.device("cpu")
        assert result.shape == (1,)


class TestWhatIsRefused:
    def test_several_outcomes_with_no_weights_is_a_pareto_problem(self) -> None:
        config = OptimizeGoalConfig(outcomes={"a": "maximize", "b": "minimize"})

        with pytest.raises(ConfigError, match="Pareto") as caught:
            weights(config, "a", "b")

        assert caught.value.field_path == "goal.weights"
        assert COMPONENT in str(caught.value)
        assert "qlog_nehvi_v1" in (caught.value.suggestion or "")

    def test_an_outcome_that_the_model_does_not_cover_is_refused_with_what_it_covers(
        self,
    ) -> None:
        config = OptimizeGoalConfig(outcomes={"z": "maximize"})

        with pytest.raises(AcquisitionContextError, match="'z'") as caught:
            weights(config, "a", "b")

        assert caught.value.field_path == "goal.outcomes.z"
        assert "'a', 'b'" in str(caught.value)
