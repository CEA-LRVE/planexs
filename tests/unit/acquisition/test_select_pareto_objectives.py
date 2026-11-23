"""Unit tests for `ParetoObjectives` and `selectParetoObjectives`, the objectives of a Pareto
problem placed on the outputs of a model (roadmap task P5-05, specification 4, "Acquisition",
5.8)."""

from __future__ import annotations

import dataclasses

import pytest
import torch

from planexs.acquisition.AcquisitionContextError import AcquisitionContextError
from planexs.acquisition.ParetoObjectives import ParetoObjectives
from planexs.acquisition.selectParetoObjectives import selectParetoObjectives
from planexs.core.ConfigError import ConfigError
from planexs.goals.OptimizeGoalConfig import OptimizeGoalConfig

COMPONENT = "acquisition 'test_v1'"


def goal(**fields: object) -> OptimizeGoalConfig:
    return OptimizeGoalConfig(**fields)  # type: ignore[arg-type]


class TestSelectParetoObjectives:
    def test_each_objective_is_placed_on_the_output_that_has_its_name(self) -> None:
        objectives = selectParetoObjectives(
            goal(outcomes={"f1": "maximize", "f2": "minimize"}),
            ("f1", "f2"),
            component=COMPONENT,
        )

        assert objectives.names == ("f1", "f2")
        assert objectives.indices == (0, 1)
        assert objectives.signs == (1.0, -1.0)
        assert objectives.num_outputs == 2

    def test_the_order_is_that_of_the_outputs_and_not_that_of_the_goal(self) -> None:
        objectives = selectParetoObjectives(
            goal(outcomes={"f2": "minimize", "f1": "maximize"}),
            ("f1", "f2"),
            component=COMPONENT,
        )

        assert objectives.names == ("f1", "f2")
        assert objectives.signs == (1.0, -1.0)

    def test_an_output_that_the_goal_does_not_optimize_is_not_an_objective(self) -> None:
        objectives = selectParetoObjectives(
            goal(
                outcomes={"f1": "maximize", "f2": "maximize"},
                outcome_constraints=[{"outcome": "g", "comparator": "<=", "bound": 1.0}],
            ),
            ("g", "f1", "f2"),
            component=COMPONENT,
        )

        assert objectives.names == ("f1", "f2")
        assert objectives.indices == (1, 2)
        assert objectives.num_outputs == 3

    def test_a_goal_with_weights_is_refused(self) -> None:
        with pytest.raises(ConfigError, match="weights") as caught:
            selectParetoObjectives(
                goal(
                    outcomes={"f1": "maximize", "f2": "maximize"},
                    weights={"f1": 1.0, "f2": 2.0},
                ),
                ("f1", "f2"),
                component=COMPONENT,
            )

        assert caught.value.field_path == "goal.weights"
        assert caught.value.component == COMPONENT

    def test_a_goal_with_one_outcome_is_refused(self) -> None:
        with pytest.raises(ConfigError, match="at least two objectives") as caught:
            selectParetoObjectives(
                goal(outcomes={"f1": "maximize"}), ("f1", "f2"), component=COMPONENT
            )

        assert caught.value.field_path == "goal.outcomes"

    def test_an_objective_that_the_surrogate_does_not_model_is_refused(self) -> None:
        with pytest.raises(AcquisitionContextError, match="'f2'") as caught:
            selectParetoObjectives(
                goal(outcomes={"f1": "maximize", "f2": "maximize"}),
                ("f1",),
                component=COMPONENT,
            )

        assert caught.value.field_path == "goal.outcomes.f2"


class TestParetoObjectives:
    @staticmethod
    def objectives() -> ParetoObjectives:
        return ParetoObjectives(
            names=("f1", "f2"), indices=(1, 2), signs=(1.0, -1.0), num_outputs=3
        )

    def test_it_is_frozen(self) -> None:
        with pytest.raises(dataclasses.FrozenInstanceError):
            self.objectives().names = ("a", "b")  # type: ignore[misc]

    def test_the_objective_selects_the_outputs_and_negates_the_minimized_ones(self) -> None:
        objective = self.objectives().objective(
            dtype=torch.float64, device=torch.device("cpu")
        )
        samples = torch.tensor([[9.0, 1.0, 2.0], [9.0, 3.0, 4.0]], dtype=torch.float64)

        values = objective(samples)

        assert torch.equal(
            values, torch.tensor([[1.0, -2.0], [3.0, -4.0]], dtype=torch.float64)
        )

    def test_the_objective_keeps_the_leading_dimensions_of_the_samples(self) -> None:
        objective = self.objectives().objective(
            dtype=torch.float64, device=torch.device("cpu")
        )
        samples = torch.zeros(5, 4, 2, 3, dtype=torch.float64)

        assert objective(samples).shape == (5, 4, 2, 2)
