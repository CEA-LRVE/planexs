"""Conversion of a `LegacyObjective` into an `optimize_v1` goal (roadmap task P4-03).

The goal replaces the roles of `LegacyObjective`, `LegacyMetric` and `LegacyOutcomeConstraint`.
These tests build the same goal from a legacy fixture and check that nothing the legacy layer
said is lost or changed on the way, and that the goal reports the capability tags that the
legacy layer's own behaviour implies. They keep the parity data (`legacy_objective_cases`) that
the tasks which port the acquisitions and the policies compare against (specification 10.11,
"parity tests whenever a component replaces another").
"""

from __future__ import annotations

import json

import pytest
from legacy_objective_cases import (
    LEGACY_OBJECTIVES,
    goalConfigFromLegacy,
    problemFor,
)

import planexs.goals  # noqa: F401  importing the package registers `optimize_v1`
from planexs.core.resolveComponent import resolveComponent
from planexs.core.toResolvedConfig import toResolvedConfig
from planexs.core.validateConfig import validateConfig
from planexs.domain.LegacyMetric import LegacyMetric
from planexs.domain.LegacyObjective import LegacyObjective
from planexs.domain.LegacyOutcomeConstraint import LegacyOutcomeConstraint
from planexs.goals.goalRegistry import goalRegistry
from planexs.goals.OptimizeGoal import OptimizeGoal
from planexs.goals.OptimizeGoalConfig import OptimizeGoalConfig

CASES = pytest.mark.parametrize(
    "objective", LEGACY_OBJECTIVES.values(), ids=list(LEGACY_OBJECTIVES)
)


class TestTheSameGoalFromALegacyObjective:
    @CASES
    def test_every_metric_becomes_an_outcome_with_its_direction(
        self, objective: LegacyObjective
    ) -> None:
        config = goalConfigFromLegacy(objective)

        assert list(config.outcomes) == [metric.name for metric in objective.metrics]
        for metric in objective.metrics:
            assert config.outcomes[metric.name] == (
                "minimize" if metric.minimize else "maximize"
            )

    @CASES
    def test_the_weights_are_kept_and_now_belong_to_an_outcome_by_name(
        self, objective: LegacyObjective
    ) -> None:
        config = goalConfigFromLegacy(objective)

        if objective.weights is None:
            assert config.weights is None
        else:
            assert config.weights == {
                metric.name: weight
                for metric, weight in zip(objective.metrics, objective.weights, strict=True)
            }

    @CASES
    def test_every_outcome_constraint_is_kept_in_order(
        self, objective: LegacyObjective
    ) -> None:
        config = goalConfigFromLegacy(objective)

        assert [(c.outcome, c.comparator, c.bound) for c in config.outcome_constraints] == [
            (c.metric_name, c.comparator, c.bound) for c in objective.outcome_constraints
        ]

    @CASES
    def test_the_goal_fits_a_problem_with_the_same_outcomes(
        self, objective: LegacyObjective
    ) -> None:
        OptimizeGoal(goalConfigFromLegacy(objective)).validateAgainst(problemFor(objective))

    @CASES
    def test_the_goal_is_built_by_the_registry_from_the_converted_configuration(
        self, objective: LegacyObjective
    ) -> None:
        config = goalConfigFromLegacy(objective)

        resolved = resolveComponent(goalRegistry(), toResolvedConfig(config))
        goal = resolved.entry.builder(resolved.config)

        assert isinstance(goal, OptimizeGoal)
        assert goal.config == config

    @CASES
    def test_the_converted_configuration_round_trips(self, objective: LegacyObjective) -> None:
        config = goalConfigFromLegacy(objective)
        resolved = toResolvedConfig(config)

        reloaded = validateConfig(OptimizeGoalConfig, json.loads(json.dumps(resolved)))

        assert reloaded == config


class TestTheTagsFollowWhatTheLegacyLayerDid:
    @CASES
    def test_the_multi_objective_tag_is_what_selected_the_legacy_multi_objective_acquisition(
        self, objective: LegacyObjective
    ) -> None:
        # `AlphaAcquisitionStrategy.buildAcquisitionFunction` used its multi-objective
        # acquisition for several metrics with no weights, and its single-objective one for a
        # lone metric and for weights.
        legacy_is_pareto = objective.is_multi_objective and objective.weights is None

        supports = OptimizeGoal(goalConfigFromLegacy(objective)).supports

        assert ("objectives:multi" in supports) is legacy_is_pareto
        assert ("objectives:single" in supports) is not legacy_is_pareto
        assert "goal:optimize" in supports

    @CASES
    def test_the_goal_needs_what_the_legacy_layer_only_warned_it_did_not_enforce(
        self, objective: LegacyObjective
    ) -> None:
        # `BayesianOptimizer._askWithAlpha` logged a warning when the objective had outcome
        # constraints; the goal says it as a need, checked when a study is built.
        needs = OptimizeGoal(goalConfigFromLegacy(objective)).needs

        assert bool(needs) is bool(objective.outcome_constraints)
        assert needs <= {"outcome_constraints:supported"}


class TestWhatTheGoalDoesNotCarryOver:
    def test_a_constraint_relative_to_a_baseline_is_refused_by_the_conversion(self) -> None:
        objective = LegacyObjective(
            [LegacyMetric("qps", minimize=False)],
            outcome_constraints=[LegacyOutcomeConstraint("qps", 0.95, ">=", relative=True)],
        )

        with pytest.raises(ValueError, match="relative to a baseline"):
            goalConfigFromLegacy(objective)

    def test_a_constraint_may_now_name_an_outcome_that_is_not_an_objective(self) -> None:
        # The legacy objective refused it (a constraint had to name one of its own metrics).
        legacy = LegacyObjective([LegacyMetric("score", minimize=False)])
        with pytest.raises(ValueError, match="not among its own metrics"):
            LegacyObjective(
                legacy.metrics,
                outcome_constraints=[LegacyOutcomeConstraint("latency", 50.0, "<=")],
            )

        config = OptimizeGoalConfig(
            outcomes={"score": "maximize"},
            outcome_constraints=[{"outcome": "latency", "comparator": "<=", "bound": 50.0}],  # type: ignore[list-item]
        )

        assert [c.outcome for c in config.outcome_constraints] == ["latency"]
