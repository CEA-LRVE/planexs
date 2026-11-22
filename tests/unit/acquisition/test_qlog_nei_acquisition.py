"""Unit tests for `qlog_nei_v1`, its builder and its configuration (roadmap task P5-03,
specification 5.8, FR10, FR15).

The acceptance tests of the task: pending points change the value; the direction of the goal
is respected; a candidate that is predicted to violate an outcome constraint scores lower than
a feasible one on a toy surrogate; the sample count is configuration. The rest tests what the
registration says (needs, supports), the weights of a goal, the determinism of the samples, and
what the builder refuses.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from pydantic import ValidationError
from toy_surrogates import (
    CONSTRAINED,
    OBJECTIVE,
    candidates,
    makeToyContext,
    makeToySurrogate,
)

import planexs.acquisition
import planexs.goals
import planexs.surrogates  # noqa: F401
from planexs.acquisition.AbstractAcquisitionBuilder import AbstractAcquisitionBuilder
from planexs.acquisition.AcquisitionBuildContext import AcquisitionBuildContext
from planexs.acquisition.AcquisitionContextError import AcquisitionContextError
from planexs.acquisition.acquisitionRegistry import acquisitionRegistry
from planexs.acquisition.QLogNEIBuilder import QLogNEIBuilder
from planexs.acquisition.QLogNEIConfig import QLogNEIConfig
from planexs.core.checkCompatibility import checkCompatibility
from planexs.core.CompatibilityError import CompatibilityError
from planexs.core.ConfigError import ConfigError
from planexs.core.DeclaredParticipant import DeclaredParticipant
from planexs.core.resolveComponent import resolveComponent
from planexs.core.RuntimeConfig import RuntimeConfig
from planexs.encoding.torch_runtime import torchGenerator
from planexs.goals.LevelSetGoal import LevelSetGoal
from planexs.goals.LevelSetGoalConfig import LevelSetGoalConfig
from planexs.goals.OptimizeGoal import OptimizeGoal
from planexs.goals.OptimizeGoalConfig import OptimizeGoalConfig

MAXIMIZE_Y: dict[str, Any] = {"outcomes": {OBJECTIVE: "maximize"}}
MINIMIZE_Y: dict[str, Any] = {"outcomes": {OBJECTIVE: "minimize"}}

#: Two places of the toy space with almost the same predicted `y`, one on each side of its
#: maximum (`x = pi / 6`), and the left of them is the one with the smaller `c`.
LEFT = 0.3
RIGHT = 0.75


def constrained(comparator: str, bound: float) -> dict[str, Any]:
    """The goal that maximizes `y` and constrains `c`."""
    return {
        **MAXIMIZE_Y,
        "outcome_constraints": [
            {"outcome": CONSTRAINED, "comparator": comparator, "bound": bound}
        ],
    }


def scores(
    goal: dict[str, Any],
    *places: float,
    surrogate: Any = None,
    pending: torch.Tensor | None = None,
    seed: int = 7,
    **settings: Any,
) -> list[float]:
    """The acquisition value of each place, for a goal, built the way a policy builds it.

    Args:
        goal: The fields of the optimize goal.
        places: The value of `x` of each single-point candidate.
        surrogate: The surrogate. `None` means the two-outcome toy one.
        pending: The pending points, shape `(n, 1)`, or `None`.
        seed: The seed of the generator of the build.
        settings: Any setting of `QLogNEIConfig` to replace.
    """
    model = makeToySurrogate() if surrogate is None else surrogate
    builder = QLogNEIBuilder(QLogNEIConfig(**settings))
    function = builder.build(makeToyContext(model, goal, pending_points=pending, seed=seed))
    return [float(value) for value in function(candidates(*places)).detach()]


class TestTheConfiguration:
    def test_the_defaults(self) -> None:
        config = QLogNEIConfig()

        assert config.type == "qlog_nei_v1"
        assert config.n_samples == 512
        assert config.prune_baseline is True

    @pytest.mark.parametrize("n_samples", [0, -1, -512])
    def test_a_sample_count_below_one_is_refused(self, n_samples: int) -> None:
        with pytest.raises(ValidationError, match="greater than or equal to 1"):
            QLogNEIConfig(n_samples=n_samples)

    @pytest.mark.parametrize("n_samples", [2.5, "many", None])
    def test_a_sample_count_that_is_not_a_whole_number_is_refused(
        self, n_samples: Any
    ) -> None:
        with pytest.raises(ValidationError):
            QLogNEIConfig(n_samples=n_samples)

    def test_an_unknown_field_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="n_sample"):
            QLogNEIConfig(n_sample=64)  # type: ignore[call-arg]

    def test_the_type_is_pinned_to_the_registered_name(self) -> None:
        with pytest.raises(ValidationError):
            QLogNEIConfig(type="log_ei_v1")  # type: ignore[arg-type]

    def test_every_field_says_what_it_means(self) -> None:
        for name, field in QLogNEIConfig.model_fields.items():
            assert field.description, name
            assert len(field.description) > 40, name

    def test_the_description_of_the_sample_count_says_why_its_default_is_what_it_is(
        self,
    ) -> None:
        description = QLogNEIConfig.model_fields["n_samples"].description or ""

        assert "512" in description
        assert "BoTorch" in description

    def test_the_settings_are_read_from_a_mapping(self) -> None:
        resolved = resolveComponent(
            acquisitionRegistry(),
            {"type": "qlog_nei_v1", "n_samples": 64, "prune_baseline": False},
        )

        assert isinstance(resolved.config, QLogNEIConfig)
        assert resolved.config.n_samples == 64
        assert resolved.config.prune_baseline is False
        builder = resolved.entry.builder(resolved.config)
        assert isinstance(builder, QLogNEIBuilder)
        assert builder.config is resolved.config


class TestTheRegistration:
    def test_it_is_registered_in_the_acquisition_registry(self) -> None:
        entry = acquisitionRegistry().lookup("qlog_nei_v1")

        assert entry.builder == QLogNEIBuilder.fromConfig
        assert entry.config_schema is QLogNEIConfig
        assert issubclass(QLogNEIBuilder, AbstractAcquisitionBuilder)
        assert entry.description.strip()

    def test_it_needs_a_gaussian_posterior_a_single_objective_and_an_optimize_goal(
        self,
    ) -> None:
        entry = acquisitionRegistry().lookup("qlog_nei_v1")

        assert entry.needs == {"posterior:gaussian", "objectives:single", "goal:optimize"}

    def test_it_supports_pending_points_outcome_constraints_and_gradients(self) -> None:
        entry = acquisitionRegistry().lookup("qlog_nei_v1")

        assert entry.supports == {
            "pending:supported",
            "outcome_constraints:supported",
            "acquisition:differentiable",
        }

    def test_it_needs_no_reference_set(self) -> None:
        assert QLogNEIBuilder(QLogNEIConfig()).referencePointCount() == 0


class TestTheCompatibilityCheck:
    """The tags of the entry, read by the check that a study runs when it is built."""

    @staticmethod
    def participants(goal: Any) -> list[Any]:
        acquisition = acquisitionRegistry().lookup("qlog_nei_v1")
        return [
            goal,
            DeclaredParticipant(
                label="acquisition 'qlog_nei_v1'",
                needs=acquisition.needs,
                supports=acquisition.supports,
            ),
            # No surrogate is registered yet, so the one that supports what this entry needs of
            # it is declared here.
            DeclaredParticipant(
                label="surrogate 'toy_v1'", supports=frozenset({"posterior:gaussian"})
            ),
        ]

    def test_a_goal_with_outcome_constraints_finds_its_need_met(self) -> None:
        goal = OptimizeGoal(OptimizeGoalConfig(**constrained("<=", 0.5)))

        assert "outcome_constraints:supported" in goal.needs
        checkCompatibility(self.participants(goal))

    def test_a_weighted_goal_over_several_outcomes_is_single_objective(self) -> None:
        goal = OptimizeGoal(
            OptimizeGoalConfig(
                outcomes={"a": "maximize", "b": "minimize"}, weights={"a": 1.0, "b": 2.0}
            )
        )

        checkCompatibility(self.participants(goal))

    def test_a_pareto_goal_is_refused_for_its_objectives(self) -> None:
        goal = OptimizeGoal(OptimizeGoalConfig(outcomes={"a": "maximize", "b": "minimize"}))

        with pytest.raises(CompatibilityError, match="objectives:single"):
            checkCompatibility(self.participants(goal))

    def test_a_level_set_goal_is_refused_for_its_kind(self) -> None:
        goal = LevelSetGoal.fromConfig(LevelSetGoalConfig(outcome=OBJECTIVE))

        with pytest.raises(CompatibilityError, match="goal:optimize"):
            checkCompatibility(self.participants(goal))


class TestPendingPoints:
    """Roadmap P5-03, done when (a): pending points change the value."""

    def test_a_pending_point_changes_the_value(self) -> None:
        pending = candidates(0.5).reshape(1, 1)

        without = scores(MAXIMIZE_Y, 0.2, 0.5, 0.85)
        with_pending = scores(MAXIMIZE_Y, 0.2, 0.5, 0.85, pending=pending)

        assert all(a != b for a, b in zip(without, with_pending, strict=True))

    def test_a_candidate_at_a_pending_point_is_worth_less_than_it_was(self) -> None:
        # The improvement is incremental over the best of the observed and the pending points,
        # so a point that is already pending, and a point next to it, has less to add.
        pending = candidates(0.5).reshape(1, 1)

        (without,) = scores(MAXIMIZE_Y, 0.5)
        (at_pending,) = scores(MAXIMIZE_Y, 0.5, pending=pending)
        (next_to_pending,) = scores(MAXIMIZE_Y, 0.52, pending=pending)
        (next_to_nothing,) = scores(MAXIMIZE_Y, 0.52)

        assert at_pending < without - 1.0
        assert next_to_pending < next_to_nothing

    def test_the_pending_points_join_the_incumbent_set(self) -> None:
        pending = candidates(0.5, 0.6).reshape(2, 1)
        context = makeToyContext(makeToySurrogate(), MAXIMIZE_Y, pending_points=pending)

        function = QLogNEIBuilder(QLogNEIConfig(prune_baseline=False)).build(context)

        assert function.X_baseline.shape == (8 + 2, 1)  # type: ignore[union-attr]

    def test_no_pending_point_is_the_same_as_an_empty_set(self) -> None:
        empty = torch.empty((0, 1), dtype=candidates(0.5).dtype)

        assert scores(MAXIMIZE_Y, 0.2, 0.5, pending=empty) == scores(MAXIMIZE_Y, 0.2, 0.5)

    def test_a_pending_point_can_be_set_on_the_function_after_it_is_built(self) -> None:
        context = makeToyContext(makeToySurrogate(), MAXIMIZE_Y)
        function = QLogNEIBuilder(QLogNEIConfig()).build(context)
        before = function(candidates(0.5)).detach()

        function.set_X_pending(candidates(0.5).reshape(1, 1))
        during = function(candidates(0.5)).detach()
        function.set_X_pending(None)
        after = function(candidates(0.5)).detach()

        assert during.item() < before.item()
        assert torch.equal(before, after)


class TestTheDirectionOfTheGoal:
    """Roadmap P5-03, done when (a): the direction is respected."""

    def test_a_maximizer_prefers_the_candidate_at_the_peak(self) -> None:
        left, peak, right = scores(MAXIMIZE_Y, 0.2, 0.5, 0.85)

        assert peak > left
        assert peak > right

    def test_a_minimizer_does_not_prefer_it(self) -> None:
        left, peak, right = scores(MINIMIZE_Y, 0.2, 0.5, 0.85)

        assert peak < left
        assert peak < right

    def test_each_direction_prefers_its_own_side_of_the_space(self) -> None:
        # y falls from its peak to the right of the space: for a minimizer the right is better
        # than the left of the peak, and a maximizer ranks them the other way round at the same
        # distance from the peak.
        (maximize_near,) = scores(MAXIMIZE_Y, 0.4)
        (maximize_far,) = scores(MAXIMIZE_Y, 0.95)
        (minimize_near,) = scores(MINIMIZE_Y, 0.4)
        (minimize_far,) = scores(MINIMIZE_Y, 0.95)

        assert maximize_near > maximize_far
        assert minimize_near < minimize_far

    def test_the_weights_of_the_goal_are_applied_to_the_outcomes_by_name(self) -> None:
        # Alone, y peaks near 0.5. With c minimized at ten times the weight, the objective
        # falls as x grows, so the left of the space is better than the peak.
        goal = {
            "outcomes": {OBJECTIVE: "maximize", CONSTRAINED: "minimize"},
            "weights": {OBJECTIVE: 1.0, CONSTRAINED: 10.0},
        }

        left, peak, right = scores(goal, 0.2, 0.5, 0.85)

        assert left > peak > right

    def test_a_goal_that_does_not_say_how_to_combine_its_outcomes_is_refused(self) -> None:
        goal = {"outcomes": {OBJECTIVE: "maximize", CONSTRAINED: "minimize"}}

        with pytest.raises(ConfigError, match="Pareto") as caught:
            scores(goal, 0.5)

        assert caught.value.field_path == "goal.weights"


class TestOutcomeConstraints:
    """Roadmap P5-03, done when (a): a candidate predicted to violate an outcome constraint
    scores lower than a feasible one on a toy surrogate."""

    def test_the_two_places_are_a_tie_for_the_objective_alone(self) -> None:
        # What the next tests rest on: without the constraint nothing tells the two apart.
        left, right = scores(MAXIMIZE_Y, LEFT, RIGHT)

        assert abs(left - right) < 1.0

    def test_a_candidate_predicted_to_exceed_an_upper_bound_scores_lower(self) -> None:
        # c = x, so c <= 0.5 holds on the left and is violated on the right.
        left, right = scores(constrained("<=", 0.5), LEFT, RIGHT)

        assert left > right + 5.0

    def test_a_candidate_predicted_to_fall_below_a_lower_bound_scores_lower(self) -> None:
        left, right = scores(constrained(">=", 0.5), LEFT, RIGHT)

        assert right > left + 5.0

    def test_the_constraint_lowers_the_violating_candidate_and_not_the_feasible_one(
        self,
    ) -> None:
        (free_left, free_right) = scores(MAXIMIZE_Y, LEFT, RIGHT)
        (bound_left, bound_right) = scores(constrained("<=", 0.5), LEFT, RIGHT)

        assert bound_right < free_right - 5.0
        assert bound_left > free_left - 1.0

    def test_the_feasible_candidate_is_the_better_one_whatever_the_objective_says(
        self,
    ) -> None:
        # A candidate at the peak of y that violates the constraint loses to a feasible one
        # that is worse for y: the constraint has the last word.
        peak, _ = scores(constrained("<=", 0.4), 0.5, 0.2)
        unconstrained_peak, unconstrained_left = scores(MAXIMIZE_Y, 0.5, 0.2)

        assert unconstrained_peak > unconstrained_left
        assert peak < unconstrained_peak - 5.0

    def test_the_unit_of_the_constrained_outcome_does_not_matter(self) -> None:
        # The same outcome in a unit a million times smaller, with its bound in that unit,
        # is the same constraint, and gives the same values.
        metre = scores(constrained("<=", 0.5), LEFT, RIGHT)
        micrometre = scores(
            constrained("<=", 0.5e6), LEFT, RIGHT, surrogate=makeToySurrogate(1.0e6)
        )

        assert micrometre == pytest.approx(metre, rel=1e-6)

    def test_a_constraint_on_an_outcome_the_surrogate_does_not_model_is_refused(self) -> None:
        surrogate = makeToySurrogate(objective_only=True)

        with pytest.raises(AcquisitionContextError, match="'c'") as caught:
            scores(constrained("<=", 0.5), LEFT, surrogate=surrogate)

        assert caught.value.field_path == "goal.outcome_constraints[0].outcome"

    def test_an_objective_outcome_the_surrogate_does_not_model_is_refused(self) -> None:
        surrogate = makeToySurrogate(objective_only=True)
        goal = {"outcomes": {CONSTRAINED: "maximize"}}

        with pytest.raises(AcquisitionContextError, match="'c'") as caught:
            scores(goal, LEFT, surrogate=surrogate)

        assert caught.value.field_path == f"goal.outcomes.{CONSTRAINED}"


class TestTheSampleCount:
    """Roadmap P5-03, done when (b): the sample count is configuration."""

    def test_the_function_samples_as_many_times_as_the_configuration_says(self) -> None:
        context = makeToyContext(makeToySurrogate(), MAXIMIZE_Y)

        for n_samples in (16, 64, 512):
            function = QLogNEIBuilder(QLogNEIConfig(n_samples=n_samples)).build(context)

            assert function.sampler.sample_shape == torch.Size([n_samples])  # type: ignore[union-attr]

    def test_the_default_is_the_default_of_the_configuration(self) -> None:
        context = makeToyContext(makeToySurrogate(), MAXIMIZE_Y)

        function = QLogNEIBuilder(QLogNEIConfig()).build(context)

        assert function.sampler.sample_shape == torch.Size([512])  # type: ignore[union-attr]

    def test_the_count_changes_the_estimate_and_not_its_meaning(self) -> None:
        few = scores(MAXIMIZE_Y, 0.2, 0.5, 0.85, n_samples=8)
        many = scores(MAXIMIZE_Y, 0.2, 0.5, 0.85, n_samples=1024)

        assert few != many
        assert many.index(max(many)) == few.index(max(few)) == 1

    def test_a_count_that_the_configuration_gives_through_a_mapping_reaches_the_function(
        self,
    ) -> None:
        resolved = resolveComponent(
            acquisitionRegistry(), {"type": "qlog_nei_v1", "n_samples": 32}
        )
        builder = resolved.entry.builder(resolved.config)

        function = builder.build(makeToyContext(makeToySurrogate(), MAXIMIZE_Y))

        assert function.sampler.sample_shape == torch.Size([32])  # type: ignore[attr-defined]


class TestThePruningOfTheBaseline:
    def test_pruning_drops_the_observed_points_that_cannot_be_the_best(self) -> None:
        context = makeToyContext(makeToySurrogate(), MAXIMIZE_Y)

        pruned = QLogNEIBuilder(QLogNEIConfig(prune_baseline=True)).build(context)
        kept = QLogNEIBuilder(QLogNEIConfig(prune_baseline=False)).build(context)

        assert kept.X_baseline.shape[0] == 8  # type: ignore[union-attr]
        assert pruned.X_baseline.shape[0] < 8  # type: ignore[union-attr]


class TestTheNativeFunction:
    def test_it_scores_a_batch_of_candidates_of_q_points(self) -> None:
        function = QLogNEIBuilder(QLogNEIConfig()).build(
            makeToyContext(makeToySurrogate(), MAXIMIZE_Y)
        )
        batch = torch.stack(
            [candidates(0.2, 0.5).reshape(2, 1), candidates(0.4, 0.9).reshape(2, 1)]
        )

        values = function(batch)

        assert values.shape == (2,)

    def test_a_batch_of_two_equal_points_adds_nothing_to_a_single_one(self) -> None:
        function = QLogNEIBuilder(QLogNEIConfig()).build(
            makeToyContext(makeToySurrogate(), MAXIMIZE_Y)
        )
        single = function(candidates(0.5)).detach()
        doubled = function(candidates(0.5, 0.5).reshape(1, 2, 1)).detach()

        # The two points are one point twice, so the batch improves no more than the single
        # point does. The maximum over the points of a batch is a smooth one, which leaves a
        # difference of the order of its temperature and no more.
        assert doubled.item() == pytest.approx(single.item(), abs=5e-2)

    def test_it_is_differentiable_in_the_candidate(self) -> None:
        function = QLogNEIBuilder(QLogNEIConfig()).build(
            makeToyContext(makeToySurrogate(), MAXIMIZE_Y)
        )
        candidate = candidates(0.3).clone().requires_grad_(True)

        function(candidate).sum().backward()

        assert candidate.grad is not None
        assert torch.isfinite(candidate.grad).all()
        assert candidate.grad.abs().sum() > 0.0
        # The peak is to the right of 0.3, so the value rises with x there.
        assert candidate.grad.item() > 0.0

    def test_the_value_is_finite_far_from_any_improvement(self) -> None:
        # A log, so that it stays finite where the improvement is almost zero.
        values = scores(MAXIMIZE_Y, 0.0, 0.05, 1.0)

        assert all(torch.isfinite(torch.tensor(value)) for value in values)

    def test_the_model_is_not_changed_by_building_or_scoring(self) -> None:
        surrogate = makeToySurrogate()
        before = [parameter.detach().clone() for parameter in surrogate.model.parameters()]

        scores(constrained("<=", 0.5), LEFT, RIGHT, surrogate=surrogate)

        after = list(surrogate.model.parameters())
        assert all(torch.equal(a, b) for a, b in zip(before, after, strict=True))


class TestTheSamplesAreDeterministic:
    def test_the_same_seed_gives_the_same_values(self) -> None:
        assert scores(MAXIMIZE_Y, 0.2, 0.5, seed=3) == scores(MAXIMIZE_Y, 0.2, 0.5, seed=3)

    def test_another_seed_gives_other_samples(self) -> None:
        assert scores(MAXIMIZE_Y, 0.2, 0.5, seed=3) != scores(MAXIMIZE_Y, 0.2, 0.5, seed=4)

    def test_the_samples_are_the_same_for_every_call_of_one_function(self) -> None:
        function = QLogNEIBuilder(QLogNEIConfig()).build(
            makeToyContext(makeToySurrogate(), MAXIMIZE_Y)
        )

        assert torch.equal(function(candidates(0.2, 0.5)), function(candidates(0.2, 0.5)))

    def test_the_pruning_of_the_baseline_is_seeded_too(self) -> None:
        # The engine prunes with an unseeded sampler: left to it, builds from one seed would
        # keep different incumbents. Many builds, all the same.
        context = makeToyContext(makeToySurrogate(), MAXIMIZE_Y)
        builder = QLogNEIBuilder(QLogNEIConfig(n_samples=16, prune_baseline=True))

        baselines = []
        for _ in range(12):
            generator_seed = torchGenerator(7, RuntimeConfig())
            fresh = AcquisitionBuildContext(
                surrogate=context.surrogate,
                goal=context.goal,
                generator=generator_seed,
            )
            function = builder.build(fresh)
            baselines.append((function.X_baseline, function(candidates(0.3, 0.5)).detach()))  # type: ignore[union-attr]

        first_baseline, first_values = baselines[0]
        for baseline, values in baselines[1:]:
            assert torch.equal(baseline, first_baseline)
            assert torch.equal(values, first_values)

    def test_building_advances_the_generator_of_the_context(self) -> None:
        generator = torchGenerator(7, RuntimeConfig())
        context = AcquisitionBuildContext(
            surrogate=makeToySurrogate(),
            goal=OptimizeGoal(OptimizeGoalConfig(**MAXIMIZE_Y)),
            generator=generator,
        )
        builder = QLogNEIBuilder(QLogNEIConfig())

        first = builder.build(context)(candidates(0.2, 0.5)).detach()
        second = builder.build(context)(candidates(0.2, 0.5)).detach()

        assert not torch.equal(first, second)


class TestWhatTheBuilderRefuses:
    def test_a_goal_that_is_not_an_optimize_goal_is_refused(self) -> None:
        context = AcquisitionBuildContext(
            surrogate=makeToySurrogate(),
            goal=LevelSetGoal.fromConfig(LevelSetGoalConfig(outcome=OBJECTIVE)),
            generator=torchGenerator(7, RuntimeConfig()),
        )

        with pytest.raises(ConfigError, match="not an optimize goal") as caught:
            QLogNEIBuilder(QLogNEIConfig()).build(context)

        assert caught.value.field_path == "goal"
