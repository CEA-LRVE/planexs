"""Unit tests for `neg_integrated_variance_v1`, its builder and its configuration (roadmap task
P5-06, specification 5.8, FR21).

The acceptance tests of the task: on a one-dimensional toy surrogate the value is higher in an
unexplored region than next to an observation; the point count is configuration. The rest
tests what the registration says (needs, supports), what the goal decides (the outcomes to
learn, and the unit of each), the pending points, the batches, the determinism of the points,
and what the builder refuses.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from botorch.sampling.normal import SobolQMCNormalSampler
from pydantic import ValidationError
from toy_pareto_surrogates import F1, F2, makeParetoSurrogate
from toy_surrogates import (
    CONSTRAINED,
    OBJECTIVE,
    candidates,
    makeExploreContext,
    makeSurrogateOf,
    makeToyContext,
    makeToySurrogate,
)

import planexs.acquisition
import planexs.goals
import planexs.surrogates  # noqa: F401
from planexs.acquisition.AbstractAcquisitionBuilder import AbstractAcquisitionBuilder
from planexs.acquisition.AcquisitionContextError import AcquisitionContextError
from planexs.acquisition.acquisitionRegistry import acquisitionRegistry
from planexs.acquisition.drawSeed import drawSeed
from planexs.acquisition.NegIntegratedVarianceBuilder import NegIntegratedVarianceBuilder
from planexs.acquisition.NegIntegratedVarianceConfig import NegIntegratedVarianceConfig
from planexs.core.checkCompatibility import checkCompatibility
from planexs.core.CompatibilityError import CompatibilityError
from planexs.core.ConfigError import ConfigError
from planexs.core.DeclaredParticipant import DeclaredParticipant
from planexs.core.resolveComponent import resolveComponent
from planexs.core.RuntimeConfig import RuntimeConfig
from planexs.encoding.torch_runtime import torchGenerator
from planexs.goals.ExploreGoal import ExploreGoal
from planexs.goals.ExploreGoalConfig import ExploreGoalConfig
from planexs.goals.LevelSetGoal import LevelSetGoal
from planexs.goals.LevelSetGoalConfig import LevelSetGoalConfig
from planexs.goals.OptimizeGoal import OptimizeGoal
from planexs.goals.OptimizeGoalConfig import OptimizeGoalConfig

#: The observed points of the toy surrogate of the exploration tests, all in the left third of
#: the space: the right two thirds are unexplored.
OBSERVED = (0.05, 0.1, 0.15, 0.2, 0.25, 0.3)

#: Places of the toy space: next to an observation, at one, and in the middle of the unexplored
#: region.
NEXT_TO_OBSERVATION = 0.15
AT_OBSERVATION = 0.1
UNEXPLORED = 0.75


def oneOutcome() -> Any:
    """A surrogate of one outcome, `y = sin(3 x)`, with data on the left only. Its problem has
    that outcome only, so that an explore goal learns it by default."""
    return makeSurrogateOf({OBJECTIVE: lambda x: torch.sin(3.0 * x)}, observed_x=OBSERVED)


def explorer(outcomes: Any = None) -> Any:
    """The build context of an explore goal over `oneOutcome()`."""
    return makeExploreContext(oneOutcome(), outcomes)


def build(context: Any, **settings: Any) -> Any:
    """The native function for a context."""
    return NegIntegratedVarianceBuilder(NegIntegratedVarianceConfig(**settings)).build(context)


def scores(context: Any, *places: float, **settings: Any) -> list[float]:
    """The acquisition value of each place, for a context."""
    function = build(context, **settings)
    return [float(value) for value in function(candidates(*places)).detach()]


class TestTheConfiguration:
    def test_the_defaults(self) -> None:
        config = NegIntegratedVarianceConfig()

        assert config.type == "neg_integrated_variance_v1"
        assert config.n_points == 512

    @pytest.mark.parametrize("n_points", [0, -1, -512])
    def test_a_point_count_below_one_is_refused(self, n_points: int) -> None:
        with pytest.raises(ValidationError, match="greater than or equal to 1"):
            NegIntegratedVarianceConfig(n_points=n_points)

    @pytest.mark.parametrize("n_points", [2.5, "many", None])
    def test_a_point_count_that_is_not_a_whole_number_is_refused(self, n_points: Any) -> None:
        with pytest.raises(ValidationError):
            NegIntegratedVarianceConfig(n_points=n_points)

    def test_an_unknown_field_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="n_point"):
            NegIntegratedVarianceConfig(n_point=64)  # type: ignore[call-arg]

    def test_the_type_is_pinned_to_the_registered_name(self) -> None:
        with pytest.raises(ValidationError):
            NegIntegratedVarianceConfig(type="niv_v1")  # type: ignore[arg-type]

    def test_every_field_says_what_it_means(self) -> None:
        for name, field in NegIntegratedVarianceConfig.model_fields.items():
            assert field.description, name
            assert len(field.description) > 40, name

    def test_the_description_of_the_point_count_says_why_its_default_is_what_it_is(
        self,
    ) -> None:
        description = NegIntegratedVarianceConfig.model_fields["n_points"].description or ""

        assert "512" in description
        assert "legacy" in description
        assert "encoder" in description

    def test_the_settings_are_read_from_a_mapping(self) -> None:
        resolved = resolveComponent(
            acquisitionRegistry(), {"type": "neg_integrated_variance_v1", "n_points": 64}
        )

        assert isinstance(resolved.config, NegIntegratedVarianceConfig)
        assert resolved.config.n_points == 64
        builder = resolved.entry.builder(resolved.config)
        assert isinstance(builder, NegIntegratedVarianceBuilder)
        assert builder.config is resolved.config


class TestTheRegistration:
    def test_it_is_registered_in_the_acquisition_registry(self) -> None:
        entry = acquisitionRegistry().lookup("neg_integrated_variance_v1")

        assert entry.builder == NegIntegratedVarianceBuilder.fromConfig
        assert entry.config_schema is NegIntegratedVarianceConfig
        assert issubclass(NegIntegratedVarianceBuilder, AbstractAcquisitionBuilder)
        assert entry.description.strip()

    def test_it_needs_a_gaussian_posterior_and_an_explore_goal(self) -> None:
        entry = acquisitionRegistry().lookup("neg_integrated_variance_v1")

        assert entry.needs == {"posterior:gaussian", "goal:explore"}

    def test_it_supports_pending_points(self) -> None:
        entry = acquisitionRegistry().lookup("neg_integrated_variance_v1")

        assert entry.supports == {"pending:supported"}

    def test_it_needs_no_reference_set(self) -> None:
        # The points that it averages over are drawn from the encoder, by the builder.
        builder = NegIntegratedVarianceBuilder(NegIntegratedVarianceConfig())

        assert builder.needs_reference_points is False
        assert builder.referencePointCount() == 0


class TestTheCompatibilityCheck:
    """The tags of the entry, read by the check that a study runs when it is built."""

    @staticmethod
    def participants(goal: Any, surrogate_supports: frozenset[str] | None = None) -> list[Any]:
        acquisition = acquisitionRegistry().lookup("neg_integrated_variance_v1")
        return [
            goal,
            DeclaredParticipant(
                label="acquisition 'neg_integrated_variance_v1'",
                needs=acquisition.needs,
                supports=acquisition.supports,
            ),
            # No surrogate is registered yet, so the one that supports what this entry needs of
            # it is declared here.
            DeclaredParticipant(
                label="surrogate 'toy_v1'",
                supports=surrogate_supports or frozenset({"posterior:gaussian"}),
            ),
        ]

    def test_an_explore_goal_finds_its_need_met(self) -> None:
        checkCompatibility(self.participants(ExploreGoal.fromConfig(ExploreGoalConfig())))

    def test_an_optimize_goal_is_refused_for_its_kind(self) -> None:
        goal = OptimizeGoal(OptimizeGoalConfig(outcomes={OBJECTIVE: "maximize"}))

        with pytest.raises(CompatibilityError, match="goal:explore"):
            checkCompatibility(self.participants(goal))

    def test_a_level_set_goal_is_refused_for_its_kind(self) -> None:
        goal = LevelSetGoal.fromConfig(LevelSetGoalConfig(outcome=OBJECTIVE))

        with pytest.raises(CompatibilityError, match="goal:explore"):
            checkCompatibility(self.participants(goal))

    def test_a_surrogate_without_a_gaussian_posterior_is_refused(self) -> None:
        goal = ExploreGoal.fromConfig(ExploreGoalConfig())

        with pytest.raises(CompatibilityError, match="posterior:gaussian"):
            checkCompatibility(
                self.participants(goal, frozenset({"posterior:marginal_normal"}))
            )


class TestExploration:
    """Roadmap P5-06, done when (a): on a one-dimensional toy surrogate the value is higher in
    an unexplored region than next to an observation."""

    def test_the_value_is_higher_in_an_unexplored_region_than_next_to_an_observation(
        self,
    ) -> None:
        next_to, unexplored = scores(explorer(), NEXT_TO_OBSERVATION, UNEXPLORED)

        # The score is minus a variance: closer to zero is better.
        assert unexplored > next_to

    def test_the_gap_is_not_a_rounding_error(self) -> None:
        next_to, unexplored = scores(explorer(), NEXT_TO_OBSERVATION, UNEXPLORED)

        assert unexplored > 0.5 * next_to  # the score is negative: this is a factor of two

    def test_the_value_falls_towards_the_data_and_rises_towards_the_gap(self) -> None:
        near, middle, far = scores(explorer(), 0.2, 0.45, UNEXPLORED)

        assert near < middle < far

    def test_a_candidate_at_an_observed_point_is_the_least_worth_measuring(self) -> None:
        values = scores(explorer(), AT_OBSERVATION, 0.4, 0.6, UNEXPLORED, 0.9)

        assert values[0] == min(values)

    def test_every_candidate_of_the_unexplored_region_beats_every_one_next_to_the_data(
        self,
    ) -> None:
        near_the_data = scores(explorer(), 0.05, 0.1, 0.15, 0.2, 0.25, 0.3)
        unexplored = scores(explorer(), 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)

        assert max(near_the_data) < min(unexplored)

    def test_the_value_is_never_positive_and_always_finite(self) -> None:
        values = scores(explorer(), 0.0, 0.1, 0.5, 1.0)

        assert all(value <= 0.0 and torch.isfinite(torch.tensor(value)) for value in values)

    def test_where_the_data_is_the_best_candidate_is_the_one_that_covers_the_gap(self) -> None:
        grid = [0.35, 0.5, 0.65, 0.8, 0.95]
        values = scores(explorer(), *grid)

        # The best place is in the middle of the unexplored region and not at its far end,
        # which teaches the model about a smaller part of the space.
        assert grid[values.index(max(values))] in (0.65, 0.8)

    def test_the_unit_of_the_outcome_does_not_matter(self) -> None:
        metre = scores(explorer(), NEXT_TO_OBSERVATION, UNEXPLORED)
        millimetre_surrogate = makeSurrogateOf(
            {OBJECTIVE: lambda x: 1000.0 * torch.sin(3.0 * x)}, observed_x=OBSERVED
        )
        millimetre = scores(
            makeExploreContext(millimetre_surrogate), NEXT_TO_OBSERVATION, UNEXPLORED
        )

        assert millimetre == pytest.approx(metre, rel=1e-6)


class TestTheOutcomesToLearn:
    @staticmethod
    def two_outcomes(
        scale: float = 1.0, order: tuple[str, str] = (OBJECTIVE, CONSTRAINED)
    ) -> Any:
        functions = {
            OBJECTIVE: lambda x: torch.sin(3.0 * x),
            CONSTRAINED: lambda x: scale * x**2,
        }
        return makeSurrogateOf({name: functions[name] for name in order}, observed_x=OBSERVED)

    def test_the_goal_names_the_outcome_and_the_other_is_ignored(self) -> None:
        both = makeExploreContext(self.two_outcomes(), [OBJECTIVE])
        alone = explorer()

        assert scores(both, NEXT_TO_OBSERVATION, UNEXPLORED) == pytest.approx(
            scores(alone, NEXT_TO_OBSERVATION, UNEXPLORED), rel=1e-6
        )

    def test_by_default_all_the_outcomes_of_the_problem_are_learned(self) -> None:
        everything = scores(
            makeExploreContext(self.two_outcomes()), NEXT_TO_OBSERVATION, UNEXPLORED
        )
        first_only = scores(
            makeExploreContext(self.two_outcomes(), [OBJECTIVE]),
            NEXT_TO_OBSERVATION,
            UNEXPLORED,
        )

        assert everything != pytest.approx(first_only)
        assert everything[1] > everything[0]

    def test_naming_every_outcome_is_the_default(self) -> None:
        named = makeExploreContext(self.two_outcomes(), [OBJECTIVE, CONSTRAINED])
        default = makeExploreContext(self.two_outcomes())

        assert scores(named, 0.2, UNEXPLORED) == scores(default, 0.2, UNEXPLORED)

    def test_the_unit_of_one_outcome_does_not_make_it_count_more(self) -> None:
        # Measured in units a thousand times larger, the second outcome would dominate a sum of
        # variances. Each is divided by its own spread, so nothing changes.
        small = scores(makeExploreContext(self.two_outcomes(1.0)), 0.2, UNEXPLORED)
        large = scores(makeExploreContext(self.two_outcomes(1000.0)), 0.2, UNEXPLORED)

        assert large == pytest.approx(small, rel=1e-6)

    def test_the_order_of_the_outputs_of_the_surrogate_does_not_matter(self) -> None:
        forward = scores(makeExploreContext(self.two_outcomes()), 0.2, UNEXPLORED)
        backward = scores(
            makeExploreContext(self.two_outcomes(order=(CONSTRAINED, OBJECTIVE))),
            0.2,
            UNEXPLORED,
        )

        assert backward == pytest.approx(forward, rel=1e-9)

    def test_the_order_of_the_names_in_the_goal_does_not_matter(self) -> None:
        one = makeExploreContext(self.two_outcomes(), [OBJECTIVE, CONSTRAINED])
        other = makeExploreContext(self.two_outcomes(), [CONSTRAINED, OBJECTIVE])

        assert scores(one, 0.2, UNEXPLORED) == pytest.approx(scores(other, 0.2, UNEXPLORED))

    def test_the_value_of_several_outcomes_is_below_that_of_each_alone(self) -> None:
        # The variances are added, each in units of its own variation: there is more that the
        # model is unsure of, and the score, which is minus it, is lower.
        both = scores(makeExploreContext(self.two_outcomes()), UNEXPLORED)
        first = scores(makeExploreContext(self.two_outcomes(), [OBJECTIVE]), UNEXPLORED)
        second = scores(makeExploreContext(self.two_outcomes(), [CONSTRAINED]), UNEXPLORED)

        assert both[0] < first[0]
        assert both[0] < second[0]

    def test_a_goal_that_learns_one_outcome_of_a_pareto_toy_reads_it_by_name(self) -> None:
        surrogate = makeParetoSurrogate((F1, F2))
        learn_f2 = scores(makeExploreContext(surrogate, [F2]), 0.2, UNEXPLORED)
        learn_f1 = scores(makeExploreContext(surrogate, [F1]), 0.2, UNEXPLORED)

        assert learn_f1 != pytest.approx(learn_f2)


class TestTheMonteCarloPoints:
    """Roadmap P5-06, done when (b): the point count is configuration."""

    def test_the_function_averages_over_as_many_points_as_the_configuration_says(self) -> None:
        for n_points in (1, 16, 64, 512):
            function = build(explorer(), n_points=n_points)

            assert function.mc_points.shape == (n_points, 1)

    def test_the_default_is_the_default_of_the_configuration(self) -> None:
        assert build(explorer()).mc_points.shape == (512, 1)

    def test_a_count_that_the_configuration_gives_through_a_mapping_reaches_the_function(
        self,
    ) -> None:
        resolved = resolveComponent(
            acquisitionRegistry(), {"type": "neg_integrated_variance_v1", "n_points": 32}
        )
        function = resolved.entry.builder(resolved.config).build(explorer())  # type: ignore[attr-defined]

        assert function.mc_points.shape == (32, 1)

    def test_the_points_are_drawn_from_the_encoder_with_a_seed_of_the_generator(self) -> None:
        context = explorer()
        expected_seed = drawSeed(torchGenerator(7, RuntimeConfig()))

        function = build(context, n_points=64)

        assert torch.equal(
            function.mc_points, context.encoder.sampleEncoded(64, seed=expected_seed)
        )

    def test_the_points_cover_the_search_space(self) -> None:
        points = build(explorer(), n_points=64).mc_points.flatten().sort().values

        assert points.min() >= 0.0
        assert points.max() <= 1.0
        # Space-filling: no gap is much larger than the spacing of an even grid.
        assert (points[1:] - points[:-1]).max() < 4.0 / 64
        assert points.min() < 4.0 / 64
        assert points.max() > 1.0 - 4.0 / 64

    def test_the_points_have_the_precision_of_the_model(self) -> None:
        context = explorer()

        function = build(context)

        assert function.mc_points.dtype == context.surrogate.model.train_inputs[0].dtype

    def test_the_count_changes_the_estimate_and_not_its_meaning(self) -> None:
        places = (NEXT_TO_OBSERVATION, 0.45, UNEXPLORED)
        few = scores(explorer(), *places, n_points=8)
        many = scores(explorer(), *places, n_points=1024)

        assert few != many
        assert few == sorted(few)
        assert many == sorted(many)

    def test_one_point_is_enough_to_build_a_function(self) -> None:
        (value,) = scores(explorer(), UNEXPLORED, n_points=1)

        assert torch.isfinite(torch.tensor(value))


class TestPendingPoints:
    @staticmethod
    def pending(*places: float) -> torch.Tensor:
        return candidates(*places).reshape(len(places), 1)

    def test_the_pending_points_of_the_context_reach_the_function(self) -> None:
        pending = self.pending(0.5, 0.6)
        context = makeExploreContext(oneOutcome(), pending_points=pending)

        assert torch.equal(build(context).X_pending, pending)

    def test_a_pending_point_changes_the_value_of_every_candidate(self) -> None:
        places = (NEXT_TO_OBSERVATION, 0.45, UNEXPLORED, 0.95)
        surrogate = oneOutcome()

        without = scores(makeExploreContext(surrogate), *places)
        with_pending = scores(
            makeExploreContext(surrogate, pending_points=self.pending(UNEXPLORED)), *places
        )

        # The model is already conditioned on the pending point, so there is less left to be
        # unsure of, whichever candidate is added.
        assert all(a > b for a, b in zip(with_pending, without, strict=True))

    def test_a_candidate_at_a_pending_point_is_no_longer_the_best_place_to_measure(
        self,
    ) -> None:
        places = (0.5, 0.65, UNEXPLORED, 0.9)
        surrogate = oneOutcome()

        without = scores(makeExploreContext(surrogate), *places)
        with_pending = scores(
            makeExploreContext(surrogate, pending_points=self.pending(UNEXPLORED)), *places
        )

        assert places[without.index(max(without))] == UNEXPLORED
        assert places[with_pending.index(max(with_pending))] != UNEXPLORED

    def test_no_pending_point_is_the_same_as_an_empty_set(self) -> None:
        empty = torch.empty((0, 1), dtype=candidates(0.5).dtype)
        surrogate = oneOutcome()

        assert scores(
            makeExploreContext(surrogate, pending_points=empty), 0.5, UNEXPLORED
        ) == scores(makeExploreContext(surrogate), 0.5, UNEXPLORED)

    def test_a_pending_point_can_be_set_and_cleared_on_the_function_after_it_is_built(
        self,
    ) -> None:
        function = build(explorer())
        before = function(candidates(UNEXPLORED)).detach()

        function.set_X_pending(self.pending(UNEXPLORED))
        during = function(candidates(UNEXPLORED)).detach()
        function.set_X_pending(None)
        after = function(candidates(UNEXPLORED)).detach()

        assert during.item() > before.item()
        assert torch.equal(before, after)


class TestTheNativeFunction:
    def test_it_scores_a_batch_of_candidates_of_q_points(self) -> None:
        function = build(explorer())
        batch = torch.stack(
            [candidates(0.2, 0.5).reshape(2, 1), candidates(0.6, 0.9).reshape(2, 1)]
        )

        assert function(batch).shape == (2,)

    def test_a_batch_is_scored_as_a_whole(self) -> None:
        function = build(explorer())
        single = function(candidates(0.55)).detach()
        pair = function(candidates(0.55, 0.9).reshape(1, 2, 1)).detach()

        # A second place, far from the first, teaches the model more.
        assert pair.item() > single.item()

    def test_a_spread_out_batch_is_worth_more_than_the_same_place_twice(self) -> None:
        function = build(explorer())
        twice = function(candidates(0.55, 0.55).reshape(1, 2, 1)).detach()
        spread = function(candidates(0.55, 0.9).reshape(1, 2, 1)).detach()

        assert spread.item() > twice.item()

    def test_the_engines_function_gives_a_gradient_in_the_candidate(self) -> None:
        # Not declared (`acquisition:differentiable` is not in the contract of this entry), and
        # recorded here so that a change of the engine is noticed.
        function = build(explorer())
        candidate = candidates(0.5).clone().requires_grad_(True)

        function(candidate).sum().backward()

        assert candidate.grad is not None
        assert torch.isfinite(candidate.grad).all()
        # The gap is to the right of 0.5, so the value rises with x there.
        assert candidate.grad.item() > 0.0

    def test_the_model_is_not_changed_by_building_or_scoring(self) -> None:
        context = explorer()
        before = [
            parameter.detach().clone() for parameter in context.surrogate.model.parameters()
        ]
        train_before = context.surrogate.model.train_inputs[0].clone()

        scores(context, 0.2, UNEXPLORED)

        after = list(context.surrogate.model.parameters())
        assert all(torch.equal(a, b) for a, b in zip(before, after, strict=True))
        assert torch.equal(train_before, context.surrogate.model.train_inputs[0])


class TestThePointsAreDeterministic:
    def test_the_same_seed_gives_the_same_values(self) -> None:
        assert scores(
            makeExploreContext(oneOutcome(), seed=3),
            0.2,
            UNEXPLORED,
        ) == scores(
            makeExploreContext(oneOutcome(), seed=3),
            0.2,
            UNEXPLORED,
        )

    def test_another_seed_gives_other_points(self) -> None:
        surrogate = oneOutcome()
        one = build(makeExploreContext(surrogate, seed=3), n_points=16)
        other = build(makeExploreContext(surrogate, seed=4), n_points=16)

        assert not torch.equal(one.mc_points, other.mc_points)

    def test_the_points_are_the_same_for_every_call_of_one_function(self) -> None:
        function = build(explorer())

        assert torch.equal(function(candidates(0.2, 0.5)), function(candidates(0.2, 0.5)))

    def test_the_value_does_not_depend_on_the_sampler_of_the_engines_function(self) -> None:
        # The sampler only draws the values that the model is conditioned on, and the variance
        # of a Gaussian model after conditioning does not depend on them.
        function = build(explorer())
        before = function(candidates(0.2, UNEXPLORED)).detach()

        function.sampler = SobolQMCNormalSampler(torch.Size([1]), seed=12345)

        assert torch.equal(function(candidates(0.2, UNEXPLORED)).detach(), before)

    def test_the_global_generator_is_left_as_it_was_by_building_and_by_scoring(self) -> None:
        torch.manual_seed(5)
        before = torch.get_rng_state()

        scores(explorer(), 0.2, UNEXPLORED)

        assert torch.equal(before, torch.get_rng_state())

    def test_the_values_do_not_depend_on_the_global_generator(self) -> None:
        results = []
        for global_seed in range(5):
            torch.manual_seed(global_seed)
            results.append(scores(explorer(), 0.2, UNEXPLORED))

        assert all(result == results[0] for result in results[1:])

    def test_building_advances_the_generator_of_the_context(self) -> None:
        context = explorer()
        builder = NegIntegratedVarianceBuilder(NegIntegratedVarianceConfig(n_points=16))

        first = builder.build(context)
        second = builder.build(context)

        assert not torch.equal(first.mc_points, second.mc_points)


class TestWhatTheBuilderRefuses:
    def test_a_goal_that_is_not_an_explore_goal_is_refused(self) -> None:
        context = makeToyContext(
            makeToySurrogate(observed_x=OBSERVED), {"outcomes": {OBJECTIVE: "maximize"}}
        )

        with pytest.raises(ConfigError, match="not an explore goal") as caught:
            build(context)

        assert caught.value.field_path == "goal"
        assert caught.value.component == "acquisition 'neg_integrated_variance_v1'"

    def test_an_outcome_that_the_problem_does_not_have_is_refused(self) -> None:
        context = makeExploreContext(makeToySurrogate(observed_x=OBSERVED), ["unknown"])

        with pytest.raises(ConfigError, match="unknown") as caught:
            build(context)

        assert caught.value.field_path == "outcomes[0]"

    def test_an_outcome_that_the_surrogate_does_not_model_is_refused(self) -> None:
        # The problem has two outcomes, so the goal learns both by default, and the surrogate
        # was fitted on one.
        context = makeExploreContext(
            makeToySurrogate(observed_x=OBSERVED, objective_only=True)
        )

        with pytest.raises(AcquisitionContextError, match="'c'") as caught:
            build(context)

        assert caught.value.field_path == "goal.outcomes.c"

    def test_a_model_with_no_observed_point_is_refused(self) -> None:
        surrogate = oneOutcome()
        surrogate.model.train_inputs = (torch.empty((0, 1), dtype=torch.float64),)

        with pytest.raises(AcquisitionContextError, match="no observed point"):
            build(makeExploreContext(surrogate, [OBJECTIVE]))
