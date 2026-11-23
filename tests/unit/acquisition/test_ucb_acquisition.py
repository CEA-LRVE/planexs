"""Unit tests for `ucb_v1`, its builder and its configuration (roadmap task P5-04,
specification 5.8).

The acceptance test of the task: a higher `beta` raises the value of an uncertain point
relative to a certain one, for a minimizer and for a maximizer. The rest tests what the
registration says (needs, supports), the closed form against its definition, the weights of a
goal, the limits of an analytic one-point bound (no pending points, no batch, no outcome
constraints), and what the builder refuses.
"""

from __future__ import annotations

import itertools
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

from planexs.acquisition.AbstractAcquisitionBuilder import AbstractAcquisitionBuilder
from planexs.acquisition.AcquisitionBuildContext import AcquisitionBuildContext
from planexs.acquisition.AcquisitionContextError import AcquisitionContextError
from planexs.acquisition.acquisitionRegistry import acquisitionRegistry
from planexs.acquisition.UCBBuilder import UCBBuilder
from planexs.acquisition.UCBConfig import UCBConfig
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

#: The places where the toy model has data: the left of the space only, so that the model knows
#: the left well and the right not at all.
LEFT_OF_THE_SPACE = (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4)

#: A place among the observed points, where the model is certain, and a place far from all of
#: them, where it is not.
CERTAIN = 0.2
UNCERTAIN = 0.9


def gapSurrogate() -> Any:
    """The toy surrogate with data on the left of the space only."""
    return makeToySurrogate(observed_x=LEFT_OF_THE_SPACE)


def scores(
    goal: dict[str, Any],
    *places: float,
    surrogate: Any = None,
    pending: torch.Tensor | None = None,
    **settings: Any,
) -> list[float]:
    """The acquisition value of each place, for a goal, built the way a policy builds it.

    Args:
        goal: The fields of the optimize goal.
        places: The value of `x` of each single-point candidate.
        surrogate: The surrogate. `None` means the toy one with data on the left only.
        pending: The pending points, shape `(n, 1)`, or `None`.
        settings: Any setting of `UCBConfig` to replace.
    """
    model = gapSurrogate() if surrogate is None else surrogate
    function = UCBBuilder(UCBConfig(**settings)).build(
        makeToyContext(model, goal, pending_points=pending)
    )
    return [float(value) for value in function(candidates(*places)).detach()]


def predicted(surrogate: Any, *places: float) -> tuple[torch.Tensor, torch.Tensor]:
    """The mean and the standard deviation of each output of the model at each place, each of
    shape `(len(places), m)`, from the posterior of the model itself."""
    with torch.no_grad():
        posterior = surrogate.model.posterior(candidates(*places).squeeze(1))
    return posterior.mean, posterior.variance.sqrt()


class TestTheConfiguration:
    def test_the_defaults(self) -> None:
        config = UCBConfig()

        assert config.type == "ucb_v1"
        assert config.beta == 1.96

    @pytest.mark.parametrize("beta", [0.0, 0.5, 1.96, 100.0])
    def test_a_beta_of_zero_or_more_is_accepted(self, beta: float) -> None:
        assert UCBConfig(beta=beta).beta == beta

    @pytest.mark.parametrize("beta", [-0.1, -1.96])
    def test_a_negative_beta_is_refused(self, beta: float) -> None:
        with pytest.raises(ValidationError, match="greater than or equal to 0"):
            UCBConfig(beta=beta)

    @pytest.mark.parametrize("beta", [float("inf"), float("-inf"), float("nan")])
    def test_a_beta_that_is_not_finite_is_refused(self, beta: float) -> None:
        with pytest.raises(ValidationError):
            UCBConfig(beta=beta)

    @pytest.mark.parametrize("beta", ["high", None])
    def test_a_beta_that_is_not_a_number_is_refused(self, beta: Any) -> None:
        with pytest.raises(ValidationError):
            UCBConfig(beta=beta)

    def test_an_unknown_field_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="bta"):
            UCBConfig(bta=2.0)  # type: ignore[call-arg]

    def test_the_type_is_pinned_to_the_registered_name(self) -> None:
        with pytest.raises(ValidationError):
            UCBConfig(type="qlog_nei_v1")  # type: ignore[arg-type]

    def test_every_field_says_what_it_means(self) -> None:
        for name, field in UCBConfig.model_fields.items():
            assert field.description, name
            assert len(field.description) > 40, name

    def test_the_description_of_beta_gives_its_meaning_its_default_and_its_convention(
        self,
    ) -> None:
        description = UCBConfig.model_fields["beta"].description or ""

        assert "standard deviations" in description
        assert "1.96" in description
        assert "95%" in description
        assert "square" in description  # BoTorch's own parameter is the square of this one

    def test_beta_is_read_from_a_mapping(self) -> None:
        resolved = resolveComponent(acquisitionRegistry(), {"type": "ucb_v1", "beta": 3.0})

        assert isinstance(resolved.config, UCBConfig)
        assert resolved.config.beta == 3.0
        builder = resolved.entry.builder(resolved.config)
        assert isinstance(builder, UCBBuilder)
        assert builder.config is resolved.config


class TestTheRegistration:
    def test_it_is_registered_in_the_acquisition_registry(self) -> None:
        entry = acquisitionRegistry().lookup("ucb_v1")

        assert entry.builder == UCBBuilder.fromConfig
        assert entry.config_schema is UCBConfig
        assert issubclass(UCBBuilder, AbstractAcquisitionBuilder)
        assert entry.description.strip()

    def test_it_needs_a_marginal_normal_posterior_a_single_objective_and_an_optimize_goal(
        self,
    ) -> None:
        entry = acquisitionRegistry().lookup("ucb_v1")

        assert entry.needs == {
            "posterior:marginal_normal",
            "objectives:single",
            "goal:optimize",
        }

    def test_it_supports_gradients_and_nothing_else(self) -> None:
        # The tags are what the code does: no pending points, no outcome constraints.
        entry = acquisitionRegistry().lookup("ucb_v1")

        assert entry.supports == {"acquisition:differentiable"}

    def test_it_needs_no_reference_set(self) -> None:
        assert UCBBuilder(UCBConfig()).referencePointCount() == 0


class TestTheCompatibilityCheck:
    """The tags of the entry, read by the check that a study runs when it is built."""

    @staticmethod
    def participants(goal: Any, surrogate_tag: str = "posterior:marginal_normal") -> list[Any]:
        acquisition = acquisitionRegistry().lookup("ucb_v1")
        return [
            goal,
            DeclaredParticipant(
                label="acquisition 'ucb_v1'",
                needs=acquisition.needs,
                supports=acquisition.supports,
            ),
            # No surrogate is registered yet, so the one that supports what this entry needs of
            # it is declared here.
            DeclaredParticipant(
                label="surrogate 'toy_v1'", supports=frozenset({surrogate_tag})
            ),
        ]

    def test_a_goal_that_optimizes_one_outcome_finds_its_needs_met(self) -> None:
        checkCompatibility(
            self.participants(OptimizeGoal(OptimizeGoalConfig(outcomes={"a": "maximize"})))
        )

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

    def test_a_goal_with_outcome_constraints_is_refused_because_the_bound_cannot_enforce_them(
        self,
    ) -> None:
        goal = OptimizeGoal(
            OptimizeGoalConfig(
                outcomes={"a": "maximize"},
                outcome_constraints=[{"outcome": "b", "comparator": "<=", "bound": 1.0}],
            )
        )

        with pytest.raises(CompatibilityError, match="outcome_constraints:supported"):
            checkCompatibility(self.participants(goal))

    def test_a_surrogate_without_a_marginal_normal_posterior_is_refused(self) -> None:
        goal = OptimizeGoal(OptimizeGoalConfig(outcomes={"a": "maximize"}))

        with pytest.raises(CompatibilityError, match="posterior:marginal_normal"):
            checkCompatibility(self.participants(goal, surrogate_tag="posterior:gaussian"))


class TestBeta:
    """Roadmap P5-04, done when: a higher `beta` raises the value of an uncertain point
    relative to a certain one; minimize and maximize."""

    @pytest.fixture(autouse=True)
    def theModelIsMoreSureOfOnePlaceThanOfTheOther(self) -> None:
        _, std = predicted(gapSurrogate(), CERTAIN, UNCERTAIN)
        assert std[1, 0] > 10.0 * std[0, 0]

    @pytest.mark.parametrize("goal", [MAXIMIZE_Y, MINIMIZE_Y], ids=["maximize", "minimize"])
    def test_a_higher_beta_raises_an_uncertain_point_relative_to_a_certain_one(
        self, goal: dict[str, Any]
    ) -> None:
        betas = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]

        advantage = []
        for beta in betas:
            certain, uncertain = scores(goal, CERTAIN, UNCERTAIN, beta=beta)
            advantage.append(uncertain - certain)

        assert advantage == sorted(advantage)
        assert all(later > earlier for earlier, later in itertools.pairwise(advantage))

    @pytest.mark.parametrize("goal", [MAXIMIZE_Y, MINIMIZE_Y], ids=["maximize", "minimize"])
    def test_a_high_enough_beta_puts_the_uncertain_point_first(
        self, goal: dict[str, Any]
    ) -> None:
        # A minimizer starts with the certain point first (it is predicted lower), and a high
        # beta turns it round; a maximizer is won by the uncertain one at once.
        small = scores(goal, CERTAIN, UNCERTAIN, beta=0.0)
        large = scores(goal, CERTAIN, UNCERTAIN, beta=8.0)

        assert large[1] > large[0]
        if goal is MINIMIZE_Y:
            assert small[0] > small[1]

    @pytest.mark.parametrize("goal", [MAXIMIZE_Y, MINIMIZE_Y], ids=["maximize", "minimize"])
    def test_a_higher_beta_never_lowers_a_point(self, goal: dict[str, Any]) -> None:
        low = scores(goal, CERTAIN, UNCERTAIN, 0.5, beta=0.5)
        high = scores(goal, CERTAIN, UNCERTAIN, 0.5, beta=2.0)

        assert all(h > lo for h, lo in zip(high, low, strict=True))

    @pytest.mark.parametrize("goal", [MAXIMIZE_Y, MINIMIZE_Y], ids=["maximize", "minimize"])
    def test_beta_is_a_number_of_standard_deviations(self, goal: dict[str, Any]) -> None:
        _, std = predicted(gapSurrogate(), CERTAIN, UNCERTAIN)

        base = scores(goal, CERTAIN, UNCERTAIN, beta=1.0)
        more = scores(goal, CERTAIN, UNCERTAIN, beta=2.0)

        for value_base, value_more, sigma in zip(base, more, std[:, 0].tolist(), strict=True):
            assert value_more - value_base == pytest.approx(sigma, rel=1e-6)

    def test_the_native_function_holds_the_square_of_beta(self) -> None:
        # BoTorch weighs the standard deviation by the square root of its `beta`.
        context = makeToyContext(gapSurrogate(), MAXIMIZE_Y)

        function = UCBBuilder(UCBConfig(beta=3.0)).build(context)

        assert function.beta.item() == pytest.approx(9.0)  # type: ignore[union-attr]

    def test_the_default_beta_is_used_when_none_is_given(self) -> None:
        assert scores(MAXIMIZE_Y, CERTAIN, UNCERTAIN) == scores(
            MAXIMIZE_Y, CERTAIN, UNCERTAIN, beta=1.96
        )


class TestTheClosedForm:
    """The value against the definition of the bound, from the posterior of the model."""

    PLACES = (0.0, 0.2, 0.45, 0.9, 1.0)

    @pytest.mark.parametrize("beta", [0.0, 1.0, 1.96, 3.5])
    def test_a_maximizer_scores_the_mean_plus_beta_standard_deviations(
        self, beta: float
    ) -> None:
        mean, std = predicted(gapSurrogate(), *self.PLACES)

        expected = mean[:, 0] + beta * std[:, 0]

        assert scores(MAXIMIZE_Y, *self.PLACES, beta=beta) == pytest.approx(
            expected.tolist(), rel=1e-6, abs=1e-9
        )

    @pytest.mark.parametrize("beta", [0.0, 1.0, 1.96, 3.5])
    def test_a_minimizer_scores_minus_the_mean_plus_beta_standard_deviations(
        self, beta: float
    ) -> None:
        mean, std = predicted(gapSurrogate(), *self.PLACES)

        expected = -mean[:, 0] + beta * std[:, 0]

        assert scores(MINIMIZE_Y, *self.PLACES, beta=beta) == pytest.approx(
            expected.tolist(), rel=1e-6, abs=1e-9
        )

    def test_a_beta_of_zero_scores_the_mean_alone(self) -> None:
        mean, _ = predicted(gapSurrogate(), *self.PLACES)

        assert scores(MAXIMIZE_Y, *self.PLACES, beta=0.0) == pytest.approx(
            mean[:, 0].tolist(), rel=1e-6, abs=1e-9
        )
        assert scores(MINIMIZE_Y, *self.PLACES, beta=0.0) == pytest.approx(
            (-mean[:, 0]).tolist(), rel=1e-6, abs=1e-9
        )

    def test_a_goal_that_weighs_two_outcomes_scores_the_weighted_sum(self) -> None:
        # y maximized with weight 1 and c minimized with weight 10: the objective is y - 10 c.
        # The outputs of the model are independent, so the variance of the sum is the sum of
        # the weighted variances.
        goal = {
            "outcomes": {OBJECTIVE: "maximize", CONSTRAINED: "minimize"},
            "weights": {OBJECTIVE: 1.0, CONSTRAINED: 10.0},
        }
        surrogate = gapSurrogate()
        mean, std = predicted(surrogate, *self.PLACES)
        weights = torch.tensor([1.0, -10.0], dtype=mean.dtype)

        expected = (mean * weights).sum(-1) + 1.96 * ((std * weights) ** 2).sum(-1).sqrt()

        assert scores(goal, *self.PLACES, surrogate=surrogate) == pytest.approx(
            expected.tolist(), rel=1e-6, abs=1e-9
        )

    def test_the_weights_are_placed_by_name_and_not_by_position(self) -> None:
        # The goal names its outcomes in the opposite order to the outputs of the model.
        in_order = {
            "outcomes": {OBJECTIVE: "maximize", CONSTRAINED: "minimize"},
            "weights": {OBJECTIVE: 1.0, CONSTRAINED: 10.0},
        }
        reversed_order = {
            "outcomes": {CONSTRAINED: "minimize", OBJECTIVE: "maximize"},
            "weights": {CONSTRAINED: 10.0, OBJECTIVE: 1.0},
        }

        assert scores(in_order, *self.PLACES) == pytest.approx(
            scores(reversed_order, *self.PLACES), rel=1e-12
        )

    def test_an_output_that_the_goal_does_not_optimize_does_not_count(self) -> None:
        # c is modelled, and not optimized: the value is that of y alone.
        mean, std = predicted(gapSurrogate(), *self.PLACES)

        expected = mean[:, 0] + 1.96 * std[:, 0]

        assert scores(MAXIMIZE_Y, *self.PLACES) == pytest.approx(
            expected.tolist(), rel=1e-6, abs=1e-9
        )

    def test_the_weight_of_a_single_outcome_scales_the_score(self) -> None:
        mean, std = predicted(gapSurrogate(), *self.PLACES)
        goal = {"outcomes": {OBJECTIVE: "maximize"}, "weights": {OBJECTIVE: 3.0}}

        expected = 3.0 * (mean[:, 0] + 1.96 * std[:, 0])

        assert scores(goal, *self.PLACES) == pytest.approx(
            expected.tolist(), rel=1e-6, abs=1e-9
        )


class TestTheDirectionOfTheGoal:
    def test_with_no_uncertainty_weight_a_maximizer_prefers_the_higher_prediction(
        self,
    ) -> None:
        mean, _ = predicted(makeToySurrogate(), 0.1, 0.5)
        assert mean[1, 0] > mean[0, 0]
        surrogate = makeToySurrogate()

        low, high = scores(MAXIMIZE_Y, 0.1, 0.5, surrogate=surrogate, beta=0.0)

        assert high > low

    def test_with_no_uncertainty_weight_a_minimizer_prefers_the_lower_prediction(self) -> None:
        surrogate = makeToySurrogate()

        low_prediction, high_prediction = scores(
            MINIMIZE_Y, 0.1, 0.5, surrogate=surrogate, beta=0.0
        )

        assert low_prediction > high_prediction

    def test_the_two_directions_are_mirror_images_without_uncertainty(self) -> None:
        up = scores(MAXIMIZE_Y, 0.1, 0.3, 0.5, 0.8, beta=0.0)
        down = scores(MINIMIZE_Y, 0.1, 0.3, 0.5, 0.8, beta=0.0)

        assert down == pytest.approx([-value for value in up], rel=1e-9)

    def test_the_uncertainty_is_added_in_both_directions(self) -> None:
        # What a minimizer is optimistic about is a low outcome: uncertainty never costs.
        for goal in (MAXIMIZE_Y, MINIMIZE_Y):
            without = scores(goal, UNCERTAIN, beta=0.0)
            with_beta = scores(goal, UNCERTAIN, beta=2.0)

            assert with_beta[0] > without[0]


class TestWhatItDoesNotDo:
    def test_the_pending_points_do_not_change_the_value(self) -> None:
        # The bound at a point depends on the posterior at that point and on nothing else: the
        # entry does not claim `pending:supported`.
        pending = candidates(UNCERTAIN).reshape(1, 1)

        assert scores(MAXIMIZE_Y, CERTAIN, UNCERTAIN, pending=pending) == scores(
            MAXIMIZE_Y, CERTAIN, UNCERTAIN
        )

    def test_a_batch_of_more_than_one_point_is_refused_by_the_native_function(self) -> None:
        function = UCBBuilder(UCBConfig()).build(makeToyContext(gapSurrogate(), MAXIMIZE_Y))
        batch = candidates(CERTAIN, UNCERTAIN).reshape(1, 2, 1)

        with pytest.raises(AssertionError, match="q=1"):
            function(batch)

    def test_it_draws_no_random_number(self) -> None:
        generator = torchGenerator(7, RuntimeConfig())
        before = generator.get_state()
        context = AcquisitionBuildContext(
            surrogate=gapSurrogate(),
            goal=OptimizeGoal(OptimizeGoalConfig(**MAXIMIZE_Y)),
            generator=generator,
        )

        function = UCBBuilder(UCBConfig()).build(context)
        function(candidates(CERTAIN, UNCERTAIN))

        assert torch.equal(generator.get_state(), before)

    def test_the_same_context_gives_the_same_values(self) -> None:
        assert scores(MAXIMIZE_Y, CERTAIN, UNCERTAIN) == scores(MAXIMIZE_Y, CERTAIN, UNCERTAIN)

    def test_the_model_is_not_changed_by_building_or_scoring(self) -> None:
        surrogate = gapSurrogate()
        before = [parameter.detach().clone() for parameter in surrogate.model.parameters()]

        scores(MINIMIZE_Y, CERTAIN, UNCERTAIN, surrogate=surrogate)

        after = list(surrogate.model.parameters())
        assert all(torch.equal(a, b) for a, b in zip(before, after, strict=True))


class TestTheNativeFunction:
    def test_it_scores_a_batch_of_single_point_candidates(self) -> None:
        function = UCBBuilder(UCBConfig()).build(makeToyContext(gapSurrogate(), MAXIMIZE_Y))

        values = function(candidates(0.1, 0.5, 0.9))

        assert values.shape == (3,)

    def test_it_is_differentiable_in_the_candidate(self) -> None:
        function = UCBBuilder(UCBConfig()).build(makeToyContext(gapSurrogate(), MAXIMIZE_Y))
        candidate = candidates(0.6).clone().requires_grad_(True)

        function(candidate).sum().backward()

        assert candidate.grad is not None
        assert torch.isfinite(candidate.grad).all()
        assert candidate.grad.abs().sum() > 0.0

    def test_the_gradient_points_towards_the_uncertain_side_for_a_large_beta(self) -> None:
        # With data on the left only, the standard deviation grows to the right.
        function = UCBBuilder(UCBConfig(beta=50.0)).build(
            makeToyContext(gapSurrogate(), MAXIMIZE_Y)
        )
        candidate = candidates(0.6).clone().requires_grad_(True)

        function(candidate).sum().backward()

        assert candidate.grad is not None
        assert candidate.grad.item() > 0.0


class TestWhatTheBuilderRefuses:
    def test_a_goal_that_is_not_an_optimize_goal_is_refused(self) -> None:
        context = AcquisitionBuildContext(
            surrogate=gapSurrogate(),
            goal=LevelSetGoal.fromConfig(LevelSetGoalConfig(outcome=OBJECTIVE)),
            generator=torchGenerator(7, RuntimeConfig()),
        )

        with pytest.raises(ConfigError, match="not an optimize goal") as caught:
            UCBBuilder(UCBConfig()).build(context)

        assert caught.value.field_path == "goal"
        assert "ucb_v1" in str(caught.value)

    def test_a_goal_that_does_not_say_how_to_combine_its_outcomes_is_refused(self) -> None:
        goal = {"outcomes": {OBJECTIVE: "maximize", CONSTRAINED: "minimize"}}

        with pytest.raises(ConfigError, match="Pareto") as caught:
            scores(goal, CERTAIN)

        assert caught.value.field_path == "goal.weights"
        assert "ucb_v1" in str(caught.value)

    def test_an_objective_outcome_the_surrogate_does_not_model_is_refused(self) -> None:
        surrogate = makeToySurrogate(objective_only=True)
        goal = {"outcomes": {CONSTRAINED: "maximize"}}

        with pytest.raises(AcquisitionContextError, match="'c'") as caught:
            scores(goal, CERTAIN, surrogate=surrogate)

        assert caught.value.field_path == f"goal.outcomes.{CONSTRAINED}"
