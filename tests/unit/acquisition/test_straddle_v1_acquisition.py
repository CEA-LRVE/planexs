"""Unit tests for `straddle_v1`, its builder, its configuration and its native function
(roadmap task P5-09, specification 5.8, FR20).

The three "done when" items of the task are these groups. The closed form against the
definition of the straddle on a grid: `TestTheClosedFormAgainstItsDefinition`. The behaviour of
the score (highest at the level where the model is uncertain, lower far from the level and
where the model is certain, `beta` as the exchange rate between the two, no direction):
`TestWhereTheScoreIsHigh`, `TestBeta` and `TestItHasNoDirection`. The compatibility check with
the exact Gaussian process and with the classifier: `TestTheCompatibilityCheck`. The rest tests
the configuration, what the registration says (needs, supports), the level in the latent space
of the classifier, the outcome placed by name, the limits of an analytic one-point score, the
gradient, and what the builder refuses.

The definition used throughout is the depth of the level inside the band `mean +- beta * std`:
the smaller of the distances from the level to the two edges of the band. It is a different
expression from the closed form that the function computes, `beta * std - |mean - level|`, so
a test that equates them checks the algebra and not a copy of it.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from itertools import pairwise
from typing import Any

import pytest
import torch
from botorch.acquisition.analytic import AnalyticAcquisitionFunction
from botorch.exceptions.errors import UnsupportedError
from botorch.models.model import Model
from botorch.posteriors.gpytorch import GPyTorchPosterior
from gpytorch.distributions import MultivariateNormal
from linear_operator.operators import DiagLinearOperator
from pydantic import ValidationError
from toy_classifier_surrogates import LABEL, makeClassifierContext, makeClassifierSurrogate
from toy_surrogates import (
    CONSTRAINED,
    OBJECTIVE,
    candidates,
    makeSurrogateOf,
    makeToyContext,
    makeToySurrogate,
)

import planexs.acquisition as acquisition_package
from planexs.acquisition.AbstractAcquisitionBuilder import AbstractAcquisitionBuilder
from planexs.acquisition.AcquisitionBuildContext import AcquisitionBuildContext
from planexs.acquisition.AcquisitionContextError import AcquisitionContextError
from planexs.acquisition.acquisitionRegistry import acquisitionRegistry
from planexs.acquisition.StraddleBuilder import StraddleBuilder
from planexs.acquisition.StraddleConfig import StraddleConfig
from planexs.acquisition.StraddleFunction import StraddleFunction
from planexs.core.checkCompatibility import checkCompatibility
from planexs.core.CompatibilityError import CompatibilityError
from planexs.core.ConfigError import ConfigError
from planexs.core.DeclaredParticipant import DeclaredParticipant
from planexs.core.KindTable import KindTable
from planexs.core.resolveComponent import resolveComponent
from planexs.core.RuntimeConfig import RuntimeConfig
from planexs.encoding.torch_runtime import torchGenerator
from planexs.goals.LevelSetGoal import LevelSetGoal
from planexs.goals.LevelSetGoalConfig import LevelSetGoalConfig
from planexs.surrogates.surrogateRegistry import surrogateRegistry

#: Where the toy model has data: both ends of the space and none in the middle, so that the
#: model knows the ends well and the middle badly. The outcome is `y = x`, so its prediction
#: crosses the level 0.5 in the middle of the gap, where the model is least sure of it.
GAP_OF_THE_SPACE = (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95)

#: The level of the gap model, in the units of the outcome: the value of `y` at the middle of
#: the gap.
LEVEL = 0.5

#: A place on the level where the model is unsure, and a place the model has data at and that
#: is far from the level.
UNCERTAIN = 0.5
CERTAIN = 0.2

#: What the exact Gaussian process (roadmap task P3-14, not registered yet) declares that it
#: supports, as the task says.
EXACT_GP_TAGS = frozenset(
    {
        "posterior:gaussian",
        "posterior:marginal_normal",
        "outputs:single",
        "outputs:multi",
        "noise:known",
        "noise:inferred",
        "outcome_kind:continuous",
    }
)


class FixedMomentsModel(Model):
    """A model whose posterior at a point is read off the point itself: the mean is its first
    coordinate and the standard deviation its second, so that a test chooses the two numbers
    that the acquisition sees and computes what it expects by hand.
    """

    @property
    def num_outputs(self) -> int:
        return 1

    def posterior(  # type: ignore[override]
        self,
        X: torch.Tensor,
        output_indices: Any = None,
        observation_noise: bool = False,
        posterior_transform: Any = None,
        **kwargs: Any,
    ) -> Any:
        distribution = MultivariateNormal(X[..., 0], DiagLinearOperator(X[..., 1] ** 2))
        posterior = GPyTorchPosterior(distribution)
        return posterior if posterior_transform is None else posterior_transform(posterior)


def momentsOf(means: torch.Tensor, deviations: torch.Tensor) -> torch.Tensor:
    """Single-point candidates of `FixedMomentsModel`, shape `(n, 1, 2)`, one for each pair of
    a mean and a standard deviation."""
    return torch.stack([means, deviations], dim=-1).unsqueeze(-2)


def gapSurrogate() -> Any:
    """The toy surrogate of `y = x` with data at the two ends of the space only."""
    return makeSurrogateOf({OBJECTIVE: lambda x: x}, observed_x=GAP_OF_THE_SPACE)


def levelContext(
    surrogate: Any,
    level: float | None = LEVEL,
    *,
    outcome: str = OBJECTIVE,
    pending: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> AcquisitionBuildContext:
    """A build context for the level-set goal of `outcome` at `level`."""
    return AcquisitionBuildContext(
        surrogate=surrogate,
        goal=LevelSetGoal.fromConfig(LevelSetGoalConfig(outcome=outcome, level=level)),
        generator=torchGenerator(7, RuntimeConfig()) if generator is None else generator,
        pending_points=pending,
    )


def scores(
    places: torch.Tensor | tuple[float, ...],
    level: float | None = LEVEL,
    *,
    surrogate: Any = None,
    **settings: Any,
) -> torch.Tensor:
    """The straddle of each place, built the way a policy builds it: from the registry's
    configuration, the build context and the builder.

    Args:
        places: The values of `x`, one for each single-point candidate.
        level: The level of the goal.
        surrogate: The surrogate. `None` means the gap model.
        settings: Any setting of `StraddleConfig` to replace.
    """
    model = gapSurrogate() if surrogate is None else surrogate
    function = StraddleBuilder(StraddleConfig(**settings)).build(levelContext(model, level))
    return function(candidates(*places)).detach()


def predicted(surrogate: Any, *places: float, output: int = 0) -> tuple[Any, Any]:
    """The mean and the standard deviation of one output of the model at each place, as lists,
    from the posterior of the model itself."""
    with torch.no_grad():
        posterior = surrogate.model.posterior(candidates(*places).squeeze(1))
    return posterior.mean[:, output], posterior.variance.sqrt()[:, output]


def definition(
    mean: torch.Tensor, deviation: torch.Tensor, level: float, beta: float
) -> torch.Tensor:
    """The straddle as its definition says it: how deep the level lies inside the band
    `mean +- beta * deviation`, the smaller of its distances to the two edges."""
    lower = mean - beta * deviation
    upper = mean + beta * deviation
    return torch.minimum(level - lower, upper - level)


@pytest.fixture(scope="module")
def classifier() -> Any:
    """The classifier surrogate of the toy label, fitted once for the module."""
    return makeClassifierSurrogate()


class TestTheConfiguration:
    def test_the_defaults(self) -> None:
        config = StraddleConfig()

        assert config.type == "straddle_v1"
        assert config.beta == 1.96

    @pytest.mark.parametrize("beta", [0.0, 0.5, 1.96, 100.0])
    def test_a_beta_of_zero_or_more_is_accepted(self, beta: float) -> None:
        assert StraddleConfig(beta=beta).beta == beta

    @pytest.mark.parametrize("beta", [-0.1, -1.96])
    def test_a_negative_beta_is_refused(self, beta: float) -> None:
        with pytest.raises(ValidationError, match="greater than or equal to 0"):
            StraddleConfig(beta=beta)

    @pytest.mark.parametrize("beta", [float("inf"), float("-inf"), float("nan")])
    def test_a_beta_that_is_not_finite_is_refused(self, beta: float) -> None:
        with pytest.raises(ValidationError):
            StraddleConfig(beta=beta)

    @pytest.mark.parametrize("beta", ["high", None])
    def test_a_beta_that_is_not_a_number_is_refused(self, beta: Any) -> None:
        with pytest.raises(ValidationError):
            StraddleConfig(beta=beta)

    def test_an_unknown_field_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="bta"):
            StraddleConfig(bta=2.0)  # type: ignore[call-arg]

    def test_the_type_is_pinned_to_the_registered_name(self) -> None:
        with pytest.raises(ValidationError):
            StraddleConfig(type="ucb_v1")  # type: ignore[arg-type]

    def test_every_field_says_what_it_means(self) -> None:
        for name, field in StraddleConfig.model_fields.items():
            assert field.description, name
            assert len(field.description) > 40, name

    def test_the_description_of_beta_gives_its_meaning_its_default_and_its_convention(
        self,
    ) -> None:
        description = StraddleConfig.model_fields["beta"].description or ""

        assert "standard deviations" in description
        assert "distance" in description
        assert "level" in description
        assert "direction" in description
        assert "1.96" in description
        assert "95%" in description

    def test_beta_is_read_from_a_mapping(self) -> None:
        resolved = resolveComponent(
            acquisitionRegistry(), {"type": "straddle_v1", "beta": 3.0}
        )

        assert isinstance(resolved.config, StraddleConfig)
        assert resolved.config.beta == 3.0
        builder = resolved.entry.builder(resolved.config)
        assert isinstance(builder, StraddleBuilder)
        assert builder.config is resolved.config

    def test_a_stored_acquisition_with_only_its_type_has_the_default_beta(self) -> None:
        resolved = resolveComponent(acquisitionRegistry(), {"type": "straddle_v1"})

        assert isinstance(resolved.config, StraddleConfig)
        assert resolved.config.beta == StraddleConfig().beta


class TestTheRegistration:
    def test_it_is_registered_in_the_acquisition_registry(self) -> None:
        entry = acquisitionRegistry().lookup("straddle_v1")

        assert entry.builder == StraddleBuilder.fromConfig
        assert entry.config_schema is StraddleConfig
        assert issubclass(StraddleBuilder, AbstractAcquisitionBuilder)

    def test_it_has_a_one_line_description(self) -> None:
        description = acquisitionRegistry().lookup("straddle_v1").description

        assert description.strip()
        assert "\n" not in description

    def test_it_is_in_the_registry_that_the_contract_test_walks(self) -> None:
        # `test_registry_contract` checks every entry of the default table, so this entry
        # is checked there as soon as it is in the table.
        table = KindTable.default()

        assert "straddle_v1" in table.registry("acquisition").names()

    def test_its_module_is_imported_by_the_package(self) -> None:
        assert sys.modules["planexs.acquisition.StraddleBuilder"] is (
            acquisition_package.StraddleBuilder
        )
        assert "StraddleBuilder" in acquisition_package.__all__

    def test_it_needs_a_level_set_goal_and_a_marginal_normal_posterior(self) -> None:
        entry = acquisitionRegistry().lookup("straddle_v1")

        assert entry.needs == {"goal:level_set", "posterior:marginal_normal"}

    def test_it_supports_gradients_and_nothing_else(self) -> None:
        # The tags are what the code does: no pending points.
        entry = acquisitionRegistry().lookup("straddle_v1")

        assert entry.supports == {"acquisition:differentiable"}

    def test_it_needs_no_reference_set(self) -> None:
        assert StraddleBuilder(StraddleConfig()).referencePointCount() == 0
        assert StraddleBuilder.needs_reference_points is False


class TestTheClosedFormAgainstItsDefinition:
    """Done when: a numerical test of the closed form against a direct evaluation of its
    definition on a grid."""

    BETAS = (0.0, 0.5, 1.96, 5.0)
    LEVELS = (-1.0, 0.0, 0.7)

    @pytest.mark.parametrize("level", LEVELS)
    @pytest.mark.parametrize("beta", BETAS)
    def test_on_a_grid_of_means_and_deviations(self, beta: float, level: float) -> None:
        means = torch.linspace(-3.0, 3.0, 25, dtype=torch.float64)
        deviations = torch.linspace(0.01, 2.0, 15, dtype=torch.float64)
        grid_means, grid_deviations = torch.meshgrid(means, deviations, indexing="ij")
        function = StraddleFunction(FixedMomentsModel(), beta=beta, level=level)

        value = function(momentsOf(grid_means.reshape(-1), grid_deviations.reshape(-1)))

        expected = definition(grid_means.reshape(-1), grid_deviations.reshape(-1), level, beta)
        assert value.shape == (means.numel() * deviations.numel(),)
        torch.testing.assert_close(value, expected, rtol=0.0, atol=1e-12)

    @pytest.mark.parametrize("level", LEVELS)
    @pytest.mark.parametrize("beta", BETAS)
    def test_on_a_grid_of_the_search_space_of_a_real_model(
        self, beta: float, level: float
    ) -> None:
        # The means and the deviations are read from the posterior of the model itself, not
        # from the function, so that the whole path (the builder, the scalarizing transform,
        # the function) is checked against the definition.
        surrogate = gapSurrogate()
        places = tuple(torch.linspace(0.0, 1.0, 201).tolist())
        mean, deviation = predicted(surrogate, *places)

        value = scores(places, level, surrogate=surrogate, beta=beta)

        torch.testing.assert_close(
            value, definition(mean, deviation, level, beta), rtol=0.0, atol=1e-12
        )

    def test_a_prediction_on_the_level_scores_beta_deviations(self) -> None:
        deviations = torch.tensor([0.1, 0.5, 2.0], dtype=torch.float64)
        function = StraddleFunction(FixedMomentsModel(), beta=1.96, level=0.4)

        value = function(momentsOf(torch.full_like(deviations, 0.4), deviations))

        torch.testing.assert_close(value, 1.96 * deviations, rtol=0.0, atol=1e-12)

    def test_a_certain_model_scores_minus_the_distance(self) -> None:
        # A model that is nearly sure (a deviation of 1e-4) still counts for `beta` times that
        # deviation, which is what separates a place on the level from a distant one only
        # by the distance.
        means = torch.tensor([-1.0, 0.2, 0.9], dtype=torch.float64)
        deviation = 1e-4
        function = StraddleFunction(FixedMomentsModel(), beta=1.96, level=0.2)

        value = function(momentsOf(means, torch.full_like(means, deviation)))

        torch.testing.assert_close(
            value, -(means - 0.2).abs(), rtol=0.0, atol=1.96 * deviation + 1e-12
        )

    def test_the_two_forms_are_the_same_number(self) -> None:
        # The claim of the docstring, on a grid: the depth in the band is the closed form.
        means = torch.linspace(-2.0, 2.0, 41, dtype=torch.float64)
        deviations = torch.linspace(0.05, 1.5, 41, dtype=torch.float64)

        torch.testing.assert_close(
            definition(means, deviations, 0.3, 1.96),
            1.96 * deviations - (means - 0.3).abs(),
            rtol=0.0,
            atol=1e-12,
        )


class TestWhereTheScoreIsHigh:
    """Done when: the score is highest at the level where the model is uncertain, lower far
    from the level, and lower where the model is certain."""

    @pytest.fixture(autouse=True)
    def theModelIsUnsureInTheGapAndPredictsTheLevelThere(self) -> None:
        mean, deviation = predicted(gapSurrogate(), UNCERTAIN, CERTAIN)
        assert abs(mean[0] - LEVEL) < 0.01
        assert deviation[0] > 4.0 * deviation[1]

    def test_it_is_highest_at_the_level_where_the_model_is_uncertain(self) -> None:
        places = tuple(torch.linspace(0.0, 1.0, 201).tolist())

        value = scores(places)

        best = places[int(value.argmax())]
        assert best == pytest.approx(UNCERTAIN, abs=0.01)

    def test_it_beats_every_place_that_the_model_has_data_at(self) -> None:
        value = scores((UNCERTAIN, *GAP_OF_THE_SPACE))

        assert value[0] > value[1:].max()

    def test_it_is_lower_far_from_the_level(self) -> None:
        # The same place and the same uncertainty, and the level moves away from the
        # prediction: the score falls, by the distance.
        levels = (0.5, 0.6, 0.8, 1.5, 3.0)

        at_the_place = [float(scores((UNCERTAIN,), level)[0]) for level in levels]

        assert all(a > b for a, b in pairwise(at_the_place))
        mean, _ = predicted(gapSurrogate(), UNCERTAIN)
        assert at_the_place[0] - at_the_place[-1] == pytest.approx(
            abs(float(mean[0]) - 3.0) - abs(float(mean[0]) - 0.5), abs=1e-9
        )

    def test_it_falls_as_a_place_is_farther_from_the_level_at_equal_uncertainty(self) -> None:
        # Two places the model has data at, with the same deviation by symmetry of the data,
        # one on the level and one far from it.
        near, far = (float(v) for v in scores((0.3, 0.05), 0.3))

        _, deviation = predicted(gapSurrogate(), 0.3, 0.05)
        assert deviation[0] != pytest.approx(float(deviation[1]), rel=0.0, abs=0.0)
        assert near > far

    def test_it_is_lower_where_the_model_is_certain_when_both_are_on_the_level(self) -> None:
        # Each place is made to sit exactly on the level, by taking the level from its own
        # prediction, so that the distance is 0 and the score is `beta * deviation`.
        mean, deviation = predicted(gapSurrogate(), UNCERTAIN, CERTAIN)

        uncertain_on_level = float(scores((UNCERTAIN,), float(mean[0]))[0])
        certain_on_level = float(scores((CERTAIN,), float(mean[1]))[0])

        assert uncertain_on_level == pytest.approx(1.96 * float(deviation[0]), abs=1e-9)
        assert certain_on_level == pytest.approx(1.96 * float(deviation[1]), abs=1e-9)
        assert uncertain_on_level > certain_on_level

    def test_it_is_lower_where_the_model_is_certain_at_equal_distance_from_the_level(
        self,
    ) -> None:
        # The level is halfway between the two predictions, so that both are as far from it,
        # and only the uncertainty differs.
        mean, deviation = predicted(gapSurrogate(), UNCERTAIN, CERTAIN)
        halfway = float((mean[0] + mean[1]) / 2.0)

        uncertain, certain = (float(v) for v in scores((UNCERTAIN, CERTAIN), halfway))

        assert uncertain > certain
        assert uncertain - certain == pytest.approx(
            1.96 * float(deviation[0] - deviation[1]), abs=1e-9
        )


class TestBeta:
    """Done when: `beta` changes the trade-off. A place on the level that the model knows well
    (A) against a place off the level that it does not (B): at 0 only the distance counts and
    A wins, and as `beta` grows the uncertainty of B pays for its distance."""

    @pytest.fixture(autouse=True)
    def theCertainPlaceIsOnTheLevelAndTheUncertainOneIsNot(self) -> None:
        mean, deviation = predicted(gapSurrogate(), CERTAIN, UNCERTAIN)
        self.level = float(mean[0])  # place A is on the level
        self.distance = float(mean[1]) - self.level  # place B is this far from it
        self.gain = float(deviation[1] - deviation[0])  # and this much less certain
        assert self.distance > 0.1
        assert self.gain > 0.05

    def advantageOfTheUncertainPlace(self, beta: float) -> float:
        certain, uncertain = (
            float(v) for v in scores((CERTAIN, UNCERTAIN), self.level, beta=beta)
        )
        return uncertain - certain

    def test_with_no_weight_on_the_uncertainty_the_place_on_the_level_wins(self) -> None:
        certain, uncertain = (
            float(v) for v in scores((CERTAIN, UNCERTAIN), self.level, beta=0.0)
        )

        assert certain > uncertain
        assert certain == pytest.approx(0.0, abs=1e-9)
        assert uncertain == pytest.approx(-self.distance, abs=1e-9)

    def test_a_higher_beta_raises_the_uncertain_place_relative_to_the_certain_one(
        self,
    ) -> None:
        advantage = [
            self.advantageOfTheUncertainPlace(beta) for beta in (0.0, 0.5, 1, 2, 4, 8)
        ]

        assert all(later > earlier for earlier, later in pairwise(advantage))

    def test_the_two_swap_places_where_the_uncertainty_pays_for_the_distance(self) -> None:
        # B is worth visiting when `beta` standard deviations of the extra doubt exceed the
        # distance: beta > distance / gain. The ranking flips there and nowhere else.
        crossover = self.distance / self.gain

        assert self.advantageOfTheUncertainPlace(0.9 * crossover) < 0.0
        assert self.advantageOfTheUncertainPlace(1.1 * crossover) > 0.0
        assert self.advantageOfTheUncertainPlace(crossover) == pytest.approx(0.0, abs=1e-9)

    def test_beta_is_a_number_of_standard_deviations(self) -> None:
        _, deviation = predicted(gapSurrogate(), UNCERTAIN)

        base = float(scores((UNCERTAIN,), beta=1.0)[0])
        more = float(scores((UNCERTAIN,), beta=2.0)[0])

        assert more - base == pytest.approx(float(deviation[0]), rel=1e-9)

    def test_the_default_beta_is_used_when_none_is_given(self) -> None:
        assert torch.equal(
            scores((CERTAIN, UNCERTAIN)), scores((CERTAIN, UNCERTAIN), beta=1.96)
        )

    def test_a_beta_of_zero_scores_minus_the_distance_to_the_level(self) -> None:
        places = (0.0, 0.2, 0.45, 0.5, 0.9, 1.0)
        mean, _ = predicted(gapSurrogate(), *places)

        value = scores(places, beta=0.0)

        torch.testing.assert_close(value, -(mean - LEVEL).abs(), rtol=0.0, atol=1e-12)


class TestItHasNoDirection:
    """Done when: the score is symmetric around the level (it has no direction)."""

    @pytest.mark.parametrize("beta", [0.0, 1.96, 4.0])
    def test_a_mean_above_the_level_scores_as_one_below_it_by_the_same_distance(
        self, beta: float
    ) -> None:
        distances = torch.linspace(0.0, 3.0, 31, dtype=torch.float64)
        deviations = torch.linspace(0.05, 1.0, 31, dtype=torch.float64)
        function = StraddleFunction(FixedMomentsModel(), beta=beta, level=0.8)

        above = function(momentsOf(0.8 + distances, deviations))
        below = function(momentsOf(0.8 - distances, deviations))

        torch.testing.assert_close(above, below, rtol=0.0, atol=1e-12)

    def test_two_places_on_either_side_of_the_level_of_a_symmetric_model_score_alike(
        self,
    ) -> None:
        # The data are symmetric about the middle of the space and the outcome is linear, so
        # the model predicts 0.396 at 0.4 and 0.604 at 0.6, each 0.104 from the level 0.5,
        # with the same standard deviation.
        mean, deviation = predicted(gapSurrogate(), 0.4, 0.6)
        assert mean[0] < LEVEL < mean[1]

        below, above = (float(v) for v in scores((0.4, 0.6)))

        assert below == pytest.approx(above, abs=1e-9)
        assert below == pytest.approx(
            1.96 * float(deviation[0]) - abs(float(mean[0]) - LEVEL), abs=1e-9
        )

    def test_flipping_the_sign_of_the_outcome_and_of_the_level_changes_nothing(self) -> None:
        # The model of -y is the reflection of the model of y: its mean is negated and its
        # standard deviation is the same, so the level -0.5 is the reflection of the level
        # 0.5, and no place may score differently.
        places = tuple(torch.linspace(0.0, 1.0, 101).tolist())
        reflected = makeSurrogateOf({OBJECTIVE: lambda x: -x}, observed_x=GAP_OF_THE_SPACE)

        original = scores(places, LEVEL)
        mirrored = scores(places, -LEVEL, surrogate=reflected)

        torch.testing.assert_close(mirrored, original, rtol=0.0, atol=1e-10)


class TestTheLevelInTheLatentSpaceOfTheClassifier:
    """The level comes from the goal, and a surrogate that reads its outcome through a link
    converts it (roadmap task P3-16): the score is of the latent function."""

    PLACES = tuple(torch.linspace(0.0, 1.0, 101).tolist())

    def latentMoments(self, classifier: Any) -> tuple[Any, Any]:
        with torch.no_grad():
            posterior = classifier.model.posterior(candidates(*self.PLACES).squeeze(1))
        return posterior.mean[:, 0], posterior.variance.sqrt()[:, 0]

    def classifierScores(self, classifier: Any, level: float | None, **settings: Any) -> Any:
        function = StraddleBuilder(StraddleConfig(**settings)).build(
            makeClassifierContext(classifier, level)
        )
        return function(candidates(*self.PLACES)).detach()

    def test_the_default_level_of_a_binary_outcome_is_the_latent_threshold_zero(
        self, classifier: Any
    ) -> None:
        mean, deviation = self.latentMoments(classifier)

        value = self.classifierScores(classifier, None)

        torch.testing.assert_close(
            value, definition(mean, deviation, 0.0, 1.96), rtol=0.0, atol=1e-10
        )

    def test_a_level_of_a_half_is_the_same_as_the_default(self, classifier: Any) -> None:
        assert torch.equal(
            self.classifierScores(classifier, 0.5), self.classifierScores(classifier, None)
        )

    def test_a_level_that_is_a_probability_is_moved_through_the_link(
        self, classifier: Any
    ) -> None:
        # 0.8 is the response at the latent value of the 0.8 quantile of the standard normal.
        quantile = torch.tensor(0.8, dtype=torch.float64)
        threshold = float(torch.distributions.Normal(0.0, 1.0).icdf(quantile))
        mean, deviation = self.latentMoments(classifier)

        value = self.classifierScores(classifier, 0.8)

        assert threshold == pytest.approx(classifier.latentLevel(0.8), abs=1e-9)
        torch.testing.assert_close(
            value, definition(mean, deviation, threshold, 1.96), rtol=0.0, atol=1e-10
        )

    def test_the_score_is_highest_near_the_boundary_of_the_probability_a_half(
        self, classifier: Any
    ) -> None:
        mean, _ = self.latentMoments(classifier)
        crossing = self.PLACES[int(mean.abs().argmin())]

        value = self.classifierScores(classifier, 0.5)

        assert self.PLACES[int(value.argmax())] == pytest.approx(crossing, abs=0.1)
        assert crossing == pytest.approx(0.5, abs=0.1)

    def test_a_higher_probability_moves_the_best_place_towards_the_label_that_is_true(
        self, classifier: Any
    ) -> None:
        half = self.PLACES[int(self.classifierScores(classifier, 0.5).argmax())]
        high = self.PLACES[int(self.classifierScores(classifier, 0.8).argmax())]

        assert high > half + 0.05

    def test_the_score_is_not_symmetric_in_the_probability_but_in_the_latent_distance(
        self, classifier: Any
    ) -> None:
        # The levels 0.2 and 0.8 are the same distance from 0.5 in probability, and the
        # same distance on the latent scale, on either side of 0: the best places mirror.
        mean, deviation = self.latentMoments(classifier)
        low = self.classifierScores(classifier, 0.2)
        high = self.classifierScores(classifier, 0.8)
        threshold = float(classifier.latentLevel(0.8))

        assert float(classifier.latentLevel(0.2)) == pytest.approx(-threshold, abs=1e-9)
        torch.testing.assert_close(
            low, definition(mean, deviation, -threshold, 1.96), rtol=0.0, atol=1e-10
        )
        assert not torch.equal(low, high)

    def test_a_level_outside_zero_to_one_is_refused_for_a_binary_outcome(
        self, classifier: Any
    ) -> None:
        with pytest.raises(ConfigError) as caught:
            StraddleBuilder(StraddleConfig()).build(makeClassifierContext(classifier, 1.5))

        assert caught.value.field_path == "level"

    def test_the_level_of_the_exact_gp_is_used_as_it_is(self) -> None:
        # A surrogate with no link models the outcome itself: nothing is converted.
        assert gapSurrogate().latentLevel(0.7) == 0.7
        assert float(scores((UNCERTAIN,), 0.7)[0]) == pytest.approx(
            1.96 * predicted(gapSurrogate(), UNCERTAIN)[1][0].item()
            - abs(predicted(gapSurrogate(), UNCERTAIN)[0][0].item() - 0.7),
            abs=1e-12,
        )


class TestTheCompatibilityCheck:
    """Done when: it works with both the exact GP and the classifier surrogate in a
    compatibility check. The classifier is the registered entry; the exact GP is declared with
    the tags that its task (P3-14) lists, since it is not registered yet."""

    @staticmethod
    def acquisitionParticipant() -> DeclaredParticipant:
        return DeclaredParticipant.fromEntry(
            "acquisition", acquisitionRegistry().lookup("straddle_v1")
        )

    @staticmethod
    def classifierParticipant() -> DeclaredParticipant:
        return DeclaredParticipant.fromEntry(
            "surrogate", surrogateRegistry().lookup("variational_gp_classifier_v1")
        )

    @staticmethod
    def exactGPParticipant() -> DeclaredParticipant:
        return DeclaredParticipant(
            label="surrogate 'single_task_gp_v1'", supports=EXACT_GP_TAGS
        )

    @staticmethod
    def goal(outcome: str = OBJECTIVE) -> LevelSetGoal:
        return LevelSetGoal.fromConfig(LevelSetGoalConfig(outcome=outcome, level=LEVEL))

    def test_a_level_set_goal_with_the_exact_gp_finds_its_needs_met(self) -> None:
        checkCompatibility(
            [self.goal(), self.acquisitionParticipant(), self.exactGPParticipant()]
        )

    def test_a_level_set_goal_with_the_classifier_finds_its_needs_met(self) -> None:
        checkCompatibility(
            [self.goal(LABEL), self.acquisitionParticipant(), self.classifierParticipant()]
        )

    def test_both_surrogates_build_the_function_and_score_with_it(
        self, classifier: Any
    ) -> None:
        # The check says the pair can work; this runs the pair: the native function of each
        # is built from the real model, scores a batch to finite values, and has a gradient.
        gap = gapSurrogate()
        for surrogate, context in (
            (gap, levelContext(gap)),
            (classifier, makeClassifierContext(classifier)),
        ):
            function = StraddleBuilder(StraddleConfig()).build(context)
            batch = candidates(0.1, 0.5, 0.9).clone().requires_grad_(True)

            value = function(batch)
            value.sum().backward()

            assert isinstance(function, AnalyticAcquisitionFunction)
            assert value.shape == (3,)
            assert torch.isfinite(value).all()
            assert batch.grad is not None
            assert torch.isfinite(batch.grad).all()
            assert surrogate.model is function.model

    def test_an_optimize_goal_is_refused_for_its_kind(self) -> None:
        from planexs.goals.OptimizeGoal import OptimizeGoal
        from planexs.goals.OptimizeGoalConfig import OptimizeGoalConfig

        goal = OptimizeGoal(OptimizeGoalConfig(outcomes={OBJECTIVE: "maximize"}))

        with pytest.raises(CompatibilityError, match="goal:level_set"):
            checkCompatibility(
                [goal, self.acquisitionParticipant(), self.exactGPParticipant()]
            )

    def test_a_surrogate_without_a_marginal_normal_posterior_is_refused(self) -> None:
        gaussian_only = DeclaredParticipant(
            label="surrogate 'toy_v1'", supports=frozenset({"posterior:gaussian"})
        )

        with pytest.raises(CompatibilityError, match="posterior:marginal_normal"):
            checkCompatibility([self.goal(), self.acquisitionParticipant(), gaussian_only])


class TestWhatTheBuilderRefuses:
    def test_a_goal_that_is_not_a_level_set_goal_is_refused(self) -> None:
        context = makeToyContext(makeToySurrogate(), {"outcomes": {OBJECTIVE: "maximize"}})

        with pytest.raises(ConfigError, match="not a level-set goal") as caught:
            StraddleBuilder(StraddleConfig()).build(context)

        assert caught.value.field_path == "goal"
        assert caught.value.component == "acquisition 'straddle_v1'"

    def test_a_continuous_outcome_with_no_level_is_refused(self) -> None:
        with pytest.raises(ConfigError, match="needs a level") as caught:
            StraddleBuilder(StraddleConfig()).build(levelContext(gapSurrogate(), None))

        assert caught.value.field_path == "level"

    def test_an_outcome_that_the_problem_does_not_have_is_refused(self) -> None:
        with pytest.raises(ConfigError) as caught:
            StraddleBuilder(StraddleConfig()).build(
                levelContext(gapSurrogate(), LEVEL, outcome="z")
            )

        assert caught.value.field_path == "outcome"

    def test_an_outcome_that_the_surrogate_does_not_model_is_refused(self) -> None:
        # The problem has `c`, and this surrogate was fitted on `y` alone.
        surrogate = makeToySurrogate(objective_only=True)

        with pytest.raises(AcquisitionContextError, match="does not model") as caught:
            StraddleBuilder(StraddleConfig()).build(
                levelContext(surrogate, LEVEL, outcome=CONSTRAINED)
            )

        assert caught.value.field_path == "goal.outcome"
        assert "'y'" in str(caught.value)


class TestTheOutcomeIsPlacedByName:
    PLACES = (0.1, 0.35, 0.6, 0.85)

    def test_the_output_of_the_goal_is_the_one_that_is_scored(self) -> None:
        # Two outputs: the goal asks for the level of `c`, the second.
        surrogate = makeToySurrogate()
        mean, deviation = predicted(surrogate, *self.PLACES, output=1)

        value = scores(self.PLACES, 0.4, surrogate=surrogate)  # outcome is y: the first output
        value_c = (
            StraddleBuilder(StraddleConfig())
            .build(levelContext(surrogate, 0.4, outcome=CONSTRAINED))(candidates(*self.PLACES))
            .detach()
        )

        torch.testing.assert_close(
            value_c, definition(mean, deviation, 0.4, 1.96), atol=1e-10, rtol=0.0
        )
        assert not torch.allclose(value, value_c)

    def test_the_order_of_the_outputs_does_not_matter(self) -> None:
        sine: Callable[[torch.Tensor], torch.Tensor] = lambda x: torch.sin(3.0 * x)  # noqa: E731
        line: Callable[[torch.Tensor], torch.Tensor] = lambda x: x  # noqa: E731
        in_order = makeSurrogateOf({OBJECTIVE: sine, CONSTRAINED: line})
        reversed_order = makeSurrogateOf({CONSTRAINED: line, OBJECTIVE: sine})

        def valueOf(surrogate: Any) -> torch.Tensor:
            function = StraddleBuilder(StraddleConfig()).build(
                levelContext(surrogate, 0.4, outcome=CONSTRAINED)
            )
            return function(candidates(*self.PLACES)).detach()

        torch.testing.assert_close(
            valueOf(in_order), valueOf(reversed_order), atol=1e-10, rtol=0.0
        )


class TestWhatItDoesNotDo:
    def test_the_pending_points_do_not_change_the_value(self) -> None:
        # The score at a point depends on the posterior at that point and on nothing else:
        # the entry does not claim `pending:supported`.
        pending = candidates(UNCERTAIN).reshape(1, 1)
        places = (CERTAIN, UNCERTAIN)

        function = StraddleBuilder(StraddleConfig()).build(
            levelContext(gapSurrogate(), pending=pending)
        )

        assert torch.equal(function(candidates(*places)).detach(), scores(places))

    def test_the_native_function_refuses_pending_points(self) -> None:
        function = StraddleBuilder(StraddleConfig()).build(levelContext(gapSurrogate()))

        with pytest.raises(UnsupportedError):
            function.set_X_pending(candidates(UNCERTAIN).reshape(1, 1))

    def test_a_batch_of_more_than_one_point_is_refused_by_the_native_function(self) -> None:
        function = StraddleBuilder(StraddleConfig()).build(levelContext(gapSurrogate()))
        batch = candidates(CERTAIN, UNCERTAIN).reshape(1, 2, 1)

        with pytest.raises(AssertionError, match="q=1"):
            function(batch)

    def test_it_draws_no_random_number(self) -> None:
        generator = torchGenerator(7, RuntimeConfig())
        before = generator.get_state()

        function = StraddleBuilder(StraddleConfig()).build(
            levelContext(gapSurrogate(), generator=generator)
        )
        function(candidates(CERTAIN, UNCERTAIN))

        assert torch.equal(generator.get_state(), before)

    def test_the_same_context_gives_the_same_values(self) -> None:
        assert torch.equal(scores((CERTAIN, UNCERTAIN)), scores((CERTAIN, UNCERTAIN)))

    def test_the_model_is_not_changed_by_building_or_scoring(self) -> None:
        surrogate = gapSurrogate()
        before = [parameter.detach().clone() for parameter in surrogate.model.parameters()]

        scores((CERTAIN, UNCERTAIN), surrogate=surrogate)

        after = list(surrogate.model.parameters())
        assert all(torch.equal(a, b) for a, b in zip(before, after, strict=True))


class TestTheNativeFunction:
    def test_it_is_an_analytic_function_of_the_engine_that_holds_its_settings(self) -> None:
        function = StraddleBuilder(StraddleConfig(beta=3.0)).build(
            levelContext(gapSurrogate(), 0.7)
        )

        assert isinstance(function, StraddleFunction)
        assert isinstance(function, AnalyticAcquisitionFunction)
        assert function.beta == 3.0
        assert function.level == 0.7

    def test_it_scores_a_batch_of_single_point_candidates(self) -> None:
        function = StraddleBuilder(StraddleConfig()).build(levelContext(gapSurrogate()))

        assert function(candidates(0.1, 0.5, 0.9)).shape == (3,)

    @pytest.mark.parametrize("beta", [-0.5, float("nan"), float("inf")])
    def test_a_beta_that_is_not_allowed_is_refused_with_its_field_path(
        self, beta: float
    ) -> None:
        with pytest.raises(ConfigError) as caught:
            StraddleFunction(FixedMomentsModel(), beta=beta, level=0.0)

        assert caught.value.field_path == "beta"

    @pytest.mark.parametrize("level", [float("nan"), float("inf"), float("-inf")])
    def test_a_level_that_is_not_finite_is_refused_with_its_field_path(
        self, level: float
    ) -> None:
        with pytest.raises(ConfigError) as caught:
            StraddleFunction(FixedMomentsModel(), beta=1.96, level=level)

        assert caught.value.field_path == "level"


class TestTheGradient:
    """The tag `acquisition:differentiable`: the score has a gradient in the candidate, which
    is the one of its definition wherever the mean is not on the level."""

    def gradientAt(self, place: float, level: float, beta: float = 1.96) -> torch.Tensor:
        function = StraddleBuilder(StraddleConfig(beta=beta)).build(
            levelContext(gapSurrogate(), level)
        )
        candidate = candidates(place).clone().requires_grad_(True)
        function(candidate).sum().backward()
        assert candidate.grad is not None
        return candidate.grad.reshape(-1)

    @pytest.mark.parametrize("place", [0.1, 0.42, 0.58, 0.9])
    def test_it_is_the_gradient_of_the_definition(self, place: float) -> None:
        # beta * d(std) - sign(mean - level) * d(mean), each from the posterior of the model.
        surrogate = gapSurrogate()
        x = candidates(place).squeeze(1).clone().requires_grad_(True)
        posterior = surrogate.model.posterior(x)
        mean = posterior.mean[:, 0]
        deviation = posterior.variance.sqrt()[:, 0]
        (d_mean,) = torch.autograd.grad(mean.sum(), x, retain_graph=True)
        (d_deviation,) = torch.autograd.grad(deviation.sum(), x)

        expected = 1.96 * d_deviation - torch.sign(mean - LEVEL).detach() * d_mean

        torch.testing.assert_close(
            self.gradientAt(place, LEVEL), expected.reshape(-1), rtol=1e-8, atol=1e-10
        )

    def test_it_agrees_with_a_finite_difference_away_from_the_level(self) -> None:
        step = 1e-6
        place = 0.42

        gradient = float(self.gradientAt(place, LEVEL)[0])
        up, down = (float(v) for v in scores((place + step, place - step)))

        assert gradient == pytest.approx((up - down) / (2.0 * step), rel=1e-5)

    def test_it_points_towards_the_level_when_beta_is_zero(self) -> None:
        below = self.gradientAt(0.3, LEVEL, beta=0.0)  # the prediction is below the level
        above = self.gradientAt(0.7, LEVEL, beta=0.0)  # and above it

        assert below.item() > 0.0
        assert above.item() < 0.0

    def test_on_the_level_the_distance_has_a_kink_and_the_gradient_is_the_subgradient_zero(
        self,
    ) -> None:
        # The limit that the docstring states: where the mean equals the level, the derivative
        # of the distance is 0 in automatic differentiation, and the uncertainty keeps its own.
        # The fixed-moments model puts the mean in the first coordinate and the deviation in
        # the second, so the gradient is read off by coordinate.
        function = StraddleFunction(FixedMomentsModel(), beta=1.96, level=0.4)
        points = torch.tensor(
            [[[0.4, 0.5]], [[0.7, 0.5]], [[0.1, 0.5]]], dtype=torch.float64, requires_grad=True
        )

        function(points).sum().backward()

        assert points.grad is not None
        on_level, above, below = points.grad[:, 0, :]
        assert on_level[0].item() == 0.0  # the kink: the subgradient 0
        assert above[0].item() == -1.0  # above the level, a lower mean is better
        assert below[0].item() == 1.0  # below the level, a higher mean is better
        for row in (on_level, above, below):
            assert row[1].item() == pytest.approx(1.96)  # d(beta * std) / d(std)
