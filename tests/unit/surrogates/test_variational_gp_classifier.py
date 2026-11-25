"""Unit tests for the `variational_gp_classifier_v1` surrogate (roadmap task P3-16,
specification 5.6, FR20, FR24).

The three "done when" items of the task are the first three classes: the accuracy of the
response-space predictions on held-out points of a synthetic problem with a known boundary, the
statement of the space of each prediction together with the conversion of a level, and the
reproducibility of a fit from its seed. The rest tests what the builder does with its data and
its configuration. Nothing here is mocked: a classifier is fitted for real, on data that a
seeded generator draws, which takes a fraction of a second.
"""

from __future__ import annotations

import math
import warnings
from typing import Any

import pytest
import torch
from botorch.exceptions.warnings import InputDataWarning
from botorch.models.approximate_gp import SingleTaskVariationalGP
from classifier_fakes import (
    CENTRE,
    FIT_SEED,
    N_TEST,
    N_TRAIN,
    NAME,
    OUTCOME,
    RADIUS,
    buildClassifier,
    builderFor,
    drawPoints,
    isInside,
    labelsOf,
    makeContext,
)
from gpytorch.kernels import MaternKernel, RBFKernel
from gpytorch.likelihoods import BernoulliLikelihood

import planexs.surrogates  # noqa: F401
from planexs import Real, SearchSpace
from planexs.core.CapabilityVocabulary import CapabilityVocabulary
from planexs.core.checkCompatibility import checkCompatibility
from planexs.core.ConfigError import ConfigError
from planexs.core.DeclaredParticipant import DeclaredParticipant
from planexs.core.KindTable import KindTable
from planexs.core.ModelNotReadyError import ModelNotReadyError
from planexs.core.RuntimeConfig import RuntimeConfig
from planexs.domain.Outcome import Outcome
from planexs.domain.Problem import Problem
from planexs.encoding.torch_runtime import torchGenerator
from planexs.surrogates.fitters.FitterError import FitterError
from planexs.surrogates.fitters.fitterRegistry import fitterRegistry
from planexs.surrogates.PredictionConfig import PredictionConfig
from planexs.surrogates.probitLink import probitLink
from planexs.surrogates.SamplingConfig import SamplingConfig
from planexs.surrogates.SurrogateContextError import SurrogateContextError
from planexs.surrogates.surrogateRegistry import surrogateRegistry

#: The tags that the classifier declares it supports (the task's, and the link tag it brings).
SUPPORTED = {
    "posterior:latent_gaussian",
    "posterior:marginal_normal",
    "outputs:single",
    "outcome_kind:binary",
    "link:probit",
}

#: The held-out points that every prediction of this module is made at. They are drawn from a
#: seed of their own, so that none of them is a training point.
HELD_OUT = drawPoints(N_TEST, 99)

#: The share of the held-out points whose response-space prediction falls on the correct side
#: of 0.5, below which a fit on the data of this module is not accepted. A fit of the default
#: classifier on the 60 training points measures 0.9225 on these 400 points (369 of them), and
#: so do the other kernels, an `n_inducing` of 20, three hundred iterations and other seeds: it
#: is the data and not the configuration that bounds it, because the 31 points that it misses
#: all lie within 0.08 of the boundary, which 60 labels cannot place any more closely. The
#: floor leaves a margin of about two points, eight of the 400, for a different platform's
#: arithmetic, and is far above the 0.5 of a surrogate that has learnt nothing (the two
#: classes are about as large as each other).
ACCURACY_FLOOR = 0.90


def accuracy(surrogate: Any, points: list[dict[str, float]] = HELD_OUT) -> float:
    """The share of `points` whose response-space probability is on the side of 0.5 that the
    known boundary says: above it inside the disc, below it outside.
    """
    prediction = surrogate.predict(points, PredictionConfig(through_link=True))
    correct = [
        (probability > 0.5) == isInside(point)
        for probability, point in zip(prediction.mean[OUTCOME], points, strict=True)
    ]
    return sum(correct) / len(points)


class TestItIsRegistered:
    def test_the_entry_declares_what_the_task_says_and_the_link(self) -> None:
        entry = surrogateRegistry().lookup(NAME)

        assert entry.supports == SUPPORTED
        assert entry.needs == frozenset()

    def test_it_has_a_one_line_description(self) -> None:
        description = surrogateRegistry().lookup(NAME).description

        assert description.strip()
        assert "\n" not in description

    def test_the_link_tag_is_in_the_vocabulary(self) -> None:
        vocabulary = CapabilityVocabulary.default()

        assert "link:probit" in vocabulary
        assert "link" in vocabulary.groups()
        assert "probit" in vocabulary.describe("link:probit")

    def test_a_study_with_the_default_fitter_is_compatible(self) -> None:
        # The fitter needs a latent Gaussian posterior, which the classifier supports.
        table = KindTable.default()
        surrogate = DeclaredParticipant.fromEntry(
            "surrogate", surrogateRegistry().lookup(NAME)
        )
        fitter = DeclaredParticipant.fromEntry(
            "fitter", fitterRegistry().lookup("variational_elbo_v1")
        )

        assert "posterior:latent_gaussian" in fitter.needs
        checkCompatibility([surrogate, fitter], table)

    def test_an_acquisition_that_needs_the_probit_link_is_met(self) -> None:
        table = KindTable.default()
        surrogate = DeclaredParticipant.fromEntry(
            "surrogate", surrogateRegistry().lookup(NAME)
        )
        acquisition = DeclaredParticipant(
            label="acquisition 'level_set_v1'",
            needs=frozenset({"link:probit", "posterior:marginal_normal"}),
        )

        checkCompatibility([acquisition, surrogate], table)

    def test_the_builder_is_made_from_its_configuration_by_the_registry(self) -> None:
        builder = builderFor({"min_observations": 5})

        assert builder.config.min_observations == 5
        assert type(builder).__name__ == "VariationalGPClassifierBuilder"


class TestPredictionsOnHeldOutPoints:
    """Done when: on a synthetic two-dimensional dataset labelled by a known boundary,
    response-space predictions on held-out points fall on the correct side of 0.5 with an
    accuracy stated in the test.
    """

    def test_the_held_out_points_are_not_training_points_and_have_both_classes(self) -> None:
        train = {tuple(point.values()) for point in drawPoints(N_TRAIN, 11)}
        inside = sum(isInside(point) for point in HELD_OUT)

        assert not train & {tuple(point.values()) for point in HELD_OUT}
        # The two classes are about as large as each other, so that "always the same class"
        # scores about one half, and the floor below is a real test.
        assert 0.35 * N_TEST < inside < 0.65 * N_TEST

    def test_the_default_classifier_is_correct_on_at_least_ninety_percent(self) -> None:
        assert accuracy(buildClassifier()) >= ACCURACY_FLOOR

    def test_every_mistake_is_close_to_the_boundary(self) -> None:
        # The misses are the points that 60 labels cannot place on one side or the other: the
        # farthest of them from the boundary is 0.08 away (0.4 is the radius of the disc).
        prediction = buildClassifier().predict(HELD_OUT, PredictionConfig(through_link=True))
        distances = [
            abs(math.hypot(point["a"] - CENTRE, point["b"] - CENTRE) - RADIUS)
            for point, probability in zip(HELD_OUT, prediction.mean[OUTCOME], strict=True)
            if (probability > 0.5) != isInside(point)
        ]

        assert distances
        assert max(distances) < 0.1

    @pytest.mark.parametrize(
        "configuration",
        [
            {"kernel": {"type": "rbf_v1"}},
            {"kernel": {"type": "matern_v1", "nu": 2.5}},
            {"n_inducing": 20},
            {"fitter": {"type": "variational_elbo_v1", "n_iterations": 300}},
        ],
        ids=["rbf", "matern", "twenty inducing points", "three hundred iterations"],
    )
    def test_so_is_every_other_configuration_of_it(
        self, configuration: dict[str, Any]
    ) -> None:
        assert accuracy(buildClassifier(configuration)) >= ACCURACY_FLOOR

    def test_a_fit_that_did_not_start_is_not_accurate_enough(self) -> None:
        # The floor is a real test: one iteration leaves the model close to its prior, which
        # is far from the boundary.
        one_step = {"fitter": {"type": "variational_elbo_v1", "n_iterations": 1}}

        assert accuracy(buildClassifier(one_step)) < ACCURACY_FLOOR

    def test_the_probability_is_on_the_side_that_the_latent_mean_is_on(self) -> None:
        surrogate = buildClassifier()

        latent = surrogate.predict(HELD_OUT).mean[OUTCOME]
        response = surrogate.predict(HELD_OUT, PredictionConfig(through_link=True)).mean[
            OUTCOME
        ]

        assert [value > 0.0 for value in latent] == [value > 0.5 for value in response]

    def test_the_model_is_least_sure_near_the_boundary(self) -> None:
        # The standard deviation of the probability is the epistemic uncertainty about it: it
        # is largest where the labels change, and the model has the least to go by.
        surrogate = buildClassifier()
        prediction = surrogate.predict(HELD_OUT, PredictionConfig(through_link=True))
        distance = [
            abs(math.hypot(point["a"] - CENTRE, point["b"] - CENTRE) - RADIUS)
            for point in HELD_OUT
        ]
        near = [
            std for std, d in zip(prediction.std[OUTCOME], distance, strict=True) if d < 0.05
        ]
        far = [
            std for std, d in zip(prediction.std[OUTCOME], distance, strict=True) if d > 0.2
        ]

        assert sum(near) / len(near) > 1.5 * sum(far) / len(far)

    def test_the_probabilities_and_their_spread_are_in_range(self) -> None:
        prediction = buildClassifier().predict(HELD_OUT, PredictionConfig(through_link=True))

        assert all(0.0 <= value <= 1.0 for value in prediction.mean[OUTCOME])
        assert all(0.0 < value <= 0.5 for value in prediction.std[OUTCOME])


class TestPredictionsSayWhichSpaceTheyDescribe:
    """Done when: predictions say which space they describe, and the level conversion is
    tested against the inverse link.
    """

    def test_by_default_a_prediction_describes_the_latent_function(self) -> None:
        prediction = buildClassifier().predict(HELD_OUT)

        assert prediction.space == "latent"
        assert prediction.includes_observation_noise is False

    def test_through_the_link_it_describes_the_response(self) -> None:
        prediction = buildClassifier().predict(HELD_OUT, PredictionConfig(through_link=True))

        assert prediction.space == "response"
        assert prediction.includes_observation_noise is False

    def test_the_latent_mean_is_a_value_of_the_function_and_not_a_probability(self) -> None:
        latent = buildClassifier().predict(HELD_OUT).mean[OUTCOME]

        assert min(latent) < -0.5
        assert max(latent) > 0.5

    def test_the_response_is_what_the_link_makes_of_the_latent_normal(self) -> None:
        surrogate = buildClassifier()
        latent = surrogate.predict(HELD_OUT)
        response = surrogate.predict(HELD_OUT, PredictionConfig(through_link=True))
        mean = torch.tensor(latent.mean[OUTCOME], dtype=torch.float64)
        variance = torch.tensor(latent.std[OUTCOME], dtype=torch.float64) ** 2

        expected, spread = probitLink().marginal(mean, variance)

        assert torch.allclose(
            torch.tensor(response.mean[OUTCOME], dtype=torch.float64), expected, atol=1e-12
        )
        assert torch.allclose(
            torch.tensor(response.std[OUTCOME], dtype=torch.float64), spread, atol=1e-12
        )

    def test_the_two_spaces_are_two_different_answers(self) -> None:
        surrogate = buildClassifier()

        latent = surrogate.predict(HELD_OUT)
        response = surrogate.predict(HELD_OUT, PredictionConfig(through_link=True))

        assert latent.mean != response.mean
        assert latent.std != response.std

    def test_a_noisy_prediction_is_refused_for_a_label(self) -> None:
        with pytest.raises(ConfigError, match="observation noise") as caught:
            buildClassifier().predict(
                HELD_OUT, PredictionConfig(include_observation_noise=True)
            )

        assert caught.value.field_path == "include_observation_noise"

    def test_the_surrogate_hands_out_the_link_it_was_built_with(self) -> None:
        link = buildClassifier().link

        assert link is not None
        assert link.name == "probit"

    def test_sampling_through_the_link_gives_probabilities(self) -> None:
        surrogate = buildClassifier()
        latent = surrogate.sample(
            HELD_OUT[:30],
            SamplingConfig(n_samples=64),
            generator=torchGenerator(1, RuntimeConfig()),
        )
        response = surrogate.sample(
            HELD_OUT[:30],
            SamplingConfig(n_samples=64, through_link=True),
            generator=torchGenerator(1, RuntimeConfig()),
        )

        assert latent.shape == response.shape == (64, 30, 1)
        assert float(latent.min()) < 0.0 < float(latent.max())
        assert float(response.min()) >= 0.0
        assert float(response.max()) <= 1.0
        # Same generator, same base samples: the response is the inverse link of the latent.
        assert torch.allclose(response, torch.special.ndtr(latent))

    def test_the_sampled_probability_averages_to_the_predicted_one(self) -> None:
        # The predictive probability is the mean of the response over the posterior of the
        # latent function, which the draws estimate (Monte Carlo error of 1 / sqrt(4096)).
        surrogate = buildClassifier()
        points = HELD_OUT[:20]
        draws = surrogate.sample(
            points,
            SamplingConfig(n_samples=4096, through_link=True),
            generator=torchGenerator(7, RuntimeConfig()),
        )
        predicted = surrogate.predict(points, PredictionConfig(through_link=True))

        # Each point is drawn jointly with the others, which does not change a marginal.
        assert torch.allclose(
            draws.mean(dim=0).squeeze(-1),
            torch.tensor(predicted.mean[OUTCOME], dtype=torch.float64),
            atol=0.03,
        )
        assert torch.allclose(
            draws.std(dim=0).squeeze(-1),
            torch.tensor(predicted.std[OUTCOME], dtype=torch.float64),
            atol=0.03,
        )


class TestAResponseLevelBecomesALatentThreshold:
    @pytest.mark.parametrize("level", [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99])
    def test_the_inverse_link_of_the_threshold_is_the_level(self, level: float) -> None:
        surrogate = buildClassifier()
        link = surrogate.link
        assert link is not None

        threshold = surrogate.latentLevel(level)

        assert float(
            link.inverse(torch.tensor(threshold, dtype=torch.float64))
        ) == pytest.approx(level, abs=1e-12)
        assert float(torch.special.ndtr(torch.tensor(threshold, dtype=torch.float64))) == (
            pytest.approx(level, abs=1e-12)
        )

    def test_one_half_is_the_zero_of_the_latent_function(self) -> None:
        assert buildClassifier().latentLevel(0.5) == 0.0

    def test_a_larger_level_is_a_larger_threshold(self) -> None:
        surrogate = buildClassifier()
        levels = [0.05, 0.3, 0.5, 0.7, 0.95]

        thresholds = [surrogate.latentLevel(level) for level in levels]

        assert thresholds == sorted(thresholds)
        assert len(set(thresholds)) == len(levels)

    def test_the_levels_on_either_side_of_one_half_are_symmetric(self) -> None:
        surrogate = buildClassifier()

        assert surrogate.latentLevel(0.2) == pytest.approx(-surrogate.latentLevel(0.8))

    def test_a_known_quantile_of_the_normal(self) -> None:
        assert buildClassifier().latentLevel(0.975) == pytest.approx(1.959963984540054)

    def test_the_boundary_of_the_classes_is_where_the_latent_mean_crosses_the_threshold(
        self,
    ) -> None:
        # A point is on the true side of 0.5 in the response space exactly when its latent
        # mean is above the threshold of 0.5, which is the point of converting a level.
        surrogate = buildClassifier()
        threshold = surrogate.latentLevel(0.5)

        latent = surrogate.predict(HELD_OUT).mean[OUTCOME]
        response = surrogate.predict(HELD_OUT, PredictionConfig(through_link=True)).mean[
            OUTCOME
        ]

        assert [value > threshold for value in latent] == [value > 0.5 for value in response]

    @pytest.mark.parametrize("level", [0.0, 1.0, -0.2, 1.5])
    def test_a_level_no_probability_has_is_refused(self, level: float) -> None:
        with pytest.raises(ConfigError, match="strictly between 0 and 1") as caught:
            buildClassifier().latentLevel(level)

        assert caught.value.field_path == "level"

    @pytest.mark.parametrize("level", [float("nan"), float("inf"), float("-inf"), True])
    def test_a_level_that_is_not_a_finite_number_is_refused(self, level: Any) -> None:
        with pytest.raises(ConfigError, match="finite number") as caught:
            buildClassifier().latentLevel(level)

        assert caught.value.field_path == "level"


class TestTrainingIsReproducibleFromASeed:
    def latentMean(self, seed: int) -> torch.Tensor:
        surrogate = buildClassifier(context=makeContext(seed=seed))
        return torch.tensor(surrogate.predict(HELD_OUT).mean[OUTCOME])

    def test_the_same_seed_gives_the_same_model(self) -> None:
        # Exactly, and not up to a tolerance: nothing but the generator is random.
        assert torch.equal(self.latentMean(FIT_SEED), self.latentMean(FIT_SEED))

    def test_the_same_seed_gives_the_same_probabilities_and_the_same_thresholds(self) -> None:
        first = buildClassifier(context=makeContext(seed=5))
        second = buildClassifier(context=makeContext(seed=5))
        options = PredictionConfig(through_link=True)

        assert first.predict(HELD_OUT, options) == second.predict(HELD_OUT, options)
        assert first.latentLevel(0.9) == second.latentLevel(0.9)

    def test_another_seed_gives_another_model(self) -> None:
        # The seed is what the fit draws from (the start of the variational distribution), so
        # that a different seed is not silently ignored: it moves the model, a little.
        difference = (self.latentMean(FIT_SEED) - self.latentMean(FIT_SEED + 1)).abs().max()

        assert float(difference) > 0.0

    def test_another_seed_changes_the_model_without_changing_what_it_has_learnt(self) -> None:
        assert (
            accuracy(buildClassifier(context=makeContext(seed=FIT_SEED + 1))) >= ACCURACY_FLOOR
        )

    def test_the_global_random_state_is_neither_read_nor_moved(self) -> None:
        # Specification 3.7: no global random state. The state is the same before and after,
        # and a draw from it that follows the build is the draw that would have been made.
        torch.manual_seed(1234)
        state = torch.get_rng_state()
        buildClassifier()
        after = torch.get_rng_state()

        assert torch.equal(state, after)

    def test_the_global_random_state_does_not_change_the_model(self) -> None:
        torch.manual_seed(1)
        first = self.latentMean(FIT_SEED)
        torch.manual_seed(2)
        _ = torch.rand(100)
        second = self.latentMean(FIT_SEED)

        assert torch.equal(first, second)

    def test_the_generator_of_the_context_is_the_one_that_advances(self) -> None:
        context = makeContext()
        before = context.generator.get_state()

        buildClassifier(context=context)

        assert not torch.equal(before, context.generator.get_state())


class TestWhatTheBuilderReturns:
    def test_the_model_is_a_botorch_variational_gp_with_a_bernoulli_likelihood(self) -> None:
        surrogate = buildClassifier()

        assert isinstance(surrogate.model, SingleTaskVariationalGP)
        assert isinstance(surrogate.model.likelihood, BernoulliLikelihood)
        assert surrogate.model.num_outputs == 1

    def test_the_model_is_left_in_evaluation_mode(self) -> None:
        surrogate = buildClassifier()

        assert surrogate.model.training is False
        assert surrogate.model.model.training is False
        assert surrogate.model.likelihood.training is False

    def test_the_surrogate_is_bound_to_the_one_outcome(self) -> None:
        surrogate = buildClassifier()

        assert surrogate.outcome_names == (OUTCOME,)

    def test_the_status_counts_the_observations(self) -> None:
        status = buildClassifier().status

        assert status.n_observations == N_TRAIN
        assert status.is_low_data is False
        assert status.warnings == ()

    def test_the_model_is_trained_on_the_labels_of_the_context(self) -> None:
        context = makeContext()

        model = buildClassifier(context=context).model

        assert torch.equal(model.model.train_targets, context.train_outcomes.squeeze(-1))
        assert torch.equal(model.model.train_inputs[0], context.train_inputs)

    def test_the_labels_are_not_standardized(self) -> None:
        model = buildClassifier().model

        assert set(model.model.train_targets.tolist()) == {0.0, 1.0}

    def test_building_does_not_change_the_context(self) -> None:
        context = makeContext()
        inputs = context.train_inputs.clone()
        outcomes = context.train_outcomes.clone()

        buildClassifier(context=context)

        assert torch.equal(context.train_inputs, inputs)
        assert torch.equal(context.train_outcomes, outcomes)

    def test_the_builder_can_build_twice_from_the_same_context_values(self) -> None:
        builder = builderFor()

        first = builder.build(makeContext())
        second = builder.build(makeContext())

        assert first.model is not second.model
        assert first.link == second.link

    def test_no_warning_escapes_for_labels_that_are_not_standardized(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            warnings.filterwarnings("ignore", category=FutureWarning)
            builderFor().build(makeContext())

    def test_the_precision_of_the_runtime_is_the_precision_of_the_model(self) -> None:
        runtime = RuntimeConfig(dtype="float32")
        context = makeContext(runtime=runtime)

        # BoTorch advises double precision, and the builder does not hide the advice from a
        # study that chose single precision: it only drops the advice about standardizing.
        with pytest.warns(InputDataWarning, match="float32"):
            surrogate = buildClassifier(context=context)
        draws = surrogate.sample(HELD_OUT[:3], generator=torchGenerator(1, runtime))

        assert draws.dtype == torch.float32
        assert (
            surrogate.model.model.variational_strategy.inducing_points.dtype == torch.float32
        )
        assert surrogate.model.model.train_inputs[0].dtype == torch.float32

    def test_the_default_precision_is_double(self) -> None:
        model = buildClassifier().model

        assert model.model.variational_strategy.inducing_points.dtype == torch.float64


class TestTheInducingPoints:
    def inducing(self, configuration: dict[str, Any] | None = None, n: int = N_TRAIN) -> Any:
        surrogate = buildClassifier(configuration, makeContext(n=n))
        return surrogate.model.model.variational_strategy.inducing_points

    def test_by_default_there_is_one_for_each_observation(self) -> None:
        context = makeContext()

        inducing = buildClassifier(
            context=context
        ).model.model.variational_strategy.inducing_points

        assert inducing.shape == context.train_inputs.shape

    def test_they_are_the_training_inputs_in_order_and_the_fit_leaves_them_there(self) -> None:
        # Exactly: the inducing points are not learnt. Learning them moved them by up to 1.5
        # encoded units on this data, for the same accuracy, and the task says they are the
        # training inputs.
        context = makeContext()

        inducing = buildClassifier(
            context=context
        ).model.model.variational_strategy.inducing_points

        assert torch.equal(inducing, context.train_inputs)

    def test_the_inducing_points_are_not_parameters_that_the_fit_learns(self) -> None:
        inducing = buildClassifier().model.model.variational_strategy.inducing_points

        assert inducing.requires_grad is False

    def test_a_chosen_subset_is_made_of_observed_inputs_and_stays_so(self) -> None:
        context = makeContext()

        inducing = buildClassifier(
            {"n_inducing": 10}, context
        ).model.model.variational_strategy.inducing_points

        rows = {tuple(row) for row in context.train_inputs.tolist()}
        assert {tuple(row) for row in inducing.tolist()} <= rows
        assert len({tuple(row) for row in inducing.tolist()}) == 10

    def test_they_are_a_copy_so_the_data_of_the_context_is_never_shared(self) -> None:
        context = makeContext()
        inputs = context.train_inputs.clone()

        surrogate = buildClassifier(context=context)
        inducing = surrogate.model.model.variational_strategy.inducing_points

        assert inducing.data_ptr() != context.train_inputs.data_ptr()
        assert torch.equal(context.train_inputs, inputs)

    @pytest.mark.parametrize("n_inducing", [1, 10, 30])
    def test_a_smaller_number_chooses_that_many(self, n_inducing: int) -> None:
        surrogate = buildClassifier({"n_inducing": n_inducing})

        assert surrogate.model.model.variational_strategy.inducing_points.shape == (
            n_inducing,
            2,
        )
        assert surrogate.status.warnings == ()

    def test_fewer_are_chosen_when_the_rest_add_nothing_and_the_status_says_so(self) -> None:
        # BoTorch's greedy selection stops when the observations left out are explained by
        # those chosen: with 59 of the 60 asked for, it ends with fewer.
        surrogate = buildClassifier({"n_inducing": N_TRAIN - 1})

        chosen = surrogate.model.model.variational_strategy.inducing_points.shape[0]
        assert 1 < chosen < N_TRAIN - 1
        (warning,) = surrogate.status.warnings
        assert (
            f"n_inducing is {N_TRAIN - 1} and {chosen} inducing points were chosen" in warning
        )

    def test_as_many_as_the_observations_is_the_default_and_says_nothing(self) -> None:
        surrogate = buildClassifier({"n_inducing": N_TRAIN})

        assert surrogate.model.model.variational_strategy.inducing_points.shape == (N_TRAIN, 2)
        assert surrogate.status.warnings == ()

    def test_more_than_the_observations_is_reduced_to_them_and_the_status_says_so(
        self,
    ) -> None:
        surrogate = buildClassifier({"n_inducing": 5 * N_TRAIN})

        assert surrogate.model.model.variational_strategy.inducing_points.shape == (N_TRAIN, 2)
        (warning,) = surrogate.status.warnings
        assert f"n_inducing is {5 * N_TRAIN}" in warning
        assert f"{N_TRAIN} observations" in warning

    def test_a_small_number_of_them_still_classifies(self) -> None:
        # Five points summarize a disc badly: 0.7525 on the held-out points, which is far
        # above the one half of a model that has learnt nothing, and below the full model.
        score = accuracy(buildClassifier({"n_inducing": 5}))

        assert 0.7 < score < ACCURACY_FLOOR


class TestTheNestedComponents:
    def kernelOf(self, configuration: dict[str, Any] | None = None) -> Any:
        return buildClassifier(configuration).model.model.covar_module

    def test_the_default_kernel_is_the_one_botorch_chooses(self) -> None:
        kernel = self.kernelOf()

        assert isinstance(kernel, RBFKernel)
        # One lengthscale for each encoded column, which is BoTorch's default for the model.
        assert kernel.lengthscale.shape == (1, 2)

    def test_the_kernel_that_is_named_is_the_kernel_that_is_used(self) -> None:
        kernel = self.kernelOf({"kernel": {"type": "matern_v1", "nu": 1.5}})

        assert isinstance(kernel, MaternKernel)
        assert kernel.nu == 1.5

    def test_the_settings_of_the_kernel_are_read(self) -> None:
        rough = self.kernelOf({"kernel": {"type": "matern_v1", "nu": 0.5}})
        smooth = self.kernelOf({"kernel": {"type": "matern_v1", "nu": 2.5}})

        assert (rough.nu, smooth.nu) == (0.5, 2.5)

    def test_the_fitter_that_is_named_is_the_one_that_fits(self) -> None:
        # `map_v1` fits exact models and refuses an approximate one, which shows that the name
        # in the configuration is looked up and not replaced by the default.
        with pytest.raises(FitterError, match="map_v1"):
            buildClassifier({"fitter": {"type": "map_v1"}})

    def test_the_settings_of_the_fitter_are_read(self) -> None:
        short = {"fitter": {"type": "variational_elbo_v1", "n_iterations": 1}}

        trained = torch.tensor(buildClassifier().predict(HELD_OUT).mean[OUTCOME])
        barely = torch.tensor(buildClassifier(short).predict(HELD_OUT).mean[OUTCOME])

        assert float((trained - barely).abs().max()) > 0.5

    def test_an_unknown_kernel_is_refused_when_the_configuration_is_read(self) -> None:
        with pytest.raises(ConfigError, match="no_such_kernel"):
            builderFor({"kernel": {"type": "no_such_kernel"}})


class TestWhenThereIsLittleData:
    def test_fewer_observations_than_the_minimum_is_not_a_fit(self) -> None:
        with pytest.raises(ModelNotReadyError) as caught:
            buildClassifier(context=makeContext(n=1))

        assert "at least 2 observation(s)" in caught.value.message
        assert "has 1" in caught.value.message
        assert caught.value.field_path == "min_observations"

    def test_a_lower_minimum_lets_one_observation_through_with_a_warning(self) -> None:
        surrogate = buildClassifier({"min_observations": 1}, makeContext(n=1))

        assert surrogate.status.n_observations == 1
        (warning,) = surrogate.status.warnings
        assert "Every label is" in warning

    def test_a_higher_minimum_asks_for_more(self) -> None:
        with pytest.raises(ModelNotReadyError, match="at least 40"):
            buildClassifier({"min_observations": 40}, makeContext(n=39))

    def test_exactly_the_minimum_fits(self) -> None:
        surrogate = buildClassifier({"min_observations": 40}, makeContext(n=40))

        assert surrogate.status.n_observations == 40

    @pytest.mark.parametrize(("n", "low"), [(5, True), (9, True), (10, False), (11, False)])
    def test_fewer_observations_than_ten_are_flagged_low_data(self, n: int, low: bool) -> None:
        assert buildClassifier(context=makeContext(n=n)).status.is_low_data is low

    def test_the_threshold_is_a_setting(self) -> None:
        context = makeContext(n=30)

        assert buildClassifier({"low_data_threshold": 31}, context).status.is_low_data is True
        assert buildClassifier({"low_data_threshold": 30}, context).status.is_low_data is False

    def test_low_data_changes_the_flag_and_not_the_fit(self) -> None:
        flagged = buildClassifier({"low_data_threshold": 1000})
        unflagged = buildClassifier({"low_data_threshold": 1})

        assert flagged.predict(HELD_OUT) == unflagged.predict(HELD_OUT)

    @pytest.mark.parametrize(("value", "word"), [(1.0, "true"), (0.0, "false")])
    def test_labels_that_are_all_the_same_are_warned_about(
        self, value: float, word: str
    ) -> None:
        context = makeContext(n=8, labels=torch.full((8, 1), value, dtype=torch.float64))

        (warning,) = buildClassifier(context=context).status.warnings

        assert warning.startswith(f"Every label is {word}")

    def test_labels_of_both_classes_are_not(self) -> None:
        labels = torch.tensor([[1.0], [0.0], [1.0], [0.0]], dtype=torch.float64)

        surrogate = buildClassifier(context=makeContext(n=4, labels=labels))

        assert surrogate.status.warnings == ()

    def test_a_model_of_one_class_still_predicts_probabilities(self) -> None:
        # No boundary can be read, but the model answers, and the probabilities are valid.
        context = makeContext(n=8, labels=torch.ones(8, 1, dtype=torch.float64))

        prediction = buildClassifier(context=context).predict(
            HELD_OUT[:10], PredictionConfig(through_link=True)
        )

        assert all(0.0 <= value <= 1.0 for value in prediction.mean[OUTCOME])


class TestWhatItRefuses:
    def test_a_problem_with_two_outcomes(self) -> None:
        problem = Problem(
            search_space=SearchSpace([Real("a", 0.0, 1.0), Real("b", 0.0, 1.0)]),
            outcomes=[Outcome(name="p", type="binary"), Outcome(name="q", type="binary")],
        )
        points = drawPoints(6, 3)
        labels = labelsOf(points).repeat(1, 2)

        with pytest.raises(ConfigError, match="models one binary outcome") as caught:
            builderFor().build(makeContext(n=6, problem=problem, labels=labels, points=points))

        assert caught.value.field_path == "problem.outcomes"
        assert "p (binary), q (binary)" in caught.value.message

    def test_a_continuous_outcome(self) -> None:
        problem = Problem(
            search_space=SearchSpace([Real("a", 0.0, 1.0), Real("b", 0.0, 1.0)]),
            outcomes=[Outcome(name="yield", type="continuous")],
        )

        with pytest.raises(ConfigError, match="continuous"):
            builderFor().build(makeContext(problem=problem))

    @pytest.mark.parametrize("bad", [0.5, 2.0, -1.0])
    def test_a_label_that_is_not_zero_or_one(self, bad: float) -> None:
        labels = labelsOf(drawPoints(N_TRAIN, 11))
        labels[3, 0] = bad

        with pytest.raises(SurrogateContextError, match="neither 0 nor 1") as caught:
            buildClassifier(context=makeContext(labels=labels))

        assert caught.value.field_path == "train_outcomes"

    def test_a_configuration_that_names_another_type_is_refused(self) -> None:
        with pytest.raises(ConfigError):
            builderFor({"type": "variational_gp_classifier_v2"})
