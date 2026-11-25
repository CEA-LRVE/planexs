"""`Surrogate` and the link of its outcomes (roadmap task P3-16, specification 5.6).

A surrogate whose model reads its outcomes from a latent function through a link says so in its
`BuiltSurrogate`, and the wrapper then answers in two spaces: the latent one, by default, and
the response one with `through_link`. These tests use the fake model of the other wrapper tests
and a logistic link made of two torch functions and a marginal of its own, so that what is
checked is the wrapper and not the probit: `test_variational_gp_classifier.py` checks the two
together, and `test_probit_link.py` the probit alone.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from posterior_fakes import FakeNativeModel, makeSurrogate
from surrogate_fakes import POINTS, makeProblem
from torch import Tensor

from planexs.core.ConfigError import ConfigError
from planexs.core.RuntimeConfig import RuntimeConfig
from planexs.encoding.torch_runtime import torchGenerator
from planexs.surrogates.PredictionConfig import PredictionConfig
from planexs.surrogates.ResponseLink import ResponseLink
from planexs.surrogates.SamplingConfig import SamplingConfig
from planexs.surrogates.SurrogateModelError import SurrogateModelError

LINKED = PredictionConfig(through_link=True)


def marginalOfTheFake(mean: Tensor, variance: Tensor) -> tuple[Tensor, Tensor]:
    """The marginal of the fake link: a probability, and a spread that is a quarter of the
    standard deviation of the latent value (a number that no other function gives).
    """
    return torch.sigmoid(mean), variance.sqrt() / 4


def fakeLink() -> ResponseLink:
    """A logistic link, whose marginal is not that of a logistic function of a normal."""
    return ResponseLink(
        name="logistic",
        inverse=torch.sigmoid,
        forward=torch.logit,
        marginal=marginalOfTheFake,
    )


def generator(seed: int = 5) -> torch.Generator:
    """A seeded generator on the device of the default runtime."""
    return torchGenerator(seed, RuntimeConfig())


class NegativeVarianceModel(FakeNativeModel):
    """A model whose variance comes out negative by a rounding error, as a real one can."""

    def posterior(self, X: Tensor, *args: Any, **kwargs: Any) -> Any:
        """The fake posterior with the sign of its variance flipped."""
        posterior = super().posterior(X, *args, **kwargs)
        posterior.variance = -posterior.variance
        return posterior


class RecordingLink:
    """The marginal of a link that records the variances it is given."""

    def __init__(self) -> None:
        self.variances: list[Tensor] = []

    def marginal(self, mean: Tensor, variance: Tensor) -> tuple[Tensor, Tensor]:
        """Records the variance, and answers as the fake link does."""
        self.variances.append(variance.clone())
        return marginalOfTheFake(mean, variance)


class TestTheLinkOfASurrogate:
    def test_a_surrogate_has_no_link_unless_it_was_built_with_one(self) -> None:
        assert makeSurrogate().link is None

    def test_it_hands_out_the_link_of_its_built_surrogate(self) -> None:
        link = fakeLink()

        assert makeSurrogate(link=link).link is link


class TestPredictingThroughTheLink:
    def test_the_default_prediction_is_of_the_latent_function_and_does_not_use_the_link(
        self,
    ) -> None:
        plain = makeSurrogate().predict(POINTS)
        linked = makeSurrogate(link=fakeLink()).predict(POINTS)

        assert linked == plain
        assert linked.space == "latent"

    def test_through_the_link_the_marginal_of_the_link_gives_the_mean_and_the_spread(
        self,
    ) -> None:
        model = FakeNativeModel(2)
        surrogate = makeSurrogate(model, link=fakeLink())
        latent = makeSurrogate().predict(POINTS)

        response = surrogate.predict(POINTS, LINKED)

        for name in surrogate.outcome_names:
            mean = torch.tensor(latent.mean[name], dtype=torch.float64)
            variance = torch.tensor(latent.std[name], dtype=torch.float64) ** 2
            assert response.mean[name] == pytest.approx(torch.sigmoid(mean).tolist())
            assert response.std[name] == pytest.approx((variance.sqrt() / 4).tolist())

    def test_the_prediction_says_that_it_is_in_the_response_space_without_noise(self) -> None:
        response = makeSurrogate(link=fakeLink()).predict(POINTS, LINKED)

        assert response.space == "response"
        assert response.includes_observation_noise is False
        assert response.outcomes == ("yield", "cost")

    def test_the_model_is_asked_for_the_latent_function_without_noise(self) -> None:
        model = FakeNativeModel(2)

        makeSurrogate(model, link=fakeLink()).predict(POINTS, LINKED)

        assert model.calls
        assert all(noisy is False for _, noisy in model.calls)

    def test_the_variance_that_reaches_the_link_is_never_negative(self) -> None:
        # A real model can answer a variance of -1e-12, which the link must not take the
        # square root of: the wrapper clamps it, as it does for the standard deviation.
        recording = RecordingLink()
        link = ResponseLink(
            name="recording",
            inverse=torch.sigmoid,
            forward=torch.logit,
            marginal=recording.marginal,
        )
        surrogate = makeSurrogate(NegativeVarianceModel(2), link=link)

        response = surrogate.predict(POINTS, LINKED)

        assert recording.variances
        assert all(float(variance.min()) >= 0.0 for variance in recording.variances)
        assert all(value == 0.0 for value in response.std["yield"])

    def test_a_model_that_answers_nan_is_still_caught_before_the_link(self) -> None:
        surrogate = makeSurrogate(FakeNativeModel(2, fault="nan"), link=fakeLink())

        with pytest.raises(SurrogateModelError, match="variance"):
            surrogate.predict(POINTS, LINKED)

    def test_the_link_is_applied_to_one_outcome_as_to_two(self) -> None:
        one = makeSurrogate(
            FakeNativeModel(1), problem=makeProblem(), outcome_names=["yield"], link=fakeLink()
        )

        response = one.predict(POINTS, LINKED)

        assert response.outcomes == ("yield",)
        assert all(0.0 < value < 1.0 for value in response.mean["yield"])

    def test_asking_for_it_does_not_change_the_surrogate(self) -> None:
        surrogate = makeSurrogate(link=fakeLink())
        before = surrogate.predict(POINTS)

        surrogate.predict(POINTS, LINKED)

        assert surrogate.predict(POINTS) == before


class TestAskingForALinkThatIsNotThere:
    def test_a_surrogate_without_a_link_refuses_to_predict_through_one(self) -> None:
        model = FakeNativeModel(2)

        with pytest.raises(ConfigError, match="has no link") as caught:
            makeSurrogate(model).predict(POINTS, LINKED)

        assert caught.value.field_path == "through_link"
        assert model.calls == []

    def test_and_to_sample_through_one(self) -> None:
        model = FakeNativeModel(2)

        with pytest.raises(ConfigError, match="has no link") as caught:
            makeSurrogate(model).sample(
                POINTS, SamplingConfig(through_link=True), generator=generator()
            )

        assert caught.value.field_path == "through_link"
        assert model.calls == []

    def test_the_link_and_the_observation_noise_cannot_be_asked_together_to_predict(
        self,
    ) -> None:
        model = FakeNativeModel(2)
        both = PredictionConfig(through_link=True, include_observation_noise=True)

        with pytest.raises(ConfigError, match="both asked for") as caught:
            makeSurrogate(model, link=fakeLink()).predict(POINTS, both)

        assert caught.value.field_path == "through_link"
        assert model.calls == []

    def test_nor_to_sample(self) -> None:
        model = FakeNativeModel(2)
        both = SamplingConfig(through_link=True, include_observation_noise=True)

        with pytest.raises(ConfigError, match="both asked for") as caught:
            makeSurrogate(model, link=fakeLink()).sample(POINTS, both, generator=generator())

        assert caught.value.field_path == "through_link"
        assert model.calls == []

    def test_noise_alone_is_still_a_measurement_of_a_surrogate_with_a_link(self) -> None:
        noisy = PredictionConfig(include_observation_noise=True)

        prediction = makeSurrogate(link=fakeLink()).predict(POINTS, noisy)

        assert prediction.space == "response"
        assert prediction.includes_observation_noise is True


class TestSamplingThroughTheLink:
    def test_each_draw_of_the_latent_function_is_read_through_the_inverse_link(self) -> None:
        surrogate = makeSurrogate(link=fakeLink())
        options = SamplingConfig(n_samples=32)
        linked = SamplingConfig(n_samples=32, through_link=True)

        latent = surrogate.sample(POINTS, options, generator=generator(9))
        response = surrogate.sample(POINTS, linked, generator=generator(9))

        assert response.shape == latent.shape == (32, len(POINTS), 2)
        assert torch.equal(response, torch.sigmoid(latent))

    def test_the_default_draws_are_the_latent_function(self) -> None:
        plain = makeSurrogate().sample(POINTS, generator=generator(2))
        linked = makeSurrogate(link=fakeLink()).sample(POINTS, generator=generator(2))

        assert torch.equal(plain, linked)

    def test_the_generator_advances_as_it_does_without_the_link(self) -> None:
        surrogate = makeSurrogate(link=fakeLink())
        one, two = generator(4), generator(4)

        surrogate.sample(POINTS, SamplingConfig(through_link=True), generator=one)
        surrogate.sample(POINTS, generator=two)

        assert torch.equal(one.get_state(), two.get_state())

    def test_the_precision_of_the_runtime_is_kept(self) -> None:
        runtime = RuntimeConfig(dtype="float32")
        surrogate = makeSurrogate(runtime=runtime, link=fakeLink())

        draws = surrogate.sample(
            POINTS, SamplingConfig(through_link=True), generator=torchGenerator(1, runtime)
        )

        assert draws.dtype == torch.float32


class TestAResponseLevelAsALatentThreshold:
    @pytest.mark.parametrize("level", [0.05, 0.5, 0.8])
    def test_it_is_the_forward_link_of_the_level(self, level: float) -> None:
        surrogate = makeSurrogate(link=fakeLink())

        threshold = surrogate.latentLevel(level)

        assert threshold == pytest.approx(float(torch.logit(torch.tensor(level).double())))
        assert float(torch.sigmoid(torch.tensor(threshold).double())) == pytest.approx(level)

    def test_the_threshold_is_a_float_and_not_a_tensor(self) -> None:
        threshold = makeSurrogate(link=fakeLink()).latentLevel(0.3)

        assert type(threshold) is float

    def test_a_surrogate_without_a_link_models_the_outcome_itself_so_the_level_is_kept(
        self,
    ) -> None:
        surrogate = makeSurrogate()

        assert surrogate.latentLevel(0.3) == 0.3
        assert surrogate.latentLevel(-12.5) == -12.5
        assert surrogate.latentLevel(250) == 250.0
        assert type(surrogate.latentLevel(250)) is float

    @pytest.mark.parametrize("level", [0.0, 1.0, -0.5, 7.0])
    def test_a_level_outside_the_open_unit_interval_is_refused_when_there_is_a_link(
        self, level: float
    ) -> None:
        with pytest.raises(ConfigError, match="strictly between 0 and 1") as caught:
            makeSurrogate(link=fakeLink()).latentLevel(level)

        assert caught.value.field_path == "level"
        assert "logistic" in caught.value.message

    @pytest.mark.parametrize("level", [float("nan"), float("inf"), float("-inf"), True, False])
    @pytest.mark.parametrize("linked", [True, False])
    def test_a_level_that_is_not_a_finite_number_is_refused_with_or_without_a_link(
        self, level: Any, linked: bool
    ) -> None:
        surrogate = makeSurrogate(link=fakeLink() if linked else None)

        with pytest.raises(ConfigError, match="finite number") as caught:
            surrogate.latentLevel(level)

        assert caught.value.field_path == "level"

    def test_the_conversion_is_made_in_the_precision_of_the_runtime(self) -> None:
        # The level is read in single precision, so it is the single-precision threshold that
        # comes back: it is close to the double-precision one and not equal to it.
        single = makeSurrogate(runtime=RuntimeConfig(dtype="float32"), link=fakeLink())
        double = makeSurrogate(link=fakeLink())

        assert single.latentLevel(0.3) == pytest.approx(double.latentLevel(0.3), abs=1e-6)
        assert single.latentLevel(0.3) != double.latentLevel(0.3)

    def test_the_conversion_does_not_ask_the_model_anything(self) -> None:
        model = FakeNativeModel(2)

        makeSurrogate(model, link=fakeLink()).latentLevel(0.4)

        assert model.calls == []
