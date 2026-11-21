"""Unit tests for the `log_v1` outcome transform (roadmap task P3-12, specification 5.6,
3.6, 8).

As for `standardize_v1`: the transform alone on exact answers, the un-mapping of a posterior
with known parameters, and a Gaussian process on a toy function. The refusal of a value that
has no logarithm is tested here too: it is what `StrictlyPositiveLog` is for.
"""

from __future__ import annotations

import copy

import pytest
import torch
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Log
from botorch.posteriors.gpytorch import GPyTorchPosterior
from botorch.posteriors.transformed import TransformedPosterior
from gpytorch.distributions import MultivariateNormal
from outcome_transform_fakes import fitModel, toyData

import planexs.surrogates  # noqa: F401
from planexs.core.ConfigError import ConfigError
from planexs.core.resolveComponent import resolveComponent
from planexs.surrogates.outcome_transforms.LogOutcomeTransformBuilder import (
    LogOutcomeTransformBuilder,
)
from planexs.surrogates.outcome_transforms.LogOutcomeTransformConfig import (
    LogOutcomeTransformConfig,
)
from planexs.surrogates.outcome_transforms.OutcomeTransformError import OutcomeTransformError
from planexs.surrogates.outcome_transforms.outcomeTransformRegistry import (
    outcomeTransformRegistry,
)
from planexs.surrogates.outcome_transforms.StrictlyPositiveLog import StrictlyPositiveLog

DOUBLE = torch.float64

# BoTorch warns that the targets of a model are not standardized when they are on the raw
# scale, and the toy model of a logarithm is fitted on log(y), which is not standardized.
pytestmark = pytest.mark.filterwarnings("ignore::botorch.exceptions.warnings.InputDataWarning")


def build(m: int = 1) -> StrictlyPositiveLog:
    transform = LogOutcomeTransformBuilder(LogOutcomeTransformConfig()).build(m)
    assert isinstance(transform, StrictlyPositiveLog)
    return transform


class TestTheConfiguration:
    def test_it_has_no_setting_but_its_type(self) -> None:
        assert set(LogOutcomeTransformConfig.model_fields) == {"type"}

    def test_a_setting_that_does_not_exist_is_refused(self) -> None:
        # The engine's `Log` can pick its outputs, but cannot un-transform a posterior when it
        # does: `outputs` is not offered, and a misspelling of it is an error.
        with pytest.raises(ConfigError, match="outputs"):
            resolveComponent(outcomeTransformRegistry(), {"type": "log_v1", "outputs": [0]})

    def test_the_transform_does_not_depend_on_the_number_of_outcomes(self) -> None:
        for n_outcomes in (1, 2, 7):
            assert type(build(n_outcomes)) is StrictlyPositiveLog


class TestTheTransformAlone:
    def test_it_is_the_engine_log_with_one_more_rule(self) -> None:
        transform = build()

        assert isinstance(transform, Log)
        assert type(transform) is StrictlyPositiveLog

    def test_it_takes_the_natural_logarithm_of_every_outcome(self) -> None:
        y = torch.tensor([[1.0, 10.0], [torch.e, 100.0]], dtype=DOUBLE)

        mapped, variance = build(2)(y)

        assert variance is None
        assert torch.allclose(mapped, torch.log(y))
        assert torch.allclose(mapped[1, 0], torch.tensor(1.0, dtype=DOUBLE))

    def test_untransform_gives_back_what_forward_took(self) -> None:
        y = torch.tensor([[0.5, 10.0], [3.0, 1e6], [1e-3, 2.0]], dtype=DOUBLE)
        transform = build(2)

        mapped, _ = transform(y)
        back, _ = transform.untransform(mapped)

        assert torch.allclose(back, y, rtol=1e-12)

    def test_a_known_observation_noise_goes_through_the_delta_method(self) -> None:
        # Var[log y] is Var[y] / y^2, to first order: the engine's rule, which the subclass
        # leaves as it is.
        y = torch.tensor([[2.0], [4.0], [10.0]], dtype=DOUBLE)
        variance = torch.tensor([[0.4], [0.4], [0.4]], dtype=DOUBLE)

        _, mapped_variance = build()(y, variance)

        assert mapped_variance is not None
        assert torch.allclose(mapped_variance, variance / y**2)

    def test_it_has_no_statistics_to_learn_and_is_the_same_in_every_mode(self) -> None:
        y = torch.tensor([[2.0], [5.0]], dtype=DOUBLE)
        transform = build()

        in_training, _ = transform(y)
        transform.eval()
        in_evaluation, _ = transform(y)

        assert torch.equal(in_training, in_evaluation)

    def test_the_smallest_positive_numbers_are_still_numbers_with_a_logarithm(self) -> None:
        tiny = torch.finfo(DOUBLE).tiny
        y = torch.tensor([[tiny], [1.0], [1e300]], dtype=DOUBLE)

        mapped, _ = build()(y)

        assert bool(torch.isfinite(mapped).all())


class TestUnmappingAPosterior:
    def test_it_is_a_lognormal_posterior_with_the_lognormal_moments(self) -> None:
        # exp(N(mu, s2)) has mean exp(mu + s2 / 2) and variance (exp(s2) - 1) exp(2 mu + s2).
        transform = build()
        transform.eval()
        mu = torch.tensor([-1.0, 0.0, 0.5, 2.0], dtype=DOUBLE)
        s2 = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=DOUBLE)
        posterior = GPyTorchPosterior(MultivariateNormal(mu, torch.diag(s2)))

        brought_back = transform.untransform_posterior(posterior)

        assert isinstance(brought_back, TransformedPosterior)
        assert torch.allclose(brought_back.mean.squeeze(-1), torch.exp(mu + s2 / 2))
        assert torch.allclose(
            brought_back.variance.squeeze(-1), (torch.exp(s2) - 1) * torch.exp(2 * mu + s2)
        )

    def test_its_samples_are_positive(self) -> None:
        # Drawn from base samples, as `Surrogate.sample` draws them.
        transform = build()
        transform.eval()
        mu = torch.tensor([-3.0, 0.0, 3.0], dtype=DOUBLE)
        posterior = transform.untransform_posterior(
            GPyTorchPosterior(MultivariateNormal(mu, torch.eye(3, dtype=DOUBLE)))
        )
        shape = torch.Size([200])
        generator = torch.Generator().manual_seed(0)
        base = torch.randn(
            torch.Size([*shape, *posterior.base_sample_shape]), generator=generator
        )

        samples = posterior.rsample_from_base_samples(shape, base.to(DOUBLE))

        assert samples.shape == (200, 3, 1)
        assert bool((samples > 0).all())


class TestAValueThatHasNoLogarithm:
    """It is refused with a named error, before it can become a `-inf` or a `nan` in a fit."""

    @pytest.mark.parametrize(
        "bad", [0.0, -1.0, -1e-300, float("inf"), float("-inf"), float("nan")]
    )
    def test_it_is_refused(self, bad: float) -> None:
        y = torch.tensor([[1.0], [bad], [3.0]], dtype=DOUBLE)

        with pytest.raises(OutcomeTransformError):
            build()(y)

    def test_the_error_names_the_transform_the_data_and_a_way_out(self) -> None:
        y = torch.tensor([[1.0, 5.0], [-0.5, 6.0], [3.0, 0.0]], dtype=DOUBLE)

        with pytest.raises(OutcomeTransformError) as caught:
            build(2)(y)

        error = caught.value
        assert "strictly positive" in error.message
        assert "2 value(s)" in error.message
        assert "[0, 1]" in error.message  # both columns have one
        assert "-0.5" in error.message  # one of the values, to find it
        assert error.component == "outcome_transform 'log_v1'"
        assert error.field_path == "train_outcomes"
        assert "standardize_v1" in (error.suggestion or "")

    def test_only_the_columns_that_are_at_fault_are_named(self) -> None:
        y = torch.tensor([[1.0, 5.0], [2.0, 6.0], [3.0, -7.0]], dtype=DOUBLE)

        with pytest.raises(OutcomeTransformError, match=r"column\(s\) \[1\]"):
            build(2)(y)

    def test_it_is_refused_in_evaluation_mode_too(self) -> None:
        transform = build()
        transform.eval()

        with pytest.raises(OutcomeTransformError):
            transform(torch.tensor([[0.0]], dtype=DOUBLE))

    def test_a_batch_of_targets_is_checked_whole(self) -> None:
        y = torch.ones(3, 4, 2, dtype=DOUBLE)
        y[2, 1, 1] = -1.0

        with pytest.raises(OutcomeTransformError, match=r"1 value\(s\).*column\(s\) \[1\]"):
            build(2)(y)

    def test_a_model_is_not_built_on_a_zero(self) -> None:
        train_x, train_y, _, _ = toyData()
        train_y = train_y.clone()
        train_y[3] = 0.0

        with pytest.raises(OutcomeTransformError, match="strictly positive"):
            SingleTaskGP(train_x, train_y, outcome_transform=build())

    def test_the_engine_log_alone_lets_a_zero_through_as_minus_infinity(self) -> None:
        # What `StrictlyPositiveLog` exists for. If this fails after an upgrade of BoTorch,
        # the engine refuses the value by itself and the subclass can be dropped
        # (specification 3.5: planexs adds to the engine only what it lacks).
        train_x, train_y, _, _ = toyData()
        train_y = train_y.clone()
        train_y[3] = 0.0

        model = SingleTaskGP(train_x, train_y, outcome_transform=Log())

        assert bool(torch.isneginf(model.train_targets).any())


class TestSubsetting:
    def test_a_subset_of_the_outputs_keeps_the_rule(self) -> None:
        # BoTorch rebuilds the transform with `self.__class__`, so the subclass survives.
        subset = build(3).subset_output([0, 2])

        assert type(subset) is StrictlyPositiveLog
        with pytest.raises(OutcomeTransformError):
            subset(torch.tensor([[1.0, -1.0]], dtype=DOUBLE))

    def test_a_copy_keeps_the_rule(self) -> None:
        # A model is copied by `deepcopy` when it is conditioned on new data, among others.
        duplicate = copy.deepcopy(build())

        assert type(duplicate) is StrictlyPositiveLog
        with pytest.raises(OutcomeTransformError):
            duplicate(torch.tensor([[0.0]], dtype=DOUBLE))


class TestOnAToyModel:
    """A Gaussian process fitted to the logarithm of a positive toy function."""

    # Measured on this toy: the posterior mean is within 0.04 of the observations at the
    # training inputs, whose standard deviation is 2.6. The tolerance is 10% of that
    # deviation, as for `standardize_v1`: a failure to un-map would be off by an order of
    # magnitude (the model sees logarithms of about two, the outcome is about ten).
    TOLERANCE = 0.25

    def test_the_model_is_fitted_to_the_logarithm_of_the_outcome(self) -> None:
        train_x, train_y, _, _ = toyData()

        model = fitModel(build(), train_x, train_y)

        assert torch.allclose(model.train_targets, torch.log(train_y.squeeze(-1)))

    def test_the_predictions_are_in_the_units_of_the_outcome(self) -> None:
        train_x, train_y, _, _ = toyData()
        model = fitModel(build(), train_x, train_y)

        with torch.no_grad():
            posterior = model.posterior(train_x)

        assert isinstance(posterior, TransformedPosterior)
        assert float((posterior.mean - train_y).abs().max()) < self.TOLERANCE

    def test_the_predictions_are_positive_even_far_from_the_data(self) -> None:
        # A log-normal outcome is positive everywhere: this is what the transform is for.
        train_x, train_y, _, _ = toyData()
        model = fitModel(build(), train_x, train_y)
        far = torch.tensor([[-5.0, 7.0], [10.0, -10.0], [0.5, 0.5]], dtype=DOUBLE)

        with torch.no_grad():
            posterior = model.posterior(far)

        assert bool((posterior.mean > 0).all())
        assert bool((posterior.variance > 0).all())

    def test_the_uncertainty_grows_away_from_the_data(self) -> None:
        train_x, train_y, _, _ = toyData()
        model = fitModel(build(), train_x, train_y)
        near, far = train_x[:1], torch.tensor([[4.0, 4.0]], dtype=DOUBLE)

        with torch.no_grad():
            variance_near = model.posterior(near).variance
            variance_far = model.posterior(far).variance

        assert float(variance_far) > float(variance_near)
