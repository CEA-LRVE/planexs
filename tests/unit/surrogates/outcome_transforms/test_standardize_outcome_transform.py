"""Unit tests for the `standardize_v1` outcome transform (roadmap task P3-12, specification
5.6, 3.4).

Three levels, as the card asks ("maps and un-maps predictions consistently on a toy model"):
the transform alone, on tensors whose answer is known exactly; the un-mapping of a posterior
whose parameters are known exactly; and a Gaussian process fitted to a toy function.
"""

from __future__ import annotations

import pytest
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.posteriors.gpytorch import GPyTorchPosterior
from gpytorch.distributions import MultivariateNormal
from gpytorch.mlls import ExactMarginalLogLikelihood
from outcome_transform_fakes import fitModel, toyData
from pydantic import ValidationError

import planexs.surrogates  # noqa: F401
from planexs.core.ConfigError import ConfigError
from planexs.core.resolveComponent import resolveComponent
from planexs.surrogates.outcome_transforms.outcomeTransformRegistry import (
    outcomeTransformRegistry,
)
from planexs.surrogates.outcome_transforms.StandardizeOutcomeTransformBuilder import (
    StandardizeOutcomeTransformBuilder,
)
from planexs.surrogates.outcome_transforms.StandardizeOutcomeTransformConfig import (
    StandardizeOutcomeTransformConfig,
)

DOUBLE = torch.float64


def build(m: int = 1, **settings: float) -> Standardize:
    transform = StandardizeOutcomeTransformBuilder(
        StandardizeOutcomeTransformConfig(**settings)
    ).build(m)
    assert isinstance(transform, Standardize)
    return transform


class TestTheConfiguration:
    def test_the_floor_defaults_to_the_one_of_botorch_and_of_the_legacy_surrogate(
        self,
    ) -> None:
        # The default is what keeps a study that sets nothing standardizing as it always did:
        # it floors and scales the same outcomes as BoTorch's own `Standardize`. One outcome
        # has a deviation of about 1.6e-9 and one of about 1.6e-8, which put the floor between
        # them, where 1e-8 is.
        assert StandardizeOutcomeTransformConfig().min_std == 1e-8
        y = torch.stack(
            [torch.arange(5, dtype=DOUBLE) * 1e-9, torch.arange(5, dtype=DOUBLE) * 1e-8],
            dim=-1,
        )
        ours, engine = build(2), Standardize(m=2)

        ours(y)
        engine(y)

        assert torch.equal(ours.stdvs, engine.stdvs)
        assert ours.stdvs[0, 0] == 1.0  # under the floor: only centred
        assert ours.stdvs[0, 1] != 1.0  # over it: scaled

    @pytest.mark.parametrize("bad", [0.0, -1e-8, float("nan"), float("inf"), float("-inf")])
    def test_a_floor_that_is_not_a_positive_finite_number_is_refused_with_its_path(
        self, bad: float
    ) -> None:
        with pytest.raises(ConfigError) as caught:
            resolveComponent(
                outcomeTransformRegistry(), {"type": "standardize_v1", "min_std": bad}
            )

        assert caught.value.field_path == "min_std"

    def test_a_zero_floor_is_refused_because_it_would_divide_a_constant_outcome_by_zero(
        self,
    ) -> None:
        with pytest.raises(ValidationError, match="greater than 0"):
            StandardizeOutcomeTransformConfig(min_std=0.0)

    def test_the_floor_decides_which_outcomes_are_scaled(self) -> None:
        # The deviation of this outcome is 0.5: above a floor of 0.25, below a floor of 1.
        y = torch.tensor([[0.0], [0.5], [1.0]], dtype=DOUBLE)
        below, above = build(min_std=0.25), build(min_std=1.0)

        below(y)
        above(y)

        assert float(below.stdvs) == pytest.approx(0.5)
        assert float(above.stdvs) == 1.0

    def test_an_unknown_field_is_refused(self) -> None:
        with pytest.raises(ConfigError, match="min_stdv"):
            resolveComponent(
                outcomeTransformRegistry(), {"type": "standardize_v1", "min_stdv": 1e-8}
            )


class TestTheTransformAlone:
    def test_it_is_a_botorch_standardize_with_one_statistic_per_outcome(self) -> None:
        transform = build(3)

        assert type(transform) is Standardize
        assert transform.means.shape == (1, 3) and transform.stdvs.shape == (1, 3)

    def test_it_centres_and_scales_each_outcome_with_its_own_mean_and_deviation(self) -> None:
        y = torch.tensor(
            [[1.0, 100.0], [2.0, 300.0], [3.0, 500.0], [6.0, 900.0]], dtype=DOUBLE
        )
        transform = build(2)

        mapped, _ = transform(y)

        assert torch.allclose(mapped.mean(dim=0), torch.zeros(2, dtype=DOUBLE), atol=1e-12)
        assert torch.allclose(mapped.std(dim=0), torch.ones(2, dtype=DOUBLE), atol=1e-12)
        assert torch.allclose(transform.means.squeeze(0), y.mean(dim=0))
        assert torch.allclose(transform.stdvs.squeeze(0), y.std(dim=0))

    def test_untransform_gives_back_what_forward_took(self) -> None:
        y = torch.tensor(
            [[1.0, 100.0], [2.0, 300.0], [3.0, 500.0], [6.0, 900.0]], dtype=DOUBLE
        )
        transform = build(2)

        mapped, _ = transform(y)
        back, _ = transform.untransform(mapped)

        assert torch.allclose(back, y, atol=1e-12)

    def test_a_known_observation_noise_is_scaled_with_the_outcome(self) -> None:
        y = torch.tensor([[1.0], [2.0], [3.0], [6.0]], dtype=DOUBLE)
        variance = torch.full_like(y, 0.5)
        transform = build()

        _, mapped_variance = transform(y, variance)

        assert mapped_variance is not None
        assert torch.allclose(mapped_variance, variance / transform.stdvs**2)

    def test_the_statistics_are_those_of_the_first_data_it_was_applied_to(self) -> None:
        transform = build()
        first = torch.tensor([[1.0], [2.0], [3.0]], dtype=DOUBLE)
        second = torch.tensor([[100.0], [200.0], [900.0]], dtype=DOUBLE)

        transform(first)
        statistics = (transform.means.clone(), transform.stdvs.clone())
        transform.eval()  # in evaluation mode it applies what it learned, and learns no more
        transform(second)

        assert torch.equal(transform.means, statistics[0])
        assert torch.equal(transform.stdvs, statistics[1])


class TestTheFloor:
    """`min_std` is the smallest deviation an outcome is divided by."""

    def test_a_constant_outcome_is_only_centred_and_never_divided_by_zero(self) -> None:
        y = torch.full((5, 1), 7.0, dtype=DOUBLE)
        transform = build()

        mapped, _ = transform(y)

        assert torch.equal(mapped, torch.zeros_like(y))
        assert torch.equal(transform.stdvs, torch.ones_like(transform.stdvs))

    def test_an_outcome_whose_spread_is_under_the_default_floor_is_not_scaled(self) -> None:
        # An outcome really measured at the 1e-9 scale: its genuine deviation (about 3e-9) is
        # below the default floor of 1e-8, so it is only centred. This is why the floor is a
        # setting.
        y = torch.arange(5, dtype=DOUBLE).unsqueeze(-1) * 1e-9
        transform = build()

        mapped, _ = transform(y)

        assert torch.equal(transform.stdvs, torch.ones_like(transform.stdvs))
        assert torch.allclose(mapped, y - y.mean())  # centred, not scaled

    def test_a_lower_floor_lets_that_outcome_be_scaled(self) -> None:
        y = torch.arange(5, dtype=DOUBLE).unsqueeze(-1) * 1e-9
        transform = build(min_std=1e-12)

        mapped, _ = transform(y)

        assert torch.allclose(transform.stdvs, y.std().reshape(1, 1))
        assert torch.allclose(mapped.std(), torch.tensor(1.0, dtype=DOUBLE))

    def test_the_floor_is_applied_to_each_outcome_on_its_own(self) -> None:
        y = torch.stack(
            [torch.arange(5, dtype=DOUBLE) * 1e-9, torch.arange(5, dtype=DOUBLE)], dim=-1
        )
        transform = build(2)

        transform(y)

        # The tiny outcome keeps a scale of one, the ordinary one is scaled by its deviation.
        assert transform.stdvs[0, 0] == 1.0
        assert torch.isclose(transform.stdvs[0, 1], y[:, 1].std())


class TestUnmappingAPosterior:
    """The posterior of the model, in the units of the outcome, from the one in the units the
    model was fitted in: the mean maps back affinely and the variance by the scale squared.
    """

    def test_the_mean_and_the_variance_are_brought_back_exactly(self) -> None:
        y = torch.tensor([[10.0], [14.0], [9.0], [20.0]], dtype=DOUBLE)
        transform = build()
        transform(y)
        transform.eval()
        mean = torch.tensor([-1.0, 0.0, 0.5, 2.0], dtype=DOUBLE)
        covariance = torch.diag(torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=DOUBLE))
        posterior = GPyTorchPosterior(MultivariateNormal(mean, covariance))

        brought_back = transform.untransform_posterior(posterior)

        scale, shift = transform.stdvs.squeeze(), transform.means.squeeze()
        assert torch.allclose(brought_back.mean.squeeze(-1), mean * scale + shift)
        assert torch.allclose(
            brought_back.variance.squeeze(-1),
            torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=DOUBLE) * scale**2,
        )


class TestOnAToyModel:
    """A Gaussian process fitted to targets of order ten, in the units of the outcome."""

    # Measured on this toy: the posterior mean is within 0.04 of the observations at the
    # training inputs, whose standard deviation is 2.6. The tolerance is 10% of that
    # deviation: it states that the predictions come back in the units of the outcome (a
    # failure to un-map would be off by the mean, about 10, and the scale, about 2.6), and
    # it has a margin of more than two.
    TOLERANCE = 0.25

    def test_the_predictions_are_in_the_units_of_the_outcome(self) -> None:
        train_x, train_y, _, _ = toyData()
        model = fitModel(build(), train_x, train_y)

        with torch.no_grad():
            posterior = model.posterior(train_x)

        assert float((posterior.mean - train_y).abs().max()) < self.TOLERANCE
        assert float(posterior.mean.mean()) == pytest.approx(float(train_y.mean()), rel=0.02)

    def test_the_model_trained_on_standardized_targets_and_not_on_the_raw_ones(self) -> None:
        train_x, train_y, _, _ = toyData()

        model = fitModel(build(), train_x, train_y)

        targets = model.train_targets
        assert float(targets.mean()) == pytest.approx(0.0, abs=1e-9)
        assert float(targets.std()) == pytest.approx(1.0, abs=1e-9)

    def test_it_is_the_model_that_botorch_builds_when_it_is_given_no_transform(self) -> None:
        # Parity with the engine default, and so with the legacy surrogate, which gave
        # `Standardize(m=...)` to the model: the same fit, to the last digit.
        train_x, train_y, test_x, _ = toyData()
        default = SingleTaskGP(train_x, train_y)
        fit_gpytorch_mll(ExactMarginalLogLikelihood(default.likelihood, default))
        default.eval()
        ours = fitModel(build(), train_x, train_y)

        with torch.no_grad():
            expected, actual = default.posterior(test_x), ours.posterior(test_x)

        assert torch.allclose(actual.mean, expected.mean, atol=1e-10)
        assert torch.allclose(actual.variance, expected.variance, atol=1e-10)

    def test_several_outcomes_are_each_standardized_on_their_own(self) -> None:
        train_x, train_y, _, _ = toyData(n_outputs=2)
        model = fitModel(build(2), train_x, train_y)

        with torch.no_grad():
            posterior = model.posterior(train_x)

        assert posterior.mean.shape == train_y.shape
        assert float((posterior.mean - train_y).abs().max()) < self.TOLERANCE
        assert torch.allclose(
            model.train_targets.mean(dim=-1), torch.zeros(2, dtype=DOUBLE), atol=1e-9
        )


class TestEachCallMakesATransformOfItsOwn:
    def test_the_statistics_of_one_model_are_not_the_statistics_of_another(self) -> None:
        # `Standardize` keeps what it learned: two models that shared one would overwrite each
        # other's statistics, which is why `build` makes a new one at every call.
        builder = StandardizeOutcomeTransformBuilder(StandardizeOutcomeTransformConfig())
        first, second = builder.build(1), builder.build(1)
        assert isinstance(first, Standardize) and isinstance(second, Standardize)

        first(torch.tensor([[1.0], [2.0], [3.0]], dtype=DOUBLE))
        second(torch.tensor([[100.0], [200.0], [900.0]], dtype=DOUBLE))

        assert float(first.means) == pytest.approx(2.0)
        assert float(second.means) == pytest.approx(400.0)
