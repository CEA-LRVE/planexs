"""Unit tests for `ResponseLink` and `probitLink` (roadmap task P3-16, specification 5.6,
3.10).

The probit link turns a normal latent value into the mean and the standard deviation of a
probability, in closed form. Three independent references check it: GPyTorch's own
`BernoulliLikelihood.marginal` (the predictive probability, the mean), SciPy's implementation
of Owen's T (the second moment, hence the standard deviation), and a brute-force integration of
the definition on a fine grid, which uses neither.
"""

from __future__ import annotations

import dataclasses

import numpy
import pytest
import torch
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import BernoulliLikelihood
from scipy.special import ndtr, owens_t

from planexs.surrogates.probitLink import probitLink
from planexs.surrogates.ResponseLink import ResponseLink

#: Latent means and variances that cover the range that a fitted classifier meets, from a
#: nearly certain latent value to a very wide posterior, as every combination.
MEANS = [-12.0, -3.0, -1.0, -0.2, 0.0, 0.3, 1.0, 2.5, 8.0]
VARIANCES = [0.0, 1e-6, 0.05, 0.5, 1.0, 4.0, 25.0, 400.0]


def grid(dtype: torch.dtype = torch.float64) -> tuple[torch.Tensor, torch.Tensor]:
    """Every combination of `MEANS` and `VARIANCES`, as two vectors of the same length."""
    means = torch.tensor(MEANS, dtype=dtype)
    variances = torch.tensor(VARIANCES, dtype=dtype)
    return (
        means.repeat_interleave(len(VARIANCES)),
        variances.repeat(len(MEANS)),
    )


def integrated(mean: float, variance: float) -> tuple[float, float]:
    """The mean and the standard deviation of `Phi(f)` for `f ~ N(mean, variance)`, by
    integrating the definition on a grid of 400001 points over 12 standard deviations, with
    the trapezoid rule: no formula of the link is used.
    """
    std = max(variance**0.5, 1e-12)
    f = numpy.linspace(mean - 12 * std, mean + 12 * std, 400001)
    density = numpy.exp(-0.5 * ((f - mean) / std) ** 2) / (std * numpy.sqrt(2 * numpy.pi))
    first = numpy.trapezoid(ndtr(f) * density, f)
    second = numpy.trapezoid(ndtr(f) ** 2 * density, f)
    return float(first), float(max(second - first**2, 0.0) ** 0.5)


class TestTheLinkIsARecord:
    def test_it_is_a_frozen_dataclass_of_a_name_and_three_functions(self) -> None:
        link = probitLink()

        assert dataclasses.is_dataclass(link)
        assert [field.name for field in dataclasses.fields(link)] == [
            "name",
            "inverse",
            "forward",
            "marginal",
        ]
        with pytest.raises(dataclasses.FrozenInstanceError):
            link.name = "logit"  # type: ignore[misc]

    def test_a_link_can_be_made_of_any_functions(self) -> None:
        link = ResponseLink(
            name="logit",
            inverse=torch.sigmoid,
            forward=torch.logit,
            marginal=lambda mean, variance: (torch.sigmoid(mean), variance.sqrt()),
        )

        assert link.name == "logit"
        assert float(link.inverse(torch.tensor(0.0))) == 0.5

    def test_every_call_makes_an_equal_link(self) -> None:
        assert probitLink() == probitLink()

    def test_the_link_is_named_probit(self) -> None:
        assert probitLink().name == "probit"


class TestTheInverseAndForwardFunctions:
    def test_the_inverse_link_is_the_standard_normal_distribution_function(self) -> None:
        inverse = probitLink().inverse

        assert float(inverse(torch.tensor(0.0, dtype=torch.float64))) == 0.5
        assert float(inverse(torch.tensor(1.959963984540054, dtype=torch.float64))) == (
            pytest.approx(0.975, abs=1e-12)
        )
        assert float(inverse(torch.tensor(-1.0, dtype=torch.float64))) == pytest.approx(
            0.15865525393145707, abs=1e-14
        )

    def test_the_forward_link_is_its_inverse_function(self) -> None:
        forward = probitLink().forward

        assert float(forward(torch.tensor(0.5, dtype=torch.float64))) == 0.0
        assert float(forward(torch.tensor(0.975, dtype=torch.float64))) == pytest.approx(
            1.959963984540054, abs=1e-12
        )

    def test_one_undoes_the_other_on_the_levels_a_study_asks_for(self) -> None:
        link = probitLink()
        levels = torch.tensor([1e-6, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 1 - 1e-6])

        assert torch.allclose(link.inverse(link.forward(levels.double())), levels.double())

    def test_the_other_way_round_too(self) -> None:
        link = probitLink()
        latent = torch.linspace(-6.0, 6.0, 25, dtype=torch.float64)

        assert torch.allclose(link.forward(link.inverse(latent)), latent, atol=1e-9)

    def test_the_inverse_is_increasing(self) -> None:
        link = probitLink()
        latent = torch.linspace(-5.0, 5.0, 50, dtype=torch.float64)

        assert bool((link.inverse(latent).diff() > 0).all())

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_both_keep_the_precision_and_the_shape(self, dtype: torch.dtype) -> None:
        link = probitLink()
        latent = torch.zeros(3, 4, 2, dtype=dtype)

        assert link.inverse(latent).dtype == dtype
        assert link.inverse(latent).shape == (3, 4, 2)
        assert link.forward(torch.full((5,), 0.4, dtype=dtype)).dtype == dtype


class TestTheMarginalOfAProbit:
    def test_its_mean_is_the_predictive_probability_of_gpytorch(self) -> None:
        # What `BernoulliLikelihood.marginal` gives for a Gaussian latent distribution.
        means, variances = grid()
        probability, _ = probitLink().marginal(means, variances)

        for index in range(means.shape[0]):
            latent = MultivariateNormal(
                means[index : index + 1], torch.diag(variances[index : index + 1] + 1e-12)
            )
            expected = BernoulliLikelihood().marginal(latent).probs
            assert float(probability[index]) == pytest.approx(float(expected), abs=1e-9)

    def test_its_standard_deviation_follows_from_owens_t(self) -> None:
        # Var Phi(f) = Phi(a) - 2 T(a, 1 / sqrt(1 + 2v)) - Phi(a)^2, with SciPy's T. Measured
        # against it, the largest difference over this grid is 2e-13 for a positive variance.
        means, variances = grid()
        _, spread = probitLink().marginal(means, variances)

        a = means / torch.sqrt(1 + variances)
        p = torch.special.ndtr(a)
        t = torch.tensor(
            owens_t(a.numpy(), (1 / torch.sqrt(1 + 2 * variances)).numpy()),
            dtype=torch.float64,
        )
        expected = torch.sqrt((p - 2 * t - p**2).clamp_min(0.0))

        positive = variances > 0.0
        assert torch.allclose(spread[positive], expected[positive], atol=1e-11)

    def test_without_any_variance_the_spread_is_the_square_root_of_a_rounding_error(
        self,
    ) -> None:
        # The exact spread is 0, and it is computed as the square root of a difference of two
        # numbers that are equal, so that rounding (1e-17) comes out as at most 1e-8.
        means = torch.tensor(MEANS, dtype=torch.float64)

        _, spread = probitLink().marginal(means, torch.zeros_like(means))

        assert float(spread.max()) < 1e-7

    @pytest.mark.parametrize("mean", MEANS)
    @pytest.mark.parametrize("variance", [0.05, 0.5, 1.0, 4.0, 25.0])
    def test_both_moments_are_those_of_the_definition_integrated_directly(
        self, mean: float, variance: float
    ) -> None:
        probability, spread = probitLink().marginal(
            torch.tensor([mean], dtype=torch.float64),
            torch.tensor([variance], dtype=torch.float64),
        )

        first, deviation = integrated(mean, variance)

        assert float(probability) == pytest.approx(first, abs=1e-9)
        assert float(spread) == pytest.approx(deviation, abs=1e-6)

    def test_without_uncertainty_the_probability_is_that_of_the_latent_value(self) -> None:
        means = torch.tensor(MEANS, dtype=torch.float64)

        probability, spread = probitLink().marginal(means, torch.zeros_like(means))

        assert torch.allclose(probability, torch.special.ndtr(means), atol=1e-14)
        assert torch.allclose(spread, torch.zeros_like(means), atol=1e-7)

    def test_with_a_very_wide_posterior_the_probability_is_one_half_or_the_other_extreme(
        self,
    ) -> None:
        # The label of a latent value of unbounded spread is a coin: the standard deviation of
        # the probability tends to 0.5, that of a Bernoulli of parameter one half.
        probability, spread = probitLink().marginal(
            torch.zeros(1, dtype=torch.float64), torch.full((1,), 1e6, dtype=torch.float64)
        )

        assert float(probability) == pytest.approx(0.5)
        assert float(spread) == pytest.approx(0.5, abs=1e-3)

    def test_the_spread_is_never_above_a_half_nor_below_zero_and_is_finite(self) -> None:
        means, variances = grid()

        probability, spread = probitLink().marginal(means, variances)

        assert bool(torch.isfinite(probability).all())
        assert bool(torch.isfinite(spread).all())
        assert bool((spread >= 0.0).all())
        assert bool((spread <= 0.5).all())
        assert bool(((probability >= 0.0) & (probability <= 1.0)).all())

    def test_a_latent_value_far_from_zero_is_certain(self) -> None:
        probability, spread = probitLink().marginal(
            torch.tensor([-40.0, 40.0], dtype=torch.float64),
            torch.full((2,), 0.5, dtype=torch.float64),
        )

        assert probability.tolist() == pytest.approx([0.0, 1.0], abs=1e-12)
        assert spread.tolist() == pytest.approx([0.0, 0.0], abs=1e-9)

    def test_the_spread_is_largest_where_the_probability_is_one_half(self) -> None:
        means = torch.linspace(-4.0, 4.0, 81, dtype=torch.float64)

        _, spread = probitLink().marginal(means, torch.ones_like(means))

        assert int(spread.argmax()) == 40

    def test_more_uncertainty_about_the_latent_value_is_more_about_the_probability(
        self,
    ) -> None:
        # At a mean of one, the probability is a steeper function of the latent value where
        # it is close to one half, so the spread grows with the variance up to a point, and
        # it does at least over this range.
        variances = torch.tensor([0.01, 0.1, 0.5, 1.0], dtype=torch.float64)

        _, spread = probitLink().marginal(torch.ones_like(variances), variances)

        assert bool((spread.diff() > 0).all())

    def test_it_is_symmetric_about_a_latent_mean_of_zero(self) -> None:
        variances = torch.tensor([0.3, 0.3], dtype=torch.float64)
        means = torch.tensor([-1.2, 1.2], dtype=torch.float64)

        probability, spread = probitLink().marginal(means, variances)

        assert float(probability.sum()) == pytest.approx(1.0, abs=1e-14)
        assert float(spread[0]) == pytest.approx(float(spread[1]), abs=1e-12)

    @pytest.mark.parametrize("shape", [(7,), (7, 1), (7, 1, 3), (2, 7, 1, 3)])
    def test_the_shape_of_the_arguments_is_the_shape_of_the_results(
        self, shape: tuple[int, ...]
    ) -> None:
        mean = torch.randn(
            shape, generator=torch.Generator().manual_seed(0), dtype=torch.float64
        )
        variance = torch.rand(shape, generator=torch.Generator().manual_seed(1)).double()

        probability, spread = probitLink().marginal(mean, variance)

        assert probability.shape == spread.shape == shape

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_the_precision_is_kept(self, dtype: torch.dtype) -> None:
        means, variances = grid(dtype)

        probability, spread = probitLink().marginal(means, variances)

        assert probability.dtype == spread.dtype == dtype

    def test_single_precision_agrees_with_double_precision(self) -> None:
        wide = grid(torch.float64)
        narrow = grid(torch.float32)

        in_double = probitLink().marginal(*wide)
        in_single = probitLink().marginal(*narrow)

        assert torch.allclose(in_single[0].double(), in_double[0], atol=1e-5)
        assert torch.allclose(in_single[1].double(), in_double[1], atol=2e-3)

    def test_each_element_is_independent_of_the_others(self) -> None:
        means, variances = grid()
        together = probitLink().marginal(means, variances)

        alone = [
            probitLink().marginal(means[i : i + 1], variances[i : i + 1])
            for i in range(means.shape[0])
        ]

        assert torch.equal(together[0], torch.cat([each[0] for each in alone]))
        assert torch.equal(together[1], torch.cat([each[1] for each in alone]))

    def test_it_does_not_change_its_arguments(self) -> None:
        means, variances = grid()
        kept = (means.clone(), variances.clone())

        probitLink().marginal(means, variances)

        assert torch.equal(means, kept[0])
        assert torch.equal(variances, kept[1])
