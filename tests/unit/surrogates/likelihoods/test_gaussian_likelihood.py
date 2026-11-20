"""Unit tests for the `gaussian_v1` likelihood (roadmap task P3-11, specification 5.6, 6).

The builder is checked against the engine's own likelihoods and, where a number only means
something once a model has been fitted (the floor of an inferred noise), against a real
`SingleTaskGP`.
"""

from __future__ import annotations

import json
import warnings
from typing import Any

import gpytorch
import pytest
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.models.utils.gpytorch_modules import (
    MIN_INFERRED_NOISE_LEVEL,
    get_gaussian_likelihood_with_lognormal_prior,
)
from gpytorch.likelihoods import FixedNoiseGaussianLikelihood, GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors import LogNormalPrior
from gpytorch.utils.warnings import NumericalWarning
from likelihood_fakes import builderOf, makeContext
from pydantic import ValidationError

from planexs.core.ConfigError import ConfigError
from planexs.core.resolveComponent import resolveComponent
from planexs.core.RuntimeConfig import RuntimeConfig
from planexs.core.toResolvedConfig import toResolvedConfig
from planexs.core.validateConfig import validateConfig
from planexs.surrogates.likelihoods.GaussianLikelihoodBuilder import GaussianLikelihoodBuilder
from planexs.surrogates.likelihoods.GaussianLikelihoodConfig import GaussianLikelihoodConfig
from planexs.surrogates.likelihoods.likelihoodRegistry import likelihoodRegistry

GAUSSIAN: dict[str, Any] = {"type": "gaussian_v1"}


def noiseOf(likelihood: Any) -> list[float]:
    """The noise of a likelihood, flattened."""
    return [float(value) for value in likelihood.noise.detach().flatten()]


def fitOneDimensional(likelihood: GaussianLikelihood, n_outputs: int = 1) -> SingleTaskGP:
    """A `SingleTaskGP` over nearly noise-free data of a smooth function, fitted with the
    likelihood. The data is standardized by the model, so a noise is in standard units.
    """
    generator = torch.Generator().manual_seed(0)
    train_x = torch.linspace(0.0, 1.0, 8, dtype=torch.double).unsqueeze(-1)
    columns = [torch.sin((position + 3) * train_x) for position in range(n_outputs)]
    train_y = torch.cat(columns, dim=-1)
    train_y = train_y + 1e-3 * torch.randn(
        train_y.shape, generator=generator, dtype=torch.double
    )
    model = SingleTaskGP(
        train_x, train_y, likelihood=likelihood, outcome_transform=Standardize(m=n_outputs)
    )
    fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
    return model


class TestTheRegisteredEntry:
    def test_it_models_continuous_outcomes_with_a_known_or_an_inferred_noise(self) -> None:
        entry = likelihoodRegistry().lookup("gaussian_v1")

        assert entry.supports == frozenset(
            {"outcome_kind:continuous", "noise:known", "noise:inferred"}
        )
        # Nothing is required of the surrogate: an exact GP and an approximate one take it.
        assert entry.needs == frozenset()

    def test_a_mapping_builds_it_through_the_registry(self) -> None:
        resolved = resolveComponent(
            likelihoodRegistry(), {"type": "gaussian_v1", "noise_lower_bound": 0.01}
        )

        builder = resolved.entry.builder(resolved.config)

        assert isinstance(builder, GaussianLikelihoodBuilder)
        assert builder.config.noise_lower_bound == 0.01


class TestItsConfiguration:
    def test_the_defaults_are_the_legacy_choices(self) -> None:
        config = GaussianLikelihoodConfig()

        # No floor of its own: BoTorch's, as the legacy surrogate had. A known noise is raised
        # to the 1e-10 that the legacy code used.
        assert config.noise_lower_bound is None
        assert config.min_observation_variance == 1e-10

    @pytest.mark.parametrize("bound", [0.0, -1e-3, float("inf"), float("nan")])
    def test_a_lower_bound_must_be_a_positive_number(self, bound: float) -> None:
        with pytest.raises(ValidationError):
            GaussianLikelihoodConfig(noise_lower_bound=bound)

    @pytest.mark.parametrize("floor", [-1e-10, float("inf"), float("nan")])
    def test_the_floor_must_be_a_number_of_zero_or_more(self, floor: float) -> None:
        with pytest.raises(ValidationError):
            GaussianLikelihoodConfig(min_observation_variance=floor)

    def test_a_floor_of_zero_is_legal(self) -> None:
        assert (
            GaussianLikelihoodConfig(min_observation_variance=0.0).min_observation_variance
            == 0
        )

    def test_an_unknown_field_is_refused_with_the_fields_of_the_schema(self) -> None:
        with pytest.raises(ConfigError, match="noise_floor") as error:
            validateConfig(
                GaussianLikelihoodConfig, {"type": "gaussian_v1", "noise_floor": 1.0}
            )

        assert error.value.suggestion is not None
        assert "noise_lower_bound" in error.value.suggestion

    def test_it_survives_json_as_a_resolved_configuration(self) -> None:
        config = GaussianLikelihoodConfig(
            noise_lower_bound=1e-6, min_observation_variance=1e-8
        )

        data = json.loads(json.dumps(toResolvedConfig(config)))

        assert validateConfig(GaussianLikelihoodConfig, data) == config
        assert data == {
            "type": "gaussian_v1",
            "noise_lower_bound": 1e-6,
            "min_observation_variance": 1e-8,
        }


class TestAnInferredNoise:
    def test_without_a_floor_it_is_the_default_likelihood_of_the_engine(self) -> None:
        likelihood = builderOf(GAUSSIAN).build(makeContext())
        engine = get_gaussian_likelihood_with_lognormal_prior(batch_shape=torch.Size())

        assert isinstance(likelihood, GaussianLikelihood)
        assert not isinstance(likelihood, FixedNoiseGaussianLikelihood)
        assert float(likelihood.noise.detach()) == pytest.approx(float(engine.noise.detach()))
        assert float(likelihood.noise_covar.raw_noise_constraint.lower_bound) == pytest.approx(
            MIN_INFERRED_NOISE_LEVEL
        )
        prior = likelihood.noise_covar.noise_prior
        assert isinstance(prior, LogNormalPrior)
        assert float(prior.loc) == float(engine.noise_covar.noise_prior.loc)
        assert float(prior.scale) == float(engine.noise_covar.noise_prior.scale)

    def test_it_is_a_parameter_that_the_fit_learns(self) -> None:
        likelihood = builderOf(GAUSSIAN).build(makeContext())

        assert any(parameter.requires_grad for parameter in likelihood.parameters())

    @pytest.mark.parametrize("bound", [1e-8, 1e-2, 0.5])
    def test_with_a_floor_the_engine_floor_is_replaced_and_its_prior_is_kept(
        self, bound: float
    ) -> None:
        engine = get_gaussian_likelihood_with_lognormal_prior(batch_shape=torch.Size())

        likelihood = builderOf({**GAUSSIAN, "noise_lower_bound": bound}).build(makeContext())

        assert isinstance(likelihood, GaussianLikelihood)
        constraint = likelihood.noise_covar.raw_noise_constraint
        assert float(constraint.lower_bound) == pytest.approx(bound, rel=1e-6)
        # The prior is the engine's, so the fit stays a MAP fit.
        prior = likelihood.noise_covar.noise_prior
        assert isinstance(prior, LogNormalPrior)
        assert float(prior.loc) == float(engine.noise_covar.noise_prior.loc)
        # It starts at the mode of the prior, or at the floor if the prior's mode is below it.
        assert float(likelihood.noise.detach()) == pytest.approx(
            max(float(engine.noise_covar.noise_prior.mode), bound), rel=1e-6
        )

    def test_a_floor_below_the_engine_floor_lowers_it(self) -> None:
        low = builderOf({**GAUSSIAN, "noise_lower_bound": 1e-8}).build(makeContext())
        default = builderOf(GAUSSIAN).build(makeContext())

        assert float(low.noise_covar.raw_noise_constraint.lower_bound) < float(
            default.noise_covar.raw_noise_constraint.lower_bound
        )

    def test_the_fit_never_infers_a_noise_below_the_floor(self) -> None:
        # Nearly noise-free data: left to itself the fit infers a very small noise (about
        # 0.003 in standard units). With a floor of 0.1 it stops at the floor.
        free = fitOneDimensional(builderOf(GAUSSIAN).build(makeContext()))
        floored = fitOneDimensional(
            builderOf({**GAUSSIAN, "noise_lower_bound": 0.1}).build(makeContext())
        )

        assert float(free.likelihood.noise.detach()) < 0.05
        assert float(floored.likelihood.noise.detach()) == pytest.approx(0.1, abs=1e-6)

    def test_each_model_of_a_batch_has_a_noise_of_its_own(self) -> None:
        likelihood = builderOf(GAUSSIAN).build(
            makeContext("continuous", "continuous", batch_shape=(2,))
        )

        assert tuple(likelihood.noise.shape) == (2, 1)

    def test_a_batched_likelihood_serves_a_model_with_several_outputs(self) -> None:
        likelihood = builderOf(GAUSSIAN).build(
            makeContext("continuous", "continuous", batch_shape=(2,))
        )

        model = fitOneDimensional(likelihood, n_outputs=2)

        assert tuple(model.likelihood.noise.shape) == (2, 1)
        assert model.num_outputs == 2


class TestAKnownNoise:
    def test_it_is_a_fixed_noise_likelihood_holding_the_reported_variances(self) -> None:
        likelihood = builderOf(GAUSSIAN).build(makeContext(noise=[0.1, 0.2, 0.3]))

        assert isinstance(likelihood, FixedNoiseGaussianLikelihood)
        assert noiseOf(likelihood) == pytest.approx([0.1, 0.2, 0.3])
        assert not any(parameter.requires_grad for parameter in likelihood.parameters())

    def test_a_floor_above_the_variances_raises_them_to_it(self) -> None:
        likelihood = builderOf({**GAUSSIAN, "min_observation_variance": 0.15}).build(
            makeContext(noise=[0.0, 0.1, 0.2, 0.5])
        )

        assert noiseOf(likelihood) == pytest.approx([0.15, 0.15, 0.2, 0.5])

    def test_a_floor_changes_nothing_that_is_already_above_it(self) -> None:
        likelihood = builderOf({**GAUSSIAN, "min_observation_variance": 0.01}).build(
            makeContext(noise=[0.1, 0.2])
        )

        assert noiseOf(likelihood) == pytest.approx([0.1, 0.2])

    def test_the_floor_does_not_change_the_tensor_of_the_caller(self) -> None:
        variances = torch.tensor([0.0, 0.5], dtype=torch.double)

        builderOf({**GAUSSIAN, "min_observation_variance": 0.1}).build(
            makeContext(noise=variances)
        )

        assert variances.tolist() == [0.0, 0.5]

    def test_a_floor_above_the_engines_own_minimum_prevents_the_engine_warning(self) -> None:
        # GPyTorch warns when it has to round a fixed noise up. A deterministic observation
        # (a variance of 0) raised by the floor first does not make it.
        with warnings.catch_warnings():
            warnings.simplefilter("error", NumericalWarning)

            builderOf({**GAUSSIAN, "min_observation_variance": 0.01}).build(
                makeContext(noise=[0.0, 0.5])
            )

    def test_the_engine_has_a_minimum_of_its_own_that_a_lower_floor_cannot_beat(self) -> None:
        # The legacy floor of 1e-10 never bound: GPyTorch raises a fixed noise to
        # `settings.min_fixed_noise` (1e-6 in double precision), and the larger wins.
        # The test states the rule with the engine's own number, which it reads.
        minimum = float(gpytorch.settings.min_fixed_noise.value(torch.float64))

        with pytest.warns(NumericalWarning, match="Rounding small noise"):
            likelihood = builderOf(GAUSSIAN).build(makeContext(noise=[0.0, 1e-12, 0.5]))

        assert noiseOf(likelihood) == pytest.approx([minimum, minimum, 0.5])

    def test_a_batch_of_models_takes_one_row_of_variances_each(self) -> None:
        variances = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=torch.double)

        likelihood = builderOf(GAUSSIAN).build(
            makeContext("continuous", "continuous", noise=variances, batch_shape=(2,))
        )

        assert tuple(likelihood.noise.shape) == (2, 3)
        assert noiseOf(likelihood) == pytest.approx(variances.flatten().tolist())

    def test_the_fit_keeps_the_noise_it_was_given(self) -> None:
        likelihood = builderOf({**GAUSSIAN, "min_observation_variance": 1e-2}).build(
            makeContext(noise=[1e-2] * 8)
        )

        model = fitOneDimensional(likelihood)

        # A fixed noise is not a parameter of the fit. The model standardizes the outcome
        # but the likelihood keeps the variances as they were built.
        assert noiseOf(model.likelihood) == pytest.approx([1e-2] * 8)


class TestThePrecisionAndTheDevice:
    @pytest.mark.parametrize(
        ("name", "dtype"), [("float32", torch.float32), ("float64", torch.float64)]
    )
    def test_an_inferred_noise_has_the_precision_of_the_runtime(
        self, name: str, dtype: torch.dtype
    ) -> None:
        context = makeContext(runtime=RuntimeConfig(dtype=name))  # type: ignore[arg-type]

        likelihood = builderOf(GAUSSIAN).build(context)

        assert likelihood.noise.dtype == dtype
        assert all(parameter.dtype == dtype for parameter in likelihood.parameters())

    def test_a_floored_inferred_noise_has_it_too(self) -> None:
        context = makeContext(runtime=RuntimeConfig(dtype="float32"))

        likelihood = builderOf({**GAUSSIAN, "noise_lower_bound": 0.01}).build(context)

        assert likelihood.noise.dtype == torch.float32
        assert likelihood.noise_covar.raw_noise_constraint.lower_bound.dtype == torch.float32

    def test_a_known_noise_has_it_too(self) -> None:
        context = makeContext(noise=[0.1, 0.2], runtime=RuntimeConfig(dtype="float32"))

        likelihood = builderOf(GAUSSIAN).build(context)

        assert likelihood.noise.dtype == torch.float32


class TestTheOutcomeKindsItSupports:
    def test_it_refuses_a_label_and_names_a_likelihood_that_takes_one(self) -> None:
        with pytest.raises(ConfigError, match="cannot model the outcome 'y0'") as error:
            builderOf(GAUSSIAN).build(makeContext("binary"))

        assert error.value.component == "likelihood 'gaussian_v1'"
        assert error.value.field_path == "outcomes[0]"
        assert "continuous" in error.value.message
        assert error.value.suggestion is not None
        assert "bernoulli_v1" in error.value.suggestion

    def test_it_names_the_position_of_the_outcome_it_cannot_take(self) -> None:
        with pytest.raises(ConfigError) as error:
            builderOf(GAUSSIAN).build(makeContext("continuous", "continuous", "binary"))

        assert error.value.field_path == "outcomes[2]"
        assert "'y2'" in error.value.message

    def test_it_makes_nothing_for_a_context_it_refuses(self) -> None:
        # Known noise on a label is refused as a kind first: the check comes before the build.
        with pytest.raises(ConfigError, match="cannot model"):
            builderOf(GAUSSIAN).build(makeContext("binary", noise=[0.1]))
