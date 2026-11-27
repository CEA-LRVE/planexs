"""Unit tests for the `single_task_gp_v1` surrogate (roadmap task P3-14, specification 5.6, 6,
3.4, 3.11, 8).

What is tested: that the surrogate is registered with the tags it declares and that each tag
is true (specification 10.6), the model it builds for each case of kernel and number of
outcomes, how known noise reaches the likelihood, the fit status and the low-data flag,
`ModelNotReadyError`, the refusal of an outcome of another kind, and that a build is
reproducible. The comparison with the legacy surrogate is in `test_single_task_gp_parity.py`.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest
import torch
from botorch.models import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.posteriors.gpytorch import GPyTorchPosterior
from gpytorch.kernels import MaternKernel
from gpytorch.likelihoods import FixedNoiseGaussianLikelihood, GaussianLikelihood
from single_task_gp_fakes import N_COLUMNS, builderOf, fixedPoints, makeContext, makeProblem

from planexs.core.checkCompatibility import checkCompatibility
from planexs.core.CompatibilityError import CompatibilityError
from planexs.core.ConfigError import ConfigError
from planexs.core.DeclaredParticipant import DeclaredParticipant
from planexs.core.ModelNotReadyError import ModelNotReadyError
from planexs.core.PlanexsError import PlanexsError
from planexs.core.RuntimeConfig import RuntimeConfig
from planexs.domain.Outcome import Outcome
from planexs.domain.Problem import Problem
from planexs.encoding.SearchSpaceEncoder import SearchSpaceEncoder
from planexs.surrogates.fitters.fitterRegistry import fitterRegistry
from planexs.surrogates.fitters.MapFitter import MapFitter
from planexs.surrogates.likelihoods.likelihoodRegistry import likelihoodRegistry
from planexs.surrogates.SingleTaskGPBuilder import SingleTaskGPBuilder
from planexs.surrogates.SurrogateBuildContext import SurrogateBuildContext
from planexs.surrogates.surrogateRegistry import surrogateRegistry

NAME = "single_task_gp_v1"
LOGGER = "planexs.surrogates.SingleTaskGPBuilder"


class TestRegistration:
    def test_it_is_registered_in_the_real_registry(self) -> None:
        # Not `== [NAME]`: it is one of the surrogates that the package registers, and a
        # registry test must not break when another one is added.
        assert NAME in surrogateRegistry().names()

    def test_it_declares_exactly_the_tags_of_the_card(self) -> None:
        entry = surrogateRegistry().lookup(NAME)

        assert entry.needs == frozenset()
        assert entry.supports == frozenset(
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

    def test_it_has_a_one_line_description(self) -> None:
        description = surrogateRegistry().lookup(NAME).description

        assert description.strip() and description.splitlines() == [description]

    def test_the_entry_builder_makes_the_builder_that_keeps_its_configuration(self) -> None:
        builder = builderOf(min_observations=3)

        assert isinstance(builder, SingleTaskGPBuilder)
        assert builder.config.min_observations == 3
        assert surrogateRegistry().lookup(NAME).builder == SingleTaskGPBuilder.fromConfig


class TestEveryDeclaredTagIsTrue:
    """A component that declares that it supports something works with it (specification
    10.6): each tag is exercised on the real build.
    """

    def test_outputs_single(self) -> None:
        model = builderOf().build(makeContext(n_outcomes=1)).model

        assert model.num_outputs == 1

    def test_outputs_multi(self) -> None:
        model = builderOf().build(makeContext(n_outcomes=3)).model

        assert model.num_outputs == 3

    def test_noise_inferred(self) -> None:
        model = builderOf().build(makeContext(noise=False)).model

        assert isinstance(model.likelihood, GaussianLikelihood)
        assert not isinstance(model.likelihood, FixedNoiseGaussianLikelihood)

    def test_noise_known(self) -> None:
        model = builderOf().build(makeContext(noise=True)).model

        assert isinstance(model.likelihood, FixedNoiseGaussianLikelihood)

    @pytest.mark.parametrize("n_outcomes", [1, 2])
    def test_the_posterior_is_gaussian_with_a_normal_marginal_at_each_point(
        self, n_outcomes: int
    ) -> None:
        # `posterior:gaussian` and `posterior:marginal_normal`: a joint Gaussian whose marginal
        # at each point has a mean and a (positive, finite) variance.
        model = builderOf().build(makeContext(n_outcomes=n_outcomes)).model

        posterior = model.posterior(fixedPoints())

        assert isinstance(posterior, GPyTorchPosterior)
        assert tuple(posterior.mean.shape) == (25, 1, n_outcomes)
        assert tuple(posterior.variance.shape) == (25, 1, n_outcomes)
        assert bool(torch.isfinite(posterior.mean).all())
        assert bool((posterior.variance > 0).all())

    def test_the_continuous_outcome_kind(self) -> None:
        builderOf().build(makeContext())

    def test_a_kind_that_it_does_not_declare_is_refused_below(self) -> None:
        # See `TestAnOutcomeOfAnotherKind`: not declaring `outcome_kind:binary` is what makes
        # the check refuse it.
        assert "outcome_kind:binary" not in surrogateRegistry().lookup(NAME).supports


class TestTheCompatibilityCheck:
    @staticmethod
    def _surrogate() -> DeclaredParticipant:
        return DeclaredParticipant.fromEntry("surrogate", surrogateRegistry().lookup(NAME))

    def test_it_meets_the_need_of_its_default_fitter(self) -> None:
        fitter = DeclaredParticipant.fromEntry("fitter", fitterRegistry().lookup("map_v1"))

        checkCompatibility([self._surrogate(), fitter])

    def test_it_does_not_meet_the_need_of_an_approximate_likelihood(self) -> None:
        # `bernoulli_v1` needs a latent Gaussian posterior, which an exact GP does not have:
        # the declaration is what makes the study refuse the combination when it is built.
        likelihood = DeclaredParticipant.fromEntry(
            "likelihood", likelihoodRegistry().lookup("bernoulli_v1")
        )

        with pytest.raises(CompatibilityError) as caught:
            checkCompatibility([self._surrogate(), likelihood])

        assert "posterior:latent_gaussian" in caught.value.message
        assert "likelihood 'bernoulli_v1'" in caught.value.message


class TestTheModelItBuilds:
    def test_the_default_kernel_gives_one_batched_model_for_all_the_outcomes(self) -> None:
        model = builderOf().build(makeContext(n_outcomes=3)).model

        assert isinstance(model, SingleTaskGP)
        assert model.num_outputs == 3

    def test_an_explicit_kernel_with_one_outcome_gives_one_model_with_that_kernel(
        self,
    ) -> None:
        model = (
            builderOf(kernel={"type": "matern_v1", "nu": 1.5, "ard": True})
            .build(makeContext(n_outcomes=1))
            .model
        )

        assert isinstance(model, SingleTaskGP)
        assert isinstance(model.covar_module, MaternKernel)
        assert model.covar_module.nu == 1.5
        assert tuple(model.covar_module.lengthscale.shape) == (1, N_COLUMNS)

    def test_an_explicit_kernel_with_several_outcomes_gives_a_model_for_each(self) -> None:
        model = (
            builderOf(kernel={"type": "matern_v1", "nu": 2.5})
            .build(makeContext(n_outcomes=3))
            .model
        )

        assert isinstance(model, ModelListGP)
        assert model.num_outputs == 3
        members = list(model.models)
        assert len(members) == 3 and all(isinstance(m, SingleTaskGP) for m in members)
        kernels = [m.covar_module for m in members]
        # A kernel of its own for each: two models never share a kernel (specification 6), or
        # the hyperparameters of one would be fitted into another.
        assert len({id(kernel) for kernel in kernels}) == 3
        assert all(isinstance(kernel, MaternKernel) for kernel in kernels)

    def test_the_outcomes_of_a_list_are_fitted_independently(self) -> None:
        # Outcome 0 is the same data in both builds, so its fit must not depend on the other
        # outcomes that the model list holds.
        together_context = makeContext(n_outcomes=3)
        alone_context = makeContext(
            n_outcomes=1, train_outcomes=together_context.train_outcomes[:, :1]
        )
        alone = builderOf(kernel={"type": "rbf_v1"}).build(alone_context).model
        together = builderOf(kernel={"type": "rbf_v1"}).build(together_context).model
        assert isinstance(together, ModelListGP)

        first_alone = alone.posterior(fixedPoints()).mean[..., 0]
        first_together = together.posterior(fixedPoints()).mean[..., 0]

        torch.testing.assert_close(first_together, first_alone, rtol=0.0, atol=1e-6)

    def test_the_model_is_fitted_and_left_in_evaluation_mode(self) -> None:
        for n_outcomes in (1, 3):
            model = builderOf().build(makeContext(n_outcomes=n_outcomes)).model

            assert model.training is False

    def test_the_fit_moved_the_hyperparameters_from_their_priors(self) -> None:
        context = makeContext(n_outcomes=1)
        unfitted = SingleTaskGP(context.train_inputs, context.train_outcomes)
        fitted = builderOf().build(context).model
        assert isinstance(fitted, SingleTaskGP)

        assert not torch.allclose(
            fitted.covar_module.lengthscale.detach(),
            unfitted.covar_module.lengthscale.detach(),
        )

    def test_the_outcomes_are_standardized_by_default_and_raw_without_a_transform(
        self,
    ) -> None:
        context = makeContext(n_outcomes=1)

        standardized = builderOf().build(context).model
        raw = builderOf(outcome_transform=[]).build(makeContext(n_outcomes=1)).model

        assert standardized.outcome_transform is not None  # type: ignore[attr-defined]
        assert getattr(raw, "outcome_transform", None) is None

    def test_a_chain_of_transforms_is_applied_in_the_order_given(self) -> None:
        # Positive outcomes, logged and then standardized: predictions come back positive.
        context = makeContext(n_outcomes=1)
        positive = context.train_outcomes.abs() + 1.0
        model = (
            builderOf(outcome_transform=[{"type": "log_v1"}, {"type": "standardize_v1"}])
            .build(makeContext(n_outcomes=1, train_outcomes=positive))
            .model
        )

        mean = model.posterior(fixedPoints()).mean

        assert bool((mean > 0).all())


class TestTheFitterIsTheConfiguredOne:
    @pytest.fixture
    def calls(self, monkeypatch: pytest.MonkeyPatch) -> list[tuple[Any, ...]]:
        recorded: list[tuple[Any, ...]] = []
        original = MapFitter.fit

        def spy(self: MapFitter, model: Any, *, generator: torch.Generator) -> None:
            recorded.append((self.config, model, generator))
            original(self, model, generator=generator)

        monkeypatch.setattr(MapFitter, "fit", spy)
        return recorded

    def test_it_is_given_the_model_that_is_returned_and_the_generator_of_the_context(
        self, calls: list[tuple[Any, ...]]
    ) -> None:
        context = makeContext(n_outcomes=2)

        built = builderOf().build(context)

        ((_, model, generator),) = calls
        assert model is built.model
        assert generator is context.generator

    def test_a_list_of_models_is_fitted_by_one_call(
        self, calls: list[tuple[Any, ...]]
    ) -> None:
        built = builderOf(kernel={"type": "matern_v1"}).build(makeContext(n_outcomes=2))

        ((_, model, _),) = calls
        assert model is built.model and isinstance(model, ModelListGP)

    def test_the_settings_of_the_fitter_reach_it(self, calls: list[tuple[Any, ...]]) -> None:
        builderOf(fitter={"type": "map_v1", "max_attempts": 3, "maxiter": 40}).build(
            makeContext()
        )

        ((config, _, _),) = calls
        assert (config.max_attempts, config.maxiter) == (3, 40)

    def test_a_build_that_is_refused_fits_nothing(self, calls: list[tuple[Any, ...]]) -> None:
        with pytest.raises(ModelNotReadyError):
            builderOf().build(makeContext(n_points=1))

        assert calls == []


class TestKnownNoise:
    """The noise reaches the likelihood fixed, squared, and in the space in which the model
    trains: standardized outcomes have their noise divided by the square of their spread.
    """

    def test_standardized_outcomes_have_their_noise_divided_by_the_variance(self) -> None:
        context = makeContext(noise=True)
        assert context.train_noise_std is not None

        model = builderOf().build(context).model

        expected = context.train_noise_std[:, 0] ** 2 / context.train_outcomes[:, 0].var()
        torch.testing.assert_close(
            model.likelihood.noise.detach().reshape(-1), expected, rtol=1e-9, atol=0.0
        )

    def test_several_outcomes_have_their_own_noise_row(self) -> None:
        context = makeContext(n_outcomes=2, noise=True)
        assert context.train_noise_std is not None

        model = builderOf().build(context).model

        expected = context.train_noise_std.T**2 / context.train_outcomes.var(dim=0)[:, None]
        noise = model.likelihood.noise.detach()
        assert tuple(noise.shape) == (2, 12)
        torch.testing.assert_close(noise, expected, rtol=1e-9, atol=0.0)

    def test_each_model_of_a_list_has_its_own_noise(self) -> None:
        context = makeContext(n_outcomes=2, noise=True)
        assert context.train_noise_std is not None

        model = builderOf(kernel={"type": "rbf_v1"}).build(context).model

        assert isinstance(model, ModelListGP)
        for column, member in enumerate(model.models):
            expected = (
                context.train_noise_std[:, column] ** 2
                / context.train_outcomes[:, column].var()
            )
            torch.testing.assert_close(
                member.likelihood.noise.detach().reshape(-1), expected, rtol=1e-9, atol=0.0
            )

    def test_without_a_transform_the_noise_is_the_variance_itself(self) -> None:
        context = makeContext(noise=True)
        assert context.train_noise_std is not None

        model = builderOf(outcome_transform=[]).build(context).model

        torch.testing.assert_close(
            model.likelihood.noise.detach().reshape(-1),
            context.train_noise_std[:, 0] ** 2,
            rtol=1e-9,
            atol=0.0,
        )

    def test_a_logarithm_gives_the_noise_the_delta_method(self) -> None:
        positive = makeContext().train_outcomes.abs() + 1.0
        context = makeContext(noise=True, train_outcomes=positive)
        assert context.train_noise_std is not None

        model = builderOf(outcome_transform=[{"type": "log_v1"}]).build(context).model

        torch.testing.assert_close(
            model.likelihood.noise.detach().reshape(-1),
            context.train_noise_std[:, 0] ** 2 / positive[:, 0] ** 2,
            rtol=1e-9,
            atol=0.0,
        )

    def test_the_known_noise_floor_of_the_likelihood_is_its_own_setting(self) -> None:
        model = (
            builderOf(likelihood={"type": "gaussian_v1", "min_observation_variance": 0.5})
            .build(makeContext(noise=True))
            .model
        )

        assert float(model.likelihood.noise.detach().min()) >= 0.5

    def test_a_lower_bound_on_the_inferred_noise_reaches_the_likelihood(self) -> None:
        model = (
            builderOf(likelihood={"type": "gaussian_v1", "noise_lower_bound": 0.2})
            .build(makeContext(noise=False))
            .model
        )

        assert float(model.likelihood.noise.detach().min()) >= 0.2 - 1e-9

    def test_the_noise_policy_is_stored_with_the_configuration_and_not_read_by_a_build(
        self,
    ) -> None:
        # `trialsToTensors` reads it when the history becomes tensors, before the build.
        builder = builderOf(noise_policy={"partial": "error"})

        assert builder.config.noise_policy.partial == "error"
        builder.build(makeContext(noise=True))


class TestTheFitStatus:
    def test_enough_observations_is_not_low_data_and_has_no_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            status = builderOf().build(makeContext(n_points=12)).status

        assert (status.n_observations, status.is_low_data, status.warnings) == (12, False, ())
        assert caplog.records == []

    @pytest.mark.parametrize("n_points", [2, 3, 4])
    def test_fewer_observations_than_the_threshold_are_flagged_low_data(
        self, n_points: int, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            built = builderOf().build(makeContext(n_points=n_points))

        status = built.status
        assert status.n_observations == n_points
        assert status.is_low_data is True
        (warning,) = status.warnings
        assert f"only {n_points} observation(s)" in warning and "fewer than 5" in warning
        assert "low-confidence" in warning
        # Reported, and logged: a low-data fit is a degraded situation (specification 8).
        assert [record.levelno for record in caplog.records] == [logging.WARNING]
        assert warning in caplog.records[0].getMessage()
        # The fit is still made, and usable.
        assert bool(torch.isfinite(built.model.posterior(fixedPoints()).mean).all())

    def test_the_threshold_itself_is_not_low_data(self) -> None:
        assert builderOf().build(makeContext(n_points=5)).status.is_low_data is False

    def test_the_threshold_is_a_setting(self) -> None:
        assert (
            builderOf(low_data_threshold=3).build(makeContext(n_points=4)).status.is_low_data
            is False
        )
        assert (
            builderOf(low_data_threshold=20).build(makeContext(n_points=12)).status.is_low_data
            is True
        )

    def test_a_threshold_of_zero_never_flags_a_fit(self) -> None:
        status = builderOf(low_data_threshold=0).build(makeContext(n_points=2)).status

        assert (status.is_low_data, status.warnings) == (False, ())

    def test_a_threshold_at_or_below_the_minimum_flags_nothing(self) -> None:
        status = (
            builderOf(min_observations=3, low_data_threshold=3)
            .build(makeContext(n_points=3))
            .status
        )

        assert status.is_low_data is False


class TestNotEnoughObservations:
    @pytest.mark.parametrize("n_points", [0, 1])
    def test_fewer_than_the_minimum_is_model_not_ready(self, n_points: int) -> None:
        context = makeContext(n_points=max(n_points, 1))
        if n_points == 0:
            context = makeContext(
                train_inputs=torch.empty(0, N_COLUMNS, dtype=torch.float64),
                train_outcomes=torch.empty(0, 1, dtype=torch.float64),
            )

        with pytest.raises(ModelNotReadyError) as caught:
            builderOf().build(context)

        error = caught.value
        assert error.component == "surrogate 'single_task_gp_v1'"
        assert error.field_path == "min_observations"
        assert f"holds {n_points}" in error.message and "at least 2" in error.message
        assert "ask again" in (error.suggestion or "")

    def test_it_is_a_lack_of_data_and_not_an_invalid_request(self) -> None:
        with pytest.raises(ModelNotReadyError) as caught:
            builderOf().build(makeContext(n_points=1))

        assert isinstance(caught.value, PlanexsError)
        assert not isinstance(caught.value, ConfigError)

    def test_the_minimum_is_a_setting(self) -> None:
        with pytest.raises(ModelNotReadyError, match="at least 6"):
            builderOf(min_observations=6).build(makeContext(n_points=5))

        builderOf(min_observations=6).build(makeContext(n_points=6))

    def test_one_observation_can_be_fitted_when_the_minimum_allows_it(self) -> None:
        built = builderOf(min_observations=1).build(makeContext(n_points=1))

        assert built.status.n_observations == 1 and built.status.is_low_data is True
        assert bool(torch.isfinite(built.model.posterior(fixedPoints()).mean).all())

    def test_the_default_minimum_is_two_observations(self) -> None:
        built = builderOf().build(makeContext(n_points=2))

        assert built.status.n_observations == 2


class TestAnOutcomeOfAnotherKind:
    @staticmethod
    def _contextWith(*kinds: str) -> SurrogateBuildContext:
        problem = Problem(
            search_space=makeProblem().search_space,
            outcomes=[Outcome(name=f"o{i}", type=kind) for i, kind in enumerate(kinds)],
        )
        generator = torch.Generator().manual_seed(1)
        n_points = 12
        return makeContext(
            n_points=n_points,
            n_outcomes=len(kinds),
            problem=problem,
            encoder=SearchSpaceEncoder(problem.search_space),
            train_outcomes=(
                torch.rand(n_points, len(kinds), generator=generator) > 0.5
            ).double(),
        )

    def test_a_binary_outcome_is_refused_with_its_name_and_kind(self) -> None:
        with pytest.raises(ConfigError) as caught:
            builderOf().build(self._contextWith("binary"))

        error = caught.value
        assert error.component == "surrogate 'single_task_gp_v1'"
        assert error.field_path == "outcomes[0]"
        assert "'o0'" in error.message and "'binary'" in error.message
        assert "continuous" in error.message
        # The error names the registered surrogate that does support the kind, which is the way
        # out: a label is modelled by the classifier and not by an exact GP.
        assert "variational_gp_classifier_v1" in (error.suggestion or "")

    def test_the_position_of_the_first_refused_outcome_is_reported(self) -> None:
        with pytest.raises(ConfigError) as caught:
            builderOf().build(self._contextWith("continuous", "binary"))

        assert caught.value.field_path == "outcomes[1]"

    def test_the_kind_is_checked_before_the_number_of_observations(self) -> None:
        # A configuration mistake is reported even when the data is also short: it will not be
        # cured by asking again.
        context = self._contextWith("binary")
        short = makeContext(
            n_points=1,
            problem=context.problem,
            encoder=context.encoder,
            train_outcomes=torch.zeros(1, 1, dtype=torch.float64),
        )

        with pytest.raises(ConfigError):
            builderOf().build(short)


class TestPrecisionAndDevice:
    @pytest.mark.parametrize("kernel", [{"type": "botorch_default_v1"}, {"type": "matern_v1"}])
    def test_a_float32_build_gives_a_float32_model(self, kernel: dict[str, Any]) -> None:
        runtime = RuntimeConfig(dtype="float32")

        model = builderOf(kernel=kernel).build(makeContext(runtime=runtime, noise=True)).model

        assert {parameter.dtype for parameter in model.parameters()} == {torch.float32}
        assert bool(torch.isfinite(model.posterior(fixedPoints().float()).mean).all())


class TestReproducibility:
    def test_the_same_generator_state_gives_the_same_fit(self) -> None:
        first = builderOf().build(makeContext(n_outcomes=2)).model
        second = builderOf().build(makeContext(n_outcomes=2)).model

        torch.testing.assert_close(
            second.posterior(fixedPoints()).mean,
            first.posterior(fixedPoints()).mean,
            rtol=0.0,
            atol=1e-10,
        )

    def test_the_fit_draws_from_the_generator_of_the_context_and_advances_it(self) -> None:
        context = makeContext()
        before = context.generator.get_state()

        builderOf().build(context)

        assert not torch.equal(context.generator.get_state(), before)

    def test_a_build_does_not_touch_the_global_random_state(self) -> None:
        torch.manual_seed(321)
        before = torch.get_rng_state()

        builderOf(
            fitter={"type": "map_v1", "max_attempts": 3, "pick_best_of_all_attempts": True}
        ).build(makeContext())

        assert torch.equal(torch.get_rng_state(), before)

    def test_the_build_does_not_change_the_data_of_the_context(self) -> None:
        context = makeContext(n_outcomes=2, noise=True)
        inputs, outcomes = context.train_inputs.clone(), context.train_outcomes.clone()
        assert context.train_noise_std is not None
        noise = context.train_noise_std.clone()

        builderOf().build(context)

        assert torch.equal(context.train_inputs, inputs)
        assert torch.equal(context.train_outcomes, outcomes)
        assert torch.equal(context.train_noise_std, noise)
