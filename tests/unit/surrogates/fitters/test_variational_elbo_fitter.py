"""Unit tests for the `variational_elbo_v1` fitter (roadmap task P3-15, specification 5.6, 6,
3.7).

The three things the card asks for are tested first: the ELBO of a toy approximate GP increases
when it is fitted, every field of the configuration changes what the fit does, and the same
generator gives the same fit. A field is checked twice where it can be: an optimizer step hook
shows what the loop was made with (the class, the rate, the number of steps), and the effect on
the parameters and on the bound shows that it changes the fit.

Fits use a thirty-point toy classifier (and a ten-point toy regression) and run in well under
a second each.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from itertools import pairwise
from typing import Any, get_args

import pytest
import torch
from botorch.models import ModelListGP, SingleTaskGP
from botorch.models.approximate_gp import SingleTaskVariationalGP
from fitter_fakes import (
    FailingBernoulliLikelihood,
    NotFiniteBernoulliLikelihood,
    SamplingBernoulliLikelihood,
    elbo,
    generatorOf,
    toyApproximateRegressor,
    toyClassifier,
    toyData,
    toyLabels,
    toyModel,
)
from gpytorch.likelihoods import BernoulliLikelihood
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.optimizer import register_optimizer_step_post_hook

from planexs.surrogates.fitters.FitterError import FitterError
from planexs.surrogates.fitters.VariationalElboFitter import _OPTIMIZERS, VariationalElboFitter
from planexs.surrogates.fitters.VariationalElboFitterConfig import VariationalElboFitterConfig

#: A learning rate that suits each optimizer on the toy classifier. The rates are not
#: comparable: RMSprop's first step is about ten times its rate, so the rate that suits Adam is
#: already too large for it.
SUITABLE_RATES = {"adam": 0.1, "sgd": 0.1, "rmsprop": 0.01, "adagrad": 0.1}


def fitterOf(**settings: Any) -> VariationalElboFitter:
    return VariationalElboFitter(VariationalElboFitterConfig(**settings))


def parametersOf(model: SingleTaskVariationalGP) -> Tensor:
    """Every parameter of the model, in one vector."""
    return torch.cat([value.detach().flatten() for value in model.parameters()])


def fitted(model: SingleTaskVariationalGP, seed: int = 0, **settings: Any) -> Tensor:
    """Fits the model and returns its parameters."""
    fitterOf(**settings).fit(model, generator=generatorOf(seed))
    return parametersOf(model)


@dataclass
class StepLog:
    """The optimizers that took a step, one entry for each step, as they were then."""

    optimizers: list[Optimizer] = field(default_factory=list)
    rates: list[float] = field(default_factory=list)

    def __call__(self, optimizer: Optimizer, args: Any, kwargs: Any) -> None:
        self.optimizers.append(optimizer)
        self.rates.append(float(optimizer.param_groups[0]["lr"]))


@pytest.fixture
def steps() -> Iterator[StepLog]:
    """A hook that records every step that any torch optimizer takes during the test."""
    log = StepLog()
    handle = register_optimizer_step_post_hook(log)
    yield log
    handle.remove()


class TestItMaximizesTheElboOnAToyProblem:
    def test_the_elbo_increases_as_the_fit_takes_more_iterations(self) -> None:
        # A toy classifier from the initial parameters: the bound that each number of steps
        # reaches is higher than the one before (it goes -1.12, -0.94, -0.71, -0.58, -0.56).
        values = [elbo(toyClassifier())]
        for n_iterations in (1, 5, 25, 100):
            model = toyClassifier()
            fitterOf(n_iterations=n_iterations).fit(model, generator=generatorOf(0))
            values.append(elbo(model))

        assert values == sorted(values)
        assert all(later > earlier + 0.005 for earlier, later in pairwise(values))

    def test_the_default_settings_improve_the_elbo_by_a_wide_margin(self) -> None:
        model = toyClassifier()
        initial = elbo(model)

        fitterOf().fit(model, generator=generatorOf(0))

        # From -1.12 to -0.56: the margin of 0.1 is far above rounding and far below it.
        assert elbo(model) > initial + 0.1

    def test_it_fits_an_approximate_gp_with_a_gaussian_likelihood_too(self) -> None:
        # The fitter needs a latent Gaussian posterior and not a Bernoulli likelihood: a
        # regression with an approximate GP is fitted by the same bound (-124 to -2).
        model = toyApproximateRegressor()
        initial = elbo(model)

        fitterOf().fit(model, generator=generatorOf(0))

        assert elbo(model) > initial + 0.1

    @pytest.mark.parametrize("name", SUITABLE_RATES)
    def test_every_optimizer_improves_the_elbo(self, name: str) -> None:
        model = toyClassifier()
        initial = elbo(model)

        fitterOf(optimizer=name, learning_rate=SUITABLE_RATES[name]).fit(
            model, generator=generatorOf(0)
        )

        assert elbo(model) > initial + 0.1

    def test_the_default_number_of_iterations_is_close_to_a_run_four_times_as_long(
        self,
    ) -> None:
        # The default is a budget that gets within a margin of what the loop can reach: 0.005
        # below a run of 400 steps on this toy, where 5 steps are 0.16 below it. The margin of
        # 0.02 separates the two.
        default, long, short = toyClassifier(), toyClassifier(), toyClassifier()
        n_default = VariationalElboFitterConfig().n_iterations

        fitterOf().fit(default, generator=generatorOf(0))
        fitterOf(n_iterations=4 * n_default).fit(long, generator=generatorOf(0))
        fitterOf(n_iterations=5).fit(short, generator=generatorOf(0))

        assert elbo(long) - elbo(default) < 0.02
        assert elbo(long) - elbo(short) > 0.02

    def test_the_parameters_are_changed_in_place_and_stay_finite(self) -> None:
        model = toyClassifier()
        before = {name: value.detach().clone() for name, value in model.named_parameters()}

        fitterOf().fit(model, generator=generatorOf(0))

        after = dict(model.named_parameters())
        assert after.keys() == before.keys()
        assert all(not torch.equal(before[name], after[name]) for name in before)
        assert all(bool(torch.isfinite(value).all()) for value in after.values())

    def test_the_model_is_left_in_evaluation_mode_and_can_predict(self) -> None:
        model = toyClassifier()
        train_x, _ = toyLabels()

        fitterOf().fit(model, generator=generatorOf(0))
        posterior = model.posterior(train_x[:3] * 0.9)

        assert model.training is False
        assert model.model.training is False
        assert bool(torch.isfinite(posterior.mean).all())
        assert bool((posterior.variance > 0).all())

    def test_the_training_data_is_not_changed(self) -> None:
        model = toyClassifier()
        inputs = model.model.train_inputs[0].clone()
        targets = model.model.train_targets.clone()

        fitterOf().fit(model, generator=generatorOf(0))

        assert torch.equal(model.model.train_inputs[0], inputs)
        assert torch.equal(model.model.train_targets, targets)

    def test_no_gradient_is_left_on_the_parameters(self) -> None:
        model = toyClassifier()

        fitterOf(n_iterations=3).fit(model, generator=generatorOf(0))

        assert all(value.grad is None for value in model.parameters())


class TestEveryParameterIsFittedInsideItsBounds:
    def test_the_optimizer_is_given_every_parameter_of_the_model(self, steps: StepLog) -> None:
        model = toyClassifier()
        expected = {id(value) for value in model.parameters()}

        fitterOf(n_iterations=1).fit(model, generator=generatorOf(0))

        (optimizer,) = steps.optimizers
        given = [value for group in optimizer.param_groups for value in group["params"]]
        assert {id(value) for value in given} == expected
        # The inducing points, the variational distribution, the mean and the kernel.
        assert len(given) == len(list(model.parameters())) == 5

    def test_a_step_that_leaves_the_lower_bound_of_a_lengthscale_is_put_back(self) -> None:
        # RMSprop at 0.1 takes a first step of about 1.0 in every parameter, which takes the
        # lengthscale of BoTorch's kernel below its bound (the constraint has no transform,
        # and the prior of the lengthscale is not defined there). The loop puts it back, as
        # BoTorch's own loop with a torch optimizer does, and the fit goes on.
        model = toyClassifier()
        kernel = model.model.covar_module
        lower = float(kernel.raw_lengthscale_constraint.lower_bound)

        fitterOf(optimizer="rmsprop", learning_rate=0.1).fit(model, generator=generatorOf(0))

        assert float(kernel.lengthscale.detach().min()) >= lower
        assert bool(torch.isfinite(parametersOf(model)).all())


class TestEveryFieldChangesBehaviour:
    @pytest.mark.parametrize("n_iterations", [1, 7, 30])
    def test_n_iterations_is_the_number_of_steps_taken(
        self, steps: StepLog, n_iterations: int
    ) -> None:
        fitterOf(n_iterations=n_iterations).fit(toyClassifier(), generator=generatorOf(0))

        assert len(steps.optimizers) == n_iterations

    def test_n_iterations_changes_the_fit(self) -> None:
        few = fitted(toyClassifier(), n_iterations=5)
        many = fitted(toyClassifier(), n_iterations=50)

        assert float((few - many).abs().max()) > 0.1

    @pytest.mark.parametrize("rate", [0.003, 0.1, 0.25])
    def test_learning_rate_is_the_rate_of_the_optimizer(
        self, steps: StepLog, rate: float
    ) -> None:
        fitterOf(n_iterations=3, learning_rate=rate).fit(
            toyClassifier(), generator=generatorOf(0)
        )

        assert steps.rates == [rate, rate, rate]

    def test_learning_rate_changes_the_fit_and_the_bound_it_reaches(self) -> None:
        slow, fast = toyClassifier(), toyClassifier()

        slow_parameters = fitted(slow, n_iterations=25, learning_rate=0.01)
        fast_parameters = fitted(fast, n_iterations=25, learning_rate=0.1)

        assert float((slow_parameters - fast_parameters).abs().max()) > 0.1
        # Ten times the rate gets further in the same number of steps: -0.59 against -0.91.
        assert elbo(fast) > elbo(slow) + 0.1

    @pytest.mark.parametrize(
        ("name", "optimizer_class"),
        [
            ("adam", torch.optim.Adam),
            ("sgd", torch.optim.SGD),
            ("rmsprop", torch.optim.RMSprop),
            ("adagrad", torch.optim.Adagrad),
        ],
    )
    def test_optimizer_names_the_class_that_takes_the_steps(
        self, steps: StepLog, name: str, optimizer_class: type[Optimizer]
    ) -> None:
        fitterOf(n_iterations=2, optimizer=name).fit(toyClassifier(), generator=generatorOf(0))

        assert [type(optimizer) for optimizer in steps.optimizers] == [optimizer_class] * 2

    def test_optimizer_changes_the_fit(self) -> None:
        results = {name: fitted(toyClassifier(), optimizer=name) for name in _OPTIMIZERS}

        names = list(results)
        for position, first in enumerate(names):
            for second in names[position + 1 :]:
                difference = float((results[first] - results[second]).abs().max())
                assert difference > 0.05, (first, second)

    def test_the_names_of_the_optimizers_are_those_of_the_schema(self) -> None:
        annotation = VariationalElboFitterConfig.model_fields["optimizer"].annotation

        assert set(_OPTIMIZERS) == set(get_args(annotation))

    @pytest.mark.parametrize("name", list(_OPTIMIZERS))
    def test_an_optimizer_is_made_with_torchs_defaults_and_no_weight_decay(
        self, steps: StepLog, name: str
    ) -> None:
        # The bound is what is maximized: a decay would add a penalty that is not part of it.
        fitterOf(n_iterations=1, optimizer=name).fit(toyClassifier(), generator=generatorOf(0))

        (optimizer,) = steps.optimizers
        assert optimizer.defaults.get("weight_decay", 0) == 0
        assert optimizer.defaults.get("momentum", 0) == 0
        assert len(optimizer.param_groups) == 1


class TestTheFitIsReproducibleAndLeavesTheGlobalRandomStateAlone:
    """The first evaluation of a variational GP perturbs its variational mean with a draw from
    torch's global random state, so even the fit of a plain Bernoulli classifier depends on it.
    The fit seeds that state from the generator it is given, for the length of the fit only
    (specification 3.7). A likelihood that integrates by sampling is a second place where the
    loop draws, and the test likelihood stands for it.
    """

    @staticmethod
    def _sampled(generator_seed: int, global_seed: int) -> Tensor:
        torch.manual_seed(global_seed)
        model = toyClassifier(likelihood=SamplingBernoulliLikelihood())
        return fitted(model, seed=generator_seed, n_iterations=20)

    def test_the_same_generator_seed_gives_the_same_fit(self) -> None:
        first = fitted(toyClassifier(), seed=7)
        again = fitted(toyClassifier(), seed=7)

        torch.testing.assert_close(first, again, rtol=0.0, atol=1e-10)

    def test_the_fit_does_not_depend_on_the_global_state_and_does_on_the_generator_seed(
        self,
    ) -> None:
        # The model is built and fitted under three states of torch's global generator: the
        # variational mean is perturbed in the fit, so the global state would reach the result
        # if the fit did not take its own.
        fits = []
        for generator_seed, global_seed in [(1, 100), (1, 200), (2, 100)]:
            torch.manual_seed(global_seed)
            fits.append(fitted(toyClassifier(), seed=generator_seed, n_iterations=20))

        torch.testing.assert_close(fits[0], fits[1], rtol=0.0, atol=1e-10)
        assert float((fits[0] - fits[2]).abs().max()) > 0.1

    def test_a_sampling_likelihood_gives_the_same_fit_whatever_the_global_state_was(
        self,
    ) -> None:
        first = self._sampled(generator_seed=1, global_seed=100)
        again = self._sampled(generator_seed=1, global_seed=200)

        torch.testing.assert_close(first, again, rtol=0.0, atol=1e-10)

    def test_a_sampling_likelihood_gives_another_fit_for_another_generator_seed(self) -> None:
        first = self._sampled(generator_seed=1, global_seed=100)
        other = self._sampled(generator_seed=2, global_seed=100)

        assert float((first - other).abs().max()) > 1e-6

    def test_the_global_random_state_is_put_back_after_the_fit(self) -> None:
        torch.manual_seed(1234)
        before = torch.get_rng_state()

        fitted(toyClassifier(), seed=5, n_iterations=5)

        assert torch.equal(torch.get_rng_state(), before)

    def test_the_generator_is_advanced_and_a_used_generator_gives_another_fit(self) -> None:
        # The seed is drawn from the generator and not read from it, so a generator that has
        # been used does not repeat the fit of a fresh one.
        generator = generatorOf(3)
        fresh_state = generator.get_state()
        fitter = fitterOf(n_iterations=20)
        first, second = toyClassifier(), toyClassifier()

        fitter.fit(first, generator=generator)
        fitter.fit(second, generator=generator)

        assert not torch.equal(generator.get_state(), fresh_state)
        assert float((parametersOf(first) - parametersOf(second)).abs().max()) > 1e-6


class TestWhatItRefuses:
    def test_an_exact_gp_is_named_and_left_as_it_was(self) -> None:
        model = toyModel()
        before = parametersOf(model)  # type: ignore[arg-type]

        with pytest.raises(FitterError, match="SingleTaskGP") as caught:
            fitterOf().fit(model, generator=generatorOf(0))

        error = caught.value
        assert error.component == "fitter 'variational_elbo_v1'"
        assert error.field_path == "model"
        assert "approximate Gaussian processes" in error.message
        assert error.suggestion
        assert torch.equal(parametersOf(model), before)  # type: ignore[arg-type]

    def test_a_list_of_models_is_refused_and_named(self) -> None:
        train_x, train_y = toyData()
        model = ModelListGP(SingleTaskGP(train_x, train_y))

        with pytest.raises(FitterError, match="ModelListGP") as caught:
            fitterOf().fit(model, generator=generatorOf(0))

        assert caught.value.field_path == "model"

    def test_an_approximate_gp_with_several_outputs_is_refused(self) -> None:
        train_x, labels = toyLabels()
        both = torch.cat([labels, 1.0 - labels], dim=-1)
        model = SingleTaskVariationalGP(train_x, both, likelihood=BernoulliLikelihood())

        with pytest.raises(FitterError, match="one output") as caught:
            fitterOf().fit(model, generator=generatorOf(0))

        assert "has 2" in caught.value.message
        assert caught.value.field_path == "model"

    @pytest.mark.parametrize("missing", ["train_inputs", "train_targets"])
    def test_a_model_that_holds_no_training_data_is_refused(self, missing: str) -> None:
        model = toyClassifier()
        setattr(model.model, missing, None)

        with pytest.raises(FitterError, match="holds none") as caught:
            fitterOf().fit(model, generator=generatorOf(0))

        assert caught.value.field_path == "model"
        assert caught.value.suggestion


class TestWhenAStepDiverges:
    def test_a_rate_that_makes_the_bound_not_finite_names_the_iteration_and_the_rate(
        self,
    ) -> None:
        # SGD at a rate of 1000 takes the parameters to where the bound is not a number.
        model = toyClassifier()

        with pytest.raises(FitterError, match=r"at iteration \d+ of 100") as caught:
            fitterOf(optimizer="sgd", learning_rate=1000.0).fit(
                model, generator=generatorOf(0)
            )

        error = caught.value
        assert error.component == "fitter 'variational_elbo_v1'"
        assert error.field_path == "learning_rate"
        assert "'sgd'" in error.message
        assert "1000.0" in error.message
        assert "learning_rate" in (error.suggestion or "")

    def test_the_parameters_stay_finite_where_the_fit_stopped(self) -> None:
        model = toyClassifier()

        with pytest.raises(FitterError):
            fitterOf(optimizer="sgd", learning_rate=1000.0).fit(
                model, generator=generatorOf(0)
            )

        assert bool(torch.isfinite(parametersOf(model)).all())

    def test_a_failure_of_the_linear_algebra_after_a_step_is_a_fitter_error(self) -> None:
        # The bound is computed once at iteration 0, and fails at iteration 1.
        model = toyClassifier(likelihood=FailingBernoulliLikelihood(after=1))

        with pytest.raises(FitterError, match="at iteration 1 of 100") as caught:
            fitterOf().fit(model, generator=generatorOf(0))

        assert caught.value.field_path == "learning_rate"
        assert "not positive definite" in caught.value.message
        assert isinstance(caught.value.__cause__, RuntimeError)

    def test_a_failure_before_the_first_step_is_not_blamed_on_the_rate(self) -> None:
        # No step has been taken, so the error is that of the model, and it is not hidden.
        model = toyClassifier(likelihood=FailingBernoulliLikelihood(after=0))

        with pytest.raises(RuntimeError, match="not positive definite") as caught:
            fitterOf().fit(model, generator=generatorOf(0))

        assert not isinstance(caught.value, FitterError)

    def test_a_bound_that_is_not_finite_before_the_first_step_points_at_the_model(
        self,
    ) -> None:
        model = toyClassifier(likelihood=NotFiniteBernoulliLikelihood())

        with pytest.raises(FitterError, match="before its first step") as caught:
            fitterOf().fit(model, generator=generatorOf(0))

        assert caught.value.field_path == "model"
        assert "nan" in caught.value.message

    def test_the_global_random_state_is_put_back_even_when_the_fit_fails(self) -> None:
        torch.manual_seed(77)
        before = torch.get_rng_state()

        with pytest.raises(FitterError):
            fitterOf(optimizer="sgd", learning_rate=1000.0).fit(
                toyClassifier(), generator=generatorOf(0)
            )

        assert torch.equal(torch.get_rng_state(), before)
