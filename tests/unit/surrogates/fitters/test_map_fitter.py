"""Unit tests for the `map_v1` fitter (roadmap task P3-13, specification 5.6, 6, 3.7).

The two things the card asks for are tested first: a toy GP is fitted and its marginal
likelihood (with the priors, which is what a MAP fit maximizes) improves, and every option that
the configuration exposes is passed to the engine. An option is checked twice where it can be:
a spy wrapped round the real engine shows that it was passed, and the effect of the option on
the engine's own behaviour shows that the engine honours it, because the engine ignores a
keyword that it does not know without a word.

Fits use a few ten-point toy datasets and run in a second or two.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any

import pytest
import torch
from botorch.exceptions.errors import ModelFittingError
from botorch.fit import fit_gpytorch_mll as engine_fit
from botorch.models import ModelListGP, SingleTaskGP
from botorch.models.approximate_gp import SingleTaskVariationalGP
from fitter_fakes import (
    generatorOf,
    hyperparameters,
    objective,
    toyData,
    toyModel,
    toyModelList,
)
from gpytorch.mlls import (
    ExactMarginalLogLikelihood,
    MarginalLogLikelihood,
    SumMarginalLogLikelihood,
)
from torch import Tensor

from planexs.surrogates.fitters.FitterError import FitterError
from planexs.surrogates.fitters.MapFitter import MapFitter
from planexs.surrogates.fitters.MapFitterConfig import MapFitterConfig

#: The module that holds `fit_gpytorch_mll` as `MapFitter` calls it. (The package attribute of
#: the same name is this module too, but `MapFitter.py` also defines a class of that name.)
MAP_FITTER_MODULE = "planexs.surrogates.fitters.MapFitter"


def fitterOf(**settings: Any) -> MapFitter:
    return MapFitter(MapFitterConfig(**settings))


@dataclass
class EngineCall:
    """What the engine was given, and the state of torch's global random generator it saw."""

    mll: MarginalLogLikelihood
    kwargs: dict[str, Any]
    rng_state: Tensor


@dataclass
class EngineSpy:
    """The real engine, wrapped: it records each call and then makes it."""

    calls: list[EngineCall] = field(default_factory=list)

    def __call__(self, mll: MarginalLogLikelihood, **kwargs: Any) -> Any:
        self.calls.append(EngineCall(mll, dict(kwargs), torch.get_rng_state()))
        return engine_fit(mll, **kwargs)


@pytest.fixture
def spy(monkeypatch: pytest.MonkeyPatch) -> EngineSpy:
    """A spy that stands where `MapFitter` looks up the engine's fitting routine."""
    recorder = EngineSpy()
    monkeypatch.setattr(sys.modules[MAP_FITTER_MODULE], "fit_gpytorch_mll", recorder)
    return recorder


MODELS = {
    "one-output": lambda: toyModel(1),
    "batched-outputs": lambda: toyModel(2),
    "model-list": lambda: toyModelList(2),
}


class TestItFitsAToyGaussianProcess:
    @pytest.mark.parametrize("kind", MODELS)
    def test_the_marginal_likelihood_improves_over_its_initial_value(self, kind: str) -> None:
        model = MODELS[kind]()
        initial = objective(model)

        fitterOf().fit(model, generator=generatorOf(0))
        fitted = objective(model)

        # The objective is the marginal likelihood plus the log priors, which is what a MAP fit
        # maximizes. On these datasets the improvement is between 0.27 and 1.0: a margin of 0.1
        # is far above rounding and far below the improvement.
        assert fitted > initial + 0.1, (initial, fitted)

    @pytest.mark.parametrize("kind", MODELS)
    def test_the_hyperparameters_are_changed_in_place_and_stay_finite(self, kind: str) -> None:
        model = MODELS[kind]()
        before = hyperparameters(model)

        fitterOf().fit(model, generator=generatorOf(0))
        after = hyperparameters(model)

        assert after.keys() == before.keys()
        assert any(not torch.equal(before[name], after[name]) for name in before)
        assert all(bool(torch.isfinite(value).all()) for value in after.values())

    @pytest.mark.parametrize("kind", MODELS)
    def test_the_model_is_left_in_evaluation_mode_and_can_predict(self, kind: str) -> None:
        model = MODELS[kind]()
        train_x, _ = toyData()

        fitterOf().fit(model, generator=generatorOf(0))
        posterior = model.posterior(train_x[:3] * 0.9)

        assert model.training is False
        assert bool(torch.isfinite(posterior.mean).all())
        assert bool((posterior.variance > 0).all())

    @pytest.mark.parametrize("kind", MODELS)
    def test_the_default_settings_make_the_call_the_legacy_surrogate_made(
        self, kind: str
    ) -> None:
        # The legacy `buildSurrogateModel` called `fit_gpytorch_mll(ExactMarginalLogLikelihood(
        # model.likelihood, model))` and nothing more: the fit that the card calls "as today".
        # The models are built the same way, so the same call must give the same fit, within a
        # tolerance that is only a guard against thread-level nondeterminism of the linear
        # algebra (the first attempt of a fit has no randomness).
        legacy, new = MODELS[kind](), MODELS[kind]()
        if isinstance(legacy, ModelListGP):
            engine_fit(SumMarginalLogLikelihood(legacy.likelihood, legacy))
        else:
            assert isinstance(legacy, SingleTaskGP)
            engine_fit(ExactMarginalLogLikelihood(legacy.likelihood, legacy))

        fitterOf().fit(new, generator=generatorOf(0))

        expected, found = hyperparameters(legacy), hyperparameters(new)
        for name in expected:
            torch.testing.assert_close(found[name], expected[name], rtol=0.0, atol=1e-8)


class TestItMaximizesTheMarginalLikelihoodWithThePriors:
    def test_an_exact_gp_is_fitted_through_its_exact_marginal_likelihood(
        self, spy: EngineSpy
    ) -> None:
        model = toyModel(1)

        fitterOf().fit(model, generator=generatorOf(0))

        (call,) = spy.calls
        assert isinstance(call.mll, ExactMarginalLogLikelihood)
        assert call.mll.model is model
        assert call.mll.likelihood is model.likelihood
        # The priors are what make the fit a MAP fit: the marginal likelihood adds the log
        # prior of each hyperparameter that has one, and BoTorch's default model has several.
        assert len(list(call.mll.named_priors())) > 0

    def test_a_list_of_exact_gps_is_fitted_through_the_sum_of_their_marginal_likelihoods(
        self, spy: EngineSpy
    ) -> None:
        model = toyModelList(2)

        fitterOf().fit(model, generator=generatorOf(0))

        (call,) = spy.calls
        assert isinstance(call.mll, SumMarginalLogLikelihood)
        assert call.mll.model is model


class TestEveryExposedOptionIsPassedToTheEngine:
    def test_the_defaults_are_passed_explicitly_and_leave_the_optimizer_to_the_engine(
        self, spy: EngineSpy
    ) -> None:
        fitterOf().fit(toyModel(1), generator=generatorOf(0))

        (call,) = spy.calls
        assert call.kwargs == {
            "optimizer_kwargs": None,
            "max_attempts": 5,
            "pick_best_of_all_attempts": False,
        }

    @pytest.mark.parametrize("value", [1, 3, 9])
    def test_max_attempts(self, spy: EngineSpy, value: int) -> None:
        fitterOf(max_attempts=value).fit(toyModel(1), generator=generatorOf(0))

        assert spy.calls[0].kwargs["max_attempts"] == value

    @pytest.mark.parametrize("value", [True, False])
    def test_pick_best_of_all_attempts(self, spy: EngineSpy, value: bool) -> None:
        fitterOf(pick_best_of_all_attempts=value).fit(toyModel(1), generator=generatorOf(0))

        assert spy.calls[0].kwargs["pick_best_of_all_attempts"] is value

    @pytest.mark.parametrize("value", [1, 7, 500])
    def test_maxiter_is_the_iteration_cap_of_the_optimizer(
        self, spy: EngineSpy, value: int
    ) -> None:
        fitterOf(maxiter=value).fit(toyModel(1), generator=generatorOf(0))

        assert spy.calls[0].kwargs["optimizer_kwargs"] == {"options": {"maxiter": value}}

    def test_a_model_list_is_given_the_same_options(self, spy: EngineSpy) -> None:
        # The engine fits a list one member after the other and passes its keywords on to each.
        fitterOf(max_attempts=2, maxiter=11).fit(toyModelList(2), generator=generatorOf(0))

        assert spy.calls[0].kwargs["max_attempts"] == 2
        assert spy.calls[0].kwargs["optimizer_kwargs"] == {"options": {"maxiter": 11}}


class TestTheEngineHonoursTheOptions:
    """The engine ignores a keyword that it does not know, so a spy proves only that an option
    was passed. What the option does is checked on the engine's own behaviour.
    """

    @pytest.mark.parametrize("n_outputs", [1, 2], ids=["scipy-optimizer", "batched-optimizer"])
    def test_a_small_iteration_cap_stops_the_fit_early_on_both_optimizers(
        self, n_outputs: int
    ) -> None:
        # One output is fitted by scipy's minimizer, and several independent outputs by the
        # batched L-BFGS-B: the cap must reach both, which is why it is the option that is
        # exposed and the solver `method`, that only the first reads, is not.
        capped, free = toyModel(n_outputs), toyModel(n_outputs)

        fitterOf(maxiter=1).fit(capped, generator=generatorOf(0))
        fitterOf().fit(free, generator=generatorOf(0))

        assert objective(capped) < objective(free) - 0.05

    @pytest.mark.parametrize("attempts", [1, 2, 3])
    def test_every_attempt_is_made_when_the_best_is_picked(
        self, engineLog: list[str], attempts: int
    ) -> None:
        fitterOf(max_attempts=attempts, pick_best_of_all_attempts=True).fit(
            toyModel(1), generator=generatorOf(0)
        )

        assert _attemptLines(engineLog) == attempts

    def test_the_fit_stops_at_the_first_attempt_that_succeeds_by_default(
        self, engineLog: list[str]
    ) -> None:
        fitterOf(max_attempts=3).fit(toyModel(1), generator=generatorOf(0))

        assert _attemptLines(engineLog) == 0


def _attemptLines(messages: list[str]) -> int:
    """How many fit attempts the engine reports in its debug log: it says `Fit attempt #n` for
    each attempt that it makes when it keeps the best of all of them, and nothing otherwise.
    """
    return sum(message.startswith("Fit attempt #") for message in messages)


class TestTheFitIsReproducibleAndLeavesTheGlobalRandomStateAlone:
    """The first attempt of a fit is deterministic, and a later one restarts from
    hyperparameters that the engine samples from their priors, drawing them from torch's global
    random state. The fit seeds that state from the generator it is given, for the length of
    the fit only (specification 3.7).
    """

    def test_the_engine_runs_in_a_state_that_depends_on_the_generator_only(
        self, spy: EngineSpy
    ) -> None:
        # Whatever the global state was before, the engine sees the one that the generator
        # chose: the same for equal generators, another for another seed.
        states: list[Tensor] = []
        for generator_seed, global_seed in [(1, 100), (1, 200), (2, 100)]:
            torch.manual_seed(global_seed)
            fitterOf().fit(toyModel(1), generator=generatorOf(generator_seed))
            states.append(spy.calls[-1].rng_state)

        assert torch.equal(states[0], states[1])
        assert not torch.equal(states[0], states[2])

    def test_the_global_random_state_is_put_back_after_the_fit(self) -> None:
        torch.manual_seed(1234)
        before = torch.get_rng_state()

        fitterOf(max_attempts=3, pick_best_of_all_attempts=True).fit(
            toyModel(1), generator=generatorOf(5)
        )

        assert torch.equal(torch.get_rng_state(), before)

    def test_the_generator_is_advanced_and_a_used_generator_gives_another_fit_seed(
        self, spy: EngineSpy
    ) -> None:
        # The seed is drawn from the generator and not read from it, so a generator that has
        # been used does not repeat the fit of a fresh one.
        generator = generatorOf(3)
        fresh_state = generator.get_state()

        fitterOf().fit(toyModel(1), generator=generator)
        fitterOf().fit(toyModel(1), generator=generator)

        assert not torch.equal(generator.get_state(), fresh_state)
        assert not torch.equal(spy.calls[0].rng_state, spy.calls[1].rng_state)

    def test_restarts_from_the_priors_are_reproducible_and_depend_on_the_seed(
        self, engineLog: list[str]
    ) -> None:
        # With every attempt made, the attempts after the first start from sampled
        # hyperparameters, so the marginal likelihood that each reaches (the engine logs it to
        # all its digits) depends on the draws. Equal generators must give equal logs, whatever
        # torch's global state was, and another seed must give another one.
        def attempts(generator_seed: int, global_seed: int) -> list[str]:
            torch.manual_seed(global_seed)
            start = len(engineLog)
            fitterOf(max_attempts=3, pick_best_of_all_attempts=True).fit(
                toyModel(1), generator=generatorOf(generator_seed)
            )
            return engineLog[start:]

        first = attempts(generator_seed=4, global_seed=1)
        again = attempts(generator_seed=4, global_seed=999)
        other = attempts(generator_seed=5, global_seed=1)

        assert len(first) == 3
        assert first == again
        assert first != other

    def test_the_same_generator_seed_gives_the_same_hyperparameters(self) -> None:
        a, b = toyModel(2), toyModel(2)

        fitterOf(max_attempts=3, pick_best_of_all_attempts=True).fit(
            a, generator=generatorOf(8)
        )
        fitterOf(max_attempts=3, pick_best_of_all_attempts=True).fit(
            b, generator=generatorOf(8)
        )

        for name, value in hyperparameters(a).items():
            torch.testing.assert_close(hyperparameters(b)[name], value, rtol=0.0, atol=1e-10)


class TestWhatItRefuses:
    def test_an_approximate_model_is_named_and_the_engine_is_not_called(
        self, spy: EngineSpy
    ) -> None:
        train_x, train_y = toyData()
        model = SingleTaskVariationalGP(train_x, (train_y > 0).double())

        with pytest.raises(FitterError, match="SingleTaskVariationalGP") as caught:
            fitterOf().fit(model, generator=generatorOf(0))

        error = caught.value
        assert error.component == "fitter 'map_v1'"
        assert error.field_path == "model"
        assert error.suggestion
        assert "not an exact GP" in error.message
        assert spy.calls == []

    def test_a_list_with_an_approximate_member_names_that_member(self, spy: EngineSpy) -> None:
        train_x, train_y = toyData()
        model = ModelListGP(
            SingleTaskGP(train_x, train_y),
            SingleTaskVariationalGP(train_x, (train_y > 0).double()),
        )

        with pytest.raises(FitterError, match="holds SingleTaskVariationalGP") as caught:
            fitterOf().fit(model, generator=generatorOf(0))

        assert caught.value.field_path == "model"
        assert spy.calls == []


class TestWhenTheEngineGivesUp:
    @pytest.fixture
    def failingEngine(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def giveUp(mll: MarginalLogLikelihood, **kwargs: Any) -> Any:
            raise ModelFittingError("All attempts to fit the model have failed.")

        monkeypatch.setattr(sys.modules[MAP_FITTER_MODULE], "fit_gpytorch_mll", giveUp)

    def test_it_is_a_fitter_error_that_names_the_attempts_and_what_to_try(
        self, failingEngine: None
    ) -> None:
        with pytest.raises(FitterError, match=r"after 4 attempt\(s\)") as caught:
            fitterOf(max_attempts=4).fit(toyModel(1), generator=generatorOf(0))

        error = caught.value
        assert error.component == "fitter 'map_v1'"
        assert error.field_path == "max_attempts"
        assert "max_attempts" in (error.suggestion or "")
        assert "SingleTaskGP" in error.message

    def test_the_engines_error_is_kept_as_the_cause(self, failingEngine: None) -> None:
        with pytest.raises(FitterError) as caught:
            fitterOf().fit(toyModel(1), generator=generatorOf(0))

        assert isinstance(caught.value.__cause__, ModelFittingError)

    def test_the_global_random_state_is_put_back_even_when_the_fit_fails(
        self, failingEngine: None
    ) -> None:
        torch.manual_seed(77)
        before = torch.get_rng_state()

        with pytest.raises(FitterError):
            fitterOf().fit(toyModel(1), generator=generatorOf(0))

        assert torch.equal(torch.get_rng_state(), before)
