"""Unit tests for `Surrogate`, on a fake native model (roadmap task P3-09, specification 5.6,
3.11).

The fake model (`posterior_fakes.py`) computes its numbers from the encoded point, so each
expectation here is worked out by hand. The wrapper is checked against a real BoTorch model in
`test_surrogate_with_botorch_model.py`.
"""

from __future__ import annotations

import importlib
import inspect
import math
import re
from typing import Any

import pytest
import torch
from posterior_fakes import NOISE_VARIANCE, FakeNativeModel, makeSurrogate
from surrogate_fakes import OUTCOME_NAMES, makeProblem

from planexs import Real, SearchSpace
from planexs.core.ConfigError import ConfigError
from planexs.core.RuntimeConfig import RuntimeConfig
from planexs.domain.constraints.LinearConstraint import LinearConstraint
from planexs.domain.Outcome import Outcome
from planexs.domain.ParameterizationError import ParameterizationError
from planexs.domain.Problem import Problem
from planexs.encoding.SearchSpaceEncoder import SearchSpaceEncoder
from planexs.encoding.torch_runtime import torchGenerator
from planexs.surrogates.BuiltSurrogate import BuiltSurrogate
from planexs.surrogates.FitStatus import FitStatus
from planexs.surrogates.Prediction import Prediction
from planexs.surrogates.PredictionConfig import PredictionConfig
from planexs.surrogates.SamplingConfig import SamplingConfig
from planexs.surrogates.Surrogate import Surrogate
from planexs.surrogates.SurrogateModelError import SurrogateModelError

#: Encoded as [0.5, 0, 0, 1, 0]: the sum is 1.5 and the sum of squares is 1.25, so the fake
#: model gives the means 1.5 and 3.0 and the variances 0.0225 and 0.045.
MIDDLE = {"x": 5.0, "rate": 1e-3, "solvent": "b"}

#: Encoded as [1, 1, 0, 0, 1]: the sum is 3 and the sum of squares is 3, so the means are 3.0
#: and 6.0 and the variances 0.04 and 0.08.
CORNER = {"x": 10.0, "rate": 1.0, "solvent": "c"}


def generator(seed: int = 7) -> torch.Generator:
    """A seeded generator on the default runtime."""
    return torchGenerator(seed, RuntimeConfig())


def mixedProblem() -> Problem:
    """The space of `makeProblem` with a continuous outcome and a binary one."""
    return Problem(
        search_space=makeProblem().search_space,
        outcomes=[
            Outcome(name="yield", type="continuous"),
            Outcome(name="phase", type="binary"),
        ],
    )


class TestBindingAModel:
    def test_it_holds_the_native_model_itself_and_its_status(self) -> None:
        model = FakeNativeModel()
        status = FitStatus(n_observations=4, is_low_data=True, warnings=("few points",))
        problem = makeProblem()
        runtime = RuntimeConfig()
        encoder = SearchSpaceEncoder(problem.search_space, runtime)

        surrogate = Surrogate(
            BuiltSurrogate(model=model, status=status),  # type: ignore[arg-type]
            problem=problem,
            encoder=encoder,
            runtime=runtime,
            outcome_names=OUTCOME_NAMES,
        )

        # The native object, not a copy: it is the model that predictions are read from (5.13).
        assert surrogate.model is model
        assert surrogate.status is status
        assert surrogate.problem is problem
        assert surrogate.encoder is encoder

    def test_it_reports_the_outcomes_it_covers_in_the_order_of_the_outputs(self) -> None:
        surrogate = makeSurrogate(outcome_names=("cost", "yield"))

        assert surrogate.outcome_names == ("cost", "yield")
        assert [outcome.name for outcome in surrogate.outcomes] == ["cost", "yield"]
        assert all(outcome.type == "continuous" for outcome in surrogate.outcomes)

    def test_it_may_cover_only_some_of_the_outcomes_of_the_problem(self) -> None:
        surrogate = makeSurrogate(outcome_names=("cost",))

        prediction = surrogate.predict([MIDDLE])

        assert prediction.outcomes == ("cost",)
        assert prediction.mean == {"cost": (1.5,)}

    def test_it_reads_the_outputs_of_the_model_as_the_outcomes_in_order(self) -> None:
        surrogate = makeSurrogate(outcome_names=("cost", "yield"))

        prediction = surrogate.predict([MIDDLE])

        # Output 0 of the model is bound to "cost", output 1 to "yield".
        assert prediction.mean["cost"] == (1.5,)
        assert prediction.mean["yield"] == (3.0,)

    @pytest.mark.parametrize(
        ("names", "message"),
        [
            ((), "at least one outcome"),
            (("yield", "yield"), "more than once"),
            (("yield", "profit"), "not among the outcomes of the problem"),
        ],
    )
    def test_it_refuses_outcome_names_that_do_not_fit_the_problem(
        self, names: tuple[str, ...], message: str
    ) -> None:
        with pytest.raises(ConfigError, match=message) as error:
            makeSurrogate(model=FakeNativeModel(2), outcome_names=names)

        assert error.value.field_path == "outcome_names"

    def test_it_refuses_a_model_with_another_number_of_outputs(self) -> None:
        with pytest.raises(SurrogateModelError, match="3 output") as error:
            makeSurrogate(model=FakeNativeModel(3), outcome_names=OUTCOME_NAMES)

        assert error.value.field_path == "outcome_names"
        assert error.value.suggestion is not None

    def test_it_refuses_an_encoder_built_for_another_search_space(self) -> None:
        other = SearchSpaceEncoder(makeProblem("extra").search_space, RuntimeConfig())
        problem = makeProblem()
        built = BuiltSurrogate(
            FakeNativeModel(), FitStatus(n_observations=4, is_low_data=True)
        )  # type: ignore[arg-type]

        with pytest.raises(ConfigError, match="'extra'") as error:
            Surrogate(
                built,
                problem=problem,
                encoder=other,
                runtime=RuntimeConfig(),
                outcome_names=OUTCOME_NAMES,
            )

        assert error.value.field_path == "encoder"

    def test_it_refuses_an_encoder_of_another_precision_than_the_runtime(self) -> None:
        problem = makeProblem()
        encoder = SearchSpaceEncoder(problem.search_space, RuntimeConfig(dtype="float32"))
        built = BuiltSurrogate(
            FakeNativeModel(), FitStatus(n_observations=4, is_low_data=True)
        )  # type: ignore[arg-type]

        with pytest.raises(ConfigError, match="precision") as error:
            Surrogate(
                built,
                problem=problem,
                encoder=encoder,
                runtime=RuntimeConfig(dtype="float64"),
                outcome_names=OUTCOME_NAMES,
            )

        assert error.value.field_path == "encoder"

    def test_importing_it_does_not_import_a_model_class(self) -> None:
        # The wrapper works for any model that honours the contract (5.6): no concrete class
        # of the engine is named, and the `Model` type is only a type.
        module = importlib.import_module("planexs.surrogates.Surrogate")

        assert "Model" not in vars(module)
        assert "SingleTaskGP" not in vars(module)


class TestPredicting:
    def test_it_gives_each_outcome_a_mean_and_a_standard_deviation_at_each_point(self) -> None:
        model = FakeNativeModel()
        surrogate = makeSurrogate(model)

        prediction = surrogate.predict([MIDDLE, CORNER, MIDDLE])

        assert isinstance(prediction, Prediction)
        assert prediction.outcomes == ("yield", "cost")
        assert prediction.n_points == 3
        for table in (prediction.mean, prediction.std):
            assert set(table) == {"yield", "cost"}
            assert all(len(numbers) == 3 for numbers in table.values())

    def test_it_encodes_the_points_and_reads_the_model_at_them(self) -> None:
        surrogate = makeSurrogate()

        prediction = surrogate.predict([MIDDLE, CORNER])

        assert prediction.mean["yield"] == pytest.approx((1.5, 3.0))
        assert prediction.mean["cost"] == pytest.approx((3.0, 6.0))
        assert prediction.std["yield"] == pytest.approx((math.sqrt(0.0225), math.sqrt(0.04)))
        assert prediction.std["cost"] == pytest.approx((math.sqrt(0.045), math.sqrt(0.08)))
        assert prediction.variance["yield"] == pytest.approx((0.0225, 0.04))

    def test_it_keeps_the_order_of_the_points(self) -> None:
        surrogate = makeSurrogate()

        forward = surrogate.predict([MIDDLE, CORNER])
        backward = surrogate.predict([CORNER, MIDDLE])

        assert backward.mean["yield"] == forward.mean["yield"][::-1]
        assert backward.std["cost"] == forward.std["cost"][::-1]

    def test_it_asks_the_model_about_each_point_on_its_own_in_one_call(self) -> None:
        model = FakeNativeModel()
        surrogate = makeSurrogate(model)

        surrogate.predict([MIDDLE, CORNER])

        # A batch of two points of one: the marginal of each, and a cost linear in the points.
        assert model.calls == [((2, 1, 5), False)]

    def test_without_noise_it_describes_the_latent_function(self) -> None:
        model = FakeNativeModel()
        surrogate = makeSurrogate(model)

        prediction = surrogate.predict([MIDDLE])

        assert prediction.space == "latent"
        assert prediction.includes_observation_noise is False
        assert model.calls[-1][1] is False

    def test_the_default_options_are_those_of_the_schema(self) -> None:
        surrogate = makeSurrogate()

        assert surrogate.predict([MIDDLE]) == surrogate.predict([MIDDLE], PredictionConfig())

    def test_with_noise_it_describes_a_new_measurement(self) -> None:
        model = FakeNativeModel()
        surrogate = makeSurrogate(model)

        latent = surrogate.predict([MIDDLE])
        noisy = surrogate.predict([MIDDLE], PredictionConfig(include_observation_noise=True))

        assert noisy.space == "response"
        assert noisy.includes_observation_noise is True
        assert model.calls[-1][1] is True
        assert noisy.mean == latent.mean
        assert noisy.variance["yield"][0] == pytest.approx(0.0225 + NOISE_VARIANCE)
        assert noisy.std["cost"][0] > latent.std["cost"][0]

    def test_a_negative_variance_from_round_off_is_a_zero_standard_deviation(self) -> None:
        class RoundOff(FakeNativeModel):
            def posterior(self, X: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
                posterior = super().posterior(X, *args, **kwargs)
                posterior.variance = torch.full_like(posterior.variance, -1e-14)
                return posterior

        prediction = makeSurrogate(RoundOff()).predict([MIDDLE])

        assert prediction.std["yield"] == (0.0,)

    def test_it_works_in_single_precision(self) -> None:
        runtime = RuntimeConfig(dtype="float32")
        model = FakeNativeModel()
        surrogate = makeSurrogate(model, runtime=runtime)

        prediction = surrogate.predict([MIDDLE])

        assert prediction.mean["yield"] == pytest.approx((1.5,))
        assert surrogate.sample([MIDDLE], generator=torchGenerator(1, runtime)).dtype == (
            torch.float32
        )

    def test_it_is_deterministic(self) -> None:
        surrogate = makeSurrogate()

        assert surrogate.predict([MIDDLE, CORNER]) == surrogate.predict([MIDDLE, CORNER])

    @pytest.mark.parametrize("fault", ["nan", "shape"])
    def test_it_never_passes_on_what_the_model_got_wrong(self, fault: Any) -> None:
        surrogate = makeSurrogate(FakeNativeModel(fault=fault))

        with pytest.raises(SurrogateModelError) as error:
            surrogate.predict([MIDDLE])

        assert error.value.component == "surrogate"
        assert error.value.field_path in {"mean", "variance"}
        assert error.value.suggestion is not None


class TestPredictingAtBadPoints:
    def test_an_empty_sequence_is_refused(self) -> None:
        model = FakeNativeModel()

        with pytest.raises(ConfigError, match="No parameterization") as error:
            makeSurrogate(model).predict([])

        assert error.value.field_path == "parameterizations"
        assert model.calls == []

    def test_a_single_mapping_is_refused_with_the_fix(self) -> None:
        with pytest.raises(ConfigError, match="single parameterization") as error:
            makeSurrogate().predict(MIDDLE)  # type: ignore[arg-type]

        assert error.value.suggestion == "wrap it in a list: [parameters]"

    def test_an_illegal_point_is_refused_with_its_position(self) -> None:
        model = FakeNativeModel()

        with pytest.raises(ParameterizationError) as error:
            makeSurrogate(model).predict([MIDDLE, {**MIDDLE, "x": 11.0}])

        assert "Parameterization 1 of 2" in error.value.message
        assert "'x'" in error.value.message
        assert error.value.field_path == "x"
        # Nothing is asked of the model about a point that is not legal.
        assert model.calls == []

    def test_a_missing_parameter_is_refused(self) -> None:
        with pytest.raises(ParameterizationError, match="has no value"):
            makeSurrogate().predict([{"x": 5.0, "rate": 1e-3}])

    def test_a_point_that_breaks_a_constraint_is_refused(self) -> None:
        problem = Problem(
            search_space=SearchSpace(
                [Real("a", 0.0, 1.0), Real("b", 0.0, 1.0)],
                [LinearConstraint({"a": 1.0, "b": 1.0}, 1.0, "<=")],
            ),
            outcomes=[Outcome(name="y", type="continuous")],
        )
        surrogate = makeSurrogate(problem=problem, outcome_names=("y",))

        assert surrogate.predict([{"a": 0.2, "b": 0.3}]).n_points == 1
        with pytest.raises(ParameterizationError, match="does not hold"):
            surrogate.predict([{"a": 0.9, "b": 0.9}])


class TestNoiseForAKindThatHasNone:
    def mixed(self, model: FakeNativeModel | None = None) -> Surrogate:
        return makeSurrogate(model, problem=mixedProblem(), outcome_names=("yield", "phase"))

    def test_a_label_has_no_observation_noise_to_include(self) -> None:
        model = FakeNativeModel()

        with pytest.raises(ConfigError, match=r"\['phase'\]") as error:
            self.mixed(model).predict(
                [MIDDLE], PredictionConfig(include_observation_noise=True)
            )

        assert error.value.field_path == "include_observation_noise"
        assert model.calls == []

    def test_samples_refuse_it_too(self) -> None:
        with pytest.raises(ConfigError, match="phase"):
            self.mixed().sample(
                [MIDDLE],
                SamplingConfig(include_observation_noise=True),
                generator=generator(),
            )

    def test_the_latent_function_of_a_label_can_be_predicted(self) -> None:
        prediction = self.mixed().predict([MIDDLE])

        assert prediction.space == "latent"
        assert prediction.outcomes == ("yield", "phase")


class TestSampling:
    def test_it_draws_samples_of_every_outcome_at_every_point(self) -> None:
        surrogate = makeSurrogate()

        draws = surrogate.sample(
            [MIDDLE, CORNER, MIDDLE], SamplingConfig(n_samples=11), generator=generator()
        )

        assert tuple(draws.shape) == (11, 3, 2)
        assert draws.dtype == torch.float64
        assert bool(torch.isfinite(draws).all())

    def test_it_draws_256_samples_by_default(self) -> None:
        draws = makeSurrogate().sample([MIDDLE], generator=generator())

        assert tuple(draws.shape) == (256, 1, 2)

    def test_it_draws_the_points_jointly(self) -> None:
        model = FakeNativeModel()
        surrogate = makeSurrogate(model)

        surrogate.sample([MIDDLE, CORNER], SamplingConfig(n_samples=3), generator=generator())

        # One joint distribution over the two points, not two marginals.
        assert model.calls == [((2, 5), False)]

    def test_the_same_generator_state_gives_the_same_samples(self) -> None:
        surrogate = makeSurrogate()
        options = SamplingConfig(n_samples=20)

        first = surrogate.sample([MIDDLE, CORNER], options, generator=generator(7))
        second = surrogate.sample([MIDDLE, CORNER], options, generator=generator(7))

        assert torch.equal(first, second)

    def test_another_seed_gives_other_samples(self) -> None:
        surrogate = makeSurrogate()
        options = SamplingConfig(n_samples=20)

        first = surrogate.sample([MIDDLE], options, generator=generator(7))
        second = surrogate.sample([MIDDLE], options, generator=generator(8))

        assert not torch.equal(first, second)

    def test_the_generator_advances_so_a_second_call_gives_other_samples(self) -> None:
        surrogate = makeSurrogate()
        options = SamplingConfig(n_samples=20)
        shared = generator(7)

        first = surrogate.sample([MIDDLE], options, generator=shared)
        second = surrogate.sample([MIDDLE], options, generator=shared)

        assert not torch.equal(first, second)

    def test_it_never_touches_the_global_random_state(self) -> None:
        surrogate = makeSurrogate()
        torch.manual_seed(123)
        before = torch.get_rng_state()

        surrogate.sample([MIDDLE], SamplingConfig(n_samples=50), generator=generator())

        assert torch.equal(torch.get_rng_state(), before)

    def test_the_samples_follow_the_posterior(self) -> None:
        # With 4000 samples the standard error of an average is the standard deviation over
        # about 63, and of a standard deviation about 1.2 %: the tolerances below are about
        # six times that, and the seed is fixed, so the test cannot flake.
        surrogate = makeSurrogate()

        draws = surrogate.sample(
            [MIDDLE, CORNER], SamplingConfig(n_samples=4000), generator=generator()
        )

        expected_mean = torch.tensor([[1.5, 3.0], [3.0, 6.0]], dtype=torch.float64)
        expected_std = torch.tensor(
            [[math.sqrt(0.0225), math.sqrt(0.045)], [math.sqrt(0.04), math.sqrt(0.08)]],
            dtype=torch.float64,
        )
        assert torch.allclose(
            draws.mean(dim=0), expected_mean, atol=0.1 * float(expected_std.max())
        )
        assert torch.allclose(draws.std(dim=0), expected_std, rtol=0.08)

    def test_with_noise_the_samples_are_more_spread_out(self) -> None:
        surrogate = makeSurrogate()
        quiet = surrogate.sample(
            [MIDDLE], SamplingConfig(n_samples=2000), generator=generator()
        )
        noisy = surrogate.sample(
            [MIDDLE],
            SamplingConfig(n_samples=2000, include_observation_noise=True),
            generator=generator(),
        )

        # The standard deviation goes from 0.15 to sqrt(0.0225 + 0.25) = 0.52.
        assert float(noisy[:, 0, 0].std()) > 2.0 * float(quiet[:, 0, 0].std())

    def test_a_generator_is_required(self) -> None:
        with pytest.raises(ConfigError, match=r"seeded torch\.Generator") as error:
            makeSurrogate().sample([MIDDLE], generator=None)  # type: ignore[arg-type]

        assert error.value.field_path == "generator"

    def test_the_points_are_checked_as_for_a_prediction(self) -> None:
        surrogate = makeSurrogate()

        with pytest.raises(ConfigError, match="No parameterization"):
            surrogate.sample([], generator=generator())
        with pytest.raises(ParameterizationError, match="Parameterization 0 of 1"):
            surrogate.sample([{**MIDDLE, "x": -1.0}], generator=generator())

    @pytest.mark.parametrize("fault", ["nan", "shape"])
    def test_it_never_passes_on_samples_that_are_wrong(self, fault: Any) -> None:
        surrogate = makeSurrogate(FakeNativeModel(fault=fault))

        with pytest.raises(SurrogateModelError) as error:
            surrogate.sample([MIDDLE], SamplingConfig(n_samples=5), generator=generator())

        assert error.value.field_path == "samples"


#: What a property name must look like: lowercase snake_case, the spelling of a stored or
#: negligible-cost value (specification 10.2).
SNAKE_CASE = re.compile(r"[a-z][a-z0-9]*(_[a-z0-9]+)*")


def propertyNames(cls: type) -> list[str]:
    """The names of the properties that a class defines itself (not those of its bases)."""
    return [name for name, member in vars(cls).items() if isinstance(member, property)]


class TestPropertyNaming:
    def test_the_properties_of_a_surrogate_are_cheap_and_in_snake_case(self) -> None:
        model = FakeNativeModel()
        surrogate = makeSurrogate(model)
        names = propertyNames(Surrogate)

        assert sorted(names) == [
            "encoder",
            "link",
            "model",
            "outcome_names",
            "outcomes",
            "problem",
            "status",
        ]
        assert all(SNAKE_CASE.fullmatch(name) for name in names)
        for name in names:
            getattr(surrogate, name)
        # Reading a property never asks the model anything: that is what makes it cheap.
        assert model.calls == []

    def test_what_asks_the_model_is_a_method(self) -> None:
        for name in ("predict", "sample", "latentLevel"):
            member = inspect.getattr_static(Surrogate, name)

            assert not isinstance(member, property)
            assert callable(member)

    def test_the_properties_of_a_prediction_are_cheap_and_in_snake_case(self) -> None:
        names = propertyNames(Prediction)

        assert sorted(names) == ["n_points", "variance"]
        assert all(SNAKE_CASE.fullmatch(name) for name in names)
