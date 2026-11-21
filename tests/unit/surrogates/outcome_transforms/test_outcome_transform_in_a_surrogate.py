"""An outcome transform behind the `Surrogate` wrapper (roadmap task P3-12, specification 5.6,
3.11).

`single_task_gp_v1` (roadmap P3-14) will give its model the transform that
`buildOutcomeTransform` makes, and read the predictions through `Surrogate`. This checks that
the two work together as that surrogate will need: the predictions come back in the units of
the outcome, whatever the chain, a log-normal posterior can be sampled as well as read, and the
observation noise can be asked for.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from outcome_transform_fakes import fitModel, toyData

import planexs.surrogates  # noqa: F401
from planexs.core.ComponentConfig import ComponentConfig
from planexs.core.resolveComponent import resolveComponent
from planexs.core.RuntimeConfig import RuntimeConfig
from planexs.domain.Outcome import Outcome
from planexs.domain.parameters.Real import Real
from planexs.domain.Problem import Problem
from planexs.domain.SearchSpace import SearchSpace
from planexs.encoding.SearchSpaceEncoder import SearchSpaceEncoder
from planexs.encoding.torch_runtime import torchGenerator
from planexs.surrogates.BuiltSurrogate import BuiltSurrogate
from planexs.surrogates.FitStatus import FitStatus
from planexs.surrogates.outcome_transforms.buildOutcomeTransform import buildOutcomeTransform
from planexs.surrogates.outcome_transforms.outcomeTransformRegistry import (
    outcomeTransformRegistry,
)
from planexs.surrogates.PredictionConfig import PredictionConfig
from planexs.surrogates.SamplingConfig import SamplingConfig
from planexs.surrogates.Surrogate import Surrogate

# The toy models of a logarithm and of a raw outcome are not fitted on standardized targets,
# and BoTorch warns about it.
pytestmark = pytest.mark.filterwarnings("ignore::botorch.exceptions.warnings.InputDataWarning")

RUNTIME = RuntimeConfig()

# Measured on the toy: the predicted mean is within 0.07 of the observations at the training
# inputs, whose standard deviation is 2.6. The tolerance is 10% of that deviation.
TOLERANCE = 0.25

CHAINS = {
    "standardize": [{"type": "standardize_v1"}],
    "log": [{"type": "log_v1"}],
    "log-then-standardize": [{"type": "log_v1"}, {"type": "standardize_v1"}],
    "none": [{"type": "none_v1"}],
}


def configs(chain: list[dict[str, Any]]) -> list[ComponentConfig]:
    return [resolveComponent(outcomeTransformRegistry(), step).config for step in chain]


def makeSurrogate(
    chain: list[dict[str, Any]],
) -> tuple[Surrogate, list[dict[str, float]], Any]:
    """A surrogate of the toy function, fitted with the transforms of `chain`.

    The two inputs are real numbers in the unit interval, so that the encoding is the identity
    and the training inputs are the encoded points.

    Returns:
        The surrogate, the training points as parameterizations, and the training outcomes.
    """
    train_x, train_y, _, _ = toyData()
    problem = Problem(
        search_space=SearchSpace([Real("a", 0.0, 1.0), Real("b", 0.0, 1.0)]),
        outcomes=[Outcome(name="y", type="continuous")],
    )
    model = fitModel(buildOutcomeTransform(configs(chain), 1), train_x, train_y)
    surrogate = Surrogate(
        BuiltSurrogate(model=model, status=FitStatus(n_observations=14, is_low_data=False)),
        problem=problem,
        encoder=SearchSpaceEncoder(problem.search_space),
        runtime=RUNTIME,
        outcome_names=["y"],
    )
    points = [{"a": float(a), "b": float(b)} for a, b in train_x]
    return surrogate, points, train_y


@pytest.mark.parametrize("chain", CHAINS.values(), ids=CHAINS.keys())
class TestPredicting:
    def test_the_mean_is_in_the_units_of_the_outcome(
        self, chain: list[dict[str, Any]]
    ) -> None:
        surrogate, points, train_y = makeSurrogate(chain)

        prediction = surrogate.predict(points)

        mean = torch.tensor(prediction.mean["y"], dtype=torch.float64)
        assert float((mean - train_y.squeeze(-1)).abs().max()) < TOLERANCE

    def test_the_standard_deviation_is_finite_and_positive(
        self, chain: list[dict[str, Any]]
    ) -> None:
        surrogate, points, _ = makeSurrogate(chain)

        std = surrogate.predict(points).std["y"]

        assert all(0.0 < value < float("inf") for value in std)

    def test_it_says_that_it_describes_the_latent_function(
        self, chain: list[dict[str, Any]]
    ) -> None:
        # A transform changes the units of a prediction, never what it describes (3.11).
        surrogate, points, _ = makeSurrogate(chain)

        prediction = surrogate.predict(points)

        assert prediction.space == "latent"
        assert prediction.includes_observation_noise is False

    def test_the_observation_noise_can_be_asked_for_and_only_widens_the_prediction(
        self, chain: list[dict[str, Any]]
    ) -> None:
        surrogate, points, _ = makeSurrogate(chain)

        latent = surrogate.predict(points)
        noisy = surrogate.predict(points, PredictionConfig(include_observation_noise=True))

        assert noisy.space == "response" and noisy.includes_observation_noise is True
        assert all(
            with_noise >= without
            for with_noise, without in zip(noisy.std["y"], latent.std["y"], strict=True)
        )


@pytest.mark.parametrize("chain", CHAINS.values(), ids=CHAINS.keys())
class TestSampling:
    def test_the_samples_have_the_shape_that_a_surrogate_promises(
        self, chain: list[dict[str, Any]]
    ) -> None:
        surrogate, points, _ = makeSurrogate(chain)

        samples = surrogate.sample(
            points[:4], SamplingConfig(n_samples=32), generator=torchGenerator(1, RUNTIME)
        )

        assert samples.shape == (32, 4, 1)
        assert bool(torch.isfinite(samples).all())

    def test_the_same_generator_seed_gives_the_same_samples(
        self, chain: list[dict[str, Any]]
    ) -> None:
        surrogate, points, _ = makeSurrogate(chain)
        options = SamplingConfig(n_samples=16)

        first = surrogate.sample(points[:3], options, generator=torchGenerator(7, RUNTIME))
        second = surrogate.sample(points[:3], options, generator=torchGenerator(7, RUNTIME))
        other = surrogate.sample(points[:3], options, generator=torchGenerator(8, RUNTIME))

        assert torch.equal(first, second)
        assert not torch.equal(first, other)

    def test_the_samples_are_centred_on_the_predicted_mean(
        self, chain: list[dict[str, Any]]
    ) -> None:
        # The samples and the prediction are in the same units: their average converges on the
        # mean at the rate of one over the square root of their number, here 5 standard errors.
        surrogate, points, _ = makeSurrogate(chain)
        n_samples = 4000

        samples = surrogate.sample(
            points[:3],
            SamplingConfig(n_samples=n_samples),
            generator=torchGenerator(3, RUNTIME),
        )
        prediction = surrogate.predict(points[:3])

        mean = torch.tensor(prediction.mean["y"], dtype=torch.float64)
        std = torch.tensor(prediction.std["y"], dtype=torch.float64)
        assert bool(
            ((samples.mean(dim=0).squeeze(-1) - mean).abs() < 5 * std / n_samples**0.5).all()
        )


class TestALogarithmGivesAPositiveOutcome:
    @pytest.mark.parametrize(
        "chain", [CHAINS["log"], CHAINS["log-then-standardize"]], ids=["log", "chain"]
    )
    def test_every_sample_is_positive(self, chain: list[dict[str, Any]]) -> None:
        surrogate, _, _ = makeSurrogate(chain)
        corners = [{"a": 0.0, "b": 0.0}, {"a": 1.0, "b": 0.0}, {"a": 0.0, "b": 1.0}]

        samples = surrogate.sample(
            corners, SamplingConfig(n_samples=500), generator=torchGenerator(5, RUNTIME)
        )

        assert bool((samples > 0).all())
