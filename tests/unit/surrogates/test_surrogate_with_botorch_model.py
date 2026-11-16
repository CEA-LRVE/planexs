"""`Surrogate` on a real BoTorch model (roadmap task P3-09, specification 5.6, 10.11).

The other tests of the wrapper use a fake model. These check that the fake is faithful: the
wrapper is given a fitted `SingleTaskGP` with an outcome transform, the model that the default
surrogate will build, and its answers are compared with what the engine says directly. The
parity test against the legacy `predictWithModel` is deleted with that function (roadmap task
P11-02), like every parity test of the migration (specification 10.11).
"""

from __future__ import annotations

import math
from typing import Any

import pytest
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from surrogate_fakes import OUTCOME_NAMES, makeProblem

from planexs.core.RuntimeConfig import RuntimeConfig
from planexs.encoding.SearchSpaceEncoder import SearchSpaceEncoder
from planexs.encoding.torch_runtime import torchGenerator
from planexs.models.predictWithModel import predictWithModel
from planexs.surrogates.BuiltSurrogate import BuiltSurrogate
from planexs.surrogates.FitStatus import FitStatus
from planexs.surrogates.PredictionConfig import PredictionConfig
from planexs.surrogates.SamplingConfig import SamplingConfig
from planexs.surrogates.Surrogate import Surrogate

#: Six observations of a smooth function of the problem of `makeProblem`: `yield` rises with
#: `x`, `cost` falls with `rate`, and the solvent shifts both.
POINTS: list[dict[str, Any]] = [
    {"x": 0.0, "rate": 1e-3, "solvent": "a"},
    {"x": 2.0, "rate": 1e-2, "solvent": "b"},
    {"x": 4.0, "rate": 1e-1, "solvent": "c"},
    {"x": 6.0, "rate": 1e-3, "solvent": "a"},
    {"x": 8.0, "rate": 1e-2, "solvent": "b"},
    {"x": 10.0, "rate": 1.0, "solvent": "c"},
]
SHIFT = {"a": 0.0, "b": 0.5, "c": 1.0}

#: Points the model was not fitted on.
QUERIES: list[dict[str, Any]] = [
    {"x": 3.0, "rate": 5e-3, "solvent": "a"},
    {"x": 7.0, "rate": 0.3, "solvent": "b"},
    {"x": 9.5, "rate": 1e-3, "solvent": "c"},
]

RUNTIME = RuntimeConfig()


def fit(
    n_outputs: int = 2, known_noise: float | None = None
) -> tuple[Surrogate, SingleTaskGP]:
    """A `Surrogate` over a `SingleTaskGP` fitted on `POINTS`, and the model itself."""
    problem = makeProblem()
    encoder = SearchSpaceEncoder(problem.search_space, RUNTIME)
    train_x = torch.stack([encoder.encodeParameters(point) for point in POINTS])
    columns = [
        [point["x"] + SHIFT[point["solvent"]] for point in POINTS],
        [-math.log10(point["rate"]) + SHIFT[point["solvent"]] for point in POINTS],
    ]
    train_y = torch.tensor(columns[:n_outputs], dtype=torch.double).T
    noise = (
        None
        if known_noise is None
        else torch.full_like(train_y, known_noise, dtype=torch.double)
    )
    model = SingleTaskGP(
        train_x, train_y, train_Yvar=noise, outcome_transform=Standardize(m=n_outputs)
    )
    fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
    model.eval()
    built = BuiltSurrogate(model, FitStatus(n_observations=len(POINTS), is_low_data=False))
    surrogate = Surrogate(
        built,
        problem=problem,
        encoder=encoder,
        runtime=RUNTIME,
        outcome_names=OUTCOME_NAMES[:n_outputs],
    )
    return surrogate, model


class TestAgainstTheEngine:
    def test_the_means_and_standard_deviations_are_the_ones_of_the_posterior(self) -> None:
        surrogate, model = fit()
        encoded = torch.stack([surrogate.encoder.encodeParameters(point) for point in QUERIES])

        prediction = surrogate.predict(QUERIES)

        with torch.no_grad():
            posterior = model.posterior(encoded.unsqueeze(1))
        for column, name in enumerate(OUTCOME_NAMES):
            assert prediction.mean[name] == pytest.approx(
                posterior.mean[:, 0, column].tolist(), abs=1e-12
            )
            assert prediction.std[name] == pytest.approx(
                posterior.variance[:, 0, column].sqrt().tolist(), abs=1e-12
            )
        assert prediction.space == "latent"

    def test_predicting_points_together_or_one_by_one_gives_the_same_numbers(self) -> None:
        surrogate, _ = fit()

        together = surrogate.predict(QUERIES)
        alone = [surrogate.predict([query]) for query in QUERIES]

        for name in OUTCOME_NAMES:
            assert together.mean[name] == pytest.approx(
                tuple(one.mean[name][0] for one in alone), abs=1e-9
            )
            assert together.std[name] == pytest.approx(
                tuple(one.std[name][0] for one in alone), abs=1e-9
            )

    def test_it_matches_the_legacy_prediction_of_one_point(self) -> None:
        # Parity (specification 10.11): `Prediction` replaces `PredictionResult`, and its
        # standard deviation is the standard error that `predictWithModel` returned, in the
        # same space (latent: no observation noise). Tolerance: the two read the same
        # posterior, so only round-off in the last digits can differ.
        surrogate, model = fit()

        for query in QUERIES:
            legacy = predictWithModel(
                model, surrogate.encoder.encodeParameters(query), list(OUTCOME_NAMES)
            )
            prediction = surrogate.predict([query])

            for name, (mean, sem) in legacy.items():
                assert prediction.mean[name][0] == pytest.approx(mean, abs=1e-12)
                assert prediction.std[name][0] == pytest.approx(sem, abs=1e-12)

    def test_the_model_is_nearly_certain_where_it_was_fitted(self) -> None:
        surrogate, _ = fit()

        at_data = surrogate.predict(POINTS[:2])
        away = surrogate.predict(QUERIES[:2])

        assert max(at_data.std["yield"]) < min(away.std["yield"])

    def test_the_observation_noise_adds_to_the_variance(self) -> None:
        surrogate, model = fit()
        encoded = torch.stack([surrogate.encoder.encodeParameters(point) for point in QUERIES])

        latent = surrogate.predict(QUERIES)
        noisy = surrogate.predict(QUERIES, PredictionConfig(include_observation_noise=True))

        with torch.no_grad():
            expected = model.posterior(encoded.unsqueeze(1), observation_noise=True).variance
        assert noisy.space == "response"
        assert noisy.mean == latent.mean
        for column, name in enumerate(OUTCOME_NAMES):
            assert noisy.variance[name] == pytest.approx(
                expected[:, 0, column].tolist(), abs=1e-12
            )
            assert all(
                n > latent_std
                for n, latent_std in zip(noisy.std[name], latent.std[name], strict=True)
            )

    def test_a_known_noise_adds_exactly_its_variance(self) -> None:
        # A surrogate fitted with known noise (the `noise:known` tag): the noise that a
        # response adds is the one the caller reported, here 0.01 for every observation.
        surrogate, _ = fit(n_outputs=1, known_noise=0.01)

        latent = surrogate.predict(QUERIES)
        noisy = surrogate.predict(QUERIES, PredictionConfig(include_observation_noise=True))

        for with_noise, without in zip(
            noisy.variance["yield"], latent.variance["yield"], strict=True
        ):
            assert with_noise - without == pytest.approx(0.01, abs=1e-8)

    def test_a_model_of_one_output_is_bound_to_one_outcome(self) -> None:
        surrogate, _ = fit(n_outputs=1)

        prediction = surrogate.predict(QUERIES)

        assert prediction.outcomes == ("yield",)
        assert prediction.n_points == 3


class TestSamplingFromTheEngine:
    def test_the_samples_have_the_shape_of_the_request(self) -> None:
        surrogate, _ = fit()

        draws = surrogate.sample(
            QUERIES, SamplingConfig(n_samples=17), generator=torchGenerator(3, RUNTIME)
        )

        assert tuple(draws.shape) == (17, 3, 2)
        assert bool(torch.isfinite(draws).all())

    def test_the_same_generator_state_gives_the_same_samples(self) -> None:
        surrogate, _ = fit()
        options = SamplingConfig(n_samples=30)

        first = surrogate.sample(QUERIES, options, generator=torchGenerator(5, RUNTIME))
        second = surrogate.sample(QUERIES, options, generator=torchGenerator(5, RUNTIME))
        other = surrogate.sample(QUERIES, options, generator=torchGenerator(6, RUNTIME))

        assert torch.equal(first, second)
        assert not torch.equal(first, other)

    def test_it_never_touches_the_global_random_state(self) -> None:
        surrogate, _ = fit()
        torch.manual_seed(99)
        before = torch.get_rng_state()

        surrogate.sample(QUERIES, generator=torchGenerator(5, RUNTIME))

        assert torch.equal(torch.get_rng_state(), before)

    def test_the_samples_converge_on_the_prediction(self) -> None:
        # 3000 samples: the average of a sample has a standard error of about 1.8 % of the
        # predicted standard deviation, so a tolerance of 15 % of it is more than eight
        # times that. The seed is fixed, so the test cannot flake.
        surrogate, _ = fit()

        draws = surrogate.sample(
            QUERIES, SamplingConfig(n_samples=3000), generator=torchGenerator(11, RUNTIME)
        )
        prediction = surrogate.predict(QUERIES)

        for column, name in enumerate(OUTCOME_NAMES):
            mean = torch.tensor(prediction.mean[name], dtype=torch.double)
            std = torch.tensor(prediction.std[name], dtype=torch.double)
            assert torch.all((draws[:, :, column].mean(dim=0) - mean).abs() < 0.15 * std)
            assert torch.allclose(draws[:, :, column].std(dim=0), std, rtol=0.1)

    def test_the_points_of_a_sample_are_drawn_jointly(self) -> None:
        # Two copies of one point are one point: in a joint draw they take the same value in
        # every sample, which independent draws of the marginals would not do.
        surrogate, _ = fit()

        draws = surrogate.sample(
            [QUERIES[0], QUERIES[0]],
            SamplingConfig(n_samples=50),
            generator=torchGenerator(2, RUNTIME),
        )

        assert torch.allclose(draws[:, 0, :], draws[:, 1, :], atol=1e-3)

    def test_with_noise_the_samples_are_more_spread_out(self) -> None:
        surrogate, _ = fit()
        generator = torchGenerator(4, RUNTIME)

        quiet = surrogate.sample(
            QUERIES[:1], SamplingConfig(n_samples=2000), generator=generator
        )
        noisy = surrogate.sample(
            QUERIES[:1],
            SamplingConfig(n_samples=2000, include_observation_noise=True),
            generator=generator,
        )

        assert float(noisy[:, 0, 0].std()) > float(quiet[:, 0, 0].std())
