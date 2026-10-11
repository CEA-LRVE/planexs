"""Unit tests for `buildSurrogateModel` and `encodeTrials`."""

from __future__ import annotations

import logging

import pytest
import torch

from planexs import Categorical, Metric, Objective, Real, SearchSpace, Trial
from planexs.models.buildSurrogateModel import buildSurrogateModel
from planexs.models.encodeTrials import encodeTrials
from planexs.models.SearchSpaceEncoder import SearchSpaceEncoder


def _data(n: int = 8, m: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(0)
    x = torch.rand(n, 2, generator=generator, dtype=torch.double)
    columns = [100.0 + (j + 1) * (x[:, 0] ** 2 + torch.sin(3 * x[:, 1])) for j in range(m)]
    return x, torch.stack(columns, dim=-1)


class TestBuildSurrogateModel:
    def test_fits_and_interpolates_training_data(self) -> None:
        x, y = _data()

        model = buildSurrogateModel(x, y)

        with torch.no_grad():
            mean = model.posterior(x).mean
        assert mean.shape == y.shape
        assert torch.allclose(mean, y, atol=0.5 * float(y.std()))
        assert not model.training

    def test_posterior_is_in_the_original_units_of_y(self) -> None:
        x, y = _data()
        model = buildSurrogateModel(x, y)
        with torch.no_grad():
            mean = model.posterior(torch.rand(20, 2, dtype=torch.double)).mean
        assert float(mean.mean()) == pytest.approx(float(y.mean()), rel=0.05)

    def test_one_output_per_column(self) -> None:
        x, y = _data(m=3)
        assert buildSurrogateModel(x, y).num_outputs == 3

    def test_uncertainty_grows_away_from_the_data(self) -> None:
        x, y = _data()
        model = buildSurrogateModel(x, y)
        with torch.no_grad():
            at_data = model.posterior(x).variance.mean()
            far = model.posterior(torch.full((1, 2), 25.0, dtype=torch.double)).variance.mean()
        assert far > at_data

    def test_known_noise_is_fixed_rather_than_inferred(self) -> None:
        x, y = _data()
        inferred = buildSurrogateModel(x, y)
        fixed = buildSurrogateModel(x, y, torch.full_like(y, 4.0))

        assert type(fixed.likelihood).__name__ == "FixedNoiseGaussianLikelihood"
        assert type(inferred.likelihood).__name__ != "FixedNoiseGaussianLikelihood"

    def test_zero_variance_observations_are_accepted(self) -> None:
        x, y = _data()
        model = buildSurrogateModel(x, y, torch.zeros_like(y))
        with torch.no_grad():
            assert torch.isfinite(model.posterior(x).mean).all()

    def test_needs_at_least_two_points(self) -> None:
        x, y = _data(n=1)
        with pytest.raises(ValueError, match="At least 2"):
            buildSurrogateModel(x, y)

    def test_rejects_mismatched_shapes(self) -> None:
        x, y = _data()
        with pytest.raises(ValueError, match="same n"):
            buildSurrogateModel(x, y[:-1])
        with pytest.raises(ValueError, match="train_yvar"):
            buildSurrogateModel(x, y, torch.ones(3, 1, dtype=torch.double))
        with pytest.raises(ValueError, match="same n"):
            buildSurrogateModel(x, y.squeeze(-1))

    def test_logs_when_fitting_on_few_points_and_not_otherwise(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        x, y = _data(n=3)
        with caplog.at_level(logging.INFO, logger="planexs.models.buildSurrogateModel"):
            buildSurrogateModel(x, y, minimum_points_for_free_fit=5)
        assert "low-confidence" in caplog.text

        caplog.clear()
        with caplog.at_level(logging.INFO, logger="planexs.models.buildSurrogateModel"):
            buildSurrogateModel(x, y, minimum_points_for_free_fit=3)
        assert caplog.text == ""


class TestEncodeTrials:
    @staticmethod
    def _encoder() -> SearchSpaceEncoder:
        return SearchSpaceEncoder(
            SearchSpace([Real("x", 0.0, 10.0), Categorical("c", ["a", "b", "c"])])
        )

    def test_encodes_parameters_and_reads_metrics_in_objective_order(self) -> None:
        objective = Objective([Metric("cost", minimize=True), Metric("util", minimize=False)])
        trials = [
            Trial({"x": 5.0, "c": "b"}, {"util": 2.0, "cost": 1.0}),
            Trial({"x": 10.0, "c": "c"}, {"util": 4.0, "cost": 3.0}),
        ]

        train_x, train_y, train_yvar = encodeTrials(self._encoder(), objective, trials)

        assert train_x.tolist() == [[0.5, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 1.0]]
        assert train_y.tolist() == [[1.0, 2.0], [3.0, 4.0]]
        assert train_yvar is None
        assert train_x.dtype == train_y.dtype == torch.double

    def test_result_std_becomes_a_variance_when_given_for_every_trial(self) -> None:
        objective = Objective([Metric("f", minimize=True)])
        trials = [
            Trial({"x": 1.0, "c": "a"}, {"f": 1.0}, result_std={"f": 0.5}),
            Trial({"x": 2.0, "c": "a"}, {"f": 2.0}, result_std={"f": 2.0}),
        ]

        _, _, train_yvar = encodeTrials(self._encoder(), objective, trials)

        assert train_yvar is not None
        assert train_yvar.tolist() == [[0.25], [4.0]]

    def test_partial_result_std_is_ignored_with_a_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        objective = Objective([Metric("f", minimize=True)])
        trials = [
            Trial({"x": 1.0, "c": "a"}, {"f": 1.0}, result_std={"f": 0.5}),
            Trial({"x": 2.0, "c": "a"}, {"f": 2.0}),
        ]

        with caplog.at_level(logging.WARNING, logger="planexs.models.encodeTrials"):
            _, _, train_yvar = encodeTrials(self._encoder(), objective, trials)

        assert train_yvar is None
        assert "Only 1 of 2 trials" in caplog.text

    def test_result_std_missing_one_metric_counts_as_not_given(self) -> None:
        objective = Objective([Metric("a", minimize=True), Metric("b", minimize=True)])
        trials = [
            Trial({"x": 1.0, "c": "a"}, {"a": 1.0, "b": 1.0}, result_std={"a": 0.1}),
            Trial({"x": 2.0, "c": "a"}, {"a": 2.0, "b": 2.0}, result_std={"a": 0.1}),
        ]
        assert encodeTrials(self._encoder(), objective, trials)[2] is None

    def test_a_trial_without_an_objective_metric_is_reported_by_position(self) -> None:
        objective = Objective([Metric("f", minimize=True)])
        trials = [
            Trial({"x": 1.0, "c": "a"}, {"f": 1.0}),
            Trial({"x": 2.0, "c": "a"}, {"g": 2.0}),
        ]

        with pytest.raises(
            ValueError, match=r"Trial #1 has no result for metric\(s\) \['f'\]"
        ):
            encodeTrials(self._encoder(), objective, trials)
