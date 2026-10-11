"""Unit tests for `predictWithModel`."""

from __future__ import annotations

import pytest
import torch

from planexs.models.buildSurrogateModel import buildSurrogateModel
from planexs.models.predictWithModel import predictWithModel


@pytest.fixture(scope="module")
def model():  # type: ignore[no-untyped-def]
    x = torch.tensor([0.0, 0.2, 0.4, 0.6], dtype=torch.double).unsqueeze(-1)
    y = torch.cat([10.0 * x, 100.0 - 5.0 * x], dim=-1)
    return buildSurrogateModel(x, y)


def _at(u: float) -> torch.Tensor:
    return torch.tensor([u], dtype=torch.double)


def test_returns_a_mean_and_sem_per_named_metric(model) -> None:  # type: ignore[no-untyped-def]
    prediction = predictWithModel(model, _at(0.4), ["cost", "util"])

    assert set(prediction) == {"cost", "util"}
    assert prediction["cost"][0] == pytest.approx(4.0, abs=0.5)  # in the metric's own units
    assert prediction["util"][0] == pytest.approx(98.0, abs=2.0)
    assert all(sem >= 0.0 for _, sem in prediction.values())


def test_outputs_are_matched_to_names_in_order(model) -> None:  # type: ignore[no-untyped-def]
    swapped = predictWithModel(model, _at(0.4), ["util", "cost"])
    assert swapped["util"][0] == pytest.approx(
        4.0, abs=0.5
    )  # first output is the cost-like one


def test_uncertainty_grows_away_from_the_data(model) -> None:  # type: ignore[no-untyped-def]
    near = predictWithModel(model, _at(0.4), ["cost", "util"])
    far = predictWithModel(model, _at(5.0), ["cost", "util"])
    assert far["cost"][1] > near["cost"][1]
    assert far["util"][1] > near["util"][1]


def test_sem_is_a_standard_deviation_not_a_variance(model) -> None:  # type: ignore[no-untyped-def]
    with torch.no_grad():
        posterior = model.posterior(_at(0.3).reshape(1, -1))
    prediction = predictWithModel(model, _at(0.3), ["cost", "util"])

    assert prediction["cost"][1] == pytest.approx(float(posterior.variance[0, 0].sqrt()))


def test_rejects_names_that_do_not_match_the_outputs(model) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValueError, match=r"2 output.*1 metric"):
        predictWithModel(model, _at(0.4), ["cost"])
