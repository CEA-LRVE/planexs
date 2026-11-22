"""Unit tests for `baselinePoints`, the observed points of a fitted native model (roadmap task
P5-03, specification 5.8)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from botorch.models.model import Model
from toy_surrogates import N_OBSERVATIONS, makeToySurrogate

from planexs.acquisition.AcquisitionContextError import AcquisitionContextError
from planexs.acquisition.baselinePoints import baselinePoints


def fakeModel(train_inputs: Any) -> Model:
    """A stand-in for a native model that holds the given training inputs."""
    return cast(Model, SimpleNamespace(train_inputs=train_inputs))


class TestBaselinePoints:
    def test_a_multi_output_model_gives_its_observed_points_once(self) -> None:
        # A multi-output SingleTaskGP stores one copy of the inputs for each output.
        model = makeToySurrogate().model

        points = baselinePoints(model)

        assert model.train_inputs[0].shape == (2, N_OBSERVATIONS, 1)
        assert points.shape == (N_OBSERVATIONS, 1)
        assert torch.equal(points, model.train_inputs[0][0])

    def test_a_single_output_model_gives_its_observed_points(self) -> None:
        model = makeToySurrogate(objective_only=True).model

        points = baselinePoints(model)

        assert points.shape == (N_OBSERVATIONS, 1)

    def test_it_is_the_tensor_of_the_model_and_not_a_copy(self) -> None:
        model = makeToySurrogate(objective_only=True).model

        assert baselinePoints(model).data_ptr() == model.train_inputs[0].data_ptr()

    @pytest.mark.parametrize("train_inputs", [None, (), []])
    def test_a_model_without_training_inputs_is_refused(self, train_inputs: Any) -> None:
        with pytest.raises(AcquisitionContextError, match="no training inputs") as caught:
            baselinePoints(fakeModel(train_inputs))

        assert caught.value.field_path == "surrogate.model"

    def test_a_model_without_the_attribute_is_refused(self) -> None:
        with pytest.raises(AcquisitionContextError, match="no training inputs"):
            baselinePoints(cast(Model, SimpleNamespace()))

    def test_a_model_with_no_observed_point_is_refused(self) -> None:
        empty = (torch.empty((0, 2), dtype=torch.float64),)

        with pytest.raises(AcquisitionContextError, match="no observed point") as caught:
            baselinePoints(fakeModel(empty))

        assert caught.value.field_path == "surrogate.model"
