"""The observed points of a fitted native model."""

from __future__ import annotations

from typing import Any, cast

from botorch.models.model import Model
from torch import Tensor

from planexs.acquisition.AcquisitionContextError import AcquisitionContextError


def baselinePoints(model: Model) -> Tensor:
    """The points a fitted model was trained on, which a noisy acquisition measures its
    improvement against (specification 5.8).

    A noisy acquisition does not take the best observed value as it is: it samples the joint
    posterior of the observed points with the candidate, so that noise in the observations is
    not mistaken for a better incumbent. The observed points are the inputs the model holds.

    Args:
        model: A fitted BoTorch model that keeps its training inputs.

    Returns:
        The training inputs, shape `(r, d)`: the encoded observed points, in the unit cube. The
        surrogates of planexs apply no input transform, so the inputs the model holds are the
        encoded points themselves. The tensor of the model itself, not a copy: do not modify
        it.

    Raises:
        AcquisitionContextError: if the model keeps no training inputs, or holds none.
    """
    inputs = getattr(cast(Any, model), "train_inputs", None)
    if not inputs:
        raise AcquisitionContextError(
            "The model of the surrogate holds no training inputs, so the observed points that "
            "a noisy acquisition measures its improvement against cannot be read from it.",
            component="acquisition build context",
            field_path="surrogate.model",
            suggestion="use a surrogate whose native model keeps its training inputs, as a "
            "fitted BoTorch model does",
        )
    train_x = cast(Tensor, inputs[0])
    while train_x.ndim > 2:  # structural: (r, d) is the shape of a set of points
        # A multi-output SingleTaskGP stores one copy of the inputs for each output.
        train_x = train_x[0]
    if train_x.shape[0] == 0:
        raise AcquisitionContextError(
            "The model of the surrogate has no observed point, so a noisy acquisition has no "
            "incumbent to improve on.",
            component="acquisition build context",
            field_path="surrogate.model",
            suggestion="build the acquisition only once the study has observations, which is "
            "what the initial design is for",
        )
    return train_x
