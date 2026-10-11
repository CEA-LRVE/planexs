"""Builds and fits the surrogate model the acquisition layer works on."""

from __future__ import annotations

import logging

from botorch.fit import fit_gpytorch_mll
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.model import Model
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch import Tensor

logger = logging.getLogger(__name__)

#: Smallest observation variance passed to the GP. A deterministic observation
#: (`y_std=0`) would otherwise make the covariance matrix singular.
_MIN_OBSERVATION_VARIANCE = 1e-10


def buildSurrogateModel(
    train_x: Tensor,
    train_y: Tensor,
    train_yvar: Tensor | None = None,
    minimum_points_for_free_fit: int = 5,
) -> Model:
    """Builds a `SingleTaskGP` on `train_x`/`train_y` and fits its
    hyperparameters. The single swap point for a scalable/sparse GP in a future
    version (section 4.4).

    The model has one independent output per column of `train_y`, standardized
    internally, so its posterior is reported in the original units of `train_y`.
    `train_x` is expected to already be in the unit cube (see
    `SearchSpaceEncoder`), so no input transform is applied.

    Observation noise is inferred when `train_yvar` is `None`, and fixed to
    `train_yvar` otherwise.

    Fitting maximizes the marginal likelihood *including* BoTorch's default
    priors on lengthscales and noise, i.e. it is a MAP fit, which is what keeps
    the fit usable on few points. `minimum_points_for_free_fit` is the number
    of observations below which the fit is considered low-confidence: in this
    Phase 2 version that only logs a message. Phase 3 replaces it with an
    explicit weakly-informative-prior fallback and a low-confidence signal
    (section 4.4, section 6).

    Args:
        train_x: Observed inputs, shape `(n, d)`, in the unit cube.
        train_y: Observed outcomes, shape `(n, m)`.
        train_yvar: Known observation variances, shape `(n, m)`, or `None`.
        minimum_points_for_free_fit: See above.

    Returns:
        The fitted model, in eval mode.

    Raises:
        ValueError: if fewer than two points are given (the outcome cannot be
            standardized), or the shapes do not agree.
    """
    if train_x.ndim != 2 or train_y.ndim != 2 or train_x.shape[0] != train_y.shape[0]:
        raise ValueError(
            "train_x must be (n, d) and train_y (n, m) with the same n; got "
            f"{tuple(train_x.shape)} and {tuple(train_y.shape)}."
        )
    if train_yvar is not None and train_yvar.shape != train_y.shape:
        raise ValueError(
            f"train_yvar must have the shape of train_y {tuple(train_y.shape)}; "
            f"got {tuple(train_yvar.shape)}."
        )
    n_points = train_x.shape[0]
    if n_points < 2:
        raise ValueError(f"At least 2 observations are needed to fit a model; got {n_points}.")
    if n_points < minimum_points_for_free_fit:
        logger.info(
            "Fitting the surrogate model on only %d observation(s) (< %d): its uncertainty "
            "estimates are low-confidence.",
            n_points,
            minimum_points_for_free_fit,
        )

    fixed_variance = (
        train_yvar.clamp_min(_MIN_OBSERVATION_VARIANCE) if train_yvar is not None else None
    )
    model = SingleTaskGP(
        train_X=train_x,
        train_Y=train_y,
        train_Yvar=fixed_variance,
        outcome_transform=Standardize(m=train_y.shape[-1]),
    )
    fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
    model.eval()
    return model
