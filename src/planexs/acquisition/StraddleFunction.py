"""The native acquisition function of the `straddle_v1` acquisition."""

from __future__ import annotations

import math
from typing import cast

from botorch.acquisition.analytic import AnalyticAcquisitionFunction
from botorch.acquisition.objective import PosteriorTransform
from botorch.models.model import Model
from botorch.utils.transforms import t_batch_mode_transform
from torch import Tensor

from planexs.core.ConfigError import ConfigError

#: Who the errors of this module are about.
_COMPONENT = "acquisition 'straddle_v1'"


class StraddleFunction(AnalyticAcquisitionFunction):
    """`beta * sigma(x) - |mu(x) - level|`: the straddle of a level, which is high where the
    model is uncertain and its prediction is close to the level (specification 5.8, FR20).

    This is the native function that `StraddleBuilder` builds, a subclass of the engine's
    `AnalyticAcquisitionFunction` so that the engine's optimizers accept it. `mu` and `sigma`
    are the mean and the standard deviation of the posterior at the point. Read as an interval,
    the score is the depth at which the level lies inside the band `mu +- beta * sigma` around
    the prediction: the smaller of the distances from the level to the two edges of the band,
    `min(level - (mu - beta * sigma), (mu + beta * sigma) - level)`, which is positive when the
    level is inside the band and negative when it is outside. The two forms are the same
    number, and a test checks one against the other.

    - It is high where the level set is least known: a prediction on the level (`mu = level`)
      that the model is unsure about (`sigma` large) scores `beta * sigma`, the highest value a
      point of that `sigma` can have.
    - It falls with the distance of the prediction from the level, one unit of the outcome for
      one unit of distance, and rises with the uncertainty, `beta` units of the outcome for one
      standard deviation. `beta` sets the exchange rate between the two: a point `d` farther
      from the level is worth visiting if the model is `d / beta` standard deviations less sure
      of it.
    - It has no direction: `mu` enters only through `|mu - level|`, so a prediction above the
      level scores the same as one below it by the same distance.

    The score is in the units of the outcome (or of the latent function, for an outcome that a
    surrogate reads through a link). The level must be in those units too: the caller converts
    it, as `StraddleBuilder` does with `Surrogate.latentLevel`.

    **Differentiability.** The score is differentiable in the candidate wherever the predicted
    mean differs from the level. On the set where it equals the level the distance has a kink,
    and automatic differentiation returns 0 for the derivative of the distance there (the
    uncertainty term keeps its own derivative), so a gradient-based optimizer sees a
    subgradient there. That is the case on the boundary the acquisition is looking for, so
    the kink is where the score of a point that the model is sure of is highest.

    Only single candidates (`q = 1`) are evaluated per call, and pending points are not
    supported: the value at a point depends on the posterior at that point and on nothing
    else, as for the other analytic functions of the engine.

    Args:
        model: A fitted model whose posterior has a normal marginal at each point.
        beta: The number of posterior standard deviations that the uncertainty counts for.
            0 or more and finite.
        level: The level, in the units of the posterior of the model (after the posterior
            transform). Finite.
        posterior_transform: A transform that selects the outcome of the level from a
            multi-output model, or `None` for a model with one output.

    Raises:
        ConfigError: if `beta` is negative or not finite, or `level` is not finite. The error
            names the field.
    """

    def __init__(
        self,
        model: Model,
        *,
        beta: float,
        level: float,
        posterior_transform: PosteriorTransform | None = None,
    ) -> None:
        """Keeps `beta` and the level after checking them.

        Arguments and errors are described in the class docstring.
        """
        if not math.isfinite(beta) or beta < 0.0:
            raise ConfigError(
                f"beta must be 0 or more and finite; got {beta!r}.",
                component=_COMPONENT,
                field_path="beta",
                suggestion="give the weight of the uncertainty as a number of standard "
                "deviations, 0 or more",
            )
        if not math.isfinite(level):
            raise ConfigError(
                f"The level must be a finite number; got {level!r}.",
                component=_COMPONENT,
                field_path="level",
                suggestion="give the level as a number in the units of the outcome",
            )
        super().__init__(model=model, posterior_transform=posterior_transform)
        self.beta = float(beta)
        self.level = float(level)

    @t_batch_mode_transform(expected_q=1)  # type: ignore[arg-type]
    def forward(self, X: Tensor) -> Tensor:
        """Straddle of the level at single candidates.

        Args:
            X: A `batch_shape x 1 x d` tensor of single candidates.

        Returns:
            A `batch_shape` tensor.
        """
        mean, sigma = self._mean_and_sigma(X)  # each (batch_shape) x 1
        return (self.beta * cast(Tensor, sigma) - (mean - self.level).abs()).squeeze(-1)
