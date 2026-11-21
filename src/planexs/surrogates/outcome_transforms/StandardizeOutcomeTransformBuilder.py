"""The `standardize_v1` outcome transform: centre and scale each outcome."""

from __future__ import annotations

from botorch.models.transforms.outcome import OutcomeTransform, Standardize

from planexs.surrogates.outcome_transforms.AbstractOutcomeTransformBuilder import (
    AbstractOutcomeTransformBuilder,
)
from planexs.surrogates.outcome_transforms.registerOutcomeTransform import (
    registerOutcomeTransform,
)
from planexs.surrogates.outcome_transforms.StandardizeOutcomeTransformConfig import (
    StandardizeOutcomeTransformConfig,
)


@registerOutcomeTransform(
    "standardize_v1",
    config=StandardizeOutcomeTransformConfig,
    description=(
        "Centres each outcome and divides it by its standard deviation, with a floor on the "
        "standard deviation."
    ),
)
class StandardizeOutcomeTransformBuilder(
    AbstractOutcomeTransformBuilder[StandardizeOutcomeTransformConfig]
):
    """Standardizes every outcome: zero mean and unit standard deviation (BoTorch's
    `Standardize`).

    The mean and the standard deviation of each outcome are those of its training targets, and
    are stored by the transform the first time the model applies it, in training mode. It is
    what BoTorch's models do when they are given no transform, so a surrogate that uses
    `standardize_v1` with the default floor fits exactly the model that the legacy surrogate
    fitted. The transform is linear, so the predicted mean is brought back to the units of the
    outcome by the inverse affine map and the variance by the square of the scale, and a known
    observation noise given to the model is scaled with it.

    An outcome whose standard deviation is below `min_std` is only centred. A new transform is
    made at each call to `build`, because the statistics it stores belong to the model that
    applied it.
    """

    def makeTransform(self, n_outcomes: int) -> OutcomeTransform:
        """Makes the standardizing transform.

        Args:
            n_outcomes: The number of outcomes, one or more.

        Returns:
            A new `Standardize` for targets of `n_outcomes` columns, whose standard-deviation
            floor is `config.min_std`.
        """
        return Standardize(m=n_outcomes, min_stdv=self.config.min_std)
