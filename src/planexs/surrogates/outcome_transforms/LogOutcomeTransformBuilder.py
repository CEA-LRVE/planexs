"""The `log_v1` outcome transform: model the logarithm of each outcome."""

from __future__ import annotations

from botorch.models.transforms.outcome import OutcomeTransform

from planexs.surrogates.outcome_transforms.AbstractOutcomeTransformBuilder import (
    AbstractOutcomeTransformBuilder,
)
from planexs.surrogates.outcome_transforms.LogOutcomeTransformConfig import (
    LogOutcomeTransformConfig,
)
from planexs.surrogates.outcome_transforms.registerOutcomeTransform import (
    registerOutcomeTransform,
)
from planexs.surrogates.outcome_transforms.StrictlyPositiveLog import StrictlyPositiveLog


@registerOutcomeTransform(
    "log_v1",
    config=LogOutcomeTransformConfig,
    description=(
        "Models the natural logarithm of each outcome, which must be strictly positive, and "
        "predicts a log-normal outcome."
    ),
)
class LogOutcomeTransformBuilder(AbstractOutcomeTransformBuilder[LogOutcomeTransformConfig]):
    """Models the natural logarithm of every outcome (BoTorch's `Log`).

    It suits an outcome that is positive and spans orders of magnitude, or whose spread grows
    with its level (a rate, a concentration, a duration): the Gaussian process is then fitted
    to the logarithm, where such an outcome is closer to a Gaussian one, and its predictions
    are brought back as a log-normal posterior, whose mean and variance are the log-normal
    ones.

    The transform applies to every outcome of the surrogate. The engine's `Log` has a way to
    pick the outputs it applies to, but it cannot un-transform a posterior when it is used, so
    it is not offered. A study that wants the logarithm of one outcome and not of the others
    models them with separate surrogates.

    It is `StrictlyPositiveLog`, the engine's own class with one more rule: a value that is
    zero, negative or not finite is refused with an `OutcomeTransformError`, and does not reach
    the fit as a `-inf` or a `nan`. A known observation noise given to the model is transformed
    with the delta method, as BoTorch does.
    """

    def makeTransform(self, n_outcomes: int) -> OutcomeTransform:
        """Makes the logarithmic transform.

        Args:
            n_outcomes: The number of outcomes. Not used: the transform is the same for every
                outcome.

        Returns:
            A new `StrictlyPositiveLog` that applies to every outcome.
        """
        return StrictlyPositiveLog()
