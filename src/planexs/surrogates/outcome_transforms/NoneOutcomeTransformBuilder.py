"""The `none_v1` outcome transform: leave the outcomes as they are."""

from __future__ import annotations

from planexs.surrogates.outcome_transforms.AbstractOutcomeTransformBuilder import (
    AbstractOutcomeTransformBuilder,
)
from planexs.surrogates.outcome_transforms.NoneOutcomeTransformConfig import (
    NoneOutcomeTransformConfig,
)
from planexs.surrogates.outcome_transforms.registerOutcomeTransform import (
    registerOutcomeTransform,
)


@registerOutcomeTransform(
    "none_v1",
    config=NoneOutcomeTransformConfig,
    description="Leaves the outcomes as they are: the model is fitted to the raw values.",
)
class NoneOutcomeTransformBuilder(AbstractOutcomeTransformBuilder[NoneOutcomeTransformConfig]):
    """No transform: the model is fitted to the outcomes in their own units.

    It returns `None`, which is how BoTorch's models are told not to rescale the outcomes. That
    is a deliberate choice and not the engine default: a model that is given no
    `outcome_transform` argument standardizes its outcomes, and one that is given
    `outcome_transform=None` does not. So a surrogate that builds its transform from the
    configuration must pass the result to the model even when it is `None`, which is the rule
    that `buildOutcomeTransform` states for a list with nothing to apply.

    It is useful for outcomes that are already on a scale that suits a Gaussian process, and as
    an explicit way to say so in a configuration: leaving the transform out of a surrogate's
    configuration is not the same thing, because the field has a default.
    """

    def makeTransform(self, n_outcomes: int) -> None:
        """Makes no transform.

        Args:
            n_outcomes: The number of outcomes. Not used.

        Returns:
            `None`: no transform at all.
        """
        return None
