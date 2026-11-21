"""BoTorch's `Log` outcome transform, made to refuse what it cannot take the logarithm of."""

from __future__ import annotations

import torch
from botorch.models.transforms.outcome import Log
from torch import Tensor

from planexs.surrogates.outcome_transforms.OutcomeTransformError import OutcomeTransformError


class StrictlyPositiveLog(Log):
    """A `botorch.models.transforms.outcome.Log` that only takes strictly positive, finite
    values (specification 1.4, 3.6, 8).

    BoTorch's `Log` takes `torch.log` of whatever it is given. A zero then becomes `-inf` and
    the model is built and fitted on it without a word, and a negative value becomes `nan`,
    which the model refuses with an error that names neither the outcome nor a way out. Neither
    is acceptable here: a non-finite value must never propagate silently, and an error must say
    what is wrong and how to fix it. This subclass checks the values before it transforms them,
    and is otherwise the engine's own class, so it works wherever BoTorch uses a `Log`: its
    `untransform`, its un-transformed posterior (a log-normal one) and its `subset_output` are
    inherited unchanged.

    The check is made in every mode, and not only when the model is trained: a value that the
    transform cannot take is a mistake whenever it meets one. It reads the values as BoTorch
    gives them to the transform, columns being the outputs of the model.
    """

    def forward(
        self, Y: Tensor, Yvar: Tensor | None = None, X: Tensor | None = None
    ) -> tuple[Tensor, Tensor | None]:
        """Takes the logarithm of the outcomes, after checking that each one has one.

        Args:
            Y: A `batch_shape x n x m` tensor of targets.
            Yvar: A `batch_shape x n x m` tensor of observation noises, if known.
            X: The training inputs, if any. Not used.

        Returns:
            The transformed targets and the transformed noise, as `Log.forward` returns them.

        Raises:
            OutcomeTransformError: if a value is zero, negative, infinite or `nan`. The message
                says how many values and which outputs, and gives one of the values.
        """
        invalid = ~(torch.isfinite(Y) & (Y > 0))
        if bool(invalid.any()):
            outputs = invalid.reshape(-1, Y.shape[-1]).any(dim=0).nonzero().flatten().tolist()
            raise OutcomeTransformError(
                f"The log_v1 outcome transform takes strictly positive, finite values, but "
                f"{int(invalid.sum())} value(s) of the outcome column(s) {outputs!r} are not "
                f"(for example {float(Y[invalid][0])!r}): the logarithm of such a value is "
                "minus infinity or not a number.",
                component="outcome_transform 'log_v1'",
                field_path="train_outcomes",
                suggestion=(
                    "use the standardize_v1 outcome transform for an outcome that can be zero "
                    "or negative, or leave the log_v1 transform out of the configuration"
                ),
            )
        return super().forward(Y=Y, Yvar=Yvar, X=X)
