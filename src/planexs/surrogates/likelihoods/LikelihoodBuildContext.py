"""The read-only information a likelihood builder receives."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

from planexs.core.RuntimeConfig import RuntimeConfig
from planexs.domain.Outcome import Outcome
from planexs.encoding.torch_runtime import torchDevice, torchDtype
from planexs.surrogates.SurrogateContextError import SurrogateContextError

#: Who the errors of this module are about.
_COMPONENT = "likelihood build context"


@dataclass(frozen=True, eq=False, kw_only=True)
class LikelihoodBuildContext:
    """Everything a likelihood builder is given to make a native likelihood (specification
    5.6, 6).

    A likelihood says how an observation relates to the latent function of the model. What it
    needs to be made is the outcomes it models (their kinds decide whether it can), the
    precision and device to compute in, the shape of the batch of models it serves, and the
    observation noise that the caller reported, if any. It is built with keywords,
    `LikelihoodBuildContext(outcomes=..., runtime=...)`.

    **Batch shape.** `batch_shape` is the batch shape of the native likelihood, which is the
    surrogate builder's choice: a surrogate that models several outcomes as independent
    outputs of one batched model gives `torch.Size([m])`, a surrogate of one model gives the
    empty shape. It is the shape the engine's own models use for their likelihood.

    **Known noise.** `noise_variance` is `None` when the noise is to be inferred. Otherwise it
    holds one variance per observation for each model of the batch, with the shape
    `batch_shape + (n,)`, which is the layout that the engine's fixed-noise likelihood takes.
    It is a variance, and not a standard deviation: a surrogate builder squares the standard
    deviations of `SurrogateBuildContext.train_noise_std`. It is in the space in which the
    model is trained, so after any outcome transform: an engine model standardizes the outcomes
    it is given, and the noise must be on that scale too. Whether partial knowledge of noise
    becomes a tensor or `None` is decided before the context is built, by configuration, never
    by a hidden rule (specification 5.4).

    **Checked when it is built.** The outcomes (at least one, each named once), and the noise
    tensor against the runtime settings and the batch shape, so that a builder can rely on
    them and a mistake surfaces with the name of the field, and not as a shape error inside
    the engine (specification 3.6): the tensor has the precision and the device that `runtime`
    names, the shape above, only finite values, and no negative variance.

    **Read-only.** The context is frozen: no field can be assigned, deleted or added. The
    noise tensor is shared with the caller and not copied, so a builder must not modify it in
    place. Contexts are compared by identity (`eq=False`): equality of tensors is not a
    boolean.

    Attributes:
        outcomes: The outcomes that the likelihood models, in the order of the outputs of the
            model.
        runtime: The precision and device of every tensor.
        batch_shape: The batch shape of the native likelihood. The empty shape by default: one
            model.
        noise_variance: The known observation variances, shape `batch_shape + (n,)`, or
            `None` to infer the noise.

    Raises:
        SurrogateContextError: if the data is malformed in any of the ways above. The message
            names the field and says how to fix it.
        ConfigError: if torch has no type or no usable device of the names that `runtime`
            gives (see `torchDtype` and `torchDevice`).
    """

    outcomes: tuple[Outcome, ...]
    runtime: RuntimeConfig
    batch_shape: torch.Size = field(default_factory=torch.Size)
    noise_variance: Tensor | None = None

    def __post_init__(self) -> None:
        """Checks the outcomes and the noise against the runtime settings and the batch shape.

        Raises:
            SurrogateContextError: if the outcomes or the noise tensor are malformed.
            ConfigError: if the runtime names a dtype or a device that torch cannot use.
        """
        self._checkOutcomes()
        # Frozen dataclass: `object.__setattr__` is the sanctioned way to normalize.
        object.__setattr__(self, "outcomes", tuple(self.outcomes))
        object.__setattr__(self, "batch_shape", torch.Size(self.batch_shape))
        if self.noise_variance is not None:
            self._checkNoise(self.noise_variance)

    def _checkOutcomes(self) -> None:
        """Checks that there is an outcome and that each is named once.

        Raises:
            SurrogateContextError: if there is none, or a name repeats.
        """
        names = [outcome.name for outcome in self.outcomes]
        if not names:
            raise _fault(
                "outcomes",
                "A likelihood models at least one outcome, and outcomes is empty.",
                "give the outcomes that the model is fitted on",
            )
        repeated = sorted({name for name in names if names.count(name) > 1})
        if repeated:
            raise _fault(
                "outcomes",
                f"outcomes names {repeated!r} more than once: each output of the model is one "
                "outcome.",
                "name each outcome once",
            )

    def _checkNoise(self, noise: Tensor) -> None:
        """Checks the noise tensor against the runtime settings and the batch shape.

        Args:
            noise: The known noise variances.

        Raises:
            SurrogateContextError: if its precision, device, shape or values are wrong.
            ConfigError: if the runtime names a dtype or a device that cannot be used.
        """
        dtype = torchDtype(self.runtime)
        device = torchDevice(self.runtime)
        if noise.dtype != dtype:
            raise _fault(
                "noise_variance",
                f"noise_variance has the dtype {noise.dtype}, but the runtime settings say "
                f"{self.runtime.dtype!r}: BoTorch does not mix precisions.",
                f"convert it with .to({dtype}), or change runtime.dtype",
            )
        if not _isOnDevice(noise.device, device):
            raise _fault(
                "noise_variance",
                f"noise_variance is on the device {noise.device}, but the runtime settings "
                f"say {self.runtime.device!r}.",
                f"move it with .to({self.runtime.device!r}), or change runtime.device",
            )
        leading = tuple(self.batch_shape)
        if noise.ndim != len(leading) + 1 or tuple(noise.shape[:-1]) != leading:
            raise _fault(
                "noise_variance",
                f"noise_variance must have the shape {(*leading, 'n')} (batch_shape and then "
                f"one variance per observation); got {tuple(noise.shape)}.",
                "give one variance per observation for each model of the batch, in the layout "
                "of the engine's fixed-noise likelihood",
            )
        if not bool(torch.isfinite(noise).all()):
            raise _fault(
                "noise_variance",
                "noise_variance holds a value that is not finite (nan or an infinity): a "
                "likelihood cannot be fitted on it.",
                "leave the noise out (None) to infer it, or report the evaluation as failed",
            )
        if bool((noise < 0.0).any()):
            raise _fault(
                "noise_variance",
                "noise_variance holds a negative value: a variance is zero or more.",
                "give variances, not standard deviations: square the standard deviations",
            )


def _fault(field_name: str, message: str, suggestion: str) -> SurrogateContextError:
    """Builds the error for one malformed field of a likelihood build context.

    Args:
        field_name: The name of the field that is at fault.
        message: What is wrong.
        suggestion: How to fix it.

    Returns:
        The error, naming the context, the field and the way to fix it.
    """
    return SurrogateContextError(
        message, component=_COMPONENT, field_path=field_name, suggestion=suggestion
    )


def _isOnDevice(actual: torch.device, expected: torch.device) -> bool:
    """Whether a tensor on `actual` lives on the device that the runtime names.

    The types must be the same. The index is compared only when both have one: the runtime
    saying `cuda` accepts any CUDA device, and a CPU tensor reports no index at all.

    Args:
        actual: The device of the tensor.
        expected: The device the runtime settings name.

    Returns:
        `True` if the tensor is where the runtime says computation happens.
    """
    if actual.type != expected.type:
        return False
    return actual.index is None or expected.index is None or actual.index == expected.index
