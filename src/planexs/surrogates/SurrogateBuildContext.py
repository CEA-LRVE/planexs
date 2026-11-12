"""The read-only information a surrogate builder receives."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from planexs.core.RuntimeConfig import RuntimeConfig
from planexs.domain.Problem import Problem
from planexs.encoding.SearchSpaceEncoder import SearchSpaceEncoder
from planexs.encoding.torch_runtime import torchDevice, torchDtype
from planexs.surrogates.SurrogateContextError import SurrogateContextError

#: Who the errors of this module are about.
_COMPONENT = "surrogate build context"


@dataclass(frozen=True, eq=False, kw_only=True)
class SurrogateBuildContext:
    """Everything a surrogate builder is given to fit a model (specification 4, 5.6).

    The build context is the "read-only information a builder receives": the problem, the
    encoder that binds it to tensors, the observations as tensors, the precision and device
    to compute in, and a random generator. A builder reads it and returns a `BuiltSurrogate`;
    it never reaches for anything that is not in it (a global random state, a study, a
    default). It is built with keywords, `SurrogateBuildContext(problem=..., encoder=...)`,
    which is how the many tensors stay unmistakable at a call site.

    **Layout of the data.** `n` is the number of observations, `d` the number of encoded
    columns of the encoder and `m` the number of outcomes of the problem.

    - `train_inputs` has shape `(n, d)`, in the unit cube: what `SearchSpaceEncoder` produces,
      so a surrogate never needs to know about log scales, grids or categories.
    - `train_outcomes` has shape `(n, m)`, one column per outcome, in the order of
      `problem.outcomes`, in the outcomes' own units. A label of a binary outcome is `0.0` or
      `1.0`, as `Observation` stores it.
    - `train_noise_std` is `None` when no observation noise is known, and otherwise has the
      shape of `train_outcomes`: the known standard deviation of each value, in the units of
      the outcome, as `Observation.noise_std` reports it. A builder that wants a variance
      squares it. Whether and how partial knowledge of noise is turned into this tensor (or
      into `None`) is decided before the context is built, by configuration, never by a hidden
      rule (specification 5.4).

    **Checked when it is built.** The data is checked once, here, so that a builder can rely on
    it and a mistake surfaces with the name of the field, and not as a shape error inside the
    engine (specification 3.6): shapes as above; every tensor has the dtype and lives on the
    device that `runtime` names (BoTorch refuses to mix them); every value is finite
    (specification 8: non-finite values never propagate silently); a noise standard deviation
    is not negative; every input is in `[0, 1]`; and the encoder only encodes parameters of the
    problem. An empty history, `n = 0`, is legal: it is the builder that says whether that is
    enough to fit a model (`ModelNotReadyError`).

    **Read-only.** The context is frozen: no field can be assigned, deleted or added. The
    tensors are shared with the caller and not copied, which is cheap and which a frozen
    dataclass cannot prevent from being abused, so a builder must not modify them in place.
    The generator is the one thing that is meant to change: drawing from it advances it. The
    study gives each build a generator of its own, derived for the ask and the component
    (specification 3.7, 5.11), so no builder disturbs the stream of another.

    Contexts are compared by identity (`eq=False`): equality of tensors is not a boolean.

    Attributes:
        problem: What can be varied and what is measured.
        encoder: The encoder of the problem's search space, which produced `train_inputs`.
        train_inputs: The encoded observed points, shape `(n, d)`, in the unit cube.
        train_outcomes: The observed values, shape `(n, m)`, one column per outcome.
        train_noise_std: The known noise standard deviations, shape `(n, m)`, or `None`.
        generator: The seeded generator that every random choice of the build draws from.
        runtime: The precision and device of every tensor.

    Raises:
        SurrogateContextError: if the data is malformed in any of the ways above. The message
            names the field and says how to fix it.
        ConfigError: if torch has no type or no usable device of the names that `runtime`
            gives (see `torchDtype` and `torchDevice`).
    """

    problem: Problem
    encoder: SearchSpaceEncoder
    train_inputs: Tensor
    train_outcomes: Tensor
    generator: torch.Generator
    runtime: RuntimeConfig
    train_noise_std: Tensor | None = None

    def __post_init__(self) -> None:
        """Checks the data of the context against the problem, the encoder and the runtime.

        Raises:
            SurrogateContextError: if a shape, a dtype, a device or a value is wrong.
            ConfigError: if the runtime names a dtype or a device that torch cannot use.
        """
        self._checkEncoder()
        self._checkShapes()
        self._checkPlacement()
        self._checkValues()

    @property
    def n_observations(self) -> int:
        """The number of observations, the rows of `train_inputs` (a shape read: cheap,
        snake_case).
        """
        return int(self.train_inputs.shape[0])

    @property
    def outcome_names(self) -> tuple[str, ...]:
        """The names of the outcomes, in the order of the columns of `train_outcomes` (a
        short tuple built from the problem: cheap, snake_case).
        """
        return tuple(outcome.name for outcome in self.problem.outcomes)

    def _checkEncoder(self) -> None:
        """Checks that the encoder only encodes parameters of the problem.

        Raises:
            SurrogateContextError: if it encodes a parameter that the problem does not have.
        """
        known = set(self.problem.search_space.parameter_names)
        foreign = sorted({encoding.name for encoding in self.encoder.encodings} - known)
        if foreign:
            raise _fault(
                "encoder",
                f"The encoder encodes {foreign!r}, which "
                f"{'is' if len(foreign) == 1 else 'are'} not among the parameters of the "
                f"problem {sorted(known)!r}: it was built for another search space.",
                "build the encoder from problem.search_space",
            )

    def _checkShapes(self) -> None:
        """Checks the shapes of the three tensors against the encoder and the problem.

        Raises:
            SurrogateContextError: if a shape is wrong.
        """
        inputs = self.train_inputs
        columns = self.encoder.n_columns
        if inputs.ndim != 2 or inputs.shape[1] != columns:
            raise _fault(
                "train_inputs",
                f"train_inputs must have shape (n, {columns}), one column per encoded column "
                f"of the encoder; got {tuple(inputs.shape)}.",
                "encode each parameterization with SearchSpaceEncoder.encodeParameters and "
                "stack the rows",
            )
        n_rows = inputs.shape[0]
        n_outcomes = len(self.problem.outcomes)
        outcomes = self.train_outcomes
        if outcomes.ndim != 2 or tuple(outcomes.shape) != (n_rows, n_outcomes):
            raise _fault(
                "train_outcomes",
                f"train_outcomes must have shape ({n_rows}, {n_outcomes}), one row per "
                f"observation and one column per outcome {list(self.outcome_names)!r}; got "
                f"{tuple(outcomes.shape)}.",
                "give one row per row of train_inputs and the outcomes in the order of "
                "problem.outcomes",
            )
        noise = self.train_noise_std
        if noise is not None and tuple(noise.shape) != tuple(outcomes.shape):
            raise _fault(
                "train_noise_std",
                f"train_noise_std must have the shape of train_outcomes "
                f"{tuple(outcomes.shape)}; got {tuple(noise.shape)}.",
                "give a standard deviation for every value, or pass None to let the surrogate "
                "infer the noise",
            )

    def _checkPlacement(self) -> None:
        """Checks that every tensor has the dtype and the device that the runtime names.

        Raises:
            SurrogateContextError: if one has not.
            ConfigError: if the runtime names a dtype or a device that cannot be used.
        """
        dtype = torchDtype(self.runtime)
        device = torchDevice(self.runtime)
        for name, tensor in self._tensors():
            if tensor.dtype != dtype:
                raise _fault(
                    name,
                    f"{name} has the dtype {tensor.dtype}, but the runtime settings say "
                    f"{self.runtime.dtype!r}: BoTorch does not mix precisions.",
                    f"convert it with .to({dtype}), or change runtime.dtype",
                )
            if not _isOnDevice(tensor.device, device):
                raise _fault(
                    name,
                    f"{name} is on the device {tensor.device}, but the runtime settings say "
                    f"{self.runtime.device!r}.",
                    f"move it with .to({self.runtime.device!r}), or change runtime.device",
                )

    def _checkValues(self) -> None:
        """Checks that every value is finite, the noise is not negative, the inputs are in the
        unit cube.

        Raises:
            SurrogateContextError: if a value is not.
        """
        for name, tensor in self._tensors():
            if not bool(torch.isfinite(tensor).all()):
                raise _fault(
                    name,
                    f"{name} holds a value that is not finite (nan or an infinity): a "
                    "surrogate cannot be fitted on it.",
                    "report an evaluation that has no value as failed, and leave it out",
                )
        if bool(((self.train_inputs < 0.0) | (self.train_inputs > 1.0)).any()):
            raise _fault(
                "train_inputs",
                "train_inputs holds a value outside [0, 1]: surrogates work on the encoding "
                "of the search space, in the unit cube, and not on natural values.",
                "encode each parameterization with SearchSpaceEncoder.encodeParameters",
            )
        noise = self.train_noise_std
        if noise is not None and bool((noise < 0.0).any()):
            raise _fault(
                "train_noise_std",
                "train_noise_std holds a negative value: a standard deviation is zero or "
                "more.",
                "give the standard deviation (not the variance), as a number of zero or more",
            )

    def _tensors(self) -> list[tuple[str, Tensor]]:
        """The tensors of the context with their field names, the noise only if it is given.

        Returns:
            One `(field name, tensor)` pair per tensor.
        """
        found = [("train_inputs", self.train_inputs), ("train_outcomes", self.train_outcomes)]
        if self.train_noise_std is not None:
            found.append(("train_noise_std", self.train_noise_std))
        return found


def _fault(field: str, message: str, suggestion: str) -> SurrogateContextError:
    """Builds the error for one malformed field of a build context.

    Args:
        field: The name of the field that is at fault.
        message: What is wrong.
        suggestion: How to fix it.

    Returns:
        The error, naming the context, the field and the way to fix it.
    """
    return SurrogateContextError(
        message, component=_COMPONENT, field_path=field, suggestion=suggestion
    )


def _isOnDevice(actual: torch.device, expected: torch.device) -> bool:
    """Whether a tensor on `actual` lives on the device that the runtime names.

    The types must be the same. The index is compared only when both have one: the runtime
    saying `cuda` accepts any CUDA device, and a CPU tensor reports no index at all, where
    `torch.device("cpu:0")` has one, so the runtime saying `cpu:0` must accept it too.

    Args:
        actual: The device of the tensor.
        expected: The device the runtime settings name.

    Returns:
        `True` if the tensor is where the runtime says computation happens.
    """
    if actual.type != expected.type:
        return False
    return actual.index is None or expected.index is None or actual.index == expected.index
