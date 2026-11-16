"""A fake native model and posterior, and a `Surrogate` built on them, for the tests of
`Surrogate` (roadmap task P3-09, specification 5.6).

The fake honours exactly what `Surrogate` asks of a model, which is BoTorch's contract as the
installed version has it (checked against a real `SingleTaskGP` in
`test_surrogate_with_botorch_model.py`): `num_outputs`, and `posterior(X, observation_noise=)`
returning an object with `mean`, `variance`, `base_sample_shape` and
`rsample_from_base_samples`. Its numbers are simple functions of the encoded point, so that a
test can compute what it expects by hand.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import torch
from surrogate_fakes import OUTCOME_NAMES, makeProblem
from torch import Tensor

from planexs.core.RuntimeConfig import RuntimeConfig
from planexs.domain.Problem import Problem
from planexs.encoding.SearchSpaceEncoder import SearchSpaceEncoder
from planexs.surrogates.BuiltSurrogate import BuiltSurrogate
from planexs.surrogates.FitStatus import FitStatus
from planexs.surrogates.Surrogate import Surrogate

#: The variance that observation noise adds, for every point and outcome.
NOISE_VARIANCE = 0.25

#: How a fake model can misbehave: a nan in the variance, or a mean of the wrong shape.
Fault = Literal["nan", "shape"]


class FakePosterior:
    """Independent normal marginals, with the interface of a BoTorch posterior that
    `Surrogate` uses.
    """

    def __init__(self, mean: Tensor, variance: Tensor) -> None:
        self.mean = mean
        self.variance = variance

    @property
    def base_sample_shape(self) -> torch.Size:
        """The shape of one base sample: one standard normal per entry of the mean."""
        return self.mean.shape

    def rsample_from_base_samples(
        self, sample_shape: torch.Size, base_samples: Tensor
    ) -> Tensor:
        """Samples from base samples, refusing a shape that BoTorch would refuse."""
        expected = torch.Size([*sample_shape, *self.base_sample_shape])
        if base_samples.shape != expected:
            raise RuntimeError(
                f"base_samples has shape {tuple(base_samples.shape)}, not {expected}"
            )
        return self.mean + self.variance.sqrt() * base_samples


class FakeNativeModel:
    """A model whose outcome `j` (counted from 0) has the mean `(j + 1) * sum(x)` and the
    variance `0.01 * (j + 1) * (1 + sum(x ** 2))` at the encoded point `x`, plus
    `NOISE_VARIANCE` when the observation noise is asked for.

    It records every call of `posterior`, as `(shape of X, observation_noise)`.
    """

    def __init__(self, n_outputs: int = 2, *, fault: Fault | None = None) -> None:
        self.num_outputs = n_outputs
        self.fault = fault
        self.calls: list[tuple[tuple[int, ...], bool]] = []

    def posterior(
        self,
        X: Tensor,
        output_indices: Sequence[int] | None = None,
        observation_noise: bool = False,
        posterior_transform: object = None,
    ) -> FakePosterior:
        """The fake posterior at `X`, of shape `X.shape[:-1] + (num_outputs,)`."""
        self.calls.append((tuple(X.shape), observation_noise))
        scale = torch.arange(1, self.num_outputs + 1, dtype=X.dtype, device=X.device)
        mean = X.sum(dim=-1, keepdim=True) * scale
        variance = 0.01 * (1.0 + X.pow(2).sum(dim=-1, keepdim=True)) * scale
        if observation_noise:
            variance = variance + NOISE_VARIANCE
        if self.fault == "nan":
            variance = torch.full_like(variance, float("nan"))
        if self.fault == "shape":
            mean = mean.unsqueeze(0)
        return FakePosterior(mean, variance)


def makeSurrogate(
    model: object = None,
    *,
    problem: Problem | None = None,
    outcome_names: Sequence[str] = OUTCOME_NAMES,
    runtime: RuntimeConfig | None = None,
) -> Surrogate:
    """A `Surrogate` over a fake model.

    Args:
        model: The native model. `None` means a `FakeNativeModel` with one output per name.
        problem: The problem. `None` means `makeProblem()`.
        outcome_names: The outcomes the surrogate covers.
        runtime: The runtime settings. `None` means `RuntimeConfig()`.

    Returns:
        The surrogate, whose fit status is a low-data one of four observations.
    """
    settings = RuntimeConfig() if runtime is None else runtime
    bound = makeProblem() if problem is None else problem
    native = FakeNativeModel(len(outcome_names)) if model is None else model
    built = BuiltSurrogate(
        model=native,  # type: ignore[arg-type]
        status=FitStatus(n_observations=4, is_low_data=True, warnings=("few points",)),
    )
    return Surrogate(
        built,
        problem=bound,
        encoder=SearchSpaceEncoder(bound.search_space, settings),
        runtime=settings,
        outcome_names=outcome_names,
    )
