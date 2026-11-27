"""Helpers shared by the tests of the `single_task_gp_v1` surrogate: continuous problems of
any number of outcomes, build contexts of any size with or without known noise, and a builder
from settings.
"""

from __future__ import annotations

from typing import Any

import torch

from planexs import Real, SearchSpace
from planexs.core.RuntimeConfig import RuntimeConfig
from planexs.domain.Outcome import Outcome
from planexs.domain.Problem import Problem
from planexs.encoding.SearchSpaceEncoder import SearchSpaceEncoder
from planexs.encoding.torch_runtime import torchDtype, torchGenerator
from planexs.surrogates.SingleTaskGPBuilder import SingleTaskGPBuilder
from planexs.surrogates.SingleTaskGPConfig import SingleTaskGPConfig
from planexs.surrogates.SurrogateBuildContext import SurrogateBuildContext

#: The searched parameters of `makeProblem`, which is also its number of encoded columns.
N_COLUMNS = 2


def makeProblem(n_outcomes: int = 1) -> Problem:
    """Two real parameters in `[0, 1]` and `n_outcomes` continuous outcomes `y0`, `y1`, ..."""
    return Problem(
        search_space=SearchSpace([Real("a", 0.0, 1.0), Real("b", 0.0, 1.0)]),
        outcomes=[Outcome(name=f"y{i}", type="continuous") for i in range(n_outcomes)],
    )


def makeContext(
    n_points: int = 12,
    n_outcomes: int = 1,
    *,
    noise: bool = False,
    runtime: RuntimeConfig | None = None,
    **overrides: Any,
) -> SurrogateBuildContext:
    """A valid build context of smooth noisy functions of two parameters, the same on every
    call with the same arguments.

    Args:
        n_points: The number of observations.
        n_outcomes: The number of outcomes. Outcome `i` is `(i + 1) * sin(4a) + b ** 2` plus
            noise, so the outcomes differ in scale.
        noise: Whether the context holds a known noise standard deviation for every value, in
            `[0.05, 0.15]`.
        runtime: The runtime settings. `None` means `RuntimeConfig()`.
        **overrides: Fields of `SurrogateBuildContext` to replace.
    """
    settings = RuntimeConfig() if runtime is None else runtime
    dtype = torchDtype(settings)
    problem = makeProblem(n_outcomes)
    encoder = SearchSpaceEncoder(problem.search_space, settings)
    data = torch.Generator().manual_seed(0)
    inputs = torch.rand(n_points, N_COLUMNS, generator=data, dtype=torch.float64)
    jitter = 0.05 * torch.randn(n_points, n_outcomes, generator=data, dtype=torch.float64)
    scales = torch.arange(1, n_outcomes + 1, dtype=torch.float64)
    outcomes = torch.sin(4 * inputs[:, :1]) * scales + inputs[:, 1:] ** 2 + jitter
    deviations = 0.05 + 0.1 * torch.rand(
        n_points, n_outcomes, generator=data, dtype=torch.float64
    )
    fields: dict[str, Any] = {
        "problem": problem,
        "encoder": encoder,
        "train_inputs": inputs.to(dtype),
        "train_outcomes": outcomes.to(dtype),
        "train_noise_std": deviations.to(dtype) if noise else None,
        "generator": torchGenerator(3, settings),
        "runtime": settings,
    }
    fields.update(overrides)
    return SurrogateBuildContext(**fields)


def builderOf(**settings: Any) -> SingleTaskGPBuilder:
    """The builder of a configuration with the given settings (nested ones as mappings)."""
    return SingleTaskGPBuilder.fromConfig(SingleTaskGPConfig(**settings))


def fixedPoints(n_points: int = 25) -> torch.Tensor:
    """Fixed points of the unit cube, shape `(n_points, 1, 2)`: n batches of one point."""
    generator = torch.Generator().manual_seed(99)
    return torch.rand(n_points, 1, N_COLUMNS, generator=generator, dtype=torch.float64)
