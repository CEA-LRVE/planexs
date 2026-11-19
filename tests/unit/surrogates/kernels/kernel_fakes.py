"""Fakes shared by the tests of `planexs.surrogates.kernels`: a kernel that stands for a
plugin's, so that the contract of the kind is tested on its own, before and beside the real
ones.
"""

from __future__ import annotations

from typing import Any, Literal

import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.kernels import Kernel, LinearKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from pydantic import Field, create_model

from planexs import Real, SearchSpace
from planexs.core.ComponentConfig import ComponentConfig
from planexs.core.RuntimeConfig import RuntimeConfig
from planexs.encoding.SearchSpaceEncoder import SearchSpaceEncoder
from planexs.surrogates.kernels.AbstractKernelBuilder import AbstractKernelBuilder
from planexs.surrogates.kernels.registerKernel import registerKernel

#: The number of encoded columns of `threeColumnEncoder`: the "3-column dummy encoder" of
#: roadmap task P3-10.
N_COLUMNS = 3


def threeColumnEncoder(runtime: RuntimeConfig | None = None) -> SearchSpaceEncoder:
    """An encoder with three encoded columns: three real parameters, one column each."""
    space = SearchSpace([Real("a", 0.0, 1.0), Real("b", 0.0, 10.0), Real("c", 1.0, 100.0)])
    encoder = SearchSpaceEncoder(space, runtime)
    assert encoder.n_columns == N_COLUMNS
    return encoder


def fitAndPredict(
    kernel: Kernel | None, n_outputs: int = 1, dtype: torch.dtype = torch.float64
) -> tuple[SingleTaskGP, torch.Tensor]:
    """Fits a `SingleTaskGP` on a smooth function of three columns, as a surrogate will, with
    `kernel` as its covariance module (`None` leaves the choice to BoTorch).

    Returns:
        The fitted model and its posterior mean at 20 held-out points.
    """
    generator = torch.Generator().manual_seed(0)
    train_x = torch.rand(12, N_COLUMNS, generator=generator, dtype=dtype)
    test_x = torch.rand(20, N_COLUMNS, generator=generator, dtype=dtype)
    outputs = [torch.sin(3 * train_x[:, 0]) + k * train_x[:, 1] for k in range(n_outputs)]
    model = SingleTaskGP(
        train_x,
        torch.stack(outputs, dim=-1),
        covar_module=kernel,
        outcome_transform=Standardize(m=n_outputs),
    )
    fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
    model.eval()
    with torch.no_grad():
        mean = model.posterior(test_x).mean
    return model, mean


def fakeConfig(name: str) -> type[ComponentConfig]:
    """A configuration schema as real kernels write theirs: every field described, the `type`
    field pinned to `name`.

    Args:
        name: The registered name the schema is for.
    """
    schema: type[ComponentConfig] = create_model(
        f"{name.title().replace('_', '')}Config",
        __base__=ComponentConfig,
        type=(Literal[name], Field(default=name, description=f"The registered name, {name}.")),  # type: ignore[valid-type]
        scale=(
            float,
            Field(default=1.0, gt=0.0, description="A made-up setting of the kernel."),
        ),
    )
    return schema


class RecordingKernelBuilder(AbstractKernelBuilder[Any]):
    """The least a kernel defines: `makeKernel`. It records the `n_columns` it was given."""

    received: list[int]

    def __init__(self, config: Any) -> None:
        """Keeps the configuration and starts an empty record of the calls."""
        super().__init__(config)
        self.received = []

    def makeKernel(self, n_columns: int) -> Kernel | None:
        """Records `n_columns` and returns a new linear kernel."""
        self.received.append(n_columns)
        return LinearKernel()


def defineFake(
    name: str = "fake_kernel_v1",
    *,
    needs: frozenset[str] = frozenset(),
    supports: frozenset[str] = frozenset(),
    description: str = "A fake kernel for tests.",
) -> type[AbstractKernelBuilder[Any]]:
    """Registers a fake kernel in the kind that is current, and returns its class.

    Call it inside a test that uses the `isolatedKernelKind` fixture: a registered name is
    permanent, and must not reach the real registry.

    Args:
        name: The registered name.
        needs: The capability tags it needs.
        supports: The capability tags it supports.
        description: Its one-line description.

    Returns:
        The registered class, a subclass of `RecordingKernelBuilder`.
    """

    @registerKernel(
        name,
        config=fakeConfig(name),
        description=description,
        needs=needs,
        supports=supports,
    )
    class FakeKernelBuilder(RecordingKernelBuilder):
        pass

    return FakeKernelBuilder
