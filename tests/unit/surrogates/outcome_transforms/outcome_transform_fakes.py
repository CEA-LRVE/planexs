"""Fakes and toy data shared by the tests of `planexs.surrogates.outcome_transforms`.

A fake transform stands for a plugin's, so that the contract of the kind is tested on its own,
before and beside the real ones. The toy data is a smooth, strictly positive function of two
inputs, so that every transform (the logarithm included) can be fitted to it.
"""

from __future__ import annotations

from typing import Any, Literal

import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import OutcomeTransform, Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from pydantic import Field, create_model
from torch import Tensor

from planexs.core.ComponentConfig import ComponentConfig
from planexs.surrogates.outcome_transforms.AbstractOutcomeTransformBuilder import (
    AbstractOutcomeTransformBuilder,
)
from planexs.surrogates.outcome_transforms.registerOutcomeTransform import (
    registerOutcomeTransform,
)

#: The number of inputs of the toy function.
N_FEATURES = 2


def toyData(
    n_outputs: int = 1, dtype: torch.dtype = torch.float64
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """A smooth, strictly positive function of two inputs, observed without noise.

    Output `k` is `exp(1.2 + sin(3 a) + 0.5 (k + 1) b)` of the inputs `(a, b)` in the unit
    square: positive, spanning a factor of a few, and smooth, which is what a Gaussian process
    fits well (on the logarithm, in particular).

    Args:
        n_outputs: The number of outputs.
        dtype: The floating-point type.

    Returns:
        `(train_x, train_y, test_x, test_y)` with 14 training points and 20 held-out ones, the
        outputs being the columns of `train_y` and `test_y`.
    """
    generator = torch.Generator().manual_seed(0)
    train_x = torch.rand(14, N_FEATURES, generator=generator, dtype=dtype)
    test_x = torch.rand(20, N_FEATURES, generator=generator, dtype=dtype)

    def function(x: Tensor) -> Tensor:
        columns = [
            torch.exp(1.2 + torch.sin(3 * x[:, 0]) + 0.5 * (k + 1) * x[:, 1])
            for k in range(n_outputs)
        ]
        return torch.stack(columns, dim=-1)

    return train_x, function(train_x), test_x, function(test_x)


def fitModel(
    transform: OutcomeTransform | None, train_x: Tensor, train_y: Tensor
) -> SingleTaskGP:
    """Fits a `SingleTaskGP` with `transform` as its outcome transform, as a surrogate will.

    `None` is passed to the model as it is: it means no transform, and not BoTorch's default.

    Returns:
        The fitted model, in evaluation mode.
    """
    model = SingleTaskGP(train_x, train_y, outcome_transform=transform)
    fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
    model.eval()
    return model


def fakeConfig(name: str) -> type[ComponentConfig]:
    """A configuration schema as real transforms write theirs: every field described, the
    `type` field pinned to `name`.

    Args:
        name: The registered name the schema is for.
    """
    schema: type[ComponentConfig] = create_model(
        f"{name.title().replace('_', '')}Config",
        __base__=ComponentConfig,
        type=(Literal[name], Field(default=name, description=f"The registered name, {name}.")),  # type: ignore[valid-type]
        scale=(
            float,
            Field(default=1.0, gt=0.0, description="A made-up setting of the transform."),
        ),
    )
    return schema


class RecordingTransformBuilder(AbstractOutcomeTransformBuilder[Any]):
    """The least a transform defines: `makeTransform`. It records the `n_outcomes` it was given
    and makes a new `Standardize`.
    """

    received: list[int]

    def __init__(self, config: Any) -> None:
        """Keeps the configuration and starts an empty record of the calls."""
        super().__init__(config)
        self.received = []

    def makeTransform(self, n_outcomes: int) -> OutcomeTransform | None:
        """Records `n_outcomes` and returns a new standardizing transform."""
        self.received.append(n_outcomes)
        return Standardize(m=n_outcomes)


def defineFake(
    name: str = "fake_transform_v1",
    *,
    needs: frozenset[str] = frozenset(),
    supports: frozenset[str] = frozenset(),
    description: str = "A fake outcome transform for tests.",
) -> type[AbstractOutcomeTransformBuilder[Any]]:
    """Registers a fake outcome transform in the kind that is current, and returns its class.

    Call it inside a test that uses the `isolatedOutcomeTransformKind` fixture: a registered
    name is permanent, and must not reach the real registry.

    Args:
        name: The registered name.
        needs: The capability tags it needs.
        supports: The capability tags it supports.
        description: Its one-line description.

    Returns:
        The registered class, a subclass of `RecordingTransformBuilder`.
    """

    @registerOutcomeTransform(
        name,
        config=fakeConfig(name),
        description=description,
        needs=needs,
        supports=supports,
    )
    class FakeOutcomeTransformBuilder(RecordingTransformBuilder):
        pass

    return FakeOutcomeTransformBuilder
