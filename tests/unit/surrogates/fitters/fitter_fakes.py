"""Fakes and toy models shared by the tests of `planexs.surrogates.fitters`: a fitter that
stands for a plugin's, so that the contract of the kind is tested on its own, before and beside
the real one, and small exact and approximate GPs to fit.
"""

from __future__ import annotations

from typing import Any, Literal

import torch
from botorch.models import ModelListGP, SingleTaskGP
from botorch.models.approximate_gp import ApproximateGPyTorchModel, SingleTaskVariationalGP
from botorch.models.model import Model
from botorch.models.transforms.outcome import Standardize
from botorch.utils.sampling import manual_seed
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import BernoulliLikelihood, Likelihood
from gpytorch.mlls import ExactMarginalLogLikelihood, SumMarginalLogLikelihood, VariationalELBO
from pydantic import Field, create_model
from torch import Tensor

from planexs.core.ComponentConfig import ComponentConfig
from planexs.surrogates.fitters.AbstractFitter import AbstractFitter
from planexs.surrogates.fitters.registerFitter import registerFitter

#: The number of input columns of the toy models.
N_COLUMNS = 2


def generatorOf(seed: int) -> torch.Generator:
    """A seeded generator on the CPU, as a surrogate's build context holds one."""
    return torch.Generator().manual_seed(seed)


def toyData(n_outputs: int = 1, n_points: int = 10) -> tuple[Tensor, Tensor]:
    """Noisy smooth functions of two columns, the same on every call.

    Returns:
        The inputs, shape `(n_points, 2)` in the unit cube, and `n_outputs` outcome columns.
    """
    generator = torch.Generator().manual_seed(0)
    train_x = torch.rand(n_points, N_COLUMNS, generator=generator, dtype=torch.float64)
    noise = 0.1 * torch.randn(n_points, generator=generator, dtype=torch.float64)
    columns = [
        torch.sin(4 * train_x[:, 0]) + noise + k * train_x[:, 1] ** 2 for k in range(n_outputs)
    ]
    return train_x, torch.stack(columns, dim=-1)


def toyModel(n_outputs: int = 1, n_points: int = 10) -> SingleTaskGP:
    """An exact GP on `toyData`, as a surrogate builds one: BoTorch's default kernel and
    priors, and the outcomes standardized. Several outputs are independent outputs of one
    model, which the engine fits in a batched way. It is not fitted, and is in training mode.
    """
    train_x, train_y = toyData(n_outputs, n_points)
    return SingleTaskGP(train_x, train_y, outcome_transform=Standardize(m=n_outputs))


def toyModelList(n_outputs: int = 2, n_points: int = 10) -> ModelListGP:
    """One exact GP for each outcome, in a `ModelListGP`, which is what a surrogate with an
    explicit kernel builds. It is not fitted.
    """
    train_x, train_y = toyData(n_outputs, n_points)
    members = [
        SingleTaskGP(train_x, train_y[:, k : k + 1], outcome_transform=Standardize(m=1))
        for k in range(n_outputs)
    ]
    return ModelListGP(*members)


def objective(model: Model) -> float:
    """The quantity that a MAP fit maximizes, at the current hyperparameters of `model`: the
    exact marginal likelihood plus the log priors of its hyperparameters, as the engine's
    marginal likelihood adds them, summed over the outputs.

    It puts the model in training mode (the marginal likelihood is a function of the training
    data), and leaves it there.
    """
    model.train()
    if isinstance(model, ModelListGP):
        sum_mll = SumMarginalLogLikelihood(model.likelihood, model)
        return float(sum_mll(model(*model.train_inputs), model.train_targets).sum().detach())
    assert isinstance(model, SingleTaskGP)
    exact_mll = ExactMarginalLogLikelihood(model.likelihood, model)
    return float(exact_mll(model(*model.train_inputs), model.train_targets).sum().detach())


def toyLabels(n_points: int = 30) -> tuple[Tensor, Tensor]:
    """Labels that a smooth function of two columns makes true or false, with noise, so that
    no function separates them: the same on every call.

    Returns:
        The inputs, shape `(n_points, 2)` in the unit cube, and the labels, 0 or 1, as a
        column.
    """
    generator = torch.Generator().manual_seed(0)
    train_x = torch.rand(n_points, N_COLUMNS, generator=generator, dtype=torch.float64)
    noise = 0.3 * torch.randn(n_points, generator=generator, dtype=torch.float64)
    latent = torch.sin(4 * train_x[:, 0]) + train_x[:, 1] - 0.5 + noise
    return train_x, (latent > 0).double().unsqueeze(-1)


def toyClassifier(
    n_points: int = 30, likelihood: Likelihood | None = None
) -> SingleTaskVariationalGP:
    """An approximate GP on `toyLabels` with a Bernoulli likelihood, as a classifier surrogate
    builds one: BoTorch's variational GP with its default kernel and inducing points. It is not
    fitted, and is in training mode.

    Args:
        n_points: The number of labels.
        likelihood: The likelihood, a `BernoulliLikelihood` by default.
    """
    train_x, labels = toyLabels(n_points)
    return SingleTaskVariationalGP(
        train_x, labels, likelihood=likelihood or BernoulliLikelihood()
    )


def toyApproximateRegressor(n_points: int = 10) -> SingleTaskVariationalGP:
    """An approximate GP on `toyData` with BoTorch's default Gaussian likelihood: an
    approximate model whose likelihood is not a label's. It is not fitted.
    """
    train_x, train_y = toyData(1, n_points)
    return SingleTaskVariationalGP(train_x, train_y)


def elbo(model: ApproximateGPyTorchModel) -> float:
    """The variational ELBO that the fitter maximizes, at the current parameters of `model`:
    GPyTorch's `VariationalELBO` (an average over the observations), on all the training data.

    It puts the model in training mode (the bound is a function of the training data), and
    leaves it there. The first evaluation of a variational GP perturbs its variational mean
    with a random draw, so the evaluation is seeded and the global random state is put back:
    the value is the same on every call, and the test that calls it is not at the mercy of the
    tests that ran before it.
    """
    model.train()
    latent = model.model
    bound = VariationalELBO(model.likelihood, latent, num_data=latent.train_targets.shape[-1])
    with torch.no_grad(), manual_seed(0):
        return float(bound(latent(*latent.train_inputs), latent.train_targets).sum())


class SamplingBernoulliLikelihood(BernoulliLikelihood):
    """A Bernoulli likelihood whose expected log probability carries an error drawn from
    torch's *global* random state, as a likelihood that integrates by sampling would: a second
    place where the loop of the fitter can depend on a random number.
    """

    def expected_log_prob(
        self,
        observations: Tensor,
        function_dist: MultivariateNormal,
        *args: Any,
        **kwargs: Any,
    ) -> Tensor:
        """The exact value, plus noise that is as large as 0.1 times the latent mean."""
        exact = super().expected_log_prob(observations, function_dist, *args, **kwargs)
        return exact + 0.1 * torch.randn_like(exact) * function_dist.mean


class NotFiniteBernoulliLikelihood(BernoulliLikelihood):
    """A Bernoulli likelihood whose expected log probability is not a number."""

    def expected_log_prob(
        self,
        observations: Tensor,
        function_dist: MultivariateNormal,
        *args: Any,
        **kwargs: Any,
    ) -> Tensor:
        """`nan` for every observation."""
        return super().expected_log_prob(observations, function_dist, *args, **kwargs) * float(
            "nan"
        )


class FailingBernoulliLikelihood(BernoulliLikelihood):
    """A Bernoulli likelihood whose expected log probability raises a `RuntimeError` after it
    has been computed `after` times, as the linear algebra of the bound does where a covariance
    is not positive definite.
    """

    def __init__(self, after: int) -> None:
        """Counts `after` computations that succeed, then fails every one."""
        super().__init__()
        self.remaining = after

    def expected_log_prob(
        self,
        observations: Tensor,
        function_dist: MultivariateNormal,
        *args: Any,
        **kwargs: Any,
    ) -> Tensor:
        """The exact value, until the count has run out."""
        if self.remaining <= 0:
            raise RuntimeError("The covariance is not positive definite.")
        self.remaining -= 1
        return super().expected_log_prob(observations, function_dist, *args, **kwargs)


def hyperparameters(model: Model) -> dict[str, Tensor]:
    """A detached copy of every parameter of the model, by name."""
    return {name: value.detach().clone() for name, value in model.named_parameters()}


def fakeConfig(name: str) -> type[ComponentConfig]:
    """A configuration schema as real fitters write theirs: every field described, the `type`
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
            Field(default=1.0, gt=0.0, description="A made-up setting of the fitter."),
        ),
    )
    return schema


class RecordingFitter(AbstractFitter[Any]):
    """The least a fitter defines: `fitParameters`. It records what it was given, and leaves
    the model in training mode, as a fit that does not care about the mode would.
    """

    received: list[tuple[Model, torch.Generator]]

    def __init__(self, config: Any) -> None:
        """Keeps the configuration and starts an empty record of the calls."""
        super().__init__(config)
        self.received = []

    def fitParameters(self, model: Model, generator: torch.Generator) -> None:
        """Records its arguments, and puts the model in training mode."""
        self.received.append((model, generator))
        model.train()


def defineFake(
    name: str = "fake_fitter_v1",
    *,
    needs: frozenset[str] = frozenset(),
    supports: frozenset[str] = frozenset(),
    description: str = "A fake fitter for tests.",
) -> type[AbstractFitter[Any]]:
    """Registers a fake fitter in the kind that is current, and returns its class.

    Call it inside a test that uses the `isolatedFitterKind` fixture: a registered name is
    permanent, and must not reach the real registry.

    Args:
        name: The registered name.
        needs: The capability tags it needs.
        supports: The capability tags it supports.
        description: Its one-line description.

    Returns:
        The registered class, a subclass of `RecordingFitter`.
    """

    @registerFitter(
        name,
        config=fakeConfig(name),
        description=description,
        needs=needs,
        supports=supports,
    )
    class FakeFitter(RecordingFitter):
        pass

    return FakeFitter
