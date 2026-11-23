"""The `map_v1` fitter: an exact GP fitted by maximum a posteriori."""

from __future__ import annotations

from typing import Any

import torch
from botorch.exceptions.errors import ModelFittingError
from botorch.fit import fit_gpytorch_mll
from botorch.models.model import Model
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.utils.sampling import manual_seed
from gpytorch.mlls import (
    ExactMarginalLogLikelihood,
    MarginalLogLikelihood,
    SumMarginalLogLikelihood,
)
from gpytorch.models import ExactGP

from planexs.surrogates.fitters.AbstractFitter import AbstractFitter
from planexs.surrogates.fitters.FitterError import FitterError
from planexs.surrogates.fitters.MapFitterConfig import MapFitterConfig
from planexs.surrogates.fitters.registerFitter import registerFitter


@registerFitter(
    "map_v1",
    config=MapFitterConfig,
    description=(
        "Fits an exact GP by maximizing the marginal likelihood with the engine's priors "
        "(a MAP fit)."
    ),
    needs=frozenset({"posterior:gaussian"}),
)
class MapFitter(AbstractFitter[MapFitterConfig]):
    """Fits the hyperparameters of an exact Gaussian process by maximum a posteriori.

    It maximizes the exact marginal likelihood of the model *including* the priors that
    BoTorch puts on the lengthscales and the noise (GPyTorch's marginal likelihood adds the log
    prior of every hyperparameter that has one), which is what makes it a MAP fit and what
    keeps it usable on few points. It is the fit that the legacy surrogate made with
    `fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))`, and with the
    default settings it makes that call with the engine's defaults written out.

    **Which models.** An exact GP (`gpytorch.models.ExactGP`), which includes BoTorch's
    `SingleTaskGP` with one output or several independent ones, and a `ModelListGP` of exact
    GPs, one model for each outcome, which a surrogate with an explicit kernel builds (see
    `MaternKernelBuilder`). The engine fits a list one model after the other, each with its own
    marginal likelihood. An exact GP has a Gaussian likelihood, which GPyTorch checks when the
    model is built. Any other model (an approximate one, whose likelihood is not Gaussian) is a
    `FitterError` that names the model: the fitter *needs* `posterior:gaussian`, so the
    compatibility check refuses it beside a surrogate that builds such a model when a study is
    built (specification 3.6).

    **Randomness.** The first attempt of the fit is deterministic. A later attempt (see
    `max_attempts`) restarts from hyperparameters that the engine samples from their priors,
    and the engine draws them from torch's global random state, which no component may use
    (specification 3.7). So the fit draws one seed from the generator it is given and runs the
    engine inside BoTorch's `manual_seed(seed)`, which seeds torch's generator for the length
    of the fit and puts its state back afterwards: the same generator state gives the same
    fit, and whatever random state the caller had is untouched.

    **Failure.** When the engine gives up (`ModelFittingError`, after every attempt) the
    model is left as the engine rolled it back and the error is a `FitterError` that names
    the fitter, the number of attempts and what to try. A fit that reaches a poor optimum
    without an error is not a failure: the engine reports it through Python warnings, which
    are not hidden.
    """

    def fitParameters(self, model: Model, generator: torch.Generator) -> None:
        """Maximizes the marginal likelihood of `model`, priors included.

        Args:
            model: An exact GP, or a `ModelListGP` of exact GPs, that holds its training data.
            generator: The seeded generator that the seed of the fit is drawn from.

        Raises:
            FitterError: if the model is not an exact GP or a list of them, or if the engine
                gave up after `max_attempts` attempts.
        """
        marginal_likelihood = self._marginalLikelihood(model)
        seed = _drawSeed(generator)
        try:
            with manual_seed(seed):
                fit_gpytorch_mll(
                    marginal_likelihood,
                    optimizer_kwargs=self._optimizerKwargs(),
                    max_attempts=self.config.max_attempts,
                    pick_best_of_all_attempts=self.config.pick_best_of_all_attempts,
                )
        except ModelFittingError as error:
            component = f"fitter {self.config.type!r}"
            raise FitterError(
                f"The {component} could not fit the {type(model).__name__}: the engine gave "
                f"up after {self.config.max_attempts} attempt(s) ({error}).",
                component=component,
                field_path="max_attempts",
                suggestion=(
                    "check the training data (constant outcomes, duplicated points, values of "
                    "very different scales), or raise max_attempts to restart from more "
                    "hyperparameters drawn from their priors"
                ),
            ) from error

    def _marginalLikelihood(self, model: Model) -> MarginalLogLikelihood:
        """The marginal likelihood that the engine maximizes for `model`.

        Args:
            model: The model to fit.

        Returns:
            An `ExactMarginalLogLikelihood` for an exact GP, and a `SumMarginalLogLikelihood`
            for a `ModelListGP` of exact GPs, as the engine does for each.

        Raises:
            FitterError: if the model is neither.
        """
        if isinstance(model, ExactGP):
            return ExactMarginalLogLikelihood(model.likelihood, model)
        if isinstance(model, ModelListGP) and all(
            isinstance(member, ExactGP) for member in model.models
        ):
            return SumMarginalLogLikelihood(model.likelihood, model)
        component = f"fitter {self.config.type!r}"
        if isinstance(model, ModelListGP):
            inexact = sorted(
                {
                    type(member).__name__
                    for member in model.models
                    if not isinstance(member, ExactGP)
                }
            )
            what = (
                f"a ModelListGP that holds {', '.join(inexact)}, which "
                f"{'is' if len(inexact) == 1 else 'are'} not exact"
            )
        else:
            what = f"a {type(model).__name__}, which is not an exact GP"
        raise FitterError(
            f"The {component} fits exact Gaussian processes (an ExactGP, or a ModelListGP of "
            f"them), and was given {what}.",
            component=component,
            field_path="model",
            suggestion=(
                "use a surrogate that builds an exact GP with this fitter, or a fitter for "
                "approximate models with this one"
            ),
        )

    def _optimizerKwargs(self) -> dict[str, Any] | None:
        """The keyword arguments for the optimizer of the engine's routine.

        Returns:
            `None` to leave the optimizer to the engine, or the options that the configuration
            sets: today the cap on iterations.
        """
        if self.config.maxiter is None:
            return None
        return {"options": {"maxiter": self.config.maxiter}}


def _drawSeed(generator: torch.Generator) -> int:
    """Draws the seed of one fit from the generator that the fit is given.

    The seed is drawn, and not read from the generator, so that a generator that has been used
    does not give the same fit as a fresh one. The generator may live on another device than
    the default one, and `randint` wants them to agree.

    Args:
        generator: The seeded generator.

    Returns:
        An integer that `torch.manual_seed` accepts.
    """
    high = torch.iinfo(torch.int32).max
    return int(torch.randint(high, (1,), generator=generator, device=generator.device))
