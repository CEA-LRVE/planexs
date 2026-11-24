"""The `variational_elbo_v1` fitter: an approximate GP fitted by maximizing its ELBO."""

from __future__ import annotations

from collections.abc import Callable

import torch
from botorch.models.approximate_gp import ApproximateGPyTorchModel
from botorch.models.model import Model
from botorch.optim.utils import get_parameters_and_bounds
from botorch.utils.sampling import manual_seed
from gpytorch.mlls import VariationalELBO
from gpytorch.models import ApproximateGP
from torch.optim import SGD, Adagrad, Adam, Optimizer, RMSprop

from planexs.surrogates.fitters.AbstractFitter import AbstractFitter
from planexs.surrogates.fitters.drawSeed import drawSeed
from planexs.surrogates.fitters.FitterError import FitterError
from planexs.surrogates.fitters.registerFitter import registerFitter
from planexs.surrogates.fitters.VariationalElboFitterConfig import VariationalElboFitterConfig

#: The optimizers that the configuration names, each made with torch's own defaults for
#: everything but the learning rate. The keys are exactly the values of the schema's
#: `optimizer` field, and a test pins that they are.
_OPTIMIZERS: dict[str, Callable[..., Optimizer]] = {
    "adam": Adam,
    "sgd": SGD,
    "rmsprop": RMSprop,
    "adagrad": Adagrad,
}


@registerFitter(
    "variational_elbo_v1",
    config=VariationalElboFitterConfig,
    description=(
        "Fits an approximate GP by maximizing the variational ELBO with a loop of a chosen "
        "optimizer, learning rate and number of iterations."
    ),
    needs=frozenset({"posterior:latent_gaussian"}),
)
class VariationalElboFitter(AbstractFitter[VariationalElboFitterConfig]):
    """Fits the parameters of an approximate Gaussian process by maximizing the variational
    evidence lower bound (ELBO).

    An approximate GP (the model of a classifier, whose likelihood is not Gaussian) has no
    closed-form marginal likelihood: it has a variational distribution over the values of the
    latent function at its inducing points, and the ELBO is the bound that is maximized over
    the variational parameters, the inducing points and the hyperparameters of the kernel and
    the likelihood together. This fitter is that loop, written out in this package rather
    than left to BoTorch's fitting routine, so that what the loop does is what the
    configuration says and nothing else: for `n_iterations` iterations it computes the
    negative ELBO on *all* the training data, takes a step of the optimizer named in
    `optimizer` at the rate in `learning_rate`, and then stops. There is no mini-batching,
    no early stopping and no learning-rate schedule. The ELBO is GPyTorch's
    `VariationalELBO`, with the weight of the KL term at 1 (the bound itself and not a
    tempered one), and it is an average over the data, as GPyTorch scales it, so its value
    does not grow with the number of observations.

    **Which models.** One BoTorch approximate GP (`ApproximateGPyTorchModel`, of which
    `SingleTaskVariationalGP` is the one a surrogate builds) with one output, whatever its
    likelihood is: a Bernoulli one for a classifier, or a Gaussian one. The model must hold its
    training data, which BoTorch's variational GP does. Any other model is a `FitterError` that
    names it: an exact GP is for `map_v1`, and the fitter *needs* `posterior:latent_gaussian`,
    so the compatibility check refuses it beside a surrogate whose model is exact when a study
    is built (specification 3.6). A model with several outputs, or a list of models, is refused
    as well: a surrogate with several outcomes fits one model for each.

    **Randomness.** The fit draws from torch's global random state, which no component may
    use (specification 3.7), in one place that matters for the built-in models: the first
    time a variational GP is evaluated, GPyTorch initializes its variational distribution
    from the prior and adds a small random perturbation to its mean. A model that has never
    been evaluated, as the one a surrogate has just built is, therefore starts the fit from a
    point that depends on a random draw, and so does the fit (a likelihood that integrates by
    sampling would be a second place). So, as `map_v1` does, the fit draws one seed from the
    generator it is given and runs the loop inside BoTorch's `manual_seed(seed)`, which
    seeds torch's generator for the length of the fit and puts its state back afterwards: the
    same generator state gives the same fit, and whatever random state the caller had is
    untouched. A model that has been evaluated before, a refit, continues from its own
    variational parameters and draws nothing there.

    **Constraints.** A step of a first-order optimizer knows nothing of the constraints of
    the model: BoTorch's kernels keep a lengthscale above its lower bound with a constraint
    that has no transform, so a large step takes it below the bound, where its prior is not
    defined. After each step the parameters are put back inside their bounds, as BoTorch's
    own loop with a torch optimizer does.

    **Failure.** A step that takes the parameters to where the model is no longer defined (the
    ELBO is not finite, or a covariance is not positive definite) is a `FitterError` that names
    the iteration and says to lower the learning rate. The model is then left as the last
    completed step left it, in training mode. A fit that has not converged after
    `n_iterations` is not a failure: the iteration count is a budget. Nor is a fit whose last
    step is worse than an earlier one, as a step that is too large makes it: the fitter
    returns the last iterate and does not keep the best one.
    """

    def fitParameters(self, model: Model, generator: torch.Generator) -> None:
        """Maximizes the variational ELBO of `model` for the configured number of iterations.

        Args:
            model: A BoTorch approximate GP with one output that holds its training data.
            generator: The seeded generator that the seed of the fit is drawn from.

        Raises:
            FitterError: if the model is not an approximate GP with one output and its data, or
                a step took the parameters to where the ELBO cannot be computed.
        """
        approximate = self._approximateModel(model)
        seed = drawSeed(generator)
        with manual_seed(seed):
            self._maximize(approximate)

    def _approximateModel(self, model: Model) -> ApproximateGPyTorchModel:
        """Checks that `model` is one this fitter fits.

        Args:
            model: The model to fit.

        Returns:
            The same model, typed as the approximate GP that it is.

        Raises:
            FitterError: if the model is not an approximate GP, has several outputs, or holds
                no training data.
        """
        component = f"fitter {self.config.type!r}"
        if not (
            isinstance(model, ApproximateGPyTorchModel)
            and isinstance(model.model, ApproximateGP)
        ):
            raise FitterError(
                f"The {component} fits approximate Gaussian processes (a BoTorch "
                f"ApproximateGPyTorchModel), and was given a {type(model).__name__}, which is "
                "not one.",
                component=component,
                field_path="model",
                suggestion=(
                    "use a surrogate that builds an approximate GP with this fitter, or a "
                    "fitter for exact models with this one"
                ),
            )
        if model.num_outputs != 1:
            raise FitterError(
                f"The {component} fits an approximate GP with one output, and the "
                f"{type(model).__name__} it was given has {model.num_outputs}.",
                component=component,
                field_path="model",
                suggestion="fit one model for each outcome, each with its own fit",
            )
        inputs = getattr(model.model, "train_inputs", None)
        targets = getattr(model.model, "train_targets", None)
        if not inputs or targets is None:
            raise FitterError(
                f"The {component} maximizes the ELBO on the training data of the model, and "
                f"the {type(model).__name__} it was given holds none.",
                component=component,
                field_path="model",
                suggestion="build the model with its training inputs and targets",
            )
        return model

    def _maximize(self, model: ApproximateGPyTorchModel) -> None:
        """The loop: `n_iterations` steps of the optimizer on the negative ELBO.

        Args:
            model: The approximate GP, which holds its training data.

        Raises:
            FitterError: if a step took the parameters to where the ELBO cannot be computed.
        """
        latent = model.model
        inputs, targets = latent.train_inputs, latent.train_targets
        elbo = VariationalELBO(model.likelihood, latent, num_data=targets.shape[-1])
        parameters, bounds = get_parameters_and_bounds(elbo, requires_grad=True)
        optimizer = _OPTIMIZERS[self.config.optimizer](
            list(parameters.values()), lr=self.config.learning_rate
        )
        model.train()
        for iteration in range(self.config.n_iterations):
            optimizer.zero_grad()
            loss = self._negativeElbo(elbo, latent, inputs, targets, iteration)
            torch.autograd.backward(loss)
            optimizer.step()
            with torch.no_grad():
                # A step of a first-order optimizer knows nothing of the constraints of the
                # model (a lengthscale must stay above its lower bound, which BoTorch's
                # kernels keep without a transform), so the parameters are put back inside
                # them, as BoTorch's own loop with a torch optimizer does after each step.
                for name, (lower, upper) in bounds.items():
                    parameters[name].clamp_(min=lower, max=upper)
        optimizer.zero_grad(set_to_none=True)

    def _negativeElbo(
        self,
        elbo: VariationalELBO,
        latent: ApproximateGP,
        inputs: tuple[torch.Tensor, ...] | list[torch.Tensor],
        targets: torch.Tensor,
        iteration: int,
    ) -> torch.Tensor:
        """The negative ELBO at the current parameters, which must be finite.

        Args:
            elbo: The bound.
            latent: The approximate GP that the bound is of.
            inputs: The training inputs of the model.
            targets: The training targets of the model.
            iteration: The iteration, counted from 0, for the message of an error.

        Returns:
            The loss, a finite scalar that holds the graph of the parameters.

        Raises:
            FitterError: if the bound is not finite, or cannot be computed after a step.
            A bound that cannot be computed before the first step raises what the engine
            raised, since the step size is not to blame for it.
        """
        try:
            loss: torch.Tensor = -elbo(latent(*inputs), targets).sum()
        except RuntimeError as error:
            # The linear algebra of the bound raises a RuntimeError (a covariance that is not
            # positive definite, or a nan in a decomposition) where a step has taken the
            # parameters. Before the first step there is no step to blame, and the error is
            # that of the model or of its data, which is not hidden behind the learning rate.
            if iteration == 0:
                raise
            raise self._divergence(iteration, f"({type(error).__name__}: {error})") from error
        if not bool(torch.isfinite(loss)):
            if iteration == 0:
                component = f"fitter {self.config.type!r}"
                raise FitterError(
                    f"The {component} found the ELBO of the model to be {float(loss)} before "
                    "its first step, so the fit cannot start: no step has been taken yet.",
                    component=component,
                    field_path="model",
                    suggestion=(
                        "check the training data (constant outcomes, duplicated points, "
                        "values of very different scales) and the likelihood of the model"
                    ),
                )
            raise self._divergence(iteration, f"(the ELBO is {float(loss)})")
        return loss

    def _divergence(self, iteration: int, reason: str) -> FitterError:
        """The error of a step that took the model to where its ELBO is undefined.

        Args:
            iteration: The iteration whose bound could not be computed, counted from 0.
            reason: What went wrong, for the message.

        Returns:
            The error to raise.
        """
        component = f"fitter {self.config.type!r}"
        return FitterError(
            f"The {component} could not compute the ELBO at iteration {iteration} of "
            f"{self.config.n_iterations} with the optimizer {self.config.optimizer!r} at the "
            f"learning rate {self.config.learning_rate}: the fit diverged {reason}.",
            component=component,
            field_path="learning_rate",
            suggestion=(
                "lower learning_rate (a step that is too large leaves the region where the "
                "model is defined), or use an optimizer that is less sensitive to the scale "
                "of the gradients, such as 'adam'"
            ),
        )
