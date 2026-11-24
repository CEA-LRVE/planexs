"""The configuration schema of the `variational_elbo_v1` fitter."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from planexs.core.ComponentConfig import ComponentConfig


class VariationalElboFitterConfig(ComponentConfig):
    """The configuration of the `variational_elbo_v1` fitter: the loop that maximizes the
    variational evidence lower bound (ELBO) of an approximate Gaussian process.

    The fit is a loop of this package's own and not a call of BoTorch's fitting routine, so
    the settings are those of the loop: how many steps it takes, how large a step is, and
    which rule takes it. The defaults are those of the GPyTorch tutorial on non-Gaussian
    likelihoods, which fits a variational GP with a Bernoulli likelihood with Adam at a
    learning rate of 0.1 (for 50 iterations), written down here so that a later
    change of an engine's default does not alter how a stored study fits (specification 3.4).
    The number of iterations is larger than the tutorial's, for reasons given in its field.

    Attributes:
        type: The registered name, always `"variational_elbo_v1"`.
        n_iterations: The number of optimizer steps.
        learning_rate: The step size of the optimizer.
        optimizer: The name of the optimizer.
    """

    type: Literal["variational_elbo_v1"] = Field(
        default="variational_elbo_v1",
        description=(
            "The registered name of this fitter, always 'variational_elbo_v1'. It selects the "
            "'variational_elbo_v1' entry of the 'fitter' registry (specification 5.6). It is "
            "the only value the field accepts."
        ),
    )
    n_iterations: int = Field(
        default=100,
        ge=1,
        description=(
            "The number of optimizer steps that the fit takes, in iterations. Every step "
            "uses all the training data (there are no mini-batches) and there is no early "
            "stopping, so this is exactly the number of steps made: a budget that bounds the "
            "cost of an ask, and a fit that has not converged by then is not an error. More "
            "iterations raise the bound further and cost in proportion; the bound of a "
            "classifier of a few dozen observations flattens within about this many steps "
            "at the default learning rate, and a regression with a Gaussian likelihood, "
            "which has a noise to learn, needs more or a smaller 'learning_rate'. It must "
            "be at least 1. The default is 100, twice the 50 of the GPyTorch tutorial on "
            "non-Gaussian likelihoods, which fits a handful of labels, and a test pins that "
            "the default gets within a small margin of a run four times as long on a toy "
            "classifier."
        ),
    )
    learning_rate: float = Field(
        default=0.1,
        gt=0.0,
        allow_inf_nan=False,
        description=(
            "The step size of the optimizer, in the unit of the parameters of the model. A "
            "larger step reaches the neighbourhood of the optimum sooner and may overshoot "
            "it, which leaves the bound worse than a smaller step would (the fit returns its "
            "last iterate and does not keep the best one) and can make it not finite, which "
            "is an error that names the iteration. A smaller step is steadier and needs more "
            "'n_iterations'. A rate is not comparable from one optimizer to another: RMSprop "
            "takes a first step of about ten times its rate, where Adam takes one of about "
            "its rate. It must be positive and finite. The default is 0.1, the learning rate "
            "of the GPyTorch tutorial on non-Gaussian likelihoods and a good one for the "
            "default optimizer. It is far above the 0.001 that torch gives Adam, which "
            "would barely move the parameters in 100 steps."
        ),
    )
    optimizer: Literal["adam", "sgd", "rmsprop", "adagrad"] = Field(
        default="adam",
        description=(
            "The name of the optimizer that takes the steps, one of 'adam', 'sgd', 'rmsprop' "
            "and 'adagrad', which are the first-order optimizers of torch.optim that "
            "maximize the bound as it is. Each takes the other options that it has at "
            "torch's own defaults (no momentum for 'sgd', for instance). L-BFGS is left out "
            "because it works with a closure and counts its iterations differently, and the "
            "optimizers that add weight decay are left out because the decay is a penalty "
            "that is not part of the bound, so that the fit would no longer maximize it. "
            "The default is 'adam', the optimizer of the GPyTorch tutorial on non-Gaussian "
            "likelihoods, which is the least sensitive of them to the scale of the "
            "gradients; the other three may want another 'learning_rate'."
        ),
    )
