"""The configuration schema of the `map_v1` fitter."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from planexs.core.ComponentConfig import ComponentConfig


class MapFitterConfig(ComponentConfig):
    """The configuration of the `map_v1` fitter: the options of BoTorch's fitting routine.

    The fitter maximizes the marginal likelihood of an exact Gaussian process together with the
    priors of its hyperparameters (a MAP fit) through `botorch.fit.fit_gpytorch_mll`. Its
    settings are the options of that routine that the installed version honours for every
    model the fitter accepts. Which these are was checked against the installed source (BoTorch
    0.18.1) and not assumed:

    - `max_attempts` and `pick_best_of_all_attempts` are options of the routine's fallback fit,
      which fits an exact model, and its fit of a list of independent models passes them on to
      each of them;
    - `maxiter` is an option of the optimizer, `options["maxiter"]`, which both of the
      optimizers that the routine can choose (the batched L-BFGS-B that fits the independent
      outputs of one model, and scipy's minimizer that fits any other model) read.

    Two options of the routine are left out on purpose, because they would be silent no-ops
    for the multi-output model that a surrogate builds by default. The solver `method` is read
    by scipy's minimizer only: the batched optimizer is always L-BFGS-B and ignores it.
    `timeout_sec` is not supported by the batched optimizer either, which reports that it was
    ignored. A setting that changes nothing is an error of the schema and never a no-op
    (specification 10.8), so they are added when the engine honours them everywhere. The
    options that are Python objects and not data (`closure`, `optimizer`, `warning_handler`,
    `caught_exception_types`) cannot be stored in a configuration file, and a new fitter is the
    way to change them (specification 3.8).

    The defaults are those of the routine in the installed version, written down here so that
    a later change of the engine does not alter how a stored study fits (specification 3.4):
    BoTorch's own defaults are `max_attempts=5` and `pick_best_of_all_attempts=False`, and a
    test pins them against it.

    Attributes:
        type: The registered name, always `"map_v1"`.
        max_attempts: The largest number of fit attempts.
        pick_best_of_all_attempts: Whether every attempt is made and the best one is kept.
        maxiter: The largest number of optimizer iterations of one attempt, or `None`.
    """

    type: Literal["map_v1"] = Field(
        default="map_v1",
        description=(
            "The registered name of this fitter, always 'map_v1'. It selects the 'map_v1' "
            "entry of the 'fitter' registry (specification 5.6). It is the only value the "
            "field accepts."
        ),
    )
    max_attempts: int = Field(
        default=5,
        ge=1,
        description=(
            "The largest number of attempts that the fit makes, in attempts. The first "
            "attempt starts from the hyperparameters that the model was built with. An "
            "attempt that fails (the covariance matrix is not positive definite, or the "
            "optimizer reports a failure) is followed by a new one that starts from "
            "hyperparameters sampled from their priors, until one succeeds or this many "
            "have been made; a fit that never succeeds is then an error. With "
            "'pick_best_of_all_attempts' every attempt is made, and this is also their "
            "number. It must be at least 1, which makes the first attempt the only one. The "
            "default is 5, the engine's own: a failure that a restart can cure is cured "
            "within a few draws, and one that survives five is a problem of the data and "
            "not of the starting point. The draws come from the generator that the "
            "surrogate gives the fit, so they are reproducible."
        ),
    )
    pick_best_of_all_attempts: bool = Field(
        default=False,
        description=(
            "Whether the fit makes all 'max_attempts' attempts and keeps the one with the "
            "best marginal likelihood (with the priors), a flag. When False the fit stops "
            "at the first attempt that succeeds, which is the usual case and the cheapest. "
            "When True the attempts after the first restart from hyperparameters sampled "
            "from their priors, so a model whose marginal likelihood has several local "
            "maxima (very few observations, or noise that the model can explain either as "
            "signal or as noise) is not left in the first one that was found, at the price "
            "of 'max_attempts' times the cost. The default is False, the engine's own: the "
            "starting point of the first attempt is the mode of the priors, which is a "
            "sound one."
        ),
    )
    maxiter: int | None = Field(
        default=None,
        ge=1,
        description=(
            "The largest number of iterations that the optimizer makes in one attempt, in "
            "iterations. The optimizer stops earlier when it has converged, so this is a "
            "budget and not a target: a small value gives a fit that is not converged, "
            "which is useful to bound the cost of an ask and is never an error. It must be "
            "at least 1. The default is None, which leaves the cap to the engine: 15000 "
            "iterations for both of the optimizers that the routine uses, a cap that a fit "
            "of a few dozen observations never reaches, so that the fit is bounded by its "
            "convergence and not by a number that a study would have to tune."
        ),
    )
