"""The contract of the `fitter` kind of components."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Generic, Self, TypeVar

import torch
from botorch.models.model import Model

from planexs.core.ComponentConfig import ComponentConfig
from planexs.core.ConfigError import ConfigError
from planexs.surrogates.fitters.FitterError import FitterError

logger = logging.getLogger(__name__)

ConfigT = TypeVar("ConfigT", bound=ComponentConfig)


class AbstractFitter(ABC, Generic[ConfigT]):
    """Fits the parameters of a native model to its own training data (specification 5.6, 6).

    A surrogate builds a model with its kernel, likelihood and outcome transform, and then
    something has to set the hyperparameters of that model from the data: maximize the marginal
    likelihood with the engine's priors (a MAP fit), maximize a variational bound, or anything
    else. That procedure is a choice that a reasonable user could want to take differently, so
    it is a component of this kind (specification 3.3): a concrete subclass, registered under a
    permanent name with `registerFitter`, together with the configuration schema that holds its
    settings. It is a nested component: a surrogate's configuration holds one in a field marked
    `ComponentOf("fitter")`, and the surrogate's builder gives it the model it has made.

    Unlike a kernel or a likelihood, a fitter does not make a native object, it acts on one. So
    it is named after what it does and not after what it builds, as a design or a policy is:
    the fitter *is* the procedure. What it receives is the model, which already holds its
    training data, and a seeded generator, and nothing else.

    What every fitter guarantees, and a plugin's must too:

    - **In place.** It changes the parameters of the model it is given and returns nothing, and
      it never replaces the model by another: the surrogate keeps the very object that it built
      (specification 5.13).
    - **Evaluation mode.** `fit` leaves the model in evaluation mode, whatever mode the fit
      left it in, so that what a surrogate returns is ready to predict.
    - **Finite parameters.** `fit` checks every parameter after the fit and refuses a model
      that holds a value that is not finite, with a `FitterError` (specification 8).
    - **Determinism.** It draws every random number from the generator it is given, and from
      nowhere else, and it leaves the global random state of torch as it found it. The same
      configuration, the same model and the same generator state give the same fit
      (specification 3.7, 6).
    - **No side effect at import or registration time**, and no change to the training data.

    A subclass provides `fitParameters`. `fit` is the one entry point, and it is not
    overridden: it checks what it is given, calls `fitParameters`, then sets the mode and
    checks the result, so that every fitter is held to the same rules in the same words.

    The registry entry's builder is `fromConfig`, as for every kind that makes an object from a
    validated configuration: it makes this fitter object, which keeps its configuration for
    `fit`. A fitter is configured once and then fits as often as it is asked.

    Args:
        config: The validated configuration of the fitter, an instance of the schema it is
            registered with.

    Attributes:
        config: The configuration the fitter was made with. It is a read-only attribute.
    """

    def __init__(self, config: ConfigT) -> None:
        """Keeps the validated configuration for `fit`."""
        self._config = config

    @property
    def config(self) -> ConfigT:
        """The configuration this fitter was made with (a stored value: cheap, snake_case)."""
        return self._config

    @classmethod
    def fromConfig(cls, config: ConfigT) -> Self:
        """The builder of a registered fitter: makes the fitter object that a validated
        configuration describes.

        This is what the registry entry stores as its builder, so
        `resolveComponent(registry, {...})` followed by `entry.builder(config)` gives the
        object whose `fit` fits a model.

        Args:
            config: The configuration, an instance of the schema that the fitter is registered
                with.

        Returns:
            The fitter, an instance of the class this is called on.
        """
        return cls(config)

    @abstractmethod
    def fitParameters(self, model: Model, generator: torch.Generator) -> None:
        """Fits the parameters of `model` in place.

        This is where a fitter differs from another. Do not call it directly: call `fit`,
        which has checked what it is given and will set the mode and check the result.

        Args:
            model: A BoTorch model that holds its training data. Its parameters are what is
                fitted.
            generator: The seeded generator that every random choice of the fit draws from.

        Raises:
            FitterError: if the model is not one this fitter can fit, or the fit fails.
        """

    def fit(self, model: Model, *, generator: torch.Generator) -> None:
        """Fits `model` in place and leaves it in evaluation mode.

        Args:
            model: A BoTorch model that holds its training data, built by a surrogate.
            generator: The seeded generator that every random choice of the fit draws from.
                The study gives each build one that it derived for the ask and the component
                (specification 3.7, 5.11).

        Raises:
            ConfigError: if `generator` is not a `torch.Generator`.
            FitterError: if `model` is not a BoTorch model or not one this fitter can fit, the
                fit fails, or it leaves a parameter with a value that is not finite.
        """
        component = f"fitter {self._config.type!r}"
        if not isinstance(model, Model):
            raise FitterError(
                f"The {component} was given a {type(model).__name__}, which is not a BoTorch "
                "model: a fitter acts on the native model that a surrogate builds.",
                component=component,
                field_path="model",
                suggestion="give the fitter the BoTorch model that the surrogate built",
            )
        if not isinstance(generator, torch.Generator):
            raise ConfigError(
                f"fit needs a seeded torch.Generator; got {type(generator).__name__}. Drawing "
                "from torch's global random state would make the fit depend on whatever ran "
                "before.",
                component=component,
                field_path="generator",
                suggestion="pass torchGenerator(seed, runtime), with a seed from RngTree.seed",
            )
        logger.debug(
            "Fitting a %s with the fitter %r.", type(model).__name__, self._config.type
        )
        self.fitParameters(model, generator)
        model.eval()
        self._requireFiniteParameters(model, component)

    @staticmethod
    def _requireFiniteParameters(model: Model, component: str) -> None:
        """Refuses a model that holds a parameter that is not finite.

        Args:
            model: The fitted model.
            component: Who the error is about, as its component hint.

        Raises:
            FitterError: if a parameter holds `nan` or an infinity. The message names the
                first such parameter.
        """
        for name, parameter in model.named_parameters():
            if not bool(torch.isfinite(parameter).all()):
                raise FitterError(
                    f"The {component} left the parameter {name!r} of the model with a value "
                    "that is not finite (nan or an infinity): a model that holds one is "
                    "unusable and is never passed on.",
                    component=component,
                    field_path="model",
                    suggestion="check the training data (constant outcomes, duplicated "
                    "points, values of very different scales) and the settings of the fitter",
                )
