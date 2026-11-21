"""The contract of the `outcome_transform` kind of components."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, Self, TypeVar

from botorch.models.transforms.outcome import OutcomeTransform

from planexs.core.ComponentConfig import ComponentConfig
from planexs.core.ConfigError import ConfigError
from planexs.surrogates.outcome_transforms.OutcomeTransformError import OutcomeTransformError

ConfigT = TypeVar("ConfigT", bound=ComponentConfig)


class AbstractOutcomeTransformBuilder(ABC, Generic[ConfigT]):
    """Makes the native outcome transform of a surrogate (specification 5.6, 6).

    A Gaussian process is fitted to rescaled outcomes (centred and divided by their spread, or
    on a log scale), and its predictions are brought back to the units of the outcomes. That
    rescaling is a modelling choice with consequences for the fit, and which one is used is a
    choice of configuration (specification 3.3), so each transform is a component of this
    kind: a concrete subclass, registered under a permanent name with
    `registerOutcomeTransform`, together with the configuration schema that holds its settings.
    It is a nested component: a surrogate's configuration holds a list of them in a field whose
    elements are marked `ComponentOf("outcome_transform")`, and `buildOutcomeTransform` turns
    that list into the one native transform that the model is given.

    What a transform receives is the number of outcomes of the surrogate, and nothing else: its
    statistics are learned from the training targets by the model itself, when it first applies
    the transform. What it returns is a native BoTorch `OutcomeTransform`, or `None`.

    **`None` means no transform at all.** This is the opposite of a kernel, where `None` leaves
    the choice to the engine. BoTorch's own models read `outcome_transform=None` as "do not
    rescale" and an omitted argument as "standardize", so a surrogate that gets `None` from a
    builder must still pass it to the model as `outcome_transform=None`, and not leave the
    argument out.

    What every transform guarantees, and a plugin's must too:

    - **A fresh transform.** Every call to `build` returns a new object that nothing else
      holds. Transforms are stateful: `Standardize` stores the mean and the standard deviation
      of the targets it was first applied to, so two models that shared one would overwrite
      each other's statistics.
    - **Determinism.** It draws no random number, and it has no side effect at import or
      registration time. The same configuration and the same `n_outcomes` give the same
      transform (specification 6).
    - **No data.** It is made before the data is read. The statistics a transform needs are
      set by the model when it applies the transform in training mode.
    - **Unbatched targets.** The training targets of a surrogate are `(n, m)`, never batched
      (`SurrogateBuildContext` checks it), so a transform is made without a batch shape.
    - **A named error for a value it cannot take.** A transform with a restricted domain
      refuses a value outside it with an `OutcomeTransformError` that names the transform and
      says how to fix it, and does not let a `-inf` or a `nan` through to the fit
      (specification 8).

    A subclass provides `makeTransform`. `build` is the one entry point, and it is not
    overridden: it checks `n_outcomes` and what `makeTransform` returns, so that every
    transform is refused in the same way when it is wrong.

    The registry entry's builder is `fromConfig`, as for every kind that makes an object from
    a validated configuration: it makes this builder object, which keeps its configuration for
    `build`.

    Args:
        config: The validated configuration of the transform, an instance of the schema it is
            registered with.

    Attributes:
        config: The configuration the builder was made with. It is a read-only attribute.
    """

    def __init__(self, config: ConfigT) -> None:
        """Keeps the validated configuration for `build`."""
        self._config = config

    @property
    def config(self) -> ConfigT:
        """The configuration this builder was made with (a stored value: cheap, snake_case)."""
        return self._config

    @classmethod
    def fromConfig(cls, config: ConfigT) -> Self:
        """The builder of a registered outcome transform: makes the builder object that a
        validated configuration describes.

        This is what the registry entry stores as its builder, so
        `resolveComponent(registry, {...})` followed by `entry.builder(config)` gives the
        object whose `build` makes the transform.

        Args:
            config: The configuration, an instance of the schema that the transform is
                registered with.

        Returns:
            The builder, an instance of the class this is called on.
        """
        return cls(config)

    @abstractmethod
    def makeTransform(self, n_outcomes: int) -> OutcomeTransform | None:
        """Makes the native transform, for `n_outcomes` of at least 1.

        This is where a transform differs from another. Do not call it directly: call `build`,
        which has checked `n_outcomes` and will check the result.

        Args:
            n_outcomes: The number of outcomes of the surrogate, the columns of its training
                targets, one or more.

        Returns:
            A new BoTorch `OutcomeTransform` for targets of `n_outcomes` columns, or `None`
            for no transform at all (see the class docstring: not the engine default).
        """

    @staticmethod
    def requireOutcomeCount(n_outcomes: object, *, component: str) -> None:
        """Checks that a number of outcomes is a whole number of one or more.

        It is the one place that says so, for `build` and for `buildOutcomeTransform`, which
        must refuse a bad count even for a list of transforms that is empty, where no builder
        is called.

        Args:
            n_outcomes: The number to check. A `bool` is not one: `True` is not a count.
            component: Who the error is about, as its component hint.

        Raises:
            ConfigError: if it is not a whole number of one or more.
        """
        if isinstance(n_outcomes, bool) or not isinstance(n_outcomes, int) or n_outcomes < 1:
            raise ConfigError(
                f"The number of outcomes of an outcome transform must be a whole number of "
                f"one or more; got {n_outcomes!r}.",
                component=component,
                field_path="n_outcomes",
                suggestion="pass the number of outcomes the surrogate models",
            )

    def build(self, n_outcomes: int) -> OutcomeTransform | None:
        """Makes the native outcome transform of a surrogate of `n_outcomes` outcomes.

        Args:
            n_outcomes: The number of outcomes the surrogate models, which is the number of
                outcomes of the problem that it is bound to: a whole number of one or more.

        Returns:
            A new BoTorch `OutcomeTransform`, or `None` when the transform is no transform, in
            which case the surrogate must give the model `outcome_transform=None`.

        Raises:
            ConfigError: if `n_outcomes` is not a whole number of one or more.
            OutcomeTransformError: if `makeTransform` returned something that is neither an
                `OutcomeTransform` nor `None`.
        """
        self.requireOutcomeCount(
            n_outcomes, component=f"outcome_transform {self._config.type!r}"
        )
        transform = self.makeTransform(n_outcomes)
        if transform is not None and not isinstance(transform, OutcomeTransform):
            raise OutcomeTransformError(
                f"The outcome transform {self._config.type!r} made a "
                f"{type(transform).__name__}, which is not a BoTorch OutcomeTransform.",
                component=f"outcome_transform {self._config.type!r}",
                suggestion="make `makeTransform` return an OutcomeTransform, or None for no "
                "transform",
            )
        return transform
