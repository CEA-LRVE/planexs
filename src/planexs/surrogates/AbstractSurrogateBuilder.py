"""The contract of the `surrogate` kind of components."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, Self, TypeVar

from planexs.core.ComponentConfig import ComponentConfig
from planexs.surrogates.BuiltSurrogate import BuiltSurrogate
from planexs.surrogates.SurrogateBuildContext import SurrogateBuildContext

ConfigT = TypeVar("ConfigT", bound=ComponentConfig)


class AbstractSurrogateBuilder(ABC, Generic[ConfigT]):
    """Fits a native model on observations and says how it fitted (specification 5.6, 6).

    A surrogate is a probabilistic model of the outcomes as a function of the parameters. What
    varies between surrogates (the model class, the likelihood, how several outcomes are
    modelled, the fitting procedure) is chosen by name in configuration, so each choice is a
    component of this kind: a concrete subclass, registered under a permanent name with
    `registerSurrogate`, together with the configuration schema that holds its settings
    (specification 3.3). It provides:

    - `build`, which receives a `SurrogateBuildContext` (the problem, the encoder, the data as
      tensors, a random generator, the runtime settings) and returns a `BuiltSurrogate`: the
      fitted native model and a `FitStatus`;
    - declared needs and supports, truthful with respect to what it does: the capability tags
      it passes to `registerSurrogate`, such as `posterior:gaussian` or `noise:known`. The
      outcome kinds it can model are among them (`outcome_kind:continuous`), so that the
      compatibility check can meet them with the problem's (specification 5.12).

    The registry entry's builder is `fromConfig`, as for every kind that makes an object from
    a validated configuration: it makes this builder object, which keeps its configuration for
    `build`. A builder is configured once and then builds as often as it is asked, each time
    from a context alone, so it is deterministic given its configuration, the context and the
    generator in it (specification 6). It draws no random number from anywhere else, has no
    side effect at import or registration time, and never modifies the tensors of the context.

    A builder with several outcomes decides how to model them (independent models, a joint
    model) and says so in its capability tags (`outputs:multi`). One that cannot fit with the
    data it is given raises `ModelNotReadyError` (too few observations), and one that fits on
    little data says so in the `FitStatus` instead of hiding it (specification 3.11). A fit
    that falls back to something else says so in the status warnings, and logs a warning
    (specification 8).

    The class is generic in the schema of its configuration, so that a subclass reads the
    fields of its own settings with their types::

        @registerSurrogate(
            "my_gp_v1",
            config=MyGPConfig,
            description="An exact GP with my kernel.",
            supports=frozenset({"posterior:gaussian", "outcome_kind:continuous"}),
        )
        class MyGPBuilder(AbstractSurrogateBuilder[MyGPConfig]):
            def build(self, context: SurrogateBuildContext) -> BuiltSurrogate:
                ...  # read self.config.<field> and the context, fit, return the model

    Args:
        config: The validated configuration of the component, an instance of the schema it is
            registered with.

    Attributes:
        config: The configuration the builder was made with. It is a read-only attribute.
    """

    def __init__(self, config: ConfigT) -> None:
        """Keeps the validated configuration for `build`."""
        self._config = config

    @property
    def config(self) -> ConfigT:
        """The configuration this builder was made with (a stored value: cheap,
        snake_case).
        """
        return self._config

    @classmethod
    def fromConfig(cls, config: ConfigT) -> Self:
        """The builder of a registered surrogate: makes the builder object that a validated
        configuration describes.

        This is what the registry entry stores as its builder, so
        `resolveComponent(registry, {...})` followed by `entry.builder(config)` gives the
        object whose `build` fits the model.

        Args:
            config: The configuration, an instance of the schema that the surrogate is
                registered with.

        Returns:
            The builder, an instance of the class this is called on.
        """
        return cls(config)

    @abstractmethod
    def build(self, context: SurrogateBuildContext) -> BuiltSurrogate:
        """Fits a model on the observations of the context.

        Args:
            context: The problem, the encoder, the data as tensors, the generator and the
                runtime settings. Read-only.

        Returns:
            The fitted native model, in evaluation mode, and how it was fitted.

        Raises:
            ModelNotReadyError: if the context holds fewer observations than this surrogate
                needs to fit.
            ConfigError: if the configuration cannot be applied to this problem.
        """
