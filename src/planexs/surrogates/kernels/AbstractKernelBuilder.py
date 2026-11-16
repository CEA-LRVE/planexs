"""The contract of the `kernel` kind of components."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, Self, TypeVar

from gpytorch.kernels import Kernel

from planexs.core.ComponentConfig import ComponentConfig
from planexs.core.ConfigError import ConfigError

ConfigT = TypeVar("ConfigT", bound=ComponentConfig)


class AbstractKernelBuilder(ABC, Generic[ConfigT]):
    """Makes the native covariance function of a surrogate (specification 5.6, 6).

    The kernel of a Gaussian process says how strongly the outcome at one point is tied to the
    outcome at another, and it is the most consequential modelling choice of a surrogate. Which
    kernel is used is a choice of configuration (specification 3.3), so each kernel is a
    component of this kind: a concrete subclass, registered under a permanent name with
    `registerKernel`, together with the configuration schema that holds its settings. It is a
    nested component: a surrogate's configuration holds one in a field marked
    `ComponentOf("kernel")`, and the surrogate's builder asks it for the native kernel.

    What a kernel receives is the number of encoded columns, and nothing else: the surrogate
    works on the unit-cube encoding of the search space (specification 5.5), so a kernel never
    needs to know about log scales, grids or categorical values, and `n_columns` is what it
    needs to give an automatic-relevance-determination kernel one lengthscale for each column.
    What it returns is a native GPyTorch `Kernel`, or `None` to leave the choice to the engine.

    What every kernel guarantees, and a plugin's must too:

    - **A fresh kernel.** Every call to `build` returns a new object that nothing else holds,
      so two models never share a kernel and the hyperparameters of one are not fitted into
      another.
    - **Determinism.** It draws no random number, and it has no side effect at import or
      registration time. The same configuration and the same `n_columns` give the same kernel
      (specification 6).
    - **No data.** It is made before the data is read: it holds the form of the covariance,
      whose hyperparameters the fitting procedure of the surrogate then sets.
    - **No batch shape.** It receives the number of columns and nothing else, so the kernel it
      makes has hyperparameters of one set, not one for each output of a batched model. A
      surrogate of several independent outcomes that uses an explicit kernel makes one model
      for each outcome, with a new kernel from each call to `build`.

    A subclass provides `makeKernel`. `build` is the one entry point, and it is not
    overridden: it checks `n_columns`, so that every kernel refuses a bad one in the same way.

    The registry entry's builder is `fromConfig`, as for every kind that makes an object from
    a validated configuration: it makes this builder object, which keeps its configuration for
    `build`.

    Args:
        config: The validated configuration of the kernel, an instance of the schema it is
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
        """The builder of a registered kernel: makes the builder object that a validated
        configuration describes.

        This is what the registry entry stores as its builder, so
        `resolveComponent(registry, {...})` followed by `entry.builder(config)` gives the
        object whose `build` makes the kernel.

        Args:
            config: The configuration, an instance of the schema that the kernel is registered
                with.

        Returns:
            The builder, an instance of the class this is called on.
        """
        return cls(config)

    @abstractmethod
    def makeKernel(self, n_columns: int) -> Kernel | None:
        """Makes the native kernel, for `n_columns` of at least 1.

        This is where a kernel differs from another. Do not call it directly: call `build`,
        which has checked `n_columns`.

        Args:
            n_columns: The number of encoded columns of the surrogate's inputs, one or more.

        Returns:
            A new GPyTorch kernel over inputs of `n_columns` columns, or `None` to leave the
            choice of kernel to the engine.
        """

    def build(self, n_columns: int) -> Kernel | None:
        """Makes the native kernel of a surrogate over inputs of `n_columns` encoded columns.

        Args:
            n_columns: The number of encoded columns, which is `SearchSpaceEncoder.n_columns`:
                a whole number of one or more.

        Returns:
            A new GPyTorch `Kernel`, or `None` when the kernel leaves the choice to the
            engine, in which case the surrogate does not give one to the model.

        Raises:
            ConfigError: if `n_columns` is not a whole number of one or more.
        """
        if isinstance(n_columns, bool) or not isinstance(n_columns, int) or n_columns < 1:
            raise ConfigError(
                f"The number of encoded columns of a kernel must be a whole number of one or "
                f"more; got {n_columns!r}.",
                component=f"kernel {self._config.type!r}",
                field_path="n_columns",
                suggestion="pass the n_columns of the encoder of the problem's search space",
            )
        return self.makeKernel(n_columns)
