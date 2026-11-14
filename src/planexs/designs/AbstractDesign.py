"""The contract of the `design` kind of components."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Generic, Self, TypeVar

import torch
from torch import Tensor

from planexs.core.ComponentConfig import ComponentConfig
from planexs.core.ConfigError import ConfigError
from planexs.designs.DesignContext import DesignContext
from planexs.domain.Trial import Trial

ConfigT = TypeVar("ConfigT", bound=ComponentConfig)


class AbstractDesign(ABC, Generic[ConfigT]):
    """A procedure that proposes points without a model (specification 5.8, 6).

    A design is what a study uses before there is a model to consult, and what it falls back
    on when one cannot be fitted: space-filling sequences or random draws. Which one is a
    choice of configuration (specification 3.3), so a concrete design is a subclass of this
    class, registered under a permanent name with `registerDesign`, together with the
    configuration schema that holds its settings.

    What every design guarantees, and a plugin's must too:

    - **Constraints.** It always honours them: every point it returns satisfies the
      constraints of the context, and is a point that `SearchSpaceEncoder.decode` turns into a
      legal parameterization, so a point is never rounded onto an infeasible one
      (specification 5.5).
    - **Reproducibility.** It draws every random number from the generator of the context, and
      from nowhere else. The same configuration, history and generator seed give the same
      points (specification 3.7, 6).
    - **Existing data.** It can count the observations that exist already, so that a warm
      start does not repeat an initial phase that the data has made unnecessary (FR3).

    A subclass provides two things. `pointsNeeded` says how many points the design still
    needs, which is the size of its phase less whatever the data has already provided.
    `drawPoints` draws a given number of points. `propose` is the one entry point, and it is
    not overridden: it resolves how many points are asked for, so that every design reads a
    count in the same way.

    The registry entry's builder is `fromConfig`, as for every kind that makes an object from
    a validated configuration: it makes the design object, which keeps its configuration.

    Args:
        config: The validated configuration of the design, an instance of the schema it is
            registered with.

    Attributes:
        config: The configuration the design was made with. It is a read-only attribute.
    """

    def __init__(self, config: ConfigT) -> None:
        """Keeps the validated configuration."""
        self._config = config

    @property
    def config(self) -> ConfigT:
        """The configuration this design was made with (a stored value: cheap, snake_case)."""
        return self._config

    @classmethod
    def fromConfig(cls, config: ConfigT) -> Self:
        """The builder of a registered design: makes the design object that a validated
        configuration describes.

        This is what the registry entry stores as its builder, so
        `resolveComponent(registry, {...})` followed by `entry.builder(config)` gives the
        design.

        Args:
            config: The configuration, an instance of the schema that the design is
                registered with.

        Returns:
            The design, an instance of the class this is called on.
        """
        return cls(config)

    @abstractmethod
    def pointsNeeded(self, trials: Sequence[Trial]) -> int:
        """How many points the design still needs to complete its phase.

        This is the size of the design less what the existing trials have already provided,
        for a design that counts them, and never less than zero. It is how a study knows how
        long the initial phase lasts, and it is what `propose` asks for when it is not given a
        count.

        Args:
            trials: Every trial of the study, in every state.

        Returns:
            A number of points, zero or more.
        """

    @abstractmethod
    def drawPoints(self, context: DesignContext, n: int) -> Tensor:
        """Draws exactly `n` points, with `n` at least 1.

        This is where a design differs from another. Each point must satisfy the constraints
        of the context and be snapped onto valid values (`SearchSpaceEncoder.snapColumns`),
        and every random number must come from `context.generator`. Do not call it directly:
        call `propose`, which has settled what `n` is.

        Args:
            context: The problem, the encoder, the constraints, the history and the generator.
            n: The number of points, one or more.

        Returns:
            A tensor of shape `(n, d)` of encoded points, `d` being the number of encoded
            columns of the encoder, of the dtype of the encoder.

        Raises:
            RuntimeError: if the constraints leave no point to propose.
        """

    def propose(self, context: DesignContext, count: int | None = None) -> Tensor:
        """Proposes encoded points, without a model.

        Args:
            context: The problem, the encoder, the constraints, the history and the generator.
            count: How many points to propose. An explicit count is honoured as it is, and it
                is the caller's: a batch that a user asked for, whatever the design's own
                size. `None` means the points the design still needs (`pointsNeeded`), which
                is the whole design, or what is left of it once the existing observations are
                counted. Zero is a count like another and gives no point.

        Returns:
            A tensor of shape `(n, d)` of encoded points: `d` is the number of encoded columns
            of the encoder, and `n` the count, or the points still needed. Every point
            satisfies the constraints.

        Raises:
            ConfigError: if `count` is negative, or is not a whole number.
            RuntimeError: if the constraints leave no point to propose.
        """
        if count is None:
            n = self.pointsNeeded(context.trials)
        elif isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ConfigError(
                f"The number of points to propose must be a whole number of zero or more; "
                f"got {count!r}.",
                component=f"design {self._config.type!r}",
                field_path="count",
                suggestion="ask for zero or more points, or leave the count out to get the "
                "points the design still needs",
            )
        else:
            n = count
        if n == 0:
            return torch.empty(0, context.encoder.n_columns, dtype=context.encoder.dtype)
        return self.drawPoints(context, n)
