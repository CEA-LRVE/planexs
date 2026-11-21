"""The contract of the `acquisition` kind of components."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar, Generic, Self, TypeVar

from torch import Tensor

from planexs.acquisition.AcquisitionBuildContext import AcquisitionBuildContext
from planexs.acquisition.AcquisitionContextError import AcquisitionContextError
from planexs.core.ComponentConfig import ComponentConfig

if TYPE_CHECKING:
    from botorch.acquisition.acquisition import AcquisitionFunction

ConfigT = TypeVar("ConfigT", bound=ComponentConfig)


class AbstractAcquisitionBuilder(ABC, Generic[ConfigT]):
    """Builds the native acquisition function that scores candidate points (specification 5.8,
    6).

    An acquisition is a score over candidate points, built from a surrogate, a goal and the
    pending points: higher means more worth evaluating (specification 4). What varies between
    acquisitions (the expected improvement, the upper confidence bound, the hypervolume
    improvement, a level-set straddle, the exploration/exploitation blend with its single
    weight, FR5) is chosen by name in configuration, so each is a component of this kind: a
    concrete subclass, registered under a permanent name with `registerAcquisition`, together
    with the configuration schema that holds its settings (specification 3.3). The catalogue is
    open: anything that fits the loop *model of the outcomes -> score of candidate points ->
    choice of points* is added by writing a builder, with no change to planexs
    (specification 1.3). A builder provides:

    - `build`, which receives an `AcquisitionBuildContext` (the surrogate, the goal, the
      pending points, a reference set when it asked for one, a random generator) and returns
      the native acquisition function of the engine: a BoTorch `AcquisitionFunction` that its
      own optimizers accept, so that an acquisition planexs writes works with them
      (specification 1.4, 3.5);
    - declared needs and supports, truthful with respect to what it does: the capability tags
      it passes to `registerAcquisition`. It needs what it reads from the goal and the
      surrogate (`goal:optimize`, `objectives:single`, `posterior:gaussian`) and supports what
      it offers the others (`acquisition:differentiable`, `pending:supported`,
      `outcome_constraints:supported`), so that the compatibility check can meet them
      (specification 5.12).

    **The reference set.** Some acquisitions normalize their terms over a space-filling set of
    candidate points (the blend of exploration and exploitation does, so that its weight means
    the same whether an outcome is measured in milliseconds or in millions). A builder that
    does says so by the class-level flag `needs_reference_points`, and says how many points it
    wants in `referencePointCount`, which reads the size from its own configuration, where a
    policy value belongs (specification 3.4). The policy asks both before it builds, draws
    that many points with the feasible sampler, so that every one satisfies the constraints,
    and hands them over in the context. Drawing the set is not the builder's business: it
    would repeat a draw that is constant within one ask (specification 8). The two are tied:
    `registerAcquisition` refuses a builder that sets the flag and does not override
    `referencePointCount`, and one that overrides it and does not set the flag, so a builder
    can neither ask for a set that is never drawn nor have one drawn that it ignores.

    The registry entry's builder is `fromConfig`, as for every kind that makes an object from a
    validated configuration: it makes this builder object, which keeps its configuration for
    `build`. A builder is configured once and then builds as often as it is asked, each time
    from a context alone, so it is deterministic given its configuration, the context and the
    generator in it (specification 6). It draws no random number from anywhere else, has no
    side effect at import or registration time, and never modifies the tensors of the context.

    The class is generic in the schema of its configuration, so that a subclass reads the
    fields of its own settings with their types::

        @registerAcquisition(
            "my_boundary_score_v1",
            config=MyBoundaryScoreConfig,
            description="A score of closeness to the level, weighted by uncertainty.",
            needs=frozenset({"goal:level_set", "posterior:marginal_normal"}),
            supports=frozenset({"acquisition:differentiable"}),
        )
        class MyBoundaryScoreBuilder(AbstractAcquisitionBuilder[MyBoundaryScoreConfig]):
            def build(self, context: AcquisitionBuildContext) -> AcquisitionFunction:
                ...  # read self.config.<field> and the context, return the native function

    Args:
        config: The validated configuration of the component, an instance of the schema it is
            registered with.

    Attributes:
        needs_reference_points: Whether the builder needs a reference set in its context. A
            class-level flag, so that it can be read without building anything. `False` unless
            a subclass says otherwise.
        config: The configuration the builder was made with. It is a read-only attribute.
    """

    needs_reference_points: ClassVar[bool] = False

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
        """The builder of a registered acquisition: makes the builder object that a validated
        configuration describes.

        This is what the registry entry stores as its builder, so
        `resolveComponent(registry, {...})` followed by `entry.builder(config)` gives the
        object whose `build` makes the acquisition function.

        Args:
            config: The configuration, an instance of the schema that the acquisition is
                registered with.

        Returns:
            The builder, an instance of the class this is called on.
        """
        return cls(config)

    def referencePointCount(self) -> int:
        """How many points of reference set the builder wants, as its configuration says.

        A builder with `needs_reference_points` overrides this to return the size that its
        configuration gives (a field of the schema, with a documented default and a reason for
        it), and the policy draws that many points before it builds. The default is for a
        builder that needs none.

        Returns:
            The number of points of the reference set, one or more for a builder that needs
            one. Zero for a builder that does not.
        """
        return 0

    def requireReferencePoints(self, context: AcquisitionBuildContext) -> Tensor:
        """The reference set of the context, for a builder that needs one.

        It exists so that every builder reports a missing set in the same words, naming the
        field and a way to fix it (specification 3.6), as `AbstractGoal.requireOutcome` does
        for a missing outcome.

        Args:
            context: The context `build` received.

        Returns:
            The reference points, shape `(n, d)`, in the unit cube. The tensor of the context
            itself, not a copy: do not modify it.

        Raises:
            AcquisitionContextError: if the context holds no reference set, which means that
                the policy did not draw one for a builder that needs it.
        """
        if context.reference_points is None:
            raise AcquisitionContextError(
                f"The acquisition {self._config.type!r} needs a reference set of points to "
                "normalize over, and the build context holds none.",
                component=f"acquisition {self._config.type!r}",
                field_path="reference_points",
                suggestion="draw referencePointCount() feasible points before building, and "
                "pass them as reference_points",
            )
        return context.reference_points

    @abstractmethod
    def build(self, context: AcquisitionBuildContext) -> AcquisitionFunction:
        """Builds the native acquisition function for the surrogate and the goal of the
        context.

        Args:
            context: The surrogate, the goal, the pending points, the reference set (when the
                builder asked for one) and the generator. Read-only.

        Returns:
            A BoTorch acquisition function, over the encoded space of the context. It takes
            the pending points of the context into account if the builder supports
            `pending:supported`.

        Raises:
            AcquisitionContextError: if the context lacks something the builder needs (a
                reference set, see `requireReferencePoints`).
            ConfigError: if the configuration cannot be applied to this goal or surrogate.
        """
