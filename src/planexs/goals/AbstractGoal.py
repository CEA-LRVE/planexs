"""The contract of the `goal` kind of components."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Collection
from typing import Any, Generic, Self, TypeVar

from planexs.core.ComponentConfig import ComponentConfig
from planexs.core.ConfigError import ConfigError
from planexs.domain.Outcome import Outcome
from planexs.domain.Problem import Problem
from planexs.goals.goalRegistry import goalRegistry

ConfigT = TypeVar("ConfigT", bound=ComponentConfig)


class AbstractGoal(ABC, Generic[ConfigT]):
    """A typed statement of what the caller wants to know or decide (specification 5.7, 6).

    A goal is configuration, not an algorithm (ADR-0013): it computes nothing. It does three
    things, and a concrete goal is a subclass of this class, registered under a permanent name
    with `registerGoal` together with the configuration schema that holds its settings:

    - It **parameterizes** acquisitions and analyses: directions and weights, a level,
      constraints on outcomes. They are stated once, on the goal, and read from `config`.
    - It **declares capability tags**, through `needs` and `supports` (below), and is validated
      against the problem it is used with, through `validateAgainst`.
    - It **proposes defaults**: `recipes`, complete policy configurations in order of
      preference, and `defaultAnalyses`, the analyses that answer it. The mapping from a goal
      to its default policy is data held by the goal, so a goal that a plugin registers brings
      its own default and no function of the study branches on the type of a goal.

    The registry entry's builder is `fromConfig`, as for every kind that makes an object from a
    validated configuration: it makes the goal object, which keeps its configuration.

    **Capability tags.** A goal is a participant of the compatibility check (`Participant`):
    it has a `label`, and the tags it `needs` some participant to support and `supports`
    itself (specification 5.12). The tags that `registerGoal` records in the registry entry
    describe the goal *type*, the union of what any configuration of it can state. A goal
    object reports what holds for *its* configuration, which a subclass narrows by overriding
    the two properties: a goal for several outcomes supports `objectives:multi` and not
    `objectives:single`, and a goal with outcome constraints needs
    `outcome_constraints:supported` where one without needs nothing. The default is the tags of
    the entry, unchanged.

    **Recipes and analyses are data.** `recipes` and `defaultAnalyses` return plain
    JSON-compatible mappings, each with the `type` of the component it configures, and not
    objects: the `policy` and `analysis` kinds are resolved by the study when it builds, with
    the validation, the resolved configuration and the compatibility check of every kind
    (ADR-0013), and persistence holds configuration as plain data (specification 5.1). A goal
    that proposes nothing, which is the default, leaves the study to take the policy the caller
    gives.

    Args:
        config: The validated configuration of the goal, an instance of the schema it is
            registered with.

    Attributes:
        config: The configuration the goal was made with. It is a read-only attribute.
    """

    def __init__(self, config: ConfigT) -> None:
        """Keeps the validated configuration."""
        self._config = config

    @property
    def config(self) -> ConfigT:
        """The configuration this goal was made with (a stored value: cheap, snake_case)."""
        return self._config

    @classmethod
    def fromConfig(cls, config: ConfigT) -> Self:
        """The builder of a registered goal: makes the goal object that a validated
        configuration describes.

        This is what the registry entry stores as its builder, so
        `resolveComponent(registry, {...})` followed by `entry.builder(config)` gives the goal.

        Args:
            config: The configuration, an instance of the schema that the goal is registered
                with.

        Returns:
            The goal, an instance of the class this is called on.
        """
        return cls(config)

    @property
    def label(self) -> str:
        """Who this goal is, as an error message of the compatibility check names it, for
        example `goal 'optimize_v1'` (a string built from the stored name: cheap, snake_case).
        """
        return f"goal {self._config.type!r}"

    @property
    def needs(self) -> frozenset[str]:
        """The capability tags this goal needs some participant to support, for its own
        configuration (a lookup of the entry's tags: cheap, snake_case).

        The default is what the registry entry declares for the type. A goal whose needs depend
        on its configuration overrides this, starting from `super().needs` or from nothing, and
        reports only tags that the entry declares.
        """
        return goalRegistry().lookup(self._config.type).needs

    @property
    def supports(self) -> frozenset[str]:
        """The capability tags this goal supports, for its own configuration (a lookup of the
        entry's tags: cheap, snake_case).

        The default is what the registry entry declares for the type. A goal whose tags depend
        on its configuration overrides this, and reports only tags that the entry declares.
        """
        return goalRegistry().lookup(self._config.type).supports

    @abstractmethod
    def validateAgainst(self, problem: Problem) -> None:
        """Checks that the goal makes sense for `problem`.

        The outcomes the goal names must exist and be of a suitable kind, a weight must have an
        outcome, and whatever else the goal says about the problem must be true of it. A goal
        built from a valid configuration can still be wrong for a problem, which is why this is
        a separate step: the study runs it when it is built (ADR-0013, decision 1), so that a
        mistake fails there and not in the middle of a run. `requireOutcome` does the two
        checks that every goal makes.

        Args:
            problem: What can be varied and what is measured.

        Raises:
            ConfigError: if the goal does not fit the problem. The message names the goal, the
                field of its configuration that is at fault (relative to the goal) and a way to
                fix it.
        """

    def recipes(self, problem: Problem) -> list[dict[str, Any]]:
        """The default policy configurations that this goal proposes, in order of preference.

        A recipe is a complete policy configuration, a mapping with the `type` of a registered
        policy, and the goal proposes several so that the first whose components are compatible
        with the problem and the goal is used (ADR-0013, decision 3). The goal does not check
        compatibility: the study does, for each recipe in order, and skips an incompatible one.
        The numbers inside a recipe are policy values: they come from a `...RecipeConfig`
        schema with documented defaults and are never written in the code of the recipe
        (specification 3.4).

        The problem is given because a recipe may depend on its size, for example a number of
        initial points per dimension.

        The default proposes nothing.

        Args:
            problem: The problem the study is built for.

        Returns:
            New dictionaries, so that a caller who changes one does not change the goal. Empty
            if the goal has no default policy.
        """
        return []

    def defaultAnalyses(self) -> list[dict[str, Any]]:
        """The analyses that answer this goal: the best point, a Pareto front, a level-set
        estimate. `Study.analyze` runs them when it is given none (ADR-0013, decision 6).

        Each is a mapping with the `type` of a registered analysis, which the study resolves
        and checks for compatibility before it runs, as it does a component of the policy.

        The default proposes nothing.

        Returns:
            New dictionaries. Empty if the goal has no default analysis.
        """
        return []

    def requireOutcome(
        self,
        problem: Problem,
        name: str,
        *,
        field_path: str,
        kinds: Collection[str] | None = None,
    ) -> Outcome:
        """Finds the outcome of `problem` that this goal names, and checks it is of a kind it
        can use: the two checks that every goal makes in `validateAgainst`.

        It exists so that every goal reports a missing outcome and an unsuitable kind in the
        same words, naming the goal, the field and a way to fix it (specification 3.6).

        Args:
            problem: The problem the goal is validated against.
            name: The name of the outcome, as the goal's configuration gives it.
            field_path: The path of the field of the goal's configuration that holds the name,
                relative to the goal, for example `outcome` or `outcomes[1]`. The error carries
                it.
            kinds: The registered names of the outcome kinds that this goal can use, for
                example the name of the continuous kind. `None` means any kind.

        Returns:
            The outcome.

        Raises:
            ConfigError: if the problem has no outcome of that name, or its kind is not one of
                `kinds`.
        """
        by_name = {outcome.name: outcome for outcome in problem.outcomes}
        if name not in by_name:
            listed = ", ".join(repr(known) for known in by_name)
            raise ConfigError(
                f"The {self.label} names the outcome {name!r}, which the problem does not "
                f"have. The outcomes of the problem are: {listed}.",
                component=self.label,
                field_path=field_path,
                suggestion="name one of the outcomes of the problem, or add the outcome to it",
            )
        outcome = by_name[name]
        if kinds is not None and outcome.type not in kinds:
            usable = ", ".join(sorted(kinds)) or "none"
            raise ConfigError(
                f"The {self.label} cannot use the outcome {name!r}: it is of the kind "
                f"{outcome.type!r}, and the goal takes an outcome of one of the kinds: "
                f"{usable}.",
                component=self.label,
                field_path=field_path,
                suggestion="name an outcome of a kind the goal can use, or change the kind of "
                "the outcome in the problem",
            )
        return outcome
