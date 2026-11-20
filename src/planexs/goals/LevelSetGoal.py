"""The goal of finding the level set of one outcome."""

from __future__ import annotations

from planexs.core.ConfigError import ConfigError
from planexs.domain.Outcome import Outcome
from planexs.domain.Problem import Problem
from planexs.goals.AbstractGoal import AbstractGoal
from planexs.goals.LevelSetGoalConfig import LevelSetGoalConfig
from planexs.goals.registerGoal import registerGoal

#: The registered names of the outcome kinds that a level set can be found for.
_CONTINUOUS = "continuous"
_BINARY = "binary"
_USABLE_KINDS = (_CONTINUOUS, _BINARY)


@registerGoal(
    "level_set_v1",
    config=LevelSetGoalConfig,
    description="Find the boundary where one outcome crosses a level (FR20).",
    supports=frozenset({"goal:level_set"}),
)
class LevelSetGoal(AbstractGoal[LevelSetGoalConfig]):
    """Estimate the level set of one outcome: where it crosses a level (specification 5.7).

    The goal names one outcome and a level. For a continuous outcome the level is a value in
    the units of the outcome, and the boundary is where the measured quantity crosses it. For
    a binary outcome the level is a probability, and the boundary is where the probability of
    the label crosses it: 0.5 is the boundary between the region where the label is more
    likely true and the one where it is more likely false, for example the boundary between
    two phases of a material.

    Like every goal it computes nothing (ADR-0013): it holds the outcome and the level, which
    the acquisitions and the analyses of a level set read through `effectiveLevel`, and it
    declares `goal:level_set`. It proposes no recipe and no default analysis yet: the recipes
    come with the roadmap task P6-08.

    What the configuration cannot say, because it depends on the kind of the outcome, is
    checked in `validateAgainst`: the outcome must exist and be continuous or binary, a
    continuous outcome needs a level, and the level of a binary outcome must be a probability
    strictly between 0 and 1. A level that the caller leaves out of a binary goal is the
    `binary_level` of the configuration, so the default is written once, in the schema.
    """

    def validateAgainst(self, problem: Problem) -> None:
        """Checks that the outcome exists, can have a level set, and has a usable level.

        Args:
            problem: What can be varied and what is measured.

        Raises:
            ConfigError: if the problem has no outcome of that name, the outcome is neither
                continuous nor binary, a continuous outcome has no level, or the level of a
                binary outcome is not strictly between 0 and 1. The error names the goal, the
                field at fault (`outcome` or `level`) and a way to fix it.
        """
        self.effectiveLevel(problem)

    def effectiveLevel(self, problem: Problem) -> float:
        """The level the boundary is drawn at, once the goal is checked against `problem`.

        It is the `level` of the configuration, and for a binary outcome that has none, the
        `binary_level` default. It is a method and not a property because it depends on the
        problem, whose outcome decides whether a missing level is an error or a default
        (specification 10.2).

        Args:
            problem: The problem the goal is used with.

        Returns:
            The level: in the units of the outcome for a continuous one, a probability for a
            binary one.

        Raises:
            ConfigError: for any of the reasons `validateAgainst` lists.
        """
        outcome = self.requireOutcome(
            problem, self.config.outcome, field_path="outcome", kinds=_USABLE_KINDS
        )
        if outcome.type == _BINARY:
            return self._binaryLevel(outcome)
        return self._continuousLevel(outcome)

    def _continuousLevel(self, outcome: Outcome) -> float:
        """The level of a continuous outcome, which the configuration must give.

        Args:
            outcome: The continuous outcome the goal names.

        Returns:
            The configured level.

        Raises:
            ConfigError: if the configuration has no level.
        """
        level = self.config.level
        if level is None:
            raise ConfigError(
                f"The {self.label} needs a level for the continuous outcome {outcome.name!r}: "
                "the level is in the units of the outcome, and the library cannot guess it.",
                component=self.label,
                field_path="level",
                suggestion="give the level, a value in the units of the outcome",
            )
        return level

    def _binaryLevel(self, outcome: Outcome) -> float:
        """The level of a binary outcome: the configured probability, or the default.

        Args:
            outcome: The binary outcome the goal names.

        Returns:
            The configured level if there is one, `binary_level` otherwise.

        Raises:
            ConfigError: if the configured level is not strictly between 0 and 1.
        """
        level = self.config.level
        if level is None:
            return self.config.binary_level
        if not 0 < level < 1:
            raise ConfigError(
                f"The {self.label} has the level {level!r} for the binary outcome "
                f"{outcome.name!r}: the level of a binary outcome is a probability, strictly "
                "between 0 and 1.",
                component=self.label,
                field_path="level",
                suggestion="give a probability strictly between 0 and 1, or leave the level "
                "out to use the default of a binary outcome",
            )
        return level
