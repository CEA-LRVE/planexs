"""The configuration schema of the `explore_v1` goal."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator

from planexs.core.ComponentConfig import ComponentConfig


class ExploreGoalConfig(ComponentConfig):
    """The configuration of the `explore_v1` goal: which outcomes to learn.

    It is the stored form of the goal (what a configuration file says), and
    `ExploreGoal.fromConfig` builds the goal from it (specification 5.7, 5.12). The goal is
    to learn the named outcomes as well as possible with no direction: nothing is maximized,
    minimized or searched for (FR21).

    The schema cannot see the problem, so it cannot tell whether a name is an outcome of it:
    `ExploreGoal.validateAgainst` checks that when the goal is validated against the problem.

    Attributes:
        type: The registered name, always `"explore_v1"`.
        outcomes: The names of the outcomes to learn, or `None` for all of them.
    """

    type: Literal["explore_v1"] = Field(
        default="explore_v1",
        description=(
            "The registered name of this goal, always 'explore_v1'. It selects the "
            "'explore_v1' entry of the 'goal' registry, so that a stored goal says which one "
            "it is (specification 5.7). It is the only value the field accepts."
        ),
    )
    outcomes: list[str] | None = Field(
        default=None,
        min_length=1,
        description=(
            "The names of the outcomes to learn as well as possible, in the order in which "
            "they are given. Each must be an outcome of the problem, which is checked when "
            "the goal is validated against the problem, and each may be named only once. The "
            "default is None, which means all the outcomes of the problem, in the order of "
            "the problem: exploring everything that is measured is what a goal with no "
            "direction means unless the caller narrows it, and the goal then follows the "
            "problem if the problem changes. A list is never empty: to learn no outcome is "
            "not a goal, so an empty list is an error and is not read as 'all'."
        ),
    )

    @field_validator("outcomes")
    @classmethod
    def _checkNames(cls, value: list[str] | None) -> list[str] | None:
        """Checks that every name is a non-empty string and that none is named twice.

        Args:
            value: The names, or `None` for all the outcomes.

        Returns:
            The names, unchanged.

        Raises:
            ValueError: if a name is empty or a name occurs more than once. The message says
                which (the value received is added to it by whoever reports the error, with
                the field path).
        """
        if value is None:
            return value
        if any(not name for name in value):
            raise ValueError("outcomes must not contain an empty name")
        duplicates = sorted({name for name in value if value.count(name) > 1})
        if duplicates:
            raise ValueError(
                f"outcomes must name each outcome once, but repeats {duplicates!r}"
            )
        return value
