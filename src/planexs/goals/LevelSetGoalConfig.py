"""The configuration schema of the `level_set_v1` goal."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from planexs.core.ComponentConfig import ComponentConfig


class LevelSetGoalConfig(ComponentConfig):
    """The configuration of the `level_set_v1` goal: one outcome and the level to find.

    It is the stored form of the goal (what a configuration file says), and
    `LevelSetGoal.fromConfig` builds the goal from it (specification 5.7, 5.12). The goal is
    the boundary where the outcome crosses `level` or, for a binary outcome, where the
    probability of the label crosses it (FR20).

    The schema cannot see the kind of the outcome, which is a fact about the problem, so it
    cannot tell whether `level` is required. `LevelSetGoal.validateAgainst` checks it when the
    goal is validated against the problem: a continuous outcome needs a `level`, and a binary
    one takes `binary_level` when `level` is not given.

    Attributes:
        type: The registered name, always `"level_set_v1"`.
        outcome: The name of the outcome whose level set is wanted.
        level: The level, or `None` to use the default for a binary outcome.
        binary_level: The probability used as the level of a binary outcome when `level` is
            not given.
    """

    type: Literal["level_set_v1"] = Field(
        default="level_set_v1",
        description=(
            "The registered name of this goal, always 'level_set_v1'. It selects the "
            "'level_set_v1' entry of the 'goal' registry, so that a stored goal says which "
            "one it is (specification 5.7). It is the only value the field accepts."
        ),
    )
    outcome: str = Field(
        min_length=1,
        description=(
            "The name of the outcome whose level set is wanted: the boundary where the "
            "outcome crosses 'level' or, for a binary outcome, where the probability of the "
            "label crosses it. It must be an outcome of the problem, of the continuous or the "
            "binary kind, which is checked when the goal is validated against the problem. It "
            "has no default: which outcome is the one of interest is the caller's question."
        ),
    )
    level: float | None = Field(
        default=None,
        allow_inf_nan=False,
        description=(
            "The level of the outcome that the boundary is drawn at. For a continuous outcome "
            "it is required and is in the units of the outcome (a yield in percent takes a "
            "level in percent), and a missing level is an error when the goal is validated "
            "against the problem: a threshold in the units of a measured quantity is not "
            "something the library can guess. For a binary outcome it is a probability, "
            "strictly between 0 and 1 (the response probability of a label never reaches "
            "either end, so a boundary at 0 or 1 does not exist), and it is optional: when it "
            "is not given the goal uses 'binary_level'. The default is None, which means "
            "'not given'. It must be a finite number."
        ),
    )
    binary_level: float = Field(
        default=0.5,
        gt=0.0,
        lt=1.0,
        description=(
            "The probability that is used as the level of a binary outcome when 'level' is "
            "not given, strictly between 0 and 1. The default is 0.5: the probability at "
            "which a label is as likely true as false, so that the boundary is the decision "
            "boundary of a classifier, which is what the boundary between two phases of a "
            "material means. It is the only documented default of the level, and it lives "
            "here because a value that changes results is policy (specification 3.4). It "
            "has no effect on a continuous outcome, which always takes 'level'."
        ),
    )
