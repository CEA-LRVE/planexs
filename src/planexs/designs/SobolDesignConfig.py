"""The configuration schema of the `sobol_v1` design."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from planexs.core.ComponentConfig import ComponentConfig


class SobolDesignConfig(ComponentConfig):
    """The configuration of the `sobol_v1` design: scrambled Sobol points in the encoded space.

    It is the stored form of the design (what a configuration file says), and
    `SobolDesign.fromConfig` builds the design from it (specification 5.8, 5.12). The values
    that the feasible sampler decides with (its rounds, its chain, its tolerance) are not here:
    they belong to the constraints that the design is given, which a study builds once from
    its `sampling` settings (`FeasibleSamplingConfig`).

    Attributes:
        type: The registered name, always `"sobol_v1"`.
        n_points: The size of the design, in points.
        scramble: Whether the Sobol sequence is scrambled.
        count_existing_observations: Whether observations that exist already count toward the
            size of the design.
    """

    type: Literal["sobol_v1"] = Field(
        default="sobol_v1",
        description=(
            "The registered name of this design, always 'sobol_v1'. It selects the "
            "'sobol_v1' entry of the 'design' registry, so that a stored design says which "
            "one it is (specification 5.8). It is the only value the field accepts."
        ),
    )
    n_points: int = Field(
        ge=1,
        description=(
            "The size of the design, in points: how many points it proposes in all when it is "
            "asked for its whole design, less the observations that already exist if they are "
            "counted (see 'count_existing_observations'). It is also the length of the "
            "initial phase of a study that uses the design, which switches to its model once "
            "there is enough data. It must be at least 1. It has no default: how many points "
            "are worth spending before a model is fitted depends on the number of parameters "
            "and on the budget of the problem, which only the caller knows, or a recipe, "
            "which computes it from the size of the problem. A power of two keeps a Sobol "
            "sequence balanced."
        ),
    )
    scramble: bool = Field(
        default=True,
        description=(
            "Whether the Sobol sequence is scrambled: its digits are randomized, by a seed "
            "that the design draws from its generator, which keeps the evenly spread "
            "structure of the sequence and makes the points depend on the seed. The default "
            "is True: an unscrambled sequence is one fixed set of points, whose first point "
            "is the corner of the unit cube where every column is 0, so every study would "
            "start from that same corner and the seed would change nothing. False gives that "
            "classical sequence, which is the choice for a design that must be identical "
            "whatever the seed. It has no effect on the points that the hit-and-run chain "
            "draws inside linear constraints: the chain draws them, and not the sequence."
        ),
    )
    count_existing_observations: bool = Field(
        default=True,
        description=(
            "Whether the observations that exist already count toward 'n_points': the "
            "completed trials, among them data attached from outside before the first ask "
            "(FR3). A pending trial has no observation yet, and a failed or abandoned one "
            "never had one, so none of them counts. When True, a study that starts with k "
            "observations needs only 'n_points' - k more design points, and none if "
            "k >= 'n_points'; when False, the design always needs 'n_points' points, whatever "
            "data exists. The default is True: data that is already there tells a model as "
            "much as design points would, and repeating an initial phase that it has made "
            "unnecessary wastes evaluations, which is what a warm start is meant to avoid. "
            "False is for data that is known not to cover the space, such as many replicates "
            "of one point."
        ),
    )
