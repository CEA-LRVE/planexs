"""The configuration schema of the `random_v1` design."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from planexs.core.ComponentConfig import ComponentConfig


class RandomDesignConfig(ComponentConfig):
    """The configuration of the `random_v1` design: uniform random points in the encoded space.

    It is the stored form of the design (what a configuration file says), and
    `RandomDesign.fromConfig` builds the design from it (specification 5.8, 5.12). Its options
    are those of `sobol_v1` less `scramble`, which only a Sobol sequence has: the size of the
    design and whether existing observations count toward it. The values that the feasible
    sampler decides with (its rounds, its chain, its tolerance) are not here: they belong to
    the constraints that the design is given, which a study builds once from its `sampling`
    settings (`FeasibleSamplingConfig`).

    Attributes:
        type: The registered name, always `"random_v1"`.
        n_points: The size of the design, in points.
        count_existing_observations: Whether observations that exist already count toward the
            size of the design.
    """

    type: Literal["random_v1"] = Field(
        default="random_v1",
        description=(
            "The registered name of this design, always 'random_v1'. It selects the "
            "'random_v1' entry of the 'design' registry, so that a stored design says which "
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
            "which computes it from the size of the problem. No size is special for random "
            "points: unlike a Sobol sequence, they are not balanced by a power of two."
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
