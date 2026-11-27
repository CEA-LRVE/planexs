"""The configuration schema of the `straddle_v1` acquisition."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from planexs.core.ComponentConfig import ComponentConfig


class StraddleConfig(ComponentConfig):
    """The configuration of the `straddle_v1` acquisition: the straddle of a level, which
    scores a point by how uncertain the surrogate is about it and how close its prediction is
    to the level (specification 5.8, FR20).

    It is the stored form of the acquisition (what a configuration file says), and
    `StraddleBuilder.fromConfig` builds the builder from it (specification 5.12). One setting
    is its policy value: how many posterior standard deviations of uncertainty make up for a
    unit of distance from the level. Everything else is read from the goal: the outcome and
    the level.

    Attributes:
        type: The registered name, always `"straddle_v1"`.
        beta: The weight of the uncertainty against the distance to the level, in posterior
            standard deviations.
    """

    type: Literal["straddle_v1"] = Field(
        default="straddle_v1",
        description=(
            "The registered name of this acquisition, always 'straddle_v1'. It selects the "
            "'straddle_v1' entry of the 'acquisition' registry, so that a stored acquisition "
            "says which one it is (specification 5.8). It is the only value the field "
            "accepts."
        ),
    )
    beta: float = Field(
        default=1.96,
        ge=0.0,
        allow_inf_nan=False,
        description=(
            "The weight of the uncertainty in the score, in posterior standard deviations: "
            "the score of a point is 'beta' times the standard deviation of the surrogate's "
            "prediction there, minus the distance between the predicted mean and the level, "
            "both in the units of the outcome (for an outcome that the surrogate reads "
            "through a link, such as a probability, in the units of the latent function the "
            "link reads it from, and the level is converted to it). A point scores high when "
            "the surrogate is unsure about it and its prediction is close to the level, "
            "which is where the boundary of the level set is least known; the score has no "
            "direction, a prediction above the level counting as much as one below it. 0 "
            "ignores the uncertainty and scores the closeness to the level alone, the point "
            "that is predicted to lie on the boundary; the larger the value, the more the "
            "search is drawn to points that the surrogate knows little about, even if their "
            "predicted value is far from the level. It must be 0 or more and finite. The "
            "default is 1.96, the number of standard deviations at the edge of a central 95% "
            "credible interval of a normal outcome, the value that the straddle was "
            "introduced with, so that the score reads as how deep the level lies inside "
            "that interval. It is a starting point, not a recommendation for every problem: "
            "the best value depends on the noise, the smoothness of the outcome and the "
            "budget. It is a number of standard deviations, as the 'beta' of 'ucb_v1' is."
        ),
    )
