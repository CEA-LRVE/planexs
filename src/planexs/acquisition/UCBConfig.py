"""The configuration schema of the `ucb_v1` acquisition."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from planexs.core.ComponentConfig import ComponentConfig


class UCBConfig(ComponentConfig):
    """The configuration of the `ucb_v1` acquisition: the analytic upper confidence bound
    (specification 5.8).

    It is the stored form of the acquisition (what a configuration file says), and
    `UCBBuilder.fromConfig` builds the builder from it (specification 5.12). One setting is the
    policy value of the acquisition: how many posterior standard deviations are added to the
    mean. Everything else is read from the goal: the outcomes, their directions and their
    weights.

    Attributes:
        type: The registered name, always `"ucb_v1"`.
        beta: The number of posterior standard deviations that the bound adds to the mean.
    """

    type: Literal["ucb_v1"] = Field(
        default="ucb_v1",
        description=(
            "The registered name of this acquisition, always 'ucb_v1'. It selects the "
            "'ucb_v1' entry of the 'acquisition' registry, so that a stored acquisition says "
            "which one it is (specification 5.8). It is the only value the field accepts."
        ),
    )
    beta: float = Field(
        default=1.96,
        ge=0.0,
        allow_inf_nan=False,
        description=(
            "The weight of the uncertainty in the bound, in posterior standard deviations: "
            "the acquisition is the mean of the objective plus 'beta' times its standard "
            "deviation, both as the surrogate predicts them, so a point scores higher the "
            "better its predicted objective is and the less certain the surrogate is about "
            "it. 0 ignores the uncertainty and scores the mean alone, which exploits what is "
            "known; the larger the value, the more an unexplored point is worth. It must be "
            "0 or more and finite. The default is 1.96, the number of standard deviations "
            "that bounds a normal outcome from above at the edge of its central 95% "
            "credible interval, so that the score reads as the optimistic end of what the "
            "surrogate considers plausible. It is a starting point, not a recommendation for "
            "every problem: the best value depends on the noise, the smoothness and the "
            "budget, and a noisy or high-dimensional problem usually asks for a different "
            "one. It is a number of standard deviations (as the 'beta' of the straddle "
            "acquisition is), whereas BoTorch's own parameter of the same name is the square "
            "of it: 'ucb_v1' squares this value when it builds the native function, so the "
            "native object holds beta squared."
        ),
    )
