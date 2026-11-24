"""The configuration schema of the `neg_integrated_variance_v1` acquisition."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from planexs.core.ComponentConfig import ComponentConfig


class NegIntegratedVarianceConfig(ComponentConfig):
    """The configuration of the `neg_integrated_variance_v1` acquisition: the negative
    integrated posterior variance (specification 5.8, FR21).

    It is the stored form of the acquisition (what a configuration file says), and
    `NegIntegratedVarianceBuilder.fromConfig` builds the builder from it (specification 5.12).
    One setting is its policy value: how many points of the search space the variance is
    averaged over. Everything else is read from the goal and the context: the outcomes to
    learn from the explore goal, the pending points from the context.

    Attributes:
        type: The registered name, always `"neg_integrated_variance_v1"`.
        n_points: The number of points of the search space that the posterior variance is
            averaged over.
    """

    type: Literal["neg_integrated_variance_v1"] = Field(
        default="neg_integrated_variance_v1",
        description=(
            "The registered name of this acquisition, always 'neg_integrated_variance_v1'. "
            "It selects the 'neg_integrated_variance_v1' entry of the 'acquisition' registry, "
            "so that a stored acquisition says which one it is (specification 5.8). It is "
            "the only value the field accepts."
        ),
    )
    n_points: int = Field(
        default=512,
        ge=1,
        description=(
            "The number of points of the search space that the posterior variance is "
            "averaged over, in points. The acquisition value is minus the mean of the "
            "variance of the model over these points, once the candidate has been added to "
            "the data: a Monte Carlo estimate of the integral over the whole space, so more "
            "points make it more accurate and cost more in proportion (every candidate is "
            "scored at every point), and a space with more parameters needs more points to "
            "cover it. The points are drawn from the encoder of the problem: a scrambled "
            "Sobol sequence over the search space, seeded from the generator of the build, "
            "and rounded onto the values that the parameters can take, so that they are "
            "space-filling and are the same for every candidate of one build. They are not "
            "restricted to the points that satisfy the constraints of the search space. It "
            "must be at least 1. The default is 512, the size of the space-filling set that "
            "the exploration/exploitation blend of the legacy layer averages its terms over "
            "(the same job: to stand for the whole space), a power of two so that a Sobol "
            "sequence stays balanced; BoTorch leaves the number to the caller."
        ),
    )
