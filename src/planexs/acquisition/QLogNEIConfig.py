"""The configuration schema of the `qlog_nei_v1` acquisition."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from planexs.core.ComponentConfig import ComponentConfig


class QLogNEIConfig(ComponentConfig):
    """The configuration of the `qlog_nei_v1` acquisition: the Monte Carlo log noisy expected
    improvement (specification 5.8, FR10, FR15).

    It is the stored form of the acquisition (what a configuration file says), and
    `QLogNEIBuilder.fromConfig` builds the builder from it (specification 5.12). Two settings
    are the policy values of the acquisition: how many Monte Carlo samples estimate the
    expectation, and whether the points that are very unlikely to be the best one are dropped
    from the incumbent set. Everything else is read from the goal and the context and is not a
    setting: the direction and the weights of the objective and the outcome constraints come
    from the goal, the pending points from the context.

    Attributes:
        type: The registered name, always `"qlog_nei_v1"`.
        n_samples: The number of Monte Carlo samples that estimate the expectation.
        prune_baseline: Whether the observed points that are very unlikely to be the best one
            are left out of the incumbent set.
    """

    type: Literal["qlog_nei_v1"] = Field(
        default="qlog_nei_v1",
        description=(
            "The registered name of this acquisition, always 'qlog_nei_v1'. It selects the "
            "'qlog_nei_v1' entry of the 'acquisition' registry, so that a stored acquisition "
            "says which one it is (specification 5.8). It is the only value the field "
            "accepts."
        ),
    )
    n_samples: int = Field(
        default=512,
        ge=1,
        description=(
            "The number of Monte Carlo samples of the joint posterior that estimate the "
            "expectation, in samples. The acquisition value is the mean of the improvement "
            "over these samples, so more samples make the estimate smoother and its maximum "
            "easier to find, at a cost that grows in proportion to the number of samples "
            "(and with the number of observed points, which are sampled jointly with each "
            "candidate). The samples are quasi-random (a scrambled Sobol sequence, seeded "
            "from the generator of the build) and are the same for every candidate of one "
            "build, so the estimate is a deterministic function of the candidate that an "
            "optimizer can follow. It is also the number of samples that decide the pruning "
            "of the baseline (see 'prune_baseline'). It must be at least 1. The default is "
            "512, the number that BoTorch uses for its own Monte Carlo acquisition functions, "
            "so that an unconfigured 'qlog_nei_v1' behaves as the engine's documentation "
            "describes; the exploration/exploitation blend of the legacy layer used 256 "
            "because it paid for a second term per sample. A power of two keeps a Sobol "
            "sequence balanced."
        ),
    )
    prune_baseline: bool = Field(
        default=True,
        description=(
            "Whether the observed points that are highly unlikely to be the best one are "
            "left out of the incumbent set, the set of observed points that the improvement "
            "is measured against. The acquisition samples the joint posterior of every "
            "incumbent point together with each candidate, so the cost grows quickly with "
            "the number of incumbent points, and a point whose outcome is almost certainly "
            "worse than another's cannot be the incumbent. A point that violates an outcome "
            "constraint is pruned in the same way. The default is True, as BoTorch "
            "recommends: pruning keeps the cost low once there are many observations and "
            "changes the value only where the pruned points could have been the best one with "
            "negligible probability. False keeps every observed point, which costs more and "
            "is the choice when a result must not depend on the pruning, for example to "
            "compare with another implementation."
        ),
    )
