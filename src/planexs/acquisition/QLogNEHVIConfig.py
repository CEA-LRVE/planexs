"""The configuration schema of the `qlog_nehvi_v1` acquisition."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from planexs.acquisition.ExplicitReferencePointConfig import ExplicitReferencePointConfig
from planexs.acquisition.InferredReferencePointConfig import InferredReferencePointConfig
from planexs.core.ComponentConfig import ComponentConfig


class QLogNEHVIConfig(ComponentConfig):
    """The configuration of the `qlog_nehvi_v1` acquisition: the multi-objective Monte Carlo
    log noisy expected hypervolume improvement (specification 5.8, FR2).

    It is the stored form of the acquisition (what a configuration file says), and
    `QLogNEHVIBuilder.fromConfig` builds the builder from it (specification 5.12). Three
    settings are its policy values: the reference point of the hypervolume, how many Monte
    Carlo samples estimate the expectation, and whether the observed points that are very
    unlikely to be on the Pareto front are dropped from the baseline. Everything else is read
    from the goal and the context: the objectives and their directions and the outcome
    constraints from the goal, the pending points from the context.

    Attributes:
        type: The registered name, always `"qlog_nehvi_v1"`.
        reference_point: Where the reference point of the hypervolume comes from.
        n_samples: The number of Monte Carlo samples that estimate the expectation.
        prune_baseline: Whether the observed points that are very unlikely to be on the Pareto
            front are left out of the baseline.
    """

    type: Literal["qlog_nehvi_v1"] = Field(
        default="qlog_nehvi_v1",
        description=(
            "The registered name of this acquisition, always 'qlog_nehvi_v1'. It selects the "
            "'qlog_nehvi_v1' entry of the 'acquisition' registry, so that a stored "
            "acquisition says which one it is (specification 5.8). It is the only value the "
            "field accepts."
        ),
    )
    reference_point: ExplicitReferencePointConfig | InferredReferencePointConfig = Field(
        default_factory=InferredReferencePointConfig,
        discriminator="kind",
        description=(
            "Where the reference point of the hypervolume comes from, which says how bad an "
            "outcome may be and still count as a result. Either 'explicit' (kind: explicit, "
            "values: one number for each objective, in the unit of the outcome) or 'inferred' "
            "(kind: inferred, rule: the name of a rule, margin: its setting), which reads it "
            "off the Pareto front that the surrogate predicts at the observed points each "
            "time the acquisition is built. The default is the inferred 'nadir_margin' rule "
            "with its own defaults, so that a first run needs no knowledge of the scale of "
            "the outcomes; give the explicit form when the worst acceptable value of each "
            "objective is known, since only the person who knows what a useless result is "
            "can say it, and the choice changes which points of the front are worth finding."
        ),
    )
    n_samples: int = Field(
        default=128,
        ge=1,
        description=(
            "The number of Monte Carlo samples of the joint posterior that estimate the "
            "expectation, in samples. The acquisition value is the mean of the hypervolume "
            "improvement over these samples, so more samples make the estimate smoother and "
            "its maximum easier to find. A Pareto front is built for each sample, so the cost "
            "grows with the number of samples, with the number of observed points and "
            "steeply with the number of objectives. The samples are quasi-random (a scrambled "
            "Sobol sequence, seeded from the generator of the build) and are the same for "
            "every candidate of one build, so the estimate is a deterministic function of "
            "the candidate that an optimizer can follow. It is also the number of samples "
            "that decide the pruning of the baseline (see 'prune_baseline'). It must be at "
            "least 1. The default is 128, the number the multi-objective blend of the legacy "
            "layer used, a quarter of the 512 that 'qlog_nei_v1' uses because a hypervolume "
            "costs a front for each sample. A power of two keeps a Sobol sequence balanced."
        ),
    )
    prune_baseline: bool = Field(
        default=True,
        description=(
            "Whether the observed points that are very unlikely to be on the Pareto front, "
            "better than the reference point and feasible are left out of the baseline, the "
            "set of observed points that the improvement is measured against. The acquisition "
            "samples the joint posterior of every baseline point together with each "
            "candidate, so the cost grows quickly with the number of baseline points, and a "
            "point that is almost surely dominated cannot change the front. The default is "
            "True, as BoTorch recommends: pruning keeps the cost low once there are many "
            "observations and changes the value only where the pruned points could have been "
            "on the front with negligible probability. False keeps every observed point, "
            "which costs more and is the choice when a result must not depend on the "
            "pruning, for example to compare with another implementation."
        ),
    )
