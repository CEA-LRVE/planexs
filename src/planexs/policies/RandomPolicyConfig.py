"""The configuration schema of the `random_policy_v1` policy."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field

from planexs.core.ComponentConfig import ComponentConfig
from planexs.core.ComponentOf import ComponentOf
from planexs.encoding.EncoderConfig import EncoderConfig
from planexs.encoding.FeasibleSamplingConfig import FeasibleSamplingConfig


class RandomPolicyConfig(ComponentConfig):
    """The configuration of the `random_policy_v1` policy: random search, which draws its
    points from a design and ignores the observations.

    It is the stored form of the policy (what a configuration file says), and
    `RandomPolicy.fromConfig` builds the policy from it (specification 5.9, 5.12). The policy
    has one choice that is a component, the design that supplies the points, and it is selected
    by name like any other component: the `type` of the nested mapping. The other settings are
    the ones that the policy needs to do for itself what a composed policy gets from its
    surrounding study: how many times it may draw again to avoid a pending point, how the
    search space is laid out for the draws (`encoder`) and how feasible points are sampled
    (`sampling`). The last two are the settings that a study holds, with the same schemas, so
    a study that uses one configuration for several components can give the same values to
    each.

    Attributes:
        type: The registered name, always `"random_policy_v1"`.
        design: The design that the points are drawn from.
        max_redraws: How many times the policy draws again to replace points that repeat a
            pending parameterization.
        encoder: The settings of the encoder that lays out the search space for the draws.
        sampling: The settings of the feasible-point sampler and of the feasibility check.
    """

    type: Literal["random_policy_v1"] = Field(
        default="random_policy_v1",
        description=(
            "The registered name of this policy, always 'random_policy_v1'. It selects the "
            "'random_policy_v1' entry of the 'policy' registry, so that a stored policy says "
            "which one it is (specification 5.9). It is the only value the field accepts."
        ),
    )
    design: Annotated[ComponentConfig, ComponentOf("design")] = Field(
        default_factory=lambda: {"type": "random_v1", "n_points": 1},
        validate_default=True,
        description=(
            "The design that the points are drawn from: a component of the 'design' registry, "
            "written as a mapping whose 'type' is the registered name of the design (for "
            "example 'random_v1' for independent uniform points, or 'sobol_v1' for a "
            "space-filling sequence) and whose other fields are that design's own settings. "
            "The default is 'random_v1', which is random search in the usual sense: every "
            "point is an independent uniform draw in the encoded space, each level of a "
            "categorical parameter equally likely. The policy asks the design for as many "
            "points as it was asked for and never for 'its whole design', so the 'n_points' "
            "that every design schema requires is not used by the policy: it only has to "
            "be a valid value. The default design says 1 for it, the smallest."
        ),
    )
    max_redraws: int = Field(
        default=10,
        ge=0,
        description=(
            "How many times the policy draws again, from a new stream of the design, to "
            "replace points that repeat the parameterization of a pending trial or of another "
            "point of the same batch, in draws. Each draw asks for as many points as are "
            "still missing. In a space with continuous parameters no two draws coincide and "
            "this is never used; in a finite one (categorical parameters, integers, grids) "
            "a draw can land on a point that is already being evaluated, and the policy keeps "
            "the points that are new and draws again for the rest. When the draws run out "
            "before the batch is complete, the policy raises an error instead of returning "
            "a repeated point or fewer points than it was asked for. 0 makes the first draw "
            "the only one. The default is 10: each redraw replaces a point by a new draw that "
            "repeats an earlier one with a probability of at most the fraction of the space "
            "that is already taken, so while at most half of the space is taken, a point "
            "survives 10 redraws as a repeat with a probability under 0.1 percent, and the "
            "error is raised mostly when the space is nearly used up, which is what it says."
        ),
    )
    encoder: EncoderConfig = Field(
        default_factory=EncoderConfig,
        description=(
            "The settings of the encoder that lays out the search space for the draws, with "
            "the schema and the defaults of a study's 'encoder' settings (for example, which "
            "choices count as ordered, which decides whether a choice is drawn as one ordered "
            "column or as one column per level). The precision and the device are not here: "
            "they are the runtime settings of the study."
        ),
    )
    sampling: FeasibleSamplingConfig = Field(
        default_factory=FeasibleSamplingConfig,
        description=(
            "The settings of the feasible-point sampler and of the feasibility check that the "
            "design is given, with the schema and the defaults of a study's 'sampling' "
            "settings (the rounds of candidates, the hit-and-run chain inside linear "
            "constraints, the tolerance). They matter only for a search space with "
            "constraints."
        ),
    )
