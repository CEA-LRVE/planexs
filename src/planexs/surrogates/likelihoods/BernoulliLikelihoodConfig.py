"""The configuration schema of the `bernoulli_v1` likelihood."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from planexs.core.ComponentConfig import ComponentConfig


class BernoulliLikelihoodConfig(ComponentConfig):
    """The configuration of the `bernoulli_v1` likelihood.

    The likelihood has no setting: a label is true with the probability that the standard
    normal distribution gives to the latent function, and nothing about that depends on the
    study. The schema holds the `type` alone, which is what makes the likelihood a component
    like any other (specification 5.12): selected by name, with a schema that rejects unknown
    fields.

    Attributes:
        type: The registered name, always `"bernoulli_v1"`.
    """

    type: Literal["bernoulli_v1"] = Field(
        default="bernoulli_v1",
        description=(
            "The registered name of this likelihood, always 'bernoulli_v1'. It selects the "
            "'bernoulli_v1' entry of the 'likelihood' registry, so that a stored likelihood "
            "says which one it is (specification 5.6). It is the only value the field "
            "accepts."
        ),
    )
