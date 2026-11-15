"""The options of `Surrogate.sample`, as a configuration schema."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class SamplingConfig(BaseModel):
    """The options that `Surrogate.sample` takes (specification 3.4, 5.6, 10.8).

    Two settings: how many joint posterior samples are drawn, and whether they include the
    observation noise. They travel in an options object, and not in defaulted arguments of
    `sample`, for the reason given by `PredictionConfig`. This class has no `type` field, is
    not registered in any kind and is not a `ComponentConfig`.

    A configuration is frozen: build a new one to change a setting.

    Attributes:
        n_samples: How many samples are drawn.
        include_observation_noise: Whether the samples include the observation noise.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    n_samples: int = Field(
        default=256,
        ge=1,
        strict=True,
        description=(
            "How many joint posterior samples `Surrogate.sample` draws, in samples. Each "
            "sample is one draw of the outcomes at all the requested points together, so "
            "the spread between samples is the uncertainty of the model and their average "
            "converges on the predicted mean at the rate of one over the square root of "
            "this number. It must be at least 1. The default is 256, the number of Monte "
            "Carlo samples that the library's own acquisition functions use by default: "
            "enough to see the shape and spread of the posterior, and cheap."
        ),
    )
    include_observation_noise: bool = Field(
        default=False,
        strict=True,
        description=(
            "Whether the samples include the observation noise, a flag. When False they are "
            "draws of the latent function of the surrogate (its epistemic uncertainty "
            "only). When True they are draws of new noisy measurements of the outcomes. It "
            "is refused for an outcome whose kind has no observation noise (a binary "
            "label). The default is False, for the reason given by `PredictionConfig`."
        ),
    )
