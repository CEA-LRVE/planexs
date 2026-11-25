"""The options of `Surrogate.predict`, as a configuration schema."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class PredictionConfig(BaseModel):
    """The options that `Surrogate.predict` takes (specification 3.4, 5.6, 10.8).

    Two settings: whether the prediction includes the observation noise, and whether it is read
    through the link of the surrogate (a classifier's probability instead of its latent
    function). They travel in an options object, and not in defaulted arguments of `predict`,
    so that a value that a caller could reasonably want to change is reachable, documented and
    stored in one place, and so that a later option is a new field and not a new signature.
    This class has no `type` field, is not registered in any kind and is not a
    `ComponentConfig`, like `RuntimeConfig` and
    `EncoderConfig`. A study-level model can hold it as a field, and `validateConfig` reads it
    like any other model, with the same field paths in its errors.

    A configuration is frozen: build a new one to change a setting.

    Attributes:
        include_observation_noise: Whether the predicted standard deviation includes the
            observation noise.
        through_link: Whether the prediction is read through the link of the surrogate, in
            the response space.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    include_observation_noise: bool = Field(
        default=False,
        strict=True,
        description=(
            "Whether the predicted standard deviation includes the observation noise, a "
            "flag. When False the prediction describes the latent function of the "
            "surrogate (its epistemic uncertainty only, in the units of the outcome), and "
            "its space is 'latent'. When True it describes a new noisy measurement of the "
            "outcome: the variance of the latent function plus the observation noise, and "
            "its space is 'response'. It is refused for an outcome whose kind has no "
            "observation noise (a binary label). The default is False: the question a "
            "caller asks of a surrogate is almost always what it believes about the "
            "function, and noise that the caller did not ask for would hide how little the "
            "model knows at a point (specification 5.6)."
        ),
    )
    through_link: bool = Field(
        default=False,
        strict=True,
        description=(
            "Whether the prediction is read through the link of the surrogate, a flag. A "
            "surrogate for an outcome that is not Gaussian, such as the probability that a "
            "label is true, models a latent function and reads the outcome from it through a "
            "link. When False the prediction describes that latent function (the mean and "
            "the standard deviation of the function, in its own units), and its space is "
            "'latent'. When True it describes the outcome as the link reads it, in the "
            "units of the response (a probability, between 0 and 1): the mean is the "
            "predictive probability, with the uncertainty about the function averaged in, "
            "and the standard deviation is that of the probability itself, the epistemic "
            "uncertainty about it, and its space is 'response'. It cannot be combined with "
            "'include_observation_noise', since the response that a link gives is not a "
            "noisy measurement of the function, and it is refused for a surrogate that has "
            "no link. The default is False, for the reason that the observation noise is "
            "left out: the question a caller asks of a surrogate is almost always what it "
            "believes about the function (specification 5.6), and the space of every "
            "prediction is stated in the result either way."
        ),
    )
