"""The options of `Surrogate.predict`, as a configuration schema."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class PredictionConfig(BaseModel):
    """The options that `Surrogate.predict` takes (specification 3.4, 5.6, 10.8).

    One setting for now: whether the prediction includes the observation noise. It travels in
    an options object, and not in a defaulted argument of `predict`, so that a value that a
    caller could reasonably want to change is reachable, documented and stored in one place,
    and so that a later option (such as the space of a prediction, for a surrogate with a
    link) is a new field and not a new signature. This class has no `type` field, is not
    registered in any kind and is not a `ComponentConfig`, like `RuntimeConfig` and
    `EncoderConfig`. A study-level model can hold it as a field, and `validateConfig` reads it
    like any other model, with the same field paths in its errors.

    A configuration is frozen: build a new one to change a setting.

    Attributes:
        include_observation_noise: Whether the predicted standard deviation includes the
            observation noise.
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
