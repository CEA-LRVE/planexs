"""What to do when only some of the observations give a noise standard deviation."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

#: The named rules that say what `trialsToTensors` does with partial knowledge of noise (see
#: `NoisePolicyConfig.partial`). A rule is a name selected in the configuration, so that the
#: choice is reachable by the caller and stored with the study (specification 3.4).
PartialNoiseRule = Literal["ignore_with_warning", "error"]


class NoisePolicyConfig(BaseModel):
    """How partial knowledge of observation noise is treated (specification 5.4, 3.4; ADR-0012,
    decision 5).

    An observation gives a noise standard deviation for the outcomes where it is known, and
    leaves the others out. That is fine when it is known for all of them or for none, but a
    Gaussian process cannot mix fixed and inferred noise in one fit, so a history in which only
    some values have a deviation needs a decision. The decision is not made by a hidden rule:
    it is this setting, which a surrogate's configuration holds and `trialsToTensors` reads.

    Knowledge is *partial* when some, but not all, of the values that can have a noise
    standard deviation have one. The values that can have one are those of the outcomes whose
    kind allows noise (a continuous outcome, not a binary one), in each completed trial. A
    history in which none of them has a deviation is not partial: the noise is simply unknown
    and is inferred, with no warning.

    It is the configuration of one decision, not of a registered component: this class has no
    `type` field, is not registered in any kind and is not a `ComponentConfig`, like
    `EncoderConfig`. A surrogate's configuration holds it as one of its fields, and
    `validateConfig` reads it like any other model, with the same field paths in its errors.

    The default is the behaviour the library had before the decision became a setting, so that
    a study that sets nothing behaves as before. It is written in the field below and nowhere
    else. A configuration is frozen: it is shared by everything that reads the history and
    must not change under it.

    Attributes:
        partial: What happens when the noise is known for only some of the values.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    partial: PartialNoiseRule = Field(
        default="ignore_with_warning",
        description=(
            "The named rule that decides what happens when the observations give a noise "
            "standard deviation for only some of the values that can have one. The rules are "
            "'ignore_with_warning' (drop every known deviation, let the surrogate infer the "
            "noise of all the observations, and log a warning that says how many deviations "
            "were known) and 'error' (refuse the data with an error that says how many were "
            "known). A history in which no deviation is known, or all of them are, is never "
            "partial, so neither rule has any effect on it. The default is "
            "'ignore_with_warning': a Gaussian process cannot mix fixed and inferred noise, "
            "so a study with a few measured deviations among many unmeasured ones is better "
            "served by inferring the noise, and the warning keeps the loss in view. Choose "
            "'error' when a missing deviation can only be a mistake in the data, and should "
            "stop the run instead of being worked around."
        ),
    )
