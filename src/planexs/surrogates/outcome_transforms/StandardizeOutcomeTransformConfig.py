"""The configuration schema of the `standardize_v1` outcome transform."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from planexs.core.ComponentConfig import ComponentConfig


class StandardizeOutcomeTransformConfig(ComponentConfig):
    """The configuration of the `standardize_v1` outcome transform: its deviation floor.

    Attributes:
        type: The registered name, always `"standardize_v1"`.
        min_std: The smallest standard deviation that an outcome is divided by.
    """

    type: Literal["standardize_v1"] = Field(
        default="standardize_v1",
        description=(
            "The registered name of this outcome transform, always 'standardize_v1'. It "
            "selects the 'standardize_v1' entry of the 'outcome_transform' registry "
            "(specification 5.6). It is the only value the field accepts."
        ),
    )
    min_std: float = Field(
        default=1e-8,
        gt=0.0,
        allow_inf_nan=False,
        description=(
            "The smallest standard deviation of an outcome, in the units of the outcome as it "
            "reaches this transform, for which the outcome is divided by its standard "
            "deviation. An outcome whose observed standard deviation is below it is only "
            "centred, and not scaled: this keeps a constant outcome (every observation "
            "equal, so a standard deviation of zero or of rounding noise) from being divided "
            "by nothing, which would give a model of nan. It must be strictly positive: zero "
            "would divide a constant outcome by zero. The default is 1e-8, which is BoTorch's "
            "own floor for its Standardize transform and the one the legacy surrogate used, "
            "so that a study that sets nothing standardizes exactly as it did: it is far "
            "above the rounding noise of a double-precision outcome of order one. Lower it "
            "for an outcome that is really measured at a very small scale (a length in "
            "metres at the nanometre scale, say), whose genuine spread would otherwise fall "
            "under the floor and leave the model to fit tiny, unscaled numbers."
        ),
    )
