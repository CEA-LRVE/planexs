"""The configuration schema of the `none_v1` outcome transform."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from planexs.core.ComponentConfig import ComponentConfig


class NoneOutcomeTransformConfig(ComponentConfig):
    """The configuration of the `none_v1` outcome transform.

    The transform has no setting: it leaves the outcomes as they are. The schema holds the
    `type` alone, which is what makes it a component like any other (specification 5.12):
    selected by name, with a schema that rejects unknown fields.

    Attributes:
        type: The registered name, always `"none_v1"`.
    """

    type: Literal["none_v1"] = Field(
        default="none_v1",
        description=(
            "The registered name of this outcome transform, always 'none_v1'. It selects the "
            "'none_v1' entry of the 'outcome_transform' registry (specification 5.6). It is "
            "the only value the field accepts."
        ),
    )
