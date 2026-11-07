"""The configuration schema of the `continuous` outcome kind."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from planexs.core.ComponentConfig import ComponentConfig


class ContinuousKindConfig(ComponentConfig):
    """The configuration of the `continuous` outcome kind.

    The kind has no setting: what a continuous value is does not depend on the study. The
    schema holds the `type` alone, which is what makes the kind a component like any other
    (specification 5.12): selected by name, with a schema that rejects unknown fields.

    Attributes:
        type: The registered name, always `"continuous"`.
    """

    type: Literal["continuous"] = Field(
        default="continuous",
        description=(
            "The registered name of this outcome kind, always 'continuous'. It selects the "
            "'continuous' entry of the 'outcome_kind' registry, so that an outcome says which "
            "kind it is (specification 5.4). It is the only value the field accepts."
        ),
    )
