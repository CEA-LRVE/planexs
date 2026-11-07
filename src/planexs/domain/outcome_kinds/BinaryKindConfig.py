"""The configuration schema of the `binary` outcome kind."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from planexs.core.ComponentConfig import ComponentConfig


class BinaryKindConfig(ComponentConfig):
    """The configuration of the `binary` outcome kind.

    The kind has no setting: what a binary label is does not depend on the study. The schema
    holds the `type` alone, which is what makes the kind a component like any other
    (specification 5.12): selected by name, with a schema that rejects unknown fields.

    Attributes:
        type: The registered name, always `"binary"`.
    """

    type: Literal["binary"] = Field(
        default="binary",
        description=(
            "The registered name of this outcome kind, always 'binary'. It selects the "
            "'binary' entry of the 'outcome_kind' registry, so that an outcome says which "
            "kind it is (specification 5.4). It is the only value the field accepts."
        ),
    )
