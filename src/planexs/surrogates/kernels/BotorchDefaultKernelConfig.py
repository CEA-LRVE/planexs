"""The configuration schema of the `botorch_default_v1` kernel."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from planexs.core.ComponentConfig import ComponentConfig


class BotorchDefaultKernelConfig(ComponentConfig):
    """The configuration of the `botorch_default_v1` kernel.

    The kernel has no setting: it names no kernel at all, and leaves the choice to the engine.
    The schema holds the `type` alone, which is what makes it a component like any other
    (specification 5.12): selected by name, with a schema that rejects unknown fields.

    Attributes:
        type: The registered name, always `"botorch_default_v1"`.
    """

    type: Literal["botorch_default_v1"] = Field(
        default="botorch_default_v1",
        description=(
            "The registered name of this kernel, always 'botorch_default_v1'. It selects "
            "the 'botorch_default_v1' entry of the 'kernel' registry, which gives the "
            "surrogate no kernel and so leaves the choice to BoTorch (specification 5.6). It "
            "is the only value the field accepts."
        ),
    )
