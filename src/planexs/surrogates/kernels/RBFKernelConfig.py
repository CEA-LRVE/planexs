"""The configuration schema of the `rbf_v1` kernel."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from planexs.core.ComponentConfig import ComponentConfig


class RBFKernelConfig(ComponentConfig):
    """The configuration of the `rbf_v1` kernel: its lengthscales.

    Attributes:
        type: The registered name, always `"rbf_v1"`.
        ard: Whether each encoded column has a lengthscale of its own.
    """

    type: Literal["rbf_v1"] = Field(
        default="rbf_v1",
        description=(
            "The registered name of this kernel, always 'rbf_v1'. It selects the 'rbf_v1' "
            "entry of the 'kernel' registry (specification 5.6). It is the only value the "
            "field accepts."
        ),
    )
    ard: bool = Field(
        default=True,
        description=(
            "Whether the kernel has one lengthscale for each encoded column (automatic "
            "relevance determination, true) or one lengthscale shared by all of them "
            "(false). A column is a column of the encoding, so a categorical parameter that "
            "is encoded as a one-hot block has one lengthscale for each of its levels. With "
            "a lengthscale of its own, a column that the outcome hardly depends on gets a "
            "long one, and the model can tell the relevant parameters from the others; the "
            "price is one more hyperparameter to fit for each column. The default is true, "
            "because that is what an experiment with several parameters needs and what "
            "BoTorch's own default kernel does. Use false for very few observations, or "
            "when all the parameters are known to matter alike."
        ),
    )
