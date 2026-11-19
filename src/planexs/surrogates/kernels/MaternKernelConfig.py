"""The configuration schema of the `matern_v1` kernel."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator

from planexs.core.ComponentConfig import ComponentConfig

#: The smoothness values that the Matern kernel of GPyTorch accepts, which are the ones it has
#: a closed form for. This is a fact about the engine and not a setting: the installed version
#: refuses any other value with a `RuntimeError` that carries no field path, and a test pins
#: the set against it. It cannot be a `Literal`, because the typing rules allow no float in
#: one.
SUPPORTED_SMOOTHNESS: tuple[float, ...] = (0.5, 1.5, 2.5)


class MaternKernelConfig(ComponentConfig):
    """The configuration of the `matern_v1` kernel: its smoothness and its lengthscales.

    Attributes:
        type: The registered name, always `"matern_v1"`.
        nu: The smoothness of the kernel: `0.5`, `1.5` or `2.5`.
        ard: Whether each encoded column has a lengthscale of its own.
    """

    type: Literal["matern_v1"] = Field(
        default="matern_v1",
        description=(
            "The registered name of this kernel, always 'matern_v1'. It selects the "
            "'matern_v1' entry of the 'kernel' registry (specification 5.6). It is the only "
            "value the field accepts."
        ),
    )
    nu: float = Field(
        default=2.5,
        description=(
            "The smoothness of the Matern kernel, a dimensionless number that is 0.5, 1.5 or "
            "2.5: the values that GPyTorch has a closed form for, and the only ones accepted. "
            "The kernel gives functions that are continuous but not differentiable (0.5), "
            "once differentiable (1.5) or twice differentiable (2.5), so a smaller value "
            "models a rougher response. The default is 2.5, the Matern 5/2 kernel that "
            "Bayesian optimization conventionally uses: the response of an experiment or of a "
            "simulation is rarely rougher than twice differentiable, and a kernel that is too "
            "rough wastes observations on noise-like wiggles. Use 0.5 or 1.5 for a response "
            "known to have kinks or abrupt changes."
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

    @field_validator("nu")
    @classmethod
    def _checkSmoothness(cls, value: float) -> float:
        """Checks that `nu` is one of the values that the Matern kernel of GPyTorch accepts.

        Args:
            value: The smoothness.

        Returns:
            The smoothness, unchanged.

        Raises:
            ValueError: if it is not one of `SUPPORTED_SMOOTHNESS`. The message names the field
                and lists the accepted values (the value received is added to it by whoever
                reports the error, with the field path).
        """
        if value not in SUPPORTED_SMOOTHNESS:
            accepted = ", ".join(str(number) for number in SUPPORTED_SMOOTHNESS)
            raise ValueError(
                f"nu must be one of {accepted}, the smoothness values that the Matern kernel "
                "has a closed form for"
            )
        return value
