"""The `botorch_default_v1` kernel: no kernel, so that the engine chooses."""

from __future__ import annotations

from gpytorch.kernels import Kernel

from planexs.surrogates.kernels.AbstractKernelBuilder import AbstractKernelBuilder
from planexs.surrogates.kernels.BotorchDefaultKernelConfig import BotorchDefaultKernelConfig
from planexs.surrogates.kernels.registerKernel import registerKernel


@registerKernel(
    "botorch_default_v1",
    config=BotorchDefaultKernelConfig,
    description="No kernel of its own: leaves the choice of the kernel to BoTorch.",
)
class BotorchDefaultKernelBuilder(AbstractKernelBuilder[BotorchDefaultKernelConfig]):
    """Leaves the choice of the kernel to the engine, by returning no kernel.

    A surrogate that gets `None` from `build` does not give its model a covariance module, and
    BoTorch builds the one it considers best for the data. That is a deliberate choice and not
    a missing one (specification 3.5: reuse the engine). In the installed version of BoTorch
    (0.18) it is an RBF kernel with one lengthscale per input column and a lognormal prior on
    each, whose location grows with the number of columns (Hvarfner et al., 2024), batched over
    the outputs of the model. That prior is what keeps the fit usable on a handful of
    observations, and it is also why this name means whatever the engine does: it follows
    BoTorch from one version to the next, which the library versions recorded with a study
    make traceable (specification 5.11).

    The other kernels of this kind are explicit: they build the kernel named in their
    configuration, and nothing more.
    """

    def makeKernel(self, n_columns: int) -> Kernel | None:
        """Returns no kernel.

        Args:
            n_columns: The number of encoded columns. Not used: the engine reads it from the
                data.

        Returns:
            `None`.
        """
        return None
