"""The `matern_v1` kernel: a Matern kernel on the encoded columns."""

from __future__ import annotations

from gpytorch.kernels import MaternKernel

from planexs.surrogates.kernels.AbstractKernelBuilder import AbstractKernelBuilder
from planexs.surrogates.kernels.MaternKernelConfig import MaternKernelConfig
from planexs.surrogates.kernels.registerKernel import registerKernel


@registerKernel(
    "matern_v1",
    config=MaternKernelConfig,
    description=(
        "A Matern kernel of smoothness 0.5, 1.5 or 2.5 on the encoded columns, with one "
        "lengthscale for each column or one shared."
    ),
)
class MaternKernelBuilder(AbstractKernelBuilder[MaternKernelConfig]):
    """A Matern kernel of the configured smoothness, over the encoded columns.

    It is the plain `gpytorch.kernels.MaternKernel`: with `ard` there is one lengthscale for
    each encoded column, without it one shared by all of them. Two things are left out on
    purpose, and a surrogate that wants them adds them:

    - **No prior.** The lengthscales carry no prior, so a fit that maximizes the marginal
      likelihood with the engine's priors (a MAP fit) is, for this kernel, a maximum
      likelihood fit of its hyperparameters. BoTorch's own default kernel carries a prior
      (see `BotorchDefaultKernelBuilder`).
    - **No output scale.** It is not wrapped in a `ScaleKernel`, as BoTorch's default is not:
      a surrogate standardizes its outcomes, so a signal variance of one is what the
      covariance needs. A surrogate that does not standardize them must scale the kernel
      itself.

    The kernel has no batch shape, so it cannot be the covariance of one batched model of
    several independent outputs: BoTorch cannot fit such a model (its fitting closure expects
    hyperparameters with the model's batch shape). BoTorch's own default kernel is built with
    that shape, which is why `botorch_default_v1` does not have this limit. A surrogate that
    models several outcomes with an explicit kernel builds one model for each outcome, with a
    kernel of its own: every call to `build` returns a new one.
    """

    def makeKernel(self, n_columns: int) -> MaternKernel:
        """Makes the Matern kernel.

        Args:
            n_columns: The number of encoded columns, one or more.

        Returns:
            A new `MaternKernel` of smoothness `config.nu`, with `n_columns` lengthscales if
            `config.ard` and one otherwise.
        """
        return MaternKernel(
            nu=self.config.nu, ard_num_dims=n_columns if self.config.ard else None
        )
