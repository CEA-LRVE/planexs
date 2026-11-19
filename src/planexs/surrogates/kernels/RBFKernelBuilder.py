"""The `rbf_v1` kernel: a squared-exponential kernel on the encoded columns."""

from __future__ import annotations

from gpytorch.kernels import RBFKernel

from planexs.surrogates.kernels.AbstractKernelBuilder import AbstractKernelBuilder
from planexs.surrogates.kernels.RBFKernelConfig import RBFKernelConfig
from planexs.surrogates.kernels.registerKernel import registerKernel


@registerKernel(
    "rbf_v1",
    config=RBFKernelConfig,
    description=(
        "A squared-exponential (RBF) kernel on the encoded columns, with one lengthscale for "
        "each column or one shared."
    ),
)
class RBFKernelBuilder(AbstractKernelBuilder[RBFKernelConfig]):
    """A squared-exponential kernel (also called radial basis function), over the encoded
    columns.

    It is the plain `gpytorch.kernels.RBFKernel`: with `ard` there is one lengthscale for each
    encoded column, without it one shared by all of them. It is the smoothest of the kernels of
    this kind (infinitely differentiable), so it suits a response that is known to be very
    smooth, and it can be too confident about one that is not.

    Like the Matern kernel it has no prior, no output scale and no batch shape (so one model of
    several outputs needs one kernel for each): see `MaternKernelBuilder`, whose notes apply to
    it as they stand.
    """

    def makeKernel(self, n_columns: int) -> RBFKernel:
        """Makes the RBF kernel.

        Args:
            n_columns: The number of encoded columns, one or more.

        Returns:
            A new `RBFKernel` with `n_columns` lengthscales if `config.ard` and one otherwise.
        """
        return RBFKernel(ard_num_dims=n_columns if self.config.ard else None)
