"""Kernels: the covariance functions that a Gaussian-process surrogate is built on
(specification 5.6, 6).

A surrogate's configuration holds a kernel by name, so the kernels are the registered entries
of the `kernel` kind of components (specification 5.12): the contract that every kernel honours
(`AbstractKernelBuilder`), the registry (`kernelRegistry`), the registration decorator
(`registerKernel`) and the kernels themselves. A kernel receives the number of encoded columns
and returns a native GPyTorch kernel, or nothing, to leave the choice to the engine:

- `botorch_default_v1` (`BotorchDefaultKernelBuilder`): no kernel of its own, so BoTorch
  chooses;
- `matern_v1` (`MaternKernelBuilder`): a Matern kernel of smoothness 0.5, 1.5 or 2.5, with or
  without one lengthscale for each column;
- `rbf_v1` (`RBFKernelBuilder`): a squared-exponential kernel, with or without one lengthscale
  for each column.

A module that registers a kernel must be imported here, one line per module, or the kernel is
silently unregistered (specification 5.12, 10.6). Importing the package also defines the
`kernel` kind, which `kernelRegistry` is imported for.
"""

from planexs.surrogates.kernels import (
    BotorchDefaultKernelBuilder,
    MaternKernelBuilder,
    RBFKernelBuilder,
    kernelRegistry,
)

__all__ = [
    "BotorchDefaultKernelBuilder",
    "MaternKernelBuilder",
    "RBFKernelBuilder",
    "kernelRegistry",
]
