"""The `encoding` layer: how a search space is laid out as tensors in the unit cube
(specification 5.1, 5.5).

Surrogates, acquisitions, optimizers and designs all work on one encoding of the search
space, so they never need to know about log scales, integer grids or categorical values.
This layer owns that encoding: the search-space encoder and the layout it gives each
parameter (`SearchSpaceEncoder`, `ParameterEncoding`), the compiler that turns a
validated constraint expression into a tensor-evaluating function (`compileExpression`), the
constraints of a search space rewritten on the encoded columns (`EncodedConstraints`,
`toBotorchNonlinearConstraints`), the sampler of feasible encoded points
(`sampleFeasibleEncoded`), and the helpers that turn the runtime settings and the study's
seeds into torch objects (`torch_runtime`).

The rule of this layer: it may import `planexs.core` and `planexs.domain`, and no higher
layer. Unlike those two, it may import torch and BoTorch; only `adapters` may import Ax.
"""
