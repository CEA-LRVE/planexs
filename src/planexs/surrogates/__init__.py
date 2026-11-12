"""The `surrogates` layer: probabilistic models of the outcomes, fitted on observations
(specification 5.1, 5.6).

A surrogate is chosen by name in configuration, so this layer is organized around the
`surrogate` kind of components: the contract that every surrogate honours
(`AbstractSurrogateBuilder`), what it is given (`SurrogateBuildContext`) and what it returns
(`BuiltSurrogate`, with its `FitStatus`), the registry (`surrogateRegistry`) and the
registration decorator (`registerSurrogate`). The surrogates themselves, the nested kinds they
are composed of (kernels, likelihoods, outcome transforms, fitters), the `Surrogate` wrapper
and the cache come with the later tasks of roadmap Phase 3.

The rule of this layer: it may import `planexs.core`, `planexs.domain` and `planexs.encoding`,
and no higher layer. Like `encoding`, it may import torch and BoTorch; only `adapters` may
import Ax.

Importing the package defines the `surrogate` kind and registers the capability tags of the
layer (`posterior:gaussian`, `outputs:multi`, `noise:known`, ...): that is what
`surrogateRegistry` is imported for below. A module that registers a surrogate must be
imported here, one line per module, or the surrogate is silently unregistered
(specification 5.12, 10.6).
"""

from planexs.surrogates import surrogateRegistry

__all__ = ["surrogateRegistry"]
