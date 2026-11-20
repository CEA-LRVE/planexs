"""The `designs` layer: procedures that propose points without a model (specification 5.1,
5.8).

A design is what a study uses before there is a model to consult: space-filling sequences or
random draws, always satisfying the constraints, and able to count the observations that exist
already so that a warm start does not repeat an initial phase unnecessarily. Which design is
used is a choice of configuration, so this layer is organized around the `design` kind of
components: the contract that every design honours (`AbstractDesign`), what it is given
(`DesignContext`), the registry (`designRegistry`) and the registration decorator
(`registerDesign`). The designs themselves are the scrambled Sobol points of `sobol_v1`
(`SobolDesign`) and the uniform random draws of `random_v1` (`RandomDesign`).

The rule of this layer: it may import `planexs.core`, `planexs.domain` and `planexs.encoding`,
and no higher layer. Like `encoding`, it may import torch and BoTorch; only `adapters` may
import Ax.

A module that registers a design must be imported here, one line per module, or the design is
silently unregistered (specification 5.12, 10.6). Importing the package also defines the
`design` kind, which `designRegistry` is imported for.
"""

from planexs.designs import RandomDesign, SobolDesign, designRegistry

__all__ = ["RandomDesign", "SobolDesign", "designRegistry"]
