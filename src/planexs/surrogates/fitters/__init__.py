"""Fitters: the procedures that fit the parameters of a surrogate's model to its data
(specification 5.6, 6).

How the hyperparameters of a model are set from the data is one of the choices that a surrogate
builder exposes as a nested kind of components, so that it is selected by name in configuration
(specification 3.3): the contract that every fitter honours (`AbstractFitter`), the error for a
model that cannot be fitted (`FitterError`), the registry (`fitterRegistry`) and the
registration decorator (`registerFitter`). A fitter does not make a native object, it acts on
one: it receives the model and a seeded generator, changes the model in place and leaves it in
evaluation mode. The built-in fitters are:

- `map_v1` (`MapFitter`): maximizes the marginal likelihood of an exact GP together with the
  engine's priors (a MAP fit) through BoTorch's fitting routine, with the options of that
  routine that the engine honours for every exact model (`MapFitterConfig`);
- `variational_elbo_v1` (`VariationalElboFitter`): maximizes the variational ELBO of an
  approximate GP, the model of a classifier, with an optimizer loop whose number of iterations,
  learning rate and optimizer are configuration fields (`VariationalElboFitterConfig`).

That there are two of them, one for each kind of model, is what makes this a kind of its own
(specification 3.10).

The rule of this package is that of the layer: it may import `planexs.core`, `planexs.domain`
and `planexs.encoding`, and no higher layer. Like `encoding`, it may import torch, BoTorch and
GPyTorch; only `adapters` may import Ax.

A module that registers a fitter must be imported here, one line per module, or the fitter is
silently unregistered (specification 5.12, 10.6). Importing the package also defines the
`fitter` kind, which `fitterRegistry` is imported for.
"""

from planexs.surrogates.fitters import MapFitter, VariationalElboFitter, fitterRegistry

__all__ = ["MapFitter", "VariationalElboFitter", "fitterRegistry"]
