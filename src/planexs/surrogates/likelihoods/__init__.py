"""Likelihoods: how an observation relates to the latent function of a surrogate (specification
5.6, 6).

The likelihood is one of the choices that a surrogate builder exposes as a nested kind of
components, so that it is selected by name in configuration (specification 3.3): the contract
that every likelihood honours (`AbstractLikelihoodBuilder`), what it is given
(`LikelihoodBuildContext`), the registry (`likelihoodRegistry`) and the registration decorator
(`registerLikelihood`). The two built-in likelihoods are the Gaussian one of `gaussian_v1`
(`GaussianLikelihoodBuilder`), for a measured quantity with a noise that is inferred or known,
and the Bernoulli one of `bernoulli_v1` (`BernoulliLikelihoodBuilder`), for a label. Each
declares the outcome kinds it supports, and `AbstractLikelihoodBuilder.build` enforces them.

The rule of this package is that of the layer: it may import `planexs.core`, `planexs.domain`
and `planexs.encoding`, and no higher layer. Like `encoding`, it may import torch, BoTorch and
GPyTorch; only `adapters` may import Ax.

A module that registers a likelihood must be imported here, one line per module, or the
likelihood is silently unregistered (specification 5.12, 10.6).
"""

from planexs.surrogates.likelihoods import (
    BernoulliLikelihoodBuilder,
    GaussianLikelihoodBuilder,
)

__all__ = ["BernoulliLikelihoodBuilder", "GaussianLikelihoodBuilder"]
