"""The `gaussian_v1` likelihood: a Gaussian observation noise, inferred or known."""

from __future__ import annotations

import torch
from botorch.models.utils.gpytorch_modules import get_gaussian_likelihood_with_lognormal_prior
from gpytorch.constraints import GreaterThan
from gpytorch.likelihoods import FixedNoiseGaussianLikelihood, GaussianLikelihood, Likelihood

from planexs.encoding.torch_runtime import torchDevice, torchDtype
from planexs.surrogates.likelihoods.AbstractLikelihoodBuilder import AbstractLikelihoodBuilder
from planexs.surrogates.likelihoods.GaussianLikelihoodConfig import GaussianLikelihoodConfig
from planexs.surrogates.likelihoods.LikelihoodBuildContext import LikelihoodBuildContext
from planexs.surrogates.likelihoods.registerLikelihood import registerLikelihood


@registerLikelihood(
    "gaussian_v1",
    config=GaussianLikelihoodConfig,
    description="A Gaussian observation noise, inferred by the fit or fixed to the reported.",
    supports=frozenset({"outcome_kind:continuous", "noise:known", "noise:inferred"}),
)
class GaussianLikelihoodBuilder(AbstractLikelihoodBuilder[GaussianLikelihoodConfig]):
    """A Gaussian likelihood: an observation is the latent function plus a normal noise.

    The noise reaches the model in one of two ways, which the context decides:

    - **Known.** The caller reported a noise variance for every observation
      (`LikelihoodBuildContext.noise_variance`). The noise is fixed to it, as the engine's
      fixed-noise likelihood does, after each variance has been raised to
      `min_observation_variance` so that a deterministic observation cannot make the
      covariance matrix singular. GPyTorch then raises it to its own minimum if that is
      higher (`gpytorch.settings.min_fixed_noise`, 1e-6 in double precision), and the engine
      warns when it does.
    - **Inferred.** No noise was reported. The noise is a parameter that the fit learns, one
      per model of the batch. With no `noise_lower_bound` it is the likelihood that BoTorch
      itself builds by default, with its prior and its floor. With one, it is that same
      likelihood with the floor replaced: BoTorch's prior is kept, so that the fit stays a MAP
      fit (specification 3.5: reuse the engine, do not re-implement it).

    It models continuous outcomes, and so every outcome of the context must be of a kind that
    supports `outcome_kind:continuous` (`build` checks it). It needs nothing from the
    surrogate that uses it: an exact GP and an approximate one both take a Gaussian
    likelihood.

    A surrogate that gives this likelihood to a BoTorch model must pass it as the model's
    `likelihood` and leave `train_Yvar` out. The model ignores `train_Yvar` once it has a
    custom likelihood (the noise it carries is only transformed and kept), so the known
    variances must be in the likelihood, on the scale on which the model trains.
    """

    def createLikelihood(self, context: LikelihoodBuildContext) -> Likelihood:
        """Makes the Gaussian likelihood with a known or an inferred noise.

        Args:
            context: The outcomes, the runtime settings, the batch shape and the known noise.

        Returns:
            A `FixedNoiseGaussianLikelihood` when `context.noise_variance` is given, and a
            `GaussianLikelihood` with a learnable noise otherwise, on the precision and device
            of `context.runtime`.
        """
        likelihood: Likelihood
        if context.noise_variance is None:
            likelihood = _inferredNoise(self.config.noise_lower_bound, context.batch_shape)
        else:
            likelihood = FixedNoiseGaussianLikelihood(
                noise=context.noise_variance.clamp_min(self.config.min_observation_variance),
                batch_shape=context.batch_shape,
            )
        return likelihood.to(
            dtype=torchDtype(context.runtime), device=torchDevice(context.runtime)
        )


def _inferredNoise(lower_bound: float | None, batch_shape: torch.Size) -> GaussianLikelihood:
    """The likelihood with an inferred noise: BoTorch's default, with its floor replaced if
    one is given.

    Args:
        lower_bound: The smallest noise variance that fitting may infer, or `None` to keep
            BoTorch's own floor.
        batch_shape: The batch shape of the likelihood.

    Returns:
        The likelihood, with BoTorch's prior on the noise. Its noise starts at the mode of
        that prior, or at the floor if that is higher.
    """
    engine_default = get_gaussian_likelihood_with_lognormal_prior(batch_shape=batch_shape)
    if lower_bound is None:
        return engine_default
    prior = engine_default.noise_covar.noise_prior
    return GaussianLikelihood(
        noise_prior=prior,
        batch_shape=batch_shape,
        noise_constraint=GreaterThan(
            lower_bound, transform=None, initial_value=max(float(prior.mode), lower_bound)
        ),
    )
