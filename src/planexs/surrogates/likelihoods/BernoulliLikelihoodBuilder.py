"""The `bernoulli_v1` likelihood: a label that is true with a probability."""

from __future__ import annotations

from gpytorch.likelihoods import BernoulliLikelihood, Likelihood

from planexs.core.ConfigError import ConfigError
from planexs.encoding.torch_runtime import torchDevice, torchDtype
from planexs.surrogates.likelihoods.AbstractLikelihoodBuilder import AbstractLikelihoodBuilder
from planexs.surrogates.likelihoods.BernoulliLikelihoodConfig import BernoulliLikelihoodConfig
from planexs.surrogates.likelihoods.LikelihoodBuildContext import LikelihoodBuildContext
from planexs.surrogates.likelihoods.registerLikelihood import registerLikelihood


@registerLikelihood(
    "bernoulli_v1",
    config=BernoulliLikelihoodConfig,
    description="A Bernoulli likelihood for a label, read from a latent function by a probit.",
    needs=frozenset({"posterior:latent_gaussian"}),
    supports=frozenset({"outcome_kind:binary"}),
)
class BernoulliLikelihoodBuilder(AbstractLikelihoodBuilder[BernoulliLikelihoodConfig]):
    """A Bernoulli likelihood: a label is true with the probability that the standard normal
    distribution gives to the latent function (a probit link).

    It models binary outcomes, and so every outcome of the context must be of a kind that
    supports `outcome_kind:binary` (`build` checks it). A label has no observation noise, so
    a context that carries known noise is refused: the likelihood does not declare
    `noise:known`.

    It **needs** a surrogate whose posterior is Gaussian over a latent function and read
    through a link (`posterior:latent_gaussian`): a Bernoulli likelihood is not Gaussian, so an
    exact GP cannot take it, and the surrogate must use approximate inference (specification
    5.6). The engine refuses the combination when the model is built, and the declared need
    makes the compatibility check refuse it when the study is built.

    The likelihood has no parameter, so its batch shape is not used.
    """

    def createLikelihood(self, context: LikelihoodBuildContext) -> Likelihood:
        """Makes the Bernoulli likelihood.

        Args:
            context: The outcomes, the runtime settings and the batch shape. It must carry no
                known noise.

        Returns:
            A `BernoulliLikelihood`, on the precision and device of `context.runtime`.

        Raises:
            ConfigError: if the context carries known noise.
        """
        if context.noise_variance is not None:
            raise ConfigError(
                "The likelihood 'bernoulli_v1' was given observation noise, but a label has "
                "none: the noise of a Bernoulli likelihood is the randomness of the label "
                "itself.",
                component="likelihood 'bernoulli_v1'",
                field_path="noise_variance",
                suggestion="leave the noise out (None) for a binary outcome",
            )
        return BernoulliLikelihood().to(
            dtype=torchDtype(context.runtime), device=torchDevice(context.runtime)
        )
