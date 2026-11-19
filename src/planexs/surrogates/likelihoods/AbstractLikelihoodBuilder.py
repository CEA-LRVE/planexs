"""The contract of the `likelihood` kind of components."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, Self, TypeVar

from planexs.core.ComponentConfig import ComponentConfig
from planexs.core.ConfigError import ConfigError
from planexs.surrogates.likelihoods.LikelihoodBuildContext import LikelihoodBuildContext
from planexs.surrogates.likelihoods.likelihoodRegistry import likelihoodRegistry

if TYPE_CHECKING:
    from gpytorch.likelihoods import Likelihood

ConfigT = TypeVar("ConfigT", bound=ComponentConfig)

#: The group of the capability tags that stand for an outcome kind (`outcome_kind:binary`),
#: which `registerOutcomeKind` registers. Like a registered name it is shared with every
#: package that declares such a tag, so it is a constant and not a setting.
_OUTCOME_KIND_GROUP = "outcome_kind"


class AbstractLikelihoodBuilder(ABC, Generic[ConfigT]):
    """Makes the native likelihood of a surrogate: how an observation relates to the latent
    function (specification 5.6, 6).

    The likelihood of a model is a choice that a reasonable user could want to take
    differently (a Gaussian one for a measured quantity, a Bernoulli one for a label), so each
    is a component of this kind (specification 3.3): a concrete subclass, registered under a
    permanent name with `registerLikelihood`, together with the configuration schema that
    holds its settings. A surrogate builder holds one in its own configuration and asks it for
    the native object. A likelihood provides:

    - `createLikelihood`, which receives a `LikelihoodBuildContext` (the outcomes, the
      runtime settings, the batch shape and the known noise, if any) and returns the native
      GPyTorch likelihood, on the precision and device of the runtime settings;
    - declared **supports** that say which outcome kinds it can model
      (`outcome_kind:continuous`, which the outcome kinds register in the vocabulary) and
      which noise it handles (`noise:known`, `noise:inferred`), and declared **needs**, the
      capabilities that the surrogate using it must provide (a likelihood that is not Gaussian
      needs approximate inference, so `posterior:latent_gaussian`). They are the tags it
      passes to `registerLikelihood`, truthful with respect to what it does (specification
      10.6).

    **The declared outcome kinds are enforced.** `build` is the one entry point, and it is not
    overridden: before it calls `createLikelihood` it checks every outcome of the context
    against the `outcome_kind:*` tags of the registry entry, and refuses an outcome whose kind
    the likelihood does not support, with an error that names the outcome, the likelihood and
    the registered likelihoods that would do. A likelihood that a plugin registers is held to
    the same rule without writing it. The compatibility check meets the same tags when a study
    is built (specification 5.12), so the error here is the last line of defence, for a
    likelihood used without a study.

    The registry entry's builder is `fromConfig`, as for every kind that makes an object from a
    validated configuration: it makes this builder object, which keeps its configuration. A
    builder is configured once and then builds as often as it is asked, each time from a
    context alone, so it is deterministic given its configuration and the context
    (specification 6): no random number is drawn, nothing happens at import or registration
    time, and the tensors of the context are never modified.

    Args:
        config: The validated configuration of the component, an instance of the schema it is
            registered with.

    Attributes:
        config: The configuration the builder was made with. It is a read-only attribute.
    """

    def __init__(self, config: ConfigT) -> None:
        """Keeps the validated configuration for `build`."""
        self._config = config

    @property
    def config(self) -> ConfigT:
        """The configuration this builder was made with (a stored value: cheap,
        snake_case).
        """
        return self._config

    @classmethod
    def fromConfig(cls, config: ConfigT) -> Self:
        """The builder of a registered likelihood: makes the builder object that a validated
        configuration describes.

        This is what the registry entry stores as its builder, so
        `resolveComponent(registry, {...})` followed by `entry.builder(config)` gives the
        object whose `build` makes the native likelihood.

        Args:
            config: The configuration, an instance of the schema that the likelihood is
                registered with.

        Returns:
            The builder, an instance of the class this is called on.
        """
        return cls(config)

    def build(self, context: LikelihoodBuildContext) -> Likelihood:
        """Checks that the likelihood can model the outcomes, and makes the native likelihood.

        Args:
            context: The outcomes, the runtime settings, the batch shape and the known noise.
                Read-only.

        Returns:
            The native GPyTorch likelihood.

        Raises:
            ConfigError: if an outcome is of a kind that the likelihood does not support, or
                the likelihood refuses the context (see `createLikelihood`).
        """
        self._requireSupportedOutcomes(context)
        return self.createLikelihood(context)

    @abstractmethod
    def createLikelihood(self, context: LikelihoodBuildContext) -> Likelihood:
        """Makes the native likelihood. This is where a likelihood differs from another.

        Do not call it directly: call `build`, which has checked that every outcome is of a
        kind that the likelihood supports. It must return a likelihood on the precision and
        the device of `context.runtime`, with the batch shape of `context.batch_shape` where
        the likelihood has parameters, and it must use the known noise when it is given and the
        likelihood handles it, or refuse it when it does not.

        Args:
            context: The outcomes, the runtime settings, the batch shape and the known noise.
                Read-only.

        Returns:
            The native GPyTorch likelihood.

        Raises:
            ConfigError: if the context asks for something that this likelihood cannot do,
                such as known noise for a likelihood that has none.
        """

    def _requireSupportedOutcomes(self, context: LikelihoodBuildContext) -> None:
        """Refuses an outcome whose kind the registry entry does not declare as supported.

        Args:
            context: The context whose outcomes are checked.

        Raises:
            ConfigError: if an outcome is of a kind that is not among the `outcome_kind:*`
                tags of the entry. The message lists the kinds the likelihood does support and
                the registered likelihoods that support the kind of the outcome.
        """
        registry = likelihoodRegistry()
        name = self._config.type
        supported = registry.lookup(name).supports
        for position, outcome in enumerate(context.outcomes):
            tag = outcome.kind().capability_tag
            if tag in supported:
                continue
            kinds = sorted(
                tag_name.partition(":")[2]
                for tag_name in supported
                if tag_name.partition(":")[0] == _OUTCOME_KIND_GROUP
            )
            alternatives = [
                other for other in registry.names() if tag in registry.lookup(other).supports
            ]
            raise ConfigError(
                f"The likelihood {name!r} cannot model the outcome {outcome.name!r}: it is of "
                f"the kind {outcome.type!r}, and the likelihood supports the kinds "
                f"{', '.join(kinds) or 'none'}.",
                component=f"likelihood {name!r}",
                field_path=f"outcomes[{position}]",
                suggestion=(
                    f"use one of the likelihoods that support {outcome.type!r}: "
                    f"{', '.join(alternatives)}"
                    if alternatives
                    else f"register a likelihood that supports the kind {outcome.type!r}"
                ),
            )
