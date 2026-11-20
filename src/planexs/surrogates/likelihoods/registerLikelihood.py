"""The registration decorator of the `likelihood` kind."""

from __future__ import annotations

from collections.abc import Callable
from operator import attrgetter
from typing import Any, TypeVar

from planexs.core.ComponentConfig import ComponentConfig
from planexs.core.registerComponentClass import registerComponentClass
from planexs.surrogates.likelihoods.AbstractLikelihoodBuilder import AbstractLikelihoodBuilder
from planexs.surrogates.likelihoods.likelihoodRegistry import likelihoodRegistry

LikelihoodBuilderT = TypeVar("LikelihoodBuilderT", bound=AbstractLikelihoodBuilder[Any])


def registerLikelihood(
    name: str,
    *,
    config: type[ComponentConfig],
    description: str,
    needs: frozenset[str] = frozenset(),
    supports: frozenset[str] = frozenset(),
) -> Callable[[type[LikelihoodBuilderT]], type[LikelihoodBuilderT]]:
    """Registers an `AbstractLikelihoodBuilder` subclass as a likelihood, under a permanent
    name.

    Used as a class decorator, at import time (specification 5.12)::

        @registerLikelihood(
            "student_t_v1",
            config=StudentTLikelihoodConfig,
            description="A Student-t likelihood for outcomes with outliers.",
            needs=frozenset({"posterior:latent_gaussian"}),
            supports=frozenset({"outcome_kind:continuous", "noise:inferred"}),
        )
        class StudentTLikelihoodBuilder(AbstractLikelihoodBuilder[StudentTLikelihoodConfig]):
            ...

    The registry entry holds the name, the configuration schema (the settings of the
    likelihood), the one-line description, the capability tags and the builder, which is the
    `fromConfig` of the class. The class is returned unchanged. The module that holds the
    decorated class must be imported by the `__init__` of its package, or the likelihood is
    silently unregistered (specification 5.12, 10.6).

    The `type` field of the schema must be pinned to the registered name
    (`Literal["student_t_v1"] = "student_t_v1"`), so that a configuration that says which
    likelihood it is can only be read by the schema of that likelihood.

    The `supports` tags say which outcome kinds the likelihood can model
    (`outcome_kind:continuous`) and `AbstractLikelihoodBuilder.build` enforces them. Needs and
    supports must be truthful (specification 10.6): a likelihood that supports a tag works
    with it, and one that does not declare it is refused with a clear message.

    Args:
        name: The registered name: lowercase snake_case with an optional `_vN` suffix. It is
            permanent: a change that alters results gets a new name (specification 3.7).
        config: The configuration schema of the likelihood, a `ComponentConfig` whose `type`
            field defaults to `name`.
        description: One line saying what the likelihood is.
        needs: The capability tags that the likelihood needs some participant to support.
        supports: The capability tags that the likelihood supports, the outcome kinds it can
            model among them.

    Returns:
        The decorator.

    Raises:
        RegistryError: when the decorator is applied, if the class is not an
            `AbstractLikelihoodBuilder` subclass, the `type` field of the schema is not pinned
            to `name`, or the entry is refused by the registry (an invalid or already taken
            name, a tag that is not in the vocabulary). Nothing is registered then.
    """

    def decorate(cls: type[LikelihoodBuilderT]) -> type[LikelihoodBuilderT]:
        """Registers `cls` and returns it unchanged."""
        registerComponentClass(
            likelihoodRegistry(),
            cls,
            base=AbstractLikelihoodBuilder,
            name=name,
            config=config,
            description=description,
            needs=needs,
            supports=supports,
            builder_of=attrgetter("fromConfig"),
        )
        return cls

    return decorate
