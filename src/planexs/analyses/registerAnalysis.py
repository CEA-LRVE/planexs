"""The registration decorator of the `analysis` kind."""

from __future__ import annotations

from collections.abc import Callable
from operator import attrgetter
from typing import Any, TypeVar

from planexs.analyses.AbstractAnalysis import AbstractAnalysis
from planexs.analyses.analysisRegistry import analysisRegistry
from planexs.core.ComponentConfig import ComponentConfig
from planexs.core.registerComponentClass import registerComponentClass

AnalysisT = TypeVar("AnalysisT", bound=AbstractAnalysis[Any, Any])


def registerAnalysis(
    name: str,
    *,
    config: type[ComponentConfig],
    description: str,
    needs: frozenset[str] = frozenset(),
    supports: frozenset[str] = frozenset(),
) -> Callable[[type[AnalysisT]], type[AnalysisT]]:
    """Registers an `AbstractAnalysis` subclass as an analysis type, under a permanent name.

    Used as a class decorator, at import time (specification 5.12)::

        @registerAnalysis(
            "best_observed_v1",
            config=BestObservedConfig,
            description="The best completed trial by observed value.",
        )
        class BestObserved(AbstractAnalysis[BestObservedConfig, BestPointResult]):
            ...

    The registry entry holds the name, the configuration schema, the one-line description, the
    capability tags and the builder, which is the `fromConfig` of the class: it takes the
    validated configuration and, by keyword, the problem, the goal and the runtime settings of
    the study. The class is returned unchanged. The module that holds the decorated class must
    be imported by the `__init__` of its package, or the analysis is silently unregistered
    (specification 5.12, 10.6).

    The `type` field of the schema must be pinned to the registered name
    (`Literal["best_observed_v1"] = "best_observed_v1"`), so that a configuration that says
    which analysis it is can only be read by the schema of that analysis.

    The tags declared here describe the analysis *type*: what any configuration of it can
    state. An analysis narrows them to its own configuration through its `needs` and
    `supports` properties (see `AbstractAnalysis`). Declarations must be truthful
    (specification 10.6): an analysis that needs a Gaussian posterior says so, and a test
    checks each tag.

    Args:
        name: The registered name: lowercase snake_case with an optional `_vN` suffix. It is
            permanent: a change that alters results gets a new name (specification 3.7).
        config: The configuration schema of the analysis, a `ComponentConfig` whose `type`
            field defaults to `name`.
        description: One line saying what the analysis computes.
        needs: The capability tags that some configuration of the analysis needs some
            participant to support.
        supports: The capability tags that some configuration of the analysis supports.

    Returns:
        The decorator.

    Raises:
        RegistryError: when the decorator is applied, if the class is not an
            `AbstractAnalysis` subclass, the `type` field of the schema is not pinned to
            `name`, or the entry is refused by the registry (an invalid or already taken name,
            a tag that is not in the vocabulary). Nothing is registered then.
    """

    def decorate(cls: type[AnalysisT]) -> type[AnalysisT]:
        """Registers `cls` and returns it unchanged."""
        registerComponentClass(
            analysisRegistry(),
            cls,
            base=AbstractAnalysis,
            name=name,
            config=config,
            description=description,
            needs=needs,
            supports=supports,
            builder_of=attrgetter("fromConfig"),
        )
        return cls

    return decorate
