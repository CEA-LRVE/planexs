"""The contract of the `analysis` kind of components."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, Self, TypeVar

from planexs.analyses.analysisRegistry import analysisRegistry
from planexs.analyses.AnalysisResult import AnalysisResult
from planexs.analyses.AnalysisState import AnalysisState
from planexs.analyses.SurrogateProvider import SurrogateProvider
from planexs.core.ComponentConfig import ComponentConfig
from planexs.core.RuntimeConfig import RuntimeConfig
from planexs.domain.Problem import Problem
from planexs.goals.AbstractGoal import AbstractGoal

ConfigT = TypeVar("ConfigT", bound=ComponentConfig)
ResultT = TypeVar("ResultT", bound=AnalysisResult)


class AbstractAnalysis(ABC, Generic[ConfigT, ResultT]):
    """A registered computation over the read-only state of a study, which returns a typed,
    serializable result (specification 5.10, 6).

    The catalogue of analyses is open: recommendations (the best point, a Pareto front, a
    level-set estimate: the answer to the goal), parameter importance, prediction grids and
    slices, and model diagnostics such as leave-one-out checks and calibration. A concrete
    analysis is a subclass of this class, registered under a permanent name with
    `registerAnalysis` together with the configuration schema that holds its settings, and it
    declares what it needs and supports.

    It has one operation, `compute`, which receives the state of the study (`AnalysisState`)
    and a provider of surrogates (`SurrogateProvider`), and returns an `AnalysisResult`. The
    result class is the second type parameter, so a caller that knows the analysis knows the
    type of what it gets: `AbstractAnalysis[BestPointConfig, BestPointResult]`.

    **An analysis never changes the study.** It reads the state, which is read-only, and it
    returns a value; the history is as it was before. It is a function of its configuration,
    the state and, if it asks for one, the surrogate: the same inputs give the same result
    (specification 3.7). Every random number comes from the sources of the state
    (`AnalysisState.seedFor` and `generatorFor`) and never from a global state.

    **A surrogate on demand.** An analysis that needs the model calls `provider.surrogate()`,
    and one that does not never pays for a fit. What the analysis needs of the model it
    declares with capability tags (see below), so that the study can refuse to run it on a
    study whose surrogate cannot satisfy it before anything is computed (specification 5.12).

    **Construction.** The registry entry's builder is `fromConfig`. An analysis is built for a
    problem and a goal, and with the runtime settings of the study, like a policy: what it
    builds from its configuration can depend on them, for example the encoder of the search
    space, which holds the precision and the device, for an analysis that evaluates the model
    on a grid. The default keeps only the configuration. A builder is free of side effects at
    import and registration time, and deterministic given its configuration and its inputs
    (specification 6).

    **Capability tags.** An analysis is a participant of the compatibility check
    (`Participant`): it has a `label`, and the tags it `needs` some participant to support and
    `supports` itself (specification 5.12). The default is the tags of the registry entry,
    which describe the analysis *type*; an analysis whose tags depend on its configuration
    overrides the two properties, as a goal does. An analysis of a Gaussian posterior declares
    that it needs the tag that says so, and an analysis that answers a level-set goal declares
    the goal tag it needs.

    Args:
        config: The validated configuration of the analysis, an instance of the schema it is
            registered with.

    Attributes:
        config: The configuration the analysis was made with. It is a read-only attribute.
    """

    def __init__(self, config: ConfigT) -> None:
        """Keeps the validated configuration."""
        self._config = config

    @property
    def config(self) -> ConfigT:
        """The configuration this analysis was made with (stored: cheap, snake_case)."""
        return self._config

    @classmethod
    def fromConfig(
        cls,
        config: ConfigT,
        *,
        problem: Problem,
        goal: AbstractGoal[Any],
        runtime: RuntimeConfig,
    ) -> Self:
        """The builder of a registered analysis: makes the analysis object that a validated
        configuration describes, for a problem and a goal.

        This is what the registry entry stores as its builder, so
        `resolveComponent(registry, {...})` followed by
        `entry.builder(config, problem=..., goal=..., runtime=...)` gives the analysis. The
        study calls it when it runs the analysis, after the goal has been validated against
        the problem.

        The default keeps the configuration and ignores the rest. An analysis that needs the
        problem, the goal or the runtime settings to be built overrides this method.

        Args:
            config: The configuration, an instance of the schema that the analysis is
                registered with.
            problem: What can be varied and what is measured.
            goal: What the study is for.
            runtime: The precision, the device and the determinism of the study.

        Returns:
            The analysis, an instance of the class this is called on.
        """
        del problem, goal, runtime  # for the analyses that override this: see the docstring
        return cls(config)

    @property
    def label(self) -> str:
        """Who this analysis is, as an error message of the compatibility check names it, for
        example `analysis 'best_observed_v1'` (a string built from the stored name: cheap,
        snake_case).
        """
        return f"analysis {self._config.type!r}"

    @property
    def needs(self) -> frozenset[str]:
        """The capability tags this analysis needs some participant to support, for its own
        configuration (a lookup of the entry's tags: cheap, snake_case).

        The default is what the registry entry declares for the type. An analysis whose needs
        depend on its configuration overrides this, and reports only tags that the entry
        declares.
        """
        return analysisRegistry().lookup(self._config.type).needs

    @property
    def supports(self) -> frozenset[str]:
        """The capability tags this analysis supports, for its own configuration (a lookup of
        the entry's tags: cheap, snake_case).

        The default is what the registry entry declares for the type. An analysis whose tags
        depend on its configuration overrides this, and reports only tags that the entry
        declares. The declaration must be truthful (specification 5.12).
        """
        return analysisRegistry().lookup(self._config.type).supports

    @abstractmethod
    def compute(self, state: AnalysisState, provider: SurrogateProvider) -> ResultT:
        """Computes the result of the analysis on the state of a study.

        The analysis reads the state and returns a value, and changes nothing. It calls
        `provider.surrogate()` only if it needs the model. The result names the analysis, with
        `analysis=self.config.type`, and carries the history version of the state it was
        computed on, with `history_version=state.history_version`.

        Args:
            state: The read-only state of the study: the problem, the goal, every trial, the
                version of the history, the ask index and the random sources.
            provider: Gives the surrogate fitted to the history, on demand.

        Returns:
            The result, an immutable value that `toDict` writes as JSON-compatible data.

        Raises:
            ConfigError: if the state is one that the analysis cannot be run on, for example
                a goal of a type it does not answer. The error names the analysis.
        """
