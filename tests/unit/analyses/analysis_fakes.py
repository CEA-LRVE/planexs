"""Fakes shared by the tests of `planexs.analyses`: problems, states, trials, surrogates and
analyses.

A fake analysis stands for a real one (roadmap tasks P7-02 to P7-07) that does not exist yet,
so that the contract of the kind (registration, tags, a state that is read-only, a surrogate on
demand, a typed and serializable result) is tested on its own. Kept in a module of its own and
not in `conftest.py`, so that the test files can import what is not a fixture.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, create_model

from planexs import Real, SearchSpace
from planexs.analyses.AbstractAnalysis import AbstractAnalysis
from planexs.analyses.analysisRegistry import analysisRegistry
from planexs.analyses.AnalysisResult import AnalysisResult
from planexs.analyses.AnalysisState import AnalysisState
from planexs.analyses.registerAnalysis import registerAnalysis
from planexs.analyses.SurrogateProvider import SurrogateProvider
from planexs.core.ComponentConfig import ComponentConfig
from planexs.core.KindTable import KindTable
from planexs.core.resolveComponent import resolveComponent
from planexs.core.RngTree import RngTree
from planexs.core.RuntimeConfig import RuntimeConfig
from planexs.domain.Observation import Observation
from planexs.domain.Outcome import Outcome
from planexs.domain.Problem import Problem
from planexs.domain.Trial import Trial
from planexs.domain.TrialState import TrialState
from planexs.encoding.SearchSpaceEncoder import SearchSpaceEncoder
from planexs.goals.AbstractGoal import AbstractGoal
from planexs.goals.ExploreGoal import ExploreGoal
from planexs.goals.ExploreGoalConfig import ExploreGoalConfig
from planexs.surrogates.BuiltSurrogate import BuiltSurrogate
from planexs.surrogates.FitStatus import FitStatus
from planexs.surrogates.Surrogate import Surrogate

#: The parameters of `makeProblem`, in order.
PARAMETERS = ("x0", "x1")

#: The path of the stream that the fake analysis draws its numbers from.
FAKE_STREAM = "analysis.fake"


def makeProblem() -> Problem:
    """A problem with two real parameters in `[0, 1]` and one continuous outcome."""
    return Problem(
        search_space=SearchSpace([Real(name, 0.0, 1.0) for name in PARAMETERS]),
        outcomes=[Outcome(name="yield", type="continuous")],
    )


def makeGoal() -> AbstractGoal[Any]:
    """The explore goal, which names every outcome of the problem."""
    return ExploreGoal.fromConfig(ExploreGoalConfig())


def makeTrial(trial_id: int, state: TrialState = TrialState.pending) -> Trial:
    """A trial in the given state, with what that state needs: an observation or a reason.

    A completed trial observes `yield = trial_id`, so the best of the completed ones is the
    last.
    """
    parameters = {name: 0.1 * (trial_id + 1) for name in PARAMETERS}
    if state is TrialState.completed:
        observation = Observation(values={"yield": float(trial_id)})
        return Trial(id=trial_id, parameters=parameters, state=state, observation=observation)
    if state is TrialState.failed:
        return Trial(id=trial_id, parameters=parameters, state=state, reason="it broke")
    return Trial(id=trial_id, parameters=parameters, state=state)


def makeTrials(**counts: int) -> list[Trial]:
    """Trials with the given number in each state, for example `makeTrials(completed=3)`.

    Args:
        **counts: The number of trials per state name (`pending`, `completed`, `failed`,
            `abandoned`). A state that is not given has none.

    Returns:
        The trials, with consecutive ids, in the order of the states as given.
    """
    trials: list[Trial] = []
    for name, number in counts.items():
        for _ in range(number):
            trials.append(makeTrial(len(trials), TrialState(name)))
    return trials


def makeState(
    trials: Sequence[Trial] | None = None,
    *,
    history_version: int = 0,
    ask_index: int = 0,
    seed: int = 7,
) -> AnalysisState:
    """An analysis state over `makeProblem` and the explore goal.

    Args:
        trials: The history. `None` means none.
        history_version: The version of the history.
        ask_index: The number of asks so far.
        seed: The seed of the random sources.
    """
    return AnalysisState(
        problem=makeProblem(),
        goal=makeGoal(),
        trials=[] if trials is None else trials,
        history_version=history_version,
        ask_index=ask_index,
        random_sources=RngTree(seed),
    )


class StubModel:
    """The least a `Surrogate` asks of a native model to be bound: it has one output."""

    num_outputs = 1


def makeSurrogate() -> Surrogate:
    """A `Surrogate` bound to `makeProblem` over a stub model, with a fit status of its own.

    The analyses of the contract tests only need to tell one surrogate from another and to read
    what it covers, so the model is a stub; the wrapper's own behaviour is tested elsewhere.
    """
    problem = makeProblem()
    runtime = RuntimeConfig()
    built = BuiltSurrogate(
        model=StubModel(),  # type: ignore[arg-type]
        status=FitStatus(n_observations=5, is_low_data=False, warnings=()),
    )
    return Surrogate(
        built,
        problem=problem,
        encoder=SearchSpaceEncoder(problem.search_space, runtime),
        runtime=runtime,
        outcome_names=("yield",),
    )


class CountingProvider:
    """A provider that counts how often it is asked, and gives the same surrogate each time.

    It honours `SurrogateProvider` by having its method, which is all the protocol asks.
    """

    def __init__(self, surrogate: Surrogate | None = None) -> None:
        self.calls = 0
        self._surrogate = makeSurrogate() if surrogate is None else surrogate

    def surrogate(self) -> Surrogate:
        """Counts the call and gives the surrogate."""
        self.calls += 1
        return self._surrogate


class FakeBand(BaseModel):
    """A nested part of a result: a frozen model, as a result's nested models must be."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    low: float
    high: float


class FakeResult(AnalysisResult):
    """The typed result of the fake analysis."""

    best_id: int | None = None
    counts: tuple[int, ...] = ()
    values: tuple[float, ...] = ()
    samples: tuple[float, ...] = ()
    band: FakeBand | None = None
    outcomes: tuple[str, ...] = ()
    fit_observations: int | None = None


def fakeConfig(name: str) -> type[ComponentConfig]:
    """A configuration schema as real analyses will write theirs: every field described, the
    `type` field pinned to `name`.

    Args:
        name: The registered name the schema is for.
    """
    schema: type[ComponentConfig] = create_model(
        f"{name.title().replace('_', '')}Config",
        __base__=ComponentConfig,
        type=(Literal[name], Field(default=name, description=f"The registered name, {name}.")),  # type: ignore[valid-type]
        use_model=(
            bool,
            Field(default=False, description="Whether the analysis asks for the surrogate."),
        ),
        draws=(
            int,
            Field(default=3, ge=1, description="How many random numbers the analysis draws."),
        ),
    )
    return schema


class FakeBehaviour(AbstractAnalysis[Any, FakeResult]):
    """What the fake analyses share: a summary of the history, some random numbers, and the
    surrogate only if the configuration says so.
    """

    def compute(self, state: AnalysisState, provider: SurrogateProvider) -> FakeResult:
        """The best completed trial, the counts per state, a draw from the analysis stream,
        and, with `use_model`, what the surrogate covers.
        """
        completed = state.trialsIn(TrialState.completed)
        values = tuple(
            float(trial.observation.values["yield"])
            for trial in completed
            if trial.observation is not None
        )
        best = (
            max(completed, key=lambda t: t.observation.values["yield"]) if completed else None
        )  # type: ignore[union-attr]
        generator = state.generatorFor(FAKE_STREAM)
        outcomes: tuple[str, ...] = ()
        fit_observations: int | None = None
        if self.config.use_model:
            surrogate = provider.surrogate()
            outcomes = surrogate.outcome_names
            fit_observations = surrogate.status.n_observations
        return FakeResult(
            analysis=self.config.type,
            history_version=state.history_version,
            best_id=None if best is None else best.id,
            counts=tuple(len(state.trialsIn(s)) for s in TrialState),
            values=values,
            samples=tuple(float(generator.random()) for _ in range(self.config.draws)),
            band=FakeBand(low=min(values), high=max(values)) if values else None,
            outcomes=outcomes,
            fit_observations=fit_observations,
        )


def defineFake(
    name: str = "fake_analysis_v1",
    *,
    needs: frozenset[str] = frozenset(),
    supports: frozenset[str] = frozenset(),
    description: str = "A fake analysis for tests.",
) -> type[AbstractAnalysis[Any, FakeResult]]:
    """Registers a fake analysis in the (swapped) registry and returns its class.

    Args:
        name: The registered name.
        needs: The tags the entry declares as needed.
        supports: The tags the entry declares as supported.
        description: The one-line description of the entry.

    Returns:
        The registered class.
    """

    @registerAnalysis(
        name,
        config=fakeConfig(name),
        description=description,
        needs=needs,
        supports=supports,
    )
    class FakeAnalysis(FakeBehaviour):
        """A fake analysis."""

    return FakeAnalysis


def defineRefining(name: str = "refining_v1") -> type[AbstractAnalysis[Any, FakeResult]]:
    """Registers a fake analysis whose tags depend on its configuration, and returns its class.

    The entry declares the union of what any configuration can state: it needs `toy:gaussian`
    and supports `toy:grid`. An instance needs `toy:gaussian` only if it asks for the surrogate
    (`use_model`), and supports `toy:grid` only if it does not. The test registers the tags in
    its own vocabulary first, with `registerToyTags`.

    Args:
        name: The registered name.

    Returns:
        The registered class.
    """

    @registerAnalysis(
        name,
        config=fakeConfig(name),
        description="A fake analysis whose tags depend on its configuration.",
        needs=frozenset({"toy:gaussian"}),
        supports=frozenset({"toy:grid"}),
    )
    class RefiningAnalysis(FakeBehaviour):
        """An analysis that narrows the tags of its entry to its own configuration."""

        @property
        def needs(self) -> frozenset[str]:
            """`toy:gaussian` if the analysis asks for the surrogate, nothing otherwise."""
            return frozenset({"toy:gaussian"}) if self.config.use_model else frozenset()

        @property
        def supports(self) -> frozenset[str]:
            """`toy:grid` if the analysis does not ask for the surrogate."""
            return frozenset() if self.config.use_model else frozenset({"toy:grid"})

    return RefiningAnalysis


def registerToyTags(table: KindTable) -> None:
    """Registers the capability tags of the fakes in the vocabulary of a swapped table.

    Args:
        table: The table that the `isolatedAnalysisKind` fixture returned, whose vocabulary is
            a copy: a tag is permanent, and must not reach the real one.
    """
    table.vocabulary.register("toy:gaussian", "The surrogate has a Gaussian posterior.")
    table.vocabulary.register("toy:grid", "The analysis can be evaluated on a grid.")
    table.vocabulary.register("toy:optimize", "The goal of the study is to optimize.")


def buildFake(name: str = "fake_analysis_v1", **fields: Any) -> AbstractAnalysis[Any, Any]:
    """Builds a registered fake analysis the way a study does: resolve, then call the builder.

    Args:
        name: The registered name.
        **fields: The fields of the configuration besides `type`.

    Returns:
        The analysis.
    """
    resolved = resolveComponent(analysisRegistry(), {"type": name, **fields})
    analysis: AbstractAnalysis[Any, Any] = resolved.entry.builder(
        resolved.config, problem=makeProblem(), goal=makeGoal(), runtime=RuntimeConfig()
    )
    return analysis
