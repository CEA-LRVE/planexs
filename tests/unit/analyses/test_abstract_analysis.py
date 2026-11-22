"""Unit tests for `AbstractAnalysis`, the contract of the `analysis` kind (roadmap task P7-01,
specification 5.10, 6).

What is tested: the contract itself (abstract, configured once, built from a configuration, a
problem, a goal and the runtime settings), the capability tags an analysis reports for its own
configuration, what `compute` is given and returns, the surrogate that is obtained only on
demand, and the result that can be written as JSON and read back. A fake analysis stands for a
real one: the contract is tested before any analysis exists.
"""

from __future__ import annotations

import json
import random
from typing import Any, Literal

import numpy as np
import pytest
import torch
from analysis_fakes import (
    CountingProvider,
    FakeBehaviour,
    FakeResult,
    buildFake,
    defineFake,
    defineRefining,
    fakeConfig,
    makeGoal,
    makeProblem,
    makeState,
    makeSurrogate,
    makeTrials,
    registerToyTags,
)
from pydantic import Field, ValidationError

import planexs.analyses  # noqa: F401
from planexs.analyses.AbstractAnalysis import AbstractAnalysis
from planexs.analyses.analysisRegistry import analysisRegistry
from planexs.analyses.AnalysisResult import AnalysisResult
from planexs.analyses.AnalysisState import AnalysisState
from planexs.analyses.registerAnalysis import registerAnalysis
from planexs.analyses.SurrogateProvider import SurrogateProvider
from planexs.core.ComponentConfig import ComponentConfig
from planexs.core.KindTable import KindTable
from planexs.core.RegistryError import RegistryError
from planexs.core.resolveComponent import resolveComponent
from planexs.core.RuntimeConfig import RuntimeConfig
from planexs.domain.TrialState import TrialState


class MinimalConfig(ComponentConfig):
    """A schema for an analysis that is built directly, without being registered."""

    type: Literal["minimal_v1"] = Field(default="minimal_v1", description="The name.")


class MinimalAnalysis(AbstractAnalysis[MinimalConfig, AnalysisResult]):
    """The least an analysis has to define: how it computes."""

    def compute(self, state: AnalysisState, provider: SurrogateProvider) -> AnalysisResult:
        """A result that only says who made it and from which history."""
        return AnalysisResult(analysis=self.config.type, history_version=state.history_version)


def buildMinimal() -> MinimalAnalysis:
    """An analysis of `MinimalAnalysis`, made the way a study makes one."""
    return MinimalAnalysis.fromConfig(
        MinimalConfig(), problem=makeProblem(), goal=makeGoal(), runtime=RuntimeConfig()
    )


def reseed(seed: int) -> None:
    """Seeds every global random state that a component could wrongly draw from."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class TestTheContract:
    def test_the_contract_cannot_be_instantiated(self) -> None:
        with pytest.raises(TypeError, match="abstract"):
            AbstractAnalysis(MinimalConfig())  # type: ignore[abstract]

    def test_an_analysis_has_one_abstract_operation(self) -> None:
        assert AbstractAnalysis.__abstractmethods__ == {"compute"}

    def test_an_analysis_that_leaves_it_out_cannot_be_instantiated(self) -> None:
        Incomplete = type("Incomplete", (AbstractAnalysis,), {})

        with pytest.raises(TypeError, match="compute"):
            Incomplete(MinimalConfig())

    def test_an_analysis_that_defines_it_is_complete(self) -> None:
        assert isinstance(MinimalAnalysis(MinimalConfig()), AbstractAnalysis)

    def test_the_configuration_is_kept_as_it_was_given(self) -> None:
        config = MinimalConfig()

        analysis = MinimalAnalysis(config)

        assert analysis.config is config

    def test_the_configuration_cannot_be_assigned(self) -> None:
        analysis = MinimalAnalysis(MinimalConfig())

        with pytest.raises(AttributeError):
            analysis.config = MinimalConfig()  # type: ignore[misc]

    def test_an_analysis_is_labelled_with_its_kind_and_name(self) -> None:
        assert buildMinimal().label == "analysis 'minimal_v1'"


class TestBuildingFromAConfiguration:
    """`fromConfig` is the builder of the registry entry: what a study calls to run one."""

    def test_by_default_the_builder_keeps_the_configuration_and_ignores_the_rest(self) -> None:
        config = MinimalConfig()

        analysis = MinimalAnalysis.fromConfig(
            config, problem=makeProblem(), goal=makeGoal(), runtime=RuntimeConfig()
        )

        assert type(analysis) is MinimalAnalysis
        assert analysis.config is config

    @pytest.mark.parametrize("name", ["problem", "goal", "runtime"])
    def test_the_problem_the_goal_and_the_runtime_are_given_by_keyword(
        self, name: str
    ) -> None:
        inputs = {"problem": makeProblem(), "goal": makeGoal(), "runtime": RuntimeConfig()}
        positional = [MinimalConfig(), *inputs.values()]

        with pytest.raises(TypeError, match="positional"):
            MinimalAnalysis.fromConfig(*positional)  # type: ignore[misc]
        del inputs[name]
        with pytest.raises(TypeError, match=name):
            MinimalAnalysis.fromConfig(MinimalConfig(), **inputs)  # type: ignore[call-arg]

    def test_an_analysis_that_needs_the_inputs_overrides_the_builder_and_receives_them(
        self, isolatedAnalysisKind: KindTable
    ) -> None:
        # An analysis that evaluates the model on a grid builds the encoder of the space with
        # the runtime settings. The registry entry holds the override, and the study calls it
        # with the very objects it has.
        received: dict[str, Any] = {}
        problem, goal, runtime = makeProblem(), makeGoal(), RuntimeConfig()

        @registerAnalysis(
            "adapting_v1", config=fakeConfig("adapting_v1"), description="Needs the inputs."
        )
        class Adapting(FakeBehaviour):
            @classmethod
            def fromConfig(cls, config: Any, *, problem: Any, goal: Any, runtime: Any) -> Any:
                received.update(problem=problem, goal=goal, runtime=runtime)
                return super().fromConfig(config, problem=problem, goal=goal, runtime=runtime)

        resolved = resolveComponent(analysisRegistry(), {"type": "adapting_v1"})

        analysis = resolved.entry.builder(
            resolved.config, problem=problem, goal=goal, runtime=runtime
        )

        assert received["problem"] is problem
        assert received["goal"] is goal
        assert received["runtime"] is runtime
        assert isinstance(analysis, Adapting)
        assert analysis.config is resolved.config

    def test_building_computes_nothing(self, isolatedAnalysisKind: KindTable) -> None:
        defineFake()

        analysis = buildFake()

        assert isinstance(analysis, AbstractAnalysis)


class TestCapabilityTags:
    """An analysis is a participant of the compatibility check: its tags are read from it."""

    def test_by_default_the_tags_are_the_ones_of_the_registry_entry(
        self, isolatedAnalysisKind: KindTable
    ) -> None:
        registerToyTags(isolatedAnalysisKind)
        defineFake(
            "plain_v1",
            needs=frozenset({"toy:gaussian"}),
            supports=frozenset({"toy:grid"}),
        )

        analysis = buildFake("plain_v1")

        assert analysis.needs == {"toy:gaussian"}
        assert analysis.supports == {"toy:grid"}
        assert isinstance(analysis.needs, frozenset)
        assert isinstance(analysis.supports, frozenset)

    def test_an_analysis_that_declares_nothing_reports_nothing(
        self, isolatedAnalysisKind: KindTable
    ) -> None:
        defineFake("bare_v1")

        analysis = buildFake("bare_v1")

        assert analysis.needs == frozenset()
        assert analysis.supports == frozenset()

    def test_an_analysis_can_narrow_the_tags_of_its_entry_to_its_configuration(
        self, isolatedAnalysisKind: KindTable
    ) -> None:
        # An analysis that does not ask for the surrogate needs no property of its posterior.
        registerToyTags(isolatedAnalysisKind)
        defineRefining()
        entry = analysisRegistry().lookup("refining_v1")

        with_model = buildFake("refining_v1", use_model=True)
        without_model = buildFake("refining_v1")

        assert entry.needs == {"toy:gaussian"}
        assert entry.supports == {"toy:grid"}
        assert with_model.needs == {"toy:gaussian"}
        assert with_model.supports == frozenset()
        assert without_model.needs == frozenset()
        assert without_model.supports == {"toy:grid"}

    def test_an_analysis_that_is_not_registered_cannot_say_what_its_entry_declares(
        self,
    ) -> None:
        analysis = buildMinimal()

        with pytest.raises(RegistryError, match="Unknown analysis 'minimal_v1'"):
            analysis.needs  # noqa: B018
        with pytest.raises(RegistryError, match="Unknown analysis 'minimal_v1'"):
            analysis.supports  # noqa: B018


class TestCompute:
    def test_it_returns_a_result_that_names_the_analysis_and_the_history_version(
        self, isolatedAnalysisKind: KindTable
    ) -> None:
        defineFake()

        result = buildFake().compute(makeState(history_version=6), CountingProvider())

        assert isinstance(result, FakeResult)
        assert isinstance(result, AnalysisResult)
        assert result.analysis == "fake_analysis_v1"
        assert result.history_version == 6

    def test_it_reads_every_trial_of_the_state_in_every_state(
        self, isolatedAnalysisKind: KindTable
    ) -> None:
        defineFake()
        trials = makeTrials(completed=3, pending=2, failed=1, abandoned=1)

        result = buildFake().compute(makeState(trials), CountingProvider())

        counts = dict(zip((s.value for s in TrialState), result.counts, strict=True))
        assert counts == {"pending": 2, "completed": 3, "failed": 1, "abandoned": 1}
        assert result.values == (0.0, 1.0, 2.0)
        assert result.best_id == 2

    def test_an_empty_history_gives_an_empty_answer_and_not_an_error(
        self, isolatedAnalysisKind: KindTable
    ) -> None:
        defineFake()

        result = buildFake().compute(makeState(), CountingProvider())

        assert result.best_id is None
        assert result.values == ()
        assert result.band is None

    def test_the_same_configuration_state_and_seed_give_the_same_result(
        self, isolatedAnalysisKind: KindTable
    ) -> None:
        # Determinism (specification 3.7): the numbers come from the sources of the state.
        defineFake()
        trials = makeTrials(completed=2, pending=1)

        first = buildFake().compute(
            makeState(trials, ask_index=4, seed=11), CountingProvider()
        )
        second = buildFake().compute(
            makeState(trials, ask_index=4, seed=11), CountingProvider()
        )

        assert first == second

    def test_asking_twice_for_the_same_state_gives_the_same_result(
        self, isolatedAnalysisKind: KindTable
    ) -> None:
        # The analysis has no memory: nothing from the first run reaches the second.
        defineFake()
        analysis = buildFake()
        state = makeState(makeTrials(completed=2), ask_index=1)

        assert analysis.compute(state, CountingProvider()) == analysis.compute(
            state, CountingProvider()
        )

    def test_another_ask_index_or_another_seed_gives_other_numbers(
        self, isolatedAnalysisKind: KindTable
    ) -> None:
        defineFake()
        analysis = buildFake()
        reference = analysis.compute(
            makeState(ask_index=0, seed=5), CountingProvider()
        ).samples

        later = analysis.compute(makeState(ask_index=1, seed=5), CountingProvider()).samples
        other = analysis.compute(makeState(ask_index=0, seed=6), CountingProvider()).samples

        assert later != reference
        assert other != reference

    def test_the_global_random_state_is_neither_read_nor_changed(
        self, isolatedAnalysisKind: KindTable
    ) -> None:
        defineFake()
        analysis = buildFake()
        state = makeState()

        reseed(1)
        first = analysis.compute(state, CountingProvider())
        torch_before = torch.get_rng_state()
        numpy_before = np.random.get_state()[1].copy()
        python_before = random.getstate()
        analysis.compute(state, CountingProvider())
        reseed(2)
        second = analysis.compute(state, CountingProvider())

        assert first == second
        reseed(1)
        analysis.compute(state, CountingProvider())
        assert torch.equal(torch.get_rng_state(), torch_before)
        assert (np.random.get_state()[1] == numpy_before).all()
        assert random.getstate() == python_before

    def test_computing_changes_nothing_in_the_state(
        self, isolatedAnalysisKind: KindTable
    ) -> None:
        # Analyses never change the history (specification 5.2).
        defineFake()
        trials = makeTrials(completed=2, pending=1)
        state = makeState(trials, history_version=6, ask_index=2)

        buildFake().compute(state, CountingProvider())

        assert state.trials == tuple(trials)
        assert state.history_version == 6
        assert state.ask_index == 2

    def test_computing_leaves_a_stream_where_a_later_run_finds_it(
        self, isolatedAnalysisKind: KindTable
    ) -> None:
        # The state hands out a new generator at each call: running an analysis does not use
        # up the stream that the next one starts from.
        defineFake()
        state = makeState(ask_index=3, seed=9)
        before = state.generatorFor("analysis.fake").random(3).tolist()

        buildFake().compute(state, CountingProvider())

        assert state.generatorFor("analysis.fake").random(3).tolist() == before


class TestTheSurrogateOnDemand:
    def test_an_analysis_that_does_not_need_the_model_never_asks_for_it(
        self, isolatedAnalysisKind: KindTable
    ) -> None:
        defineFake()
        provider = CountingProvider()

        result = buildFake().compute(makeState(makeTrials(completed=2)), provider)

        assert provider.calls == 0
        assert result.outcomes == ()
        assert result.fit_observations is None

    def test_an_analysis_that_needs_the_model_asks_for_it_once(
        self, isolatedAnalysisKind: KindTable
    ) -> None:
        defineFake()
        provider = CountingProvider()

        buildFake(use_model=True).compute(makeState(makeTrials(completed=2)), provider)

        assert provider.calls == 1

    def test_the_analysis_works_with_the_surrogate_that_it_was_given(
        self, isolatedAnalysisKind: KindTable
    ) -> None:
        defineFake()
        provider = CountingProvider(makeSurrogate())

        result = buildFake(use_model=True).compute(makeState(), provider)

        assert result.outcomes == ("yield",)
        assert result.fit_observations == 5

    def test_the_provider_is_asked_at_each_run_and_the_caching_is_its_own(
        self, isolatedAnalysisKind: KindTable
    ) -> None:
        # The study's provider keeps the fit (the cache is keyed by the history version): an
        # analysis does not, and asks each time it runs.
        defineFake()
        provider = CountingProvider()
        analysis = buildFake(use_model=True)
        state = makeState()

        analysis.compute(state, provider)
        analysis.compute(state, provider)

        assert provider.calls == 2

    def test_what_the_provider_cannot_provide_is_not_caught_by_the_analysis(
        self, isolatedAnalysisKind: KindTable
    ) -> None:
        defineFake()

        class TooFewTrials:
            def surrogate(self) -> Any:
                raise RuntimeError("only 1 completed trial: a surrogate needs 3")

        state = makeState(makeTrials(completed=1))
        provider: SurrogateProvider = TooFewTrials()

        with pytest.raises(RuntimeError, match="a surrogate needs 3"):
            buildFake(use_model=True).compute(state, provider)
        assert state.trials == tuple(makeTrials(completed=1))

    def test_an_analysis_that_does_not_need_the_model_works_when_none_can_be_provided(
        self, isolatedAnalysisKind: KindTable
    ) -> None:
        # The point of asking on demand: the best observed point needs no fit.
        defineFake()

        class NoModel:
            def surrogate(self) -> Any:
                raise RuntimeError("no surrogate suits this problem")

        result = buildFake().compute(makeState(makeTrials(completed=1)), NoModel())

        assert result.best_id == 0

    def test_anything_with_the_method_is_a_provider(
        self, isolatedAnalysisKind: KindTable
    ) -> None:
        # A caller who only has a surrogate wraps it, and the analysis cannot tell.
        defineFake()
        surrogate = makeSurrogate()

        class Wrapper:
            def surrogate(self) -> Any:
                return surrogate

        result = buildFake(use_model=True).compute(makeState(), Wrapper())

        assert result.outcomes == surrogate.outcome_names


class TestTheResultIsSerializable:
    def test_the_result_that_an_analysis_returns_survives_json_exactly(
        self, isolatedAnalysisKind: KindTable
    ) -> None:
        defineFake()
        state = makeState(makeTrials(completed=3, pending=1), history_version=4, ask_index=2)
        result = buildFake(use_model=True).compute(state, CountingProvider())

        stored = json.loads(json.dumps(result.toDict(), allow_nan=False))

        assert FakeResult.fromDict(stored) == result

    def test_the_result_cannot_be_changed_by_the_caller_that_receives_it(
        self, isolatedAnalysisKind: KindTable
    ) -> None:
        defineFake()
        result = buildFake().compute(makeState(makeTrials(completed=2)), CountingProvider())

        with pytest.raises(ValidationError, match="frozen"):
            result.best_id = 7  # type: ignore[misc]
        assert isinstance(result.values, tuple)
