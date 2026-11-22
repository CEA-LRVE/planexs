"""Unit tests for `AnalysisState` (roadmap task P7-01, specification 5.2, 5.10, 6).

What is tested: what the state holds, that it is read-only (every attempt to change the study
through it fails), the selection of trials by state, and the random sources it hands out.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest
from analysis_fakes import makeGoal, makeProblem, makeState, makeTrials
from pydantic import ValidationError

from planexs.analyses.AnalysisState import AnalysisState
from planexs.core.ConfigError import ConfigError
from planexs.core.RngTree import RngTree
from planexs.domain.Trial import Trial
from planexs.domain.TrialState import TrialState

FIELDS = ("problem", "goal", "trials", "history_version", "ask_index", "random_sources")


class TestWhatItHolds:
    def test_it_holds_what_the_study_gave_it(self) -> None:
        problem = makeProblem()
        goal = makeGoal()
        tree = RngTree(3)
        trials = makeTrials(completed=2, pending=1)

        state = AnalysisState(
            problem=problem,
            goal=goal,
            trials=trials,
            history_version=5,
            ask_index=2,
            random_sources=tree,
        )

        assert state.problem is problem
        assert state.goal is goal
        assert list(state.trials) == trials
        assert state.history_version == 5
        assert state.ask_index == 2
        assert state.random_sources is tree

    def test_every_trial_in_every_state_is_there(self) -> None:
        state = makeState(makeTrials(pending=2, completed=3, failed=1, abandoned=1))

        assert len(state.trials) == 7
        assert {trial.state for trial in state.trials} == set(TrialState)

    def test_the_trials_are_in_the_order_they_were_made(self) -> None:
        state = makeState(makeTrials(completed=2, pending=2))

        assert [trial.id for trial in state.trials] == [0, 1, 2, 3]

    def test_an_empty_history_is_a_state_of_a_study_that_has_not_started(self) -> None:
        state = makeState()

        assert state.trials == ()
        assert state.history_version == 0
        assert state.ask_index == 0

    def test_it_is_built_with_keywords_only(self) -> None:
        with pytest.raises(TypeError):
            AnalysisState(makeProblem(), makeGoal(), [], 0, 0, RngTree(1))  # type: ignore[misc]

    def test_every_field_is_required(self) -> None:
        for name in FIELDS:
            arguments: dict[str, Any] = {
                "problem": makeProblem(),
                "goal": makeGoal(),
                "trials": [],
                "history_version": 0,
                "ask_index": 0,
                "random_sources": RngTree(1),
            }
            del arguments[name]

            with pytest.raises(TypeError, match=name):
                AnalysisState(**arguments)

    def test_states_are_compared_by_identity(self) -> None:
        first = makeState()
        second = makeState()

        assert first == first
        assert first != second


class TestTheCounters:
    @pytest.mark.parametrize("value", [0, 1, 7, 10**12])
    def test_a_non_negative_integer_is_a_counter(self, value: int) -> None:
        state = makeState(history_version=value, ask_index=value)

        assert state.history_version == value
        assert state.ask_index == value

    @pytest.mark.parametrize("field", ["history_version", "ask_index"])
    @pytest.mark.parametrize("value", [-1, 1.0, 2.5, "3", None, True, False, [0]])
    def test_anything_else_is_refused_and_the_error_names_the_field(
        self, field: str, value: Any
    ) -> None:
        with pytest.raises(ConfigError) as caught:
            makeState(**{field: value})

        assert caught.value.field_path == field
        assert repr(value) in str(caught.value)
        assert caught.value.suggestion


class TestItIsReadOnly:
    """Acceptance test of the task: mutation attempts fail."""

    @pytest.mark.parametrize("name", FIELDS)
    def test_no_field_can_be_assigned(self, name: str) -> None:
        state = makeState(makeTrials(pending=1))

        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(state, name, getattr(state, name))

    @pytest.mark.parametrize("name", FIELDS)
    def test_no_field_can_be_deleted(self, name: str) -> None:
        with pytest.raises(dataclasses.FrozenInstanceError):
            delattr(makeState(), name)

    def test_nothing_can_be_added_to_it(self) -> None:
        state = makeState()

        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError, TypeError)):
            state.shortcut = 1  # type: ignore[attr-defined]

    def test_it_has_no_dictionary_to_write_to(self) -> None:
        state = makeState()

        with pytest.raises(TypeError):
            vars(state)
        assert not hasattr(state, "__dict__")

    def test_the_history_cannot_be_appended_to_removed_from_or_reordered(self) -> None:
        state = makeState(makeTrials(pending=2))

        assert isinstance(state.trials, tuple)
        for operation in ("append", "extend", "insert", "remove", "pop", "clear", "sort"):
            assert not hasattr(state.trials, operation), operation
        with pytest.raises(TypeError):
            state.trials[0] = state.trials[1]  # type: ignore[index]

    def test_the_history_does_not_follow_the_list_it_was_built_from(self) -> None:
        trials = makeTrials(pending=2)
        state = makeState(trials)

        trials.append(makeTrials(completed=1)[0])
        trials.clear()

        assert len(state.trials) == 2

    def test_a_trial_that_the_study_adds_afterwards_is_not_in_a_state_already_made(
        self,
    ) -> None:
        trials = makeTrials(pending=1)
        before = makeState(trials)

        trials.append(makeTrials(completed=1)[0])
        after = makeState(trials)

        assert len(before.trials) == 1
        assert len(after.trials) == 2

    def test_a_trial_cannot_be_changed_through_the_state(self) -> None:
        state = makeState(makeTrials(pending=1))
        trial = state.trials[0]

        with pytest.raises(ValidationError, match="frozen"):
            trial.state = TrialState.completed  # type: ignore[misc]
        with pytest.raises(ValidationError, match="frozen"):
            trial.id = 99  # type: ignore[misc]
        with pytest.raises(TypeError):
            trial.parameters["x0"] = 5.0  # type: ignore[index]
        with pytest.raises(TypeError):
            trial.metadata["k"] = 1  # type: ignore[index]
        assert state.trials[0].state is TrialState.pending

    def test_a_trial_can_only_move_by_becoming_a_new_trial_and_the_state_is_unchanged(
        self,
    ) -> None:
        state = makeState(makeTrials(pending=1))

        moved = state.trials[0].abandon()

        assert moved.state is TrialState.abandoned
        assert state.trials[0].state is TrialState.pending

    def test_the_problem_cannot_be_changed_through_the_state(self) -> None:
        state = makeState()

        with pytest.raises(ValidationError, match="frozen"):
            state.problem.outcomes = ()  # type: ignore[misc]
        with pytest.raises(ValidationError, match="frozen"):
            state.problem.search_space = None  # type: ignore[misc, assignment]

    def test_the_random_sources_cannot_be_reseeded(self) -> None:
        state = makeState(seed=11)
        before = state.seedFor("analysis")

        with pytest.raises(AttributeError):
            state.random_sources.entropy = 12  # type: ignore[misc]

        assert state.random_sources.entropy == 11
        assert state.seedFor("analysis") == before

    def test_asking_for_a_stream_changes_no_other_stream(self) -> None:
        state = makeState(seed=11)
        first = state.seedFor("analysis.design")

        state.generatorFor("analysis.surrogate").random(100)
        state.seedFor("analysis.optimizer")

        assert state.seedFor("analysis.design") == first

    def test_reading_the_state_changes_nothing_in_it(self) -> None:
        state = makeState(makeTrials(completed=2, pending=1), history_version=4, ask_index=1)
        snapshot = (state.trials, state.history_version, state.ask_index, state.seedFor("p"))

        state.trialsIn(TrialState.pending)
        state.trialsIn()
        state.generatorFor("p").random(10)

        assert (
            state.trials,
            state.history_version,
            state.ask_index,
            state.seedFor("p"),
        ) == snapshot


class TestTrialsIn:
    def history(self) -> list[Trial]:
        return makeTrials(completed=2, pending=2, failed=1, abandoned=1)

    def test_it_selects_the_trials_in_a_state(self) -> None:
        state = makeState(self.history())

        assert [t.id for t in state.trialsIn(TrialState.pending)] == [2, 3]
        assert [t.id for t in state.trialsIn(TrialState.completed)] == [0, 1]
        assert [t.id for t in state.trialsIn(TrialState.failed)] == [4]
        assert [t.id for t in state.trialsIn(TrialState.abandoned)] == [5]

    def test_it_selects_the_trials_in_any_of_several_states_in_history_order(self) -> None:
        state = makeState(self.history())

        found = state.trialsIn(TrialState.abandoned, TrialState.completed)

        assert [t.id for t in found] == [0, 1, 5]

    def test_the_four_states_together_are_the_whole_history(self) -> None:
        state = makeState(self.history())

        assert state.trialsIn(*TrialState) == state.trials

    def test_a_repeated_state_is_not_counted_twice(self) -> None:
        state = makeState(self.history())

        assert len(state.trialsIn(TrialState.pending, TrialState.pending)) == 2

    def test_no_state_selects_nothing(self) -> None:
        assert makeState(self.history()).trialsIn() == ()

    def test_a_state_with_no_trial_selects_nothing(self) -> None:
        assert makeState(makeTrials(completed=2)).trialsIn(TrialState.pending) == ()

    def test_the_answer_is_a_new_tuple_each_time(self) -> None:
        state = makeState(self.history())

        first = state.trialsIn(TrialState.pending)
        second = state.trialsIn(TrialState.pending)

        assert isinstance(first, tuple)
        assert first == second


class TestTheRandomSources:
    def test_a_seed_is_the_one_of_the_tree_for_the_ask_and_the_path(self) -> None:
        tree = RngTree(21)
        state = makeState(ask_index=3, seed=21)

        assert state.seedFor("analysis.design") == tree.seed(3, "analysis.design")

    def test_a_generator_is_the_one_of_the_tree_for_the_ask_and_the_path(self) -> None:
        tree = RngTree(21)
        state = makeState(ask_index=3, seed=21)

        mine = state.generatorFor("analysis.design").random(5)
        theirs = tree.generator(3, "analysis.design").random(5)

        assert list(mine) == list(theirs)

    def test_the_same_study_ask_and_path_give_the_same_stream_in_two_states(self) -> None:
        first = makeState(ask_index=2, seed=5).generatorFor("analysis").random(4)
        second = (
            makeState(makeTrials(completed=3), ask_index=2, seed=5)
            .generatorFor("analysis")
            .random(4)
        )

        assert list(first) == list(second)

    def test_another_ask_index_gives_another_stream(self) -> None:
        first = makeState(ask_index=0, seed=5).seedFor("analysis")
        second = makeState(ask_index=1, seed=5).seedFor("analysis")

        assert first != second

    def test_another_path_gives_another_stream(self) -> None:
        state = makeState(seed=5)

        assert state.seedFor("analysis.design") != state.seedFor("analysis.optimizer")

    def test_another_study_seed_gives_another_stream(self) -> None:
        assert makeState(seed=5).seedFor("analysis") != makeState(seed=6).seedFor("analysis")

    def test_each_call_gives_a_generator_at_the_start_of_the_stream(self) -> None:
        state = makeState(seed=5)

        first = state.generatorFor("analysis")
        first.random(10)
        second = state.generatorFor("analysis")

        assert second.random() == makeState(seed=5).generatorFor("analysis").random()

    @pytest.mark.parametrize("path", [None, 3, ["analysis"], b"analysis"])
    def test_a_path_that_is_not_a_string_is_refused(self, path: Any) -> None:
        state = makeState()

        with pytest.raises(ConfigError):
            state.seedFor(path)
        with pytest.raises(ConfigError):
            state.generatorFor(path)
