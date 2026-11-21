"""Unit tests for `StudyStateView` (roadmap task P6-01, specification 5.2, 5.9, 6, ADR-0013).

What is tested: what the view holds, that it is read-only (every attempt to change the study
through it fails), the selection of trials by state, and the random sources it hands out.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest
from policy_fakes import makeGoal, makeProblem, makeTrials, makeView
from pydantic import ValidationError

from planexs.core.ConfigError import ConfigError
from planexs.core.RngTree import RngTree
from planexs.domain.Trial import Trial
from planexs.domain.TrialState import TrialState
from planexs.policies.StudyStateView import StudyStateView

FIELDS = ("problem", "goal", "trials", "history_version", "ask_index", "random_sources")


class TestWhatItHolds:
    def test_it_holds_what_the_study_gave_it(self) -> None:
        problem = makeProblem()
        goal = makeGoal()
        tree = RngTree(3)
        trials = makeTrials(completed=2, pending=1)

        view = StudyStateView(
            problem=problem,
            goal=goal,
            trials=trials,
            history_version=5,
            ask_index=2,
            random_sources=tree,
        )

        assert view.problem is problem
        assert view.goal is goal
        assert list(view.trials) == trials
        assert view.history_version == 5
        assert view.ask_index == 2
        assert view.random_sources is tree

    def test_every_trial_in_every_state_is_there(self) -> None:
        view = makeView(makeTrials(pending=2, completed=3, failed=1, abandoned=1))

        assert len(view.trials) == 7
        assert {trial.state for trial in view.trials} == set(TrialState)

    def test_the_trials_are_in_the_order_they_were_made(self) -> None:
        view = makeView(makeTrials(completed=2, pending=2))

        assert [trial.id for trial in view.trials] == [0, 1, 2, 3]

    def test_an_empty_history_is_a_view_of_a_study_that_has_not_started(self) -> None:
        view = makeView()

        assert view.trials == ()
        assert view.history_version == 0
        assert view.ask_index == 0

    def test_it_is_built_with_keywords_only(self) -> None:
        with pytest.raises(TypeError):
            StudyStateView(makeProblem(), makeGoal(), [], 0, 0, RngTree(1))  # type: ignore[misc]

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
                StudyStateView(**arguments)

    def test_views_are_compared_by_identity(self) -> None:
        first = makeView()
        second = makeView()

        assert first == first
        assert first != second


class TestTheCounters:
    @pytest.mark.parametrize("value", [0, 1, 7, 10**12])
    def test_a_non_negative_integer_is_a_counter(self, value: int) -> None:
        view = makeView(history_version=value, ask_index=value)

        assert view.history_version == value
        assert view.ask_index == value

    @pytest.mark.parametrize("field", ["history_version", "ask_index"])
    @pytest.mark.parametrize("value", [-1, 1.0, 2.5, "3", None, True, False, [0]])
    def test_anything_else_is_refused_and_the_error_names_the_field(
        self, field: str, value: Any
    ) -> None:
        with pytest.raises(ConfigError) as caught:
            makeView(**{field: value})

        assert caught.value.field_path == field
        assert repr(value) in str(caught.value)
        assert caught.value.suggestion


class TestItIsReadOnly:
    """Acceptance test of the task: mutation attempts fail."""

    @pytest.mark.parametrize("name", FIELDS)
    def test_no_field_can_be_assigned(self, name: str) -> None:
        view = makeView(makeTrials(pending=1))

        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(view, name, getattr(view, name))

    @pytest.mark.parametrize("name", FIELDS)
    def test_no_field_can_be_deleted(self, name: str) -> None:
        with pytest.raises(dataclasses.FrozenInstanceError):
            delattr(makeView(), name)

    def test_nothing_can_be_added_to_it(self) -> None:
        view = makeView()

        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError, TypeError)):
            view.shortcut = 1  # type: ignore[attr-defined]

    def test_it_has_no_dictionary_to_write_to(self) -> None:
        view = makeView()

        with pytest.raises(TypeError):
            vars(view)
        assert not hasattr(view, "__dict__")

    def test_the_history_cannot_be_appended_to_removed_from_or_reordered(self) -> None:
        view = makeView(makeTrials(pending=2))

        assert isinstance(view.trials, tuple)
        for operation in ("append", "extend", "insert", "remove", "pop", "clear", "sort"):
            assert not hasattr(view.trials, operation), operation
        with pytest.raises(TypeError):
            view.trials[0] = view.trials[1]  # type: ignore[index]

    def test_the_history_does_not_follow_the_list_it_was_built_from(self) -> None:
        trials = makeTrials(pending=2)
        view = makeView(trials)

        trials.append(makeTrials(completed=1)[0])
        trials.clear()

        assert len(view.trials) == 2

    def test_a_trial_that_the_study_adds_afterwards_is_not_in_a_view_already_made(
        self,
    ) -> None:
        trials = makeTrials(pending=1)
        before = makeView(trials)

        trials.append(makeTrials(completed=1)[0])
        after = makeView(trials)

        assert len(before.trials) == 1
        assert len(after.trials) == 2

    def test_a_trial_cannot_be_changed_through_the_view(self) -> None:
        view = makeView(makeTrials(pending=1))
        trial = view.trials[0]

        with pytest.raises(ValidationError, match="frozen"):
            trial.state = TrialState.completed  # type: ignore[misc]
        with pytest.raises(ValidationError, match="frozen"):
            trial.id = 99  # type: ignore[misc]
        with pytest.raises(TypeError):
            trial.parameters["x0"] = 5.0  # type: ignore[index]
        with pytest.raises(TypeError):
            trial.metadata["k"] = 1  # type: ignore[index]
        assert view.trials[0].state is TrialState.pending

    def test_a_trial_can_only_move_by_becoming_a_new_trial_and_the_view_is_unchanged(
        self,
    ) -> None:
        view = makeView(makeTrials(pending=1))

        moved = view.trials[0].abandon()

        assert moved.state is TrialState.abandoned
        assert view.trials[0].state is TrialState.pending

    def test_the_problem_cannot_be_changed_through_the_view(self) -> None:
        view = makeView()

        with pytest.raises(ValidationError, match="frozen"):
            view.problem.outcomes = ()  # type: ignore[misc]
        with pytest.raises(ValidationError, match="frozen"):
            view.problem.search_space = None  # type: ignore[misc, assignment]

    def test_the_random_sources_cannot_be_reseeded(self) -> None:
        view = makeView(seed=11)
        before = view.seedFor("policy")

        with pytest.raises(AttributeError):
            view.random_sources.entropy = 12  # type: ignore[misc]

        assert view.random_sources.entropy == 11
        assert view.seedFor("policy") == before

    def test_asking_for_a_stream_changes_no_other_stream(self) -> None:
        view = makeView(seed=11)
        first = view.seedFor("policy.design")

        view.generatorFor("policy.surrogate").random(100)
        view.seedFor("policy.optimizer")

        assert view.seedFor("policy.design") == first

    def test_reading_the_view_changes_nothing_in_it(self) -> None:
        view = makeView(makeTrials(completed=2, pending=1), history_version=4, ask_index=1)
        snapshot = (view.trials, view.history_version, view.ask_index, view.seedFor("p"))

        view.trialsIn(TrialState.pending)
        view.trialsIn()
        view.generatorFor("p").random(10)

        assert (
            view.trials,
            view.history_version,
            view.ask_index,
            view.seedFor("p"),
        ) == snapshot


class TestTrialsIn:
    def history(self) -> list[Trial]:
        return makeTrials(completed=2, pending=2, failed=1, abandoned=1)

    def test_it_selects_the_trials_in_a_state(self) -> None:
        view = makeView(self.history())

        assert [t.id for t in view.trialsIn(TrialState.pending)] == [2, 3]
        assert [t.id for t in view.trialsIn(TrialState.completed)] == [0, 1]
        assert [t.id for t in view.trialsIn(TrialState.failed)] == [4]
        assert [t.id for t in view.trialsIn(TrialState.abandoned)] == [5]

    def test_it_selects_the_trials_in_any_of_several_states_in_history_order(self) -> None:
        view = makeView(self.history())

        found = view.trialsIn(TrialState.abandoned, TrialState.completed)

        assert [t.id for t in found] == [0, 1, 5]

    def test_the_four_states_together_are_the_whole_history(self) -> None:
        view = makeView(self.history())

        assert view.trialsIn(*TrialState) == view.trials

    def test_a_repeated_state_is_not_counted_twice(self) -> None:
        view = makeView(self.history())

        assert len(view.trialsIn(TrialState.pending, TrialState.pending)) == 2

    def test_no_state_selects_nothing(self) -> None:
        assert makeView(self.history()).trialsIn() == ()

    def test_a_state_with_no_trial_selects_nothing(self) -> None:
        assert makeView(makeTrials(completed=2)).trialsIn(TrialState.pending) == ()

    def test_the_answer_is_a_new_tuple_each_time(self) -> None:
        view = makeView(self.history())

        first = view.trialsIn(TrialState.pending)
        second = view.trialsIn(TrialState.pending)

        assert isinstance(first, tuple)
        assert first == second


class TestTheRandomSources:
    def test_a_seed_is_the_one_of_the_tree_for_the_ask_and_the_path(self) -> None:
        tree = RngTree(21)
        view = makeView(ask_index=3, seed=21)

        assert view.seedFor("policy.design") == tree.seed(3, "policy.design")

    def test_a_generator_is_the_one_of_the_tree_for_the_ask_and_the_path(self) -> None:
        tree = RngTree(21)
        view = makeView(ask_index=3, seed=21)

        mine = view.generatorFor("policy.design").random(5)
        theirs = tree.generator(3, "policy.design").random(5)

        assert list(mine) == list(theirs)

    def test_the_same_study_ask_and_path_give_the_same_stream_in_two_views(self) -> None:
        first = makeView(ask_index=2, seed=5).generatorFor("policy").random(4)
        second = (
            makeView(makeTrials(completed=3), ask_index=2, seed=5)
            .generatorFor("policy")
            .random(4)
        )

        assert list(first) == list(second)

    def test_another_ask_gives_another_stream(self) -> None:
        first = makeView(ask_index=0, seed=5).seedFor("policy")
        second = makeView(ask_index=1, seed=5).seedFor("policy")

        assert first != second

    def test_another_path_gives_another_stream(self) -> None:
        view = makeView(seed=5)

        assert view.seedFor("policy.design") != view.seedFor("policy.optimizer")

    def test_another_study_seed_gives_another_stream(self) -> None:
        assert makeView(seed=5).seedFor("policy") != makeView(seed=6).seedFor("policy")

    def test_each_call_gives_a_generator_at_the_start_of_the_stream(self) -> None:
        view = makeView(seed=5)

        first = view.generatorFor("policy")
        first.random(10)
        second = view.generatorFor("policy")

        assert second.random() == makeView(seed=5).generatorFor("policy").random()

    @pytest.mark.parametrize("path", [None, 3, ["policy"], b"policy"])
    def test_a_path_that_is_not_a_string_is_refused(self, path: Any) -> None:
        view = makeView()

        with pytest.raises(ConfigError):
            view.seedFor(path)
        with pytest.raises(ConfigError):
            view.generatorFor(path)
