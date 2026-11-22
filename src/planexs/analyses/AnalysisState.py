"""The read-only state of a study that an analysis is given."""

from __future__ import annotations

import operator
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from planexs.core.ConfigError import ConfigError
from planexs.core.RngTree import RngTree
from planexs.domain.Problem import Problem
from planexs.domain.Trial import Trial
from planexs.domain.TrialState import TrialState
from planexs.goals.AbstractGoal import AbstractGoal


@dataclass(frozen=True, eq=False, kw_only=True, slots=True)
class AnalysisState:
    """Everything an analysis is given to compute its result: the read-only state of a study
    (specification 5.2, 5.10, 6).

    The study builds one for each run of an analysis: the problem, the goal, every trial in
    every state, the version of the history, the number of asks so far and the random sources.
    It is built with keywords, `AnalysisState(problem=..., goal=..., ...)`. It is the view
    that a policy gets, without the part that belongs to an ask: the policy's view is for one
    ask, and an analysis can be run at any time, between asks, while trials are pending
    (specification 5.2).

    **It is read-only**, and an analysis has no way to change the study through it ("analyses
    never change the history"). No attribute can be assigned, deleted or added, and the state
    has no `__dict__` to write to. The history is a tuple, copied when the state is built, so a
    trial that the study adds afterwards is not in a state that was already made, and an
    analysis cannot append to the history, remove from it or reorder it. A trial is an
    immutable record, and so is the problem. The random sources are a `RngTree`, which holds
    only its entropy and never changes. The one thing that a state does not copy is the goal,
    which is the study's own object: it is configuration and has no operation that changes it,
    and an analysis treats it as read-only (specification 5.13).

    **The history version** is a counter that the study raises at each change of a trial's
    state and at nothing else, so two states with the same version hold the same trials in the
    same states. A result carries the version it was computed on.

    **The random sources** are the study's `RngTree`. An analysis never draws from a global
    state: it asks for a stream by the path of the component that will use it, derived for the
    ask index, with `seedFor` or `generatorFor`. The ask index here is the number of asks the
    study has made so far, so an analysis that is run twice with no ask in between draws the
    same stream and returns the same result, and the same seed, ask index and path give the
    same stream in every process (specification 3.7, 5.11).

    States are compared by identity (`eq=False`): the goal and the random sources are not
    values to compare.

    Attributes:
        problem: What can be varied and what is measured.
        goal: What the study is for.
        trials: Every trial of the study, in every state, in the order they were made.
        history_version: The version of the history, a non-negative integer that grows at
            each change of a trial's state.
        ask_index: The number of asks the study has made so far, a non-negative integer: the
            index that the next ask will have.
        random_sources: The random-number tree of the study.
    """

    problem: Problem
    goal: AbstractGoal[Any]
    trials: Sequence[Trial]
    history_version: int
    ask_index: int
    random_sources: RngTree

    def __post_init__(self) -> None:
        """Checks the two counters, and stores the trials as a tuple, so that the history
        cannot be changed through the state and the state does not follow the study's own list.

        Raises:
            ConfigError: if `history_version` or `ask_index` is not a non-negative integer.
        """
        _requireNonNegativeInt(self.history_version, "history_version")
        _requireNonNegativeInt(self.ask_index, "ask_index")
        # Frozen dataclass: `object.__setattr__` is the sanctioned way to normalize.
        object.__setattr__(self, "trials", tuple(self.trials))

    def trialsIn(self, *states: TrialState) -> tuple[Trial, ...]:
        """The trials that are in one of the given states, in the order they were made.

        It reads the whole history at each call, so it is a method and not a property
        (specification 10.2).

        Args:
            *states: The states to keep, for example `TrialState.completed`. With none, no
                trial is kept.

        Returns:
            A new tuple, empty if no trial is in one of the states.
        """
        wanted = frozenset(states)
        return tuple(trial for trial in self.trials if trial.state in wanted)

    def seedFor(self, path: str) -> int:
        """The integer seed of the stream of the component at `path`, for this ask index.

        Args:
            path: The path of the component that will use the stream, for example
                `analysis.parameter_importance`. Compared exactly, as written: any string is a
                path.

        Returns:
            The seed, the same for the same study seed, ask index and path.

        Raises:
            ConfigError: if `path` is not a string.
        """
        return self.random_sources.seed(self.ask_index, path)

    def generatorFor(self, path: str) -> np.random.Generator:
        """A new numpy generator on the stream of the component at `path`, for this ask index.

        Each call returns a new generator at the start of the stream: ask once and keep it for
        every draw of the analysis.

        Args:
            path: The path of the component that will use the stream, for example
                `analysis.parameter_importance`. Compared exactly, as written: any string is a
                path.

        Returns:
            The generator.

        Raises:
            ConfigError: if `path` is not a string.
        """
        return self.random_sources.generator(self.ask_index, path)


def _requireNonNegativeInt(value: object, name: str) -> None:
    """Checks that a counter of the state is a non-negative integer (a bool is not one).

    Args:
        value: The value to check.
        name: The name of the field, for the error.

    Raises:
        ConfigError: if it is not.
    """
    try:
        valid = not isinstance(value, bool) and operator.index(value) >= 0  # type: ignore[arg-type]
    except TypeError:
        valid = False
    if not valid:
        raise ConfigError(
            f"The analysis state needs {name} to be a non-negative integer, but got "
            f"{value!r}.",
            component="analysis state",
            field_path=name,
            suggestion=f"build the state with a whole number from 0 up as {name}",
        )
