"""The read-only view of a study that a policy is given."""

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
class StudyStateView:
    """Everything a policy is given to answer which parameterizations to evaluate next
    (specification 5.2, 5.9, 6; ADR-0013).

    The study builds a view for each ask and hands it to the policy: the problem, the goal,
    every trial in every state (pending ones included, so that batches and asynchronous
    workers are first-class), the version of the history, the index of the ask and the random
    sources. It is built with keywords, `StudyStateView(problem=..., goal=..., ...)`.

    **It is read-only**, and a policy has no way to change the study through it. No attribute
    can be assigned, deleted or added, and the view has no `__dict__` to write to. The history
    is a tuple, copied when the view is built, so a trial that the study adds afterwards is not
    in a view that was already made, and a policy cannot append to the history, remove from it
    or reorder it. A trial is an immutable record, and so is the problem. The random sources
    are a `RngTree`, which holds only its entropy and never changes. The one thing that a view
    does not copy is the goal, which is the study's own object: it is configuration and has no
    operation that changes it, and a policy treats it as read-only (specification 5.13: what a
    policy reaches beyond the view is outside what planexs keeps in sync).

    **The history version** is a counter that the study raises at each change of a trial's
    state and at nothing else, so a policy or a cache can tell whether the history is the one
    it saw: two views with the same version hold the same trials in the same states. The
    surrogate cache is keyed by it.

    **The random sources** are the study's `RngTree`. A policy never draws from a global state:
    it asks for a stream by the path of the component that will use it, derived for the index
    of this ask, with `seedFor` or `generatorFor`. The same seed, ask index and path give the
    same stream in every process (specification 3.7, 5.11), and a path that another component
    uses is another stream.

    Views are compared by identity (`eq=False`): the goal and the random sources are not values
    to compare.

    Attributes:
        problem: What can be varied and what is measured.
        goal: What the study is for.
        trials: Every trial of the study, in every state, in the order they were made.
        history_version: The version of the history, a non-negative integer that grows at
            each change of a trial's state.
        ask_index: The index of the ask that this view is for, counted from 0: the number of
            asks that were made before it.
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
        cannot be changed through the view and the view does not follow the study's own list.

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
            *states: The states to keep, for example `TrialState.pending`. With none, no
                trial is kept.

        Returns:
            A new tuple, empty if no trial is in one of the states.
        """
        wanted = frozenset(states)
        return tuple(trial for trial in self.trials if trial.state in wanted)

    def seedFor(self, path: str) -> int:
        """The integer seed of the stream of the component at `path`, for this ask.

        Args:
            path: The path of the component that will use the stream, for example
                `policy.design`. Compared exactly, as written: any string is a path.

        Returns:
            The seed, the same for the same study seed, ask index and path.

        Raises:
            ConfigError: if `path` is not a string.
        """
        return self.random_sources.seed(self.ask_index, path)

    def generatorFor(self, path: str) -> np.random.Generator:
        """A new numpy generator on the stream of the component at `path`, for this ask.

        Each call returns a new generator at the start of the stream: ask once and keep it for
        every draw of the ask.

        Args:
            path: The path of the component that will use the stream, for example
                `policy.design`. Compared exactly, as written: any string is a path.

        Returns:
            The generator.

        Raises:
            ConfigError: if `path` is not a string.
        """
        return self.random_sources.generator(self.ask_index, path)


def _requireNonNegativeInt(value: object, name: str) -> None:
    """Checks that a counter of the view is a non-negative integer (a bool is not one).

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
            f"The study-state view needs {name} to be a non-negative integer, but got "
            f"{value!r}.",
            component="study-state view",
            field_path=name,
            suggestion=f"build the view with a whole number from 0 up as {name}",
        )
