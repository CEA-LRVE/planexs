"""The read-only information a design is given to propose points."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from planexs.domain.Problem import Problem
from planexs.domain.Trial import Trial
from planexs.domain.TrialState import TrialState
from planexs.encoding.EncodedConstraints import EncodedConstraints
from planexs.encoding.SearchSpaceEncoder import SearchSpaceEncoder


@dataclass(frozen=True, eq=False, kw_only=True)
class DesignContext:
    """Everything a design receives to propose points (specification 5.8, 6).

    A design proposes points without a model, so what it needs is the space it proposes in
    (`problem`, `encoder`), the constraints every point must satisfy (`constraints`), what
    exists already (`trials`) and a source of randomness (`generator`). It is built with
    keywords, `DesignContext(problem=..., encoder=..., ...)`, and the number of points is not
    in it: that is asked for in each call (`AbstractDesign.propose`).

    It is frozen: no field can be assigned, deleted or added, and `trials` is stored as a
    tuple, so a design cannot rewrite the history through it. The history is the study's, and
    a design only reads it (specification 5.2). The generator is the one thing that is meant to
    change: drawing from it advances it. The study gives each call a generator of its own,
    derived for the ask and the component (specification 3.7, 5.11), so that no design
    disturbs the stream of another and the same seed gives the same points.

    Contexts are compared by identity (`eq=False`): the encoder and the constraints are not
    values to compare.

    Attributes:
        problem: What can be varied and what is measured.
        encoder: The encoder of the problem's search space: the points a design returns are
            points of its unit cube.
        constraints: The constraints of the search space, rewritten on the encoded columns. A
            design returns only points that satisfy them (specification 5.5, 5.8).
        trials: Every trial of the study, in every state, in the order they were made.
        generator: The seeded generator that every random choice of the design draws from.
    """

    problem: Problem
    encoder: SearchSpaceEncoder
    constraints: EncodedConstraints
    trials: Sequence[Trial]
    generator: torch.Generator

    def __post_init__(self) -> None:
        """Stores the trials as a tuple, so that the history cannot be changed through it."""
        # Frozen dataclass: `object.__setattr__` is the sanctioned way to normalize.
        object.__setattr__(self, "trials", tuple(self.trials))

    def countObservations(self) -> int:
        """How many observations exist: the trials that are completed.

        A pending trial has no observation yet, a failed one never produced one and an
        abandoned one was withdrawn, so none of them is counted. Data attached from outside
        enters as a completed trial (specification 5.4) and is counted like any other.

        It reads the whole history at each call, so it is a method and not a property
        (specification 10.2).

        Returns:
            The number of completed trials.
        """
        return sum(1 for trial in self.trials if trial.state is TrialState.completed)
