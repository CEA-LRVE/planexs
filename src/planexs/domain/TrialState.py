"""The lifecycle states of a trial."""

from __future__ import annotations

from enum import StrEnum


class TrialState(StrEnum):
    """Where a trial is in its lifecycle (specification 4, 5.4; ADR-0012).

    The value of a member is its name, which is how a state is written in a snapshot.

    Attributes:
        pending: Suggested and not yet reported. The only state a trial can leave.
        completed: Evaluated, with an observation. Terminal.
        failed: The evaluation did not produce an observation, and a reason says why.
            Terminal.
        abandoned: Withdrawn by the caller before it was reported. Terminal.
    """

    pending = "pending"
    completed = "completed"
    failed = "failed"
    abandoned = "abandoned"

    @property
    def is_terminal(self) -> bool:
        """Whether a trial in this state has ended and never changes again (a comparison:
        cheap, snake_case).
        """
        return self is not TrialState.pending

    def canMoveTo(self, target: TrialState) -> bool:
        """Whether the table of ADR-0012 allows a trial to move from this state to `target`.

        Exactly three moves are legal: from pending to completed, to failed and to
        abandoned. Every other pair is illegal, including a move from a state to itself and
        any move to pending, because a terminal trial is never reopened and creation is not
        a transition.

        Args:
            target: The state the trial would move to.

        Returns:
            `True` for one of the three legal moves.
        """
        return self is TrialState.pending and target.is_terminal
