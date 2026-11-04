"""Raised when a trial is asked to make a move its lifecycle does not allow."""

from __future__ import annotations

from planexs.core.PlanexsError import PlanexsError


class TrialStateError(PlanexsError):
    """A trial was asked to change state in a way its lifecycle forbids.

    Raised by the transition methods of a trial for every move that is not one of the three
    legal ones of the table of ADR-0012: from pending to completed, to failed or to abandoned
    (specification 5.4). The component hint names the trial, the field hint is `state`, and
    the suggestion says what to do instead.
    """
