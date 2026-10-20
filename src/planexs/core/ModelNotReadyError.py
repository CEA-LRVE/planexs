"""Raised when a surrogate model cannot be used yet."""

from __future__ import annotations

from planexs.core.PlanexsError import PlanexsError


class ModelNotReadyError(PlanexsError):
    """A surrogate model cannot be built or used yet, for lack of data.

    Raised when an operation needs a fitted surrogate and the study holds fewer observations
    than the surrogate needs. It is not an invalid request: asking again once more trials have
    been completed can succeed (specification 5.6).
    """
