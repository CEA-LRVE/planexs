"""Raised when an observation does not match the outcomes it is checked against."""

from __future__ import annotations

from planexs.core.PlanexsError import PlanexsError


class ObservationError(PlanexsError, ValueError):
    """An observation is malformed, or does not match the outcomes of its problem.

    Raised by `Observation` for what is wrong with the observation as a whole: an outcome
    that has no value, a value or a noise standard deviation for a name that is no outcome,
    a noise standard deviation for an outcome whose kind has none, and a noise standard
    deviation that is not a finite number that is zero or more. The message names the
    outcomes involved and says how to fix the problem (specification 3.6, 5.4).

    A value that is not legal for the kind of its outcome is reported by the kind instead,
    as an `OutcomeValueError`.

    It is also a `ValueError`, so that a Pydantic validator that raises it (the validation of
    an `Observation` does) reports it as a validation error with its location, as it does
    for any other value error, instead of letting it through as it is.
    """
