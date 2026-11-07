"""Raised when an observed value is not legal for the kind of its outcome."""

from __future__ import annotations

from planexs.core.PlanexsError import PlanexsError


class OutcomeValueError(PlanexsError, ValueError):
    """An observed value is not legal for the kind of its outcome.

    Raised by the `validateValue` of an outcome kind (a continuous outcome given `nan`, a
    binary outcome given `2`, ...). The message names the outcome and the value, and says what
    the kind accepts (specification 3.6, 5.4).

    It is also a `ValueError`, so that a Pydantic validator that calls it (the validation of an
    observation does) reports it as a validation error with its location, as it does for any
    other value error, instead of letting it through as it is.
    """
