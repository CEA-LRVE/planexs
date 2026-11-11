"""Raised when a value is not legal for the parameter it is given to."""

from __future__ import annotations

from planexs.core.PlanexsError import PlanexsError


class ParameterValueError(PlanexsError, ValueError):
    """A value is not legal for its parameter.

    Raised by the `validateValue` of a parameter type (a `Range` given a value above its upper
    bound, a `Choice` given something that is not one of its values, ...). The message names
    the parameter and the value, and says what the parameter accepts (specification 3.6, 5.3).

    It is also a `ValueError`, like `OutcomeValueError`, so that code that validates data with
    pydantic reports it with its location as it does any other value error.
    """
