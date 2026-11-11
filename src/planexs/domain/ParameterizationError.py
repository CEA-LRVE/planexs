"""Raised when a parameterization is not legal for the problem it is checked against."""

from __future__ import annotations

from planexs.core.PlanexsError import PlanexsError


class ParameterizationError(PlanexsError, ValueError):
    """A parameterization is not legal for its problem.

    Raised by `Problem.validateParameterization` for what is wrong with the parameterization
    as a whole: a parameter that does not exist, a value that is not legal for its parameter,
    a parameter that is missing or that must be absent because the choice that switches it on
    was made otherwise, and a constraint that does not hold. The message lists every problem of
    the first stage that finds any, each with the name of the parameter, and says how to fix
    them (specification 3.6, 5.3).

    It is also a `ValueError`, like `ObservationError`, so that code that validates data with
    pydantic reports it with its location as it does any other value error.
    """
