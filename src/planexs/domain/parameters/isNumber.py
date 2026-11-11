"""Tells whether a value is a number in the sense of a parameter value."""

from __future__ import annotations

import math
from typing import TypeGuard


def isNumber(value: object) -> TypeGuard[int | float]:
    """Whether `value` is a finite `int` or `float`.

    A `bool` is not a number here, although Python makes it an `int`: a flag that stands where
    a quantity belongs is a mistake, not a `1`. Neither is a string, whatever it spells, nor a
    numpy scalar that is not a `float`: convert it first. `nan` and the infinities are not
    values of any parameter.

    Args:
        value: Anything.

    Returns:
        `True` for a finite `int` or `float` that is not a `bool`. An `int` too large for a
        float is not finite in this sense, and is refused. It is a type guard: after a `True`
        the type checker knows that the value is an `int` or a `float`.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False
