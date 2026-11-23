"""Raised when a fitter cannot fit the model it is given."""

from __future__ import annotations

from planexs.core.PlanexsError import PlanexsError


class FitterError(PlanexsError):
    """A fitter was given a model it cannot fit, or its fit failed or left the model unusable.

    Raised for a model that the fitter does not know how to fit (an approximate model given
    to a fitter of exact ones), for a fit that the engine gave up on after every attempt, and
    for a fit that left a hyperparameter with a value that is not finite: such a value never
    propagates silently (specification 8). The component hint names the fitter, the field hint
    names what is at fault, and the suggestion says how to fix it.

    It is not a lack of data. A surrogate that has too few observations to be fitted at all
    raises `ModelNotReadyError` before any fitter is called (specification 5.6).
    """
