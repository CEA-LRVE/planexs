"""Raised when the native model of a surrogate does not answer as a surrogate needs."""

from __future__ import annotations

from planexs.core.PlanexsError import PlanexsError


class SurrogateModelError(PlanexsError):
    """The native model behind a `Surrogate` cannot be bound, or answered with something that
    a surrogate cannot pass on.

    Raised when a model has a different number of outputs than the outcomes it is bound to,
    when its posterior has a shape that does not fit the points asked about, and when a
    posterior mean, variance or sample holds a number that is not finite: such a value never
    propagates silently (specification 8). The component hint names the surrogate, the field
    hint names what is at fault, and the suggestion says how to fix it. It is a defect of the
    model or of the builder that made it, and not of the request.
    """
