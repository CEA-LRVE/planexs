"""Raised when the data given to an acquisition builder is malformed or incomplete."""

from __future__ import annotations

from planexs.core.PlanexsError import PlanexsError


class AcquisitionContextError(PlanexsError):
    """The data of an acquisition build context does not fit the surrogate it is given with,
    or lacks what the builder needs.

    Raised when an `AcquisitionBuildContext` is built, and not deep inside an engine call
    (specification 3.6): a set of points of the wrong shape or precision, a value that is not
    finite, a point outside the unit cube of the encoding. It is also what a builder raises,
    through `AbstractAcquisitionBuilder.requireReferencePoints`, when it needs a reference set
    and the context holds none. The component hint says which context it is, the field hint
    names the field that is at fault, and the suggestion says how to fix it.
    """
