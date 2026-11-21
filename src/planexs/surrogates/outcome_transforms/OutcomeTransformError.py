"""Raised when an outcome transform cannot be built or cannot take the outcomes it is given."""

from __future__ import annotations

from planexs.core.PlanexsError import PlanexsError


class OutcomeTransformError(PlanexsError):
    """An outcome transform is faulty, or was given outcomes that it cannot transform.

    Raised when the builder of a transform returns something that is not a native outcome
    transform, and when a transform meets a value outside its domain (the logarithm of a value
    that is not strictly positive). The second case is raised by the transform itself, as soon
    as a model applies it, so that such a value never reaches the fit as a `-inf` or a `nan`
    (specification 8: non-finite values never propagate silently). The component hint names
    the transform, the field hint names what is at fault, and the suggestion says how to fix
    it.
    """
