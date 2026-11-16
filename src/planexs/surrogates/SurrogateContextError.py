"""Raised when the data given to a surrogate builder is malformed."""

from __future__ import annotations

from planexs.core.PlanexsError import PlanexsError


class SurrogateContextError(PlanexsError):
    """The data of a surrogate build context does not fit its problem or its runtime settings.

    Raised when a `SurrogateBuildContext` is built, and not deep inside an engine call
    (specification 3.6): a tensor of the wrong shape, of another precision or on another device
    than the runtime settings name, a value that is not finite, or an input that is not in the
    unit cube of the encoding. It is also what `trialsToTensors` raises when the trials cannot
    be turned into those tensors: a completed trial without a value for an outcome, parameters
    that the encoder cannot encode, or partial noise under the `error` noise policy. The
    component hint says which context, trial or setting it is, the field hint names the field
    that is at fault, and the suggestion says how to fix it.

    It is not a lack of data. A context with too few observations is legal, and the builder
    that needs more raises `ModelNotReadyError` (specification 5.6).
    """
