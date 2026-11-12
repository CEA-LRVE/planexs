"""What a surrogate builder returns: a fitted native model and how it was fitted."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from planexs.surrogates.FitStatus import FitStatus

if TYPE_CHECKING:
    from botorch.models.model import Model


@dataclass(frozen=True)
class BuiltSurrogate:
    """A fitted native model together with its fit status (specification 5.6).

    This is the result of `AbstractSurrogateBuilder.build`. The `Surrogate` wrapper (roadmap
    task P3-09) binds it to the encoding and the outcomes and offers the surrogate operations;
    until then this pair is all a builder has to deliver.

    It is a frozen record, and the model it holds is the native object itself, not a copy: it
    is the BoTorch model that the policy really uses and that the escape hatch returns
    (specification 5.13). Nothing here imports a concrete model class, so a builder may return
    any model of the engine, or of a plugin that satisfies the contract (specification 1.4).

    Attributes:
        model: The fitted native model, in evaluation mode.
        status: How it was fitted.
    """

    model: Model
    status: FitStatus
