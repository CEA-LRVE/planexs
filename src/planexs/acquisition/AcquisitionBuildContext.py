"""The read-only information an acquisition builder receives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from planexs.acquisition.AcquisitionContextError import AcquisitionContextError
from planexs.domain.Problem import Problem
from planexs.encoding.SearchSpaceEncoder import SearchSpaceEncoder
from planexs.goals.AbstractGoal import AbstractGoal
from planexs.surrogates.Surrogate import Surrogate

#: Who the errors of this module are about.
_COMPONENT = "acquisition build context"


@dataclass(frozen=True, eq=False, kw_only=True)
class AcquisitionBuildContext:
    """Everything an acquisition builder is given to build a native acquisition function
    (specification 5.8, 6).

    An acquisition is a score over candidate points, built from a surrogate, a goal and the
    points already chosen (specification 4, "Acquisition"). The context holds exactly that,
    and a source of randomness: a builder reads it and returns the native acquisition function
    of the engine, and never reaches for anything that is not in it (a global random state, a
    study, a default). It is built with keywords,
    `AcquisitionBuildContext(surrogate=..., goal=..., generator=...)`.

    **What it holds.** `n` is the number of points of a set, and `d` the number of encoded
    columns of the encoder.

    - `surrogate` is the fitted `Surrogate`, which holds the native model (`surrogate.model`,
      the one the policy really uses, specification 5.13), the problem and the encoder.
    - `goal` says what is wanted: the directions and the weights of an optimize goal, the level
      of a level-set goal. It parameterizes the acquisition and the context does not check it
      against the surrogate: a goal is validated against its problem when the study is built
      (ADR-0013).
    - `pending_points` has shape `(n, d)`, in the unit cube: the points that were chosen for
      the current batch, or suggested earlier and not reported yet (specification 5.2, FR10).
      It is `None` when there are none, which is what the engine's own `X_pending` expects.
      An empty set of shape `(0, d)`, which a caller that collects the pending points into a
      tensor gets when none is pending, is read as `None`, so that "no pending point" has one
      representation.
    - `reference_points` has shape `(n, d)`, in the unit cube: a space-filling set of
      candidate points that spans the search space, for an acquisition that normalizes its
      terms over the space. It is `None` unless the builder asked for one. A builder asks by
      its class-level flag `needs_reference_points`, and says how many points it wants through
      its own configuration (`AbstractAcquisitionBuilder.referencePointCount`); the policy
      draws the set with the feasible sampler, so that every point satisfies the constraints.
    - `generator` is the seeded generator that every random choice of the build draws from.
      Drawing from it advances it. The study gives each build a generator of its own, derived
      for the ask and the component (specification 3.7, 5.11), so that no builder disturbs the
      stream of another.

    The problem and the encoder are not fields: they are the surrogate's own, and are read
    from it (`problem`, `encoder`), so that they can never disagree with the model that the
    acquisition is built on. The encoder is how a builder turns natural values into encoded
    points or draws more of them.

    **Checked when it is built.** The sets of points are checked once, here, so that a builder
    can rely on them and a mistake surfaces with the name of the field, and not as a shape
    error inside the engine (specification 3.6): each is a tensor of shape `(n, d)` with `d`
    the encoder's columns, in the precision of the encoder (BoTorch refuses to mix precisions),
    holding finite values in `[0, 1]` (specification 8: non-finite values never propagate
    silently). A reference set must not be empty: a set that normalizes nothing is a mistake.

    **Read-only.** The context is frozen: no field can be assigned, deleted or added. The
    tensors are shared with the caller and not copied, which is cheap and which a frozen
    dataclass cannot prevent from being abused, so a builder must not modify them in place.
    The generator is the one thing that is meant to change.

    Contexts are compared by identity (`eq=False`): equality of tensors is not a boolean.

    Attributes:
        surrogate: The fitted surrogate the acquisition is built on.
        goal: What is wanted, which parameterizes the acquisition.
        generator: The seeded generator that every random choice of the build draws from.
        pending_points: The encoded points chosen but not evaluated yet, shape `(n, d)`, or
            `None`.
        reference_points: The encoded space-filling set, shape `(n, d)`, or `None`.

    Raises:
        AcquisitionContextError: if a set of points is malformed in any of the ways above. The
            message names the field and says how to fix it.
    """

    surrogate: Surrogate
    goal: AbstractGoal[Any]
    generator: torch.Generator
    pending_points: Tensor | None = None
    reference_points: Tensor | None = None

    def __post_init__(self) -> None:
        """Checks the sets of points against the encoder, and reads an empty set of pending
        points as `None`.

        Raises:
            AcquisitionContextError: if a set of points is malformed.
        """
        pending = _checkedPoints("pending_points", self.pending_points, self.encoder)
        if pending is not None and pending.shape[0] == 0:
            pending = None
        reference = _checkedPoints("reference_points", self.reference_points, self.encoder)
        if reference is not None and reference.shape[0] == 0:
            raise _fault(
                "reference_points",
                "reference_points is empty: a set that normalizes nothing is a mistake.",
                "draw at least one point, or pass None when the builder does not need a "
                "reference set",
            )
        # Frozen dataclass: `object.__setattr__` is the sanctioned way to normalize.
        object.__setattr__(self, "pending_points", pending)

    @property
    def problem(self) -> Problem:
        """The problem the surrogate was fitted for (read from the surrogate, a stored value:
        cheap, snake_case).
        """
        return self.surrogate.problem

    @property
    def encoder(self) -> SearchSpaceEncoder:
        """The encoder of the problem's search space, which the model works on (read from the
        surrogate, a stored value: cheap, snake_case).
        """
        return self.surrogate.encoder


def _checkedPoints(
    field: str, points: Tensor | None, encoder: SearchSpaceEncoder
) -> Tensor | None:
    """Checks a set of encoded points against the encoder.

    Args:
        field: The name of the field that holds the points, for the errors.
        points: The set, or `None`.
        encoder: The encoder that the points are encoded with.

    Returns:
        The points, unchanged (not copied), or `None`.

    Raises:
        AcquisitionContextError: if the points are not a tensor of shape `(n, d)` in the
            precision of the encoder, hold a value that is not finite, or lie outside the unit
            cube.
    """
    if points is None:
        return None
    if not isinstance(points, Tensor):
        raise _fault(
            field,
            f"{field} must be a tensor or None; got {type(points).__name__}.",
            "encode the points with SearchSpaceEncoder.encodeParameters and stack the rows",
        )
    columns = encoder.n_columns
    if points.ndim != 2 or points.shape[1] != columns:
        raise _fault(
            field,
            f"{field} must have shape (n, {columns}), one column per encoded column of the "
            f"encoder; got {tuple(points.shape)}.",
            "encode each parameterization with SearchSpaceEncoder.encodeParameters and stack "
            "the rows",
        )
    if points.dtype != encoder.dtype:
        raise _fault(
            field,
            f"{field} has the dtype {points.dtype}, but the encoder works in {encoder.dtype}: "
            "BoTorch does not mix precisions.",
            f"convert it with .to({encoder.dtype})",
        )
    if not bool(torch.isfinite(points).all()):
        raise _fault(
            field,
            f"{field} holds a value that is not finite (nan or an infinity).",
            "check how the points were drawn or encoded",
        )
    if bool(((points < 0.0) | (points > 1.0)).any()):
        raise _fault(
            field,
            f"{field} holds a value outside [0, 1]: acquisitions work on the encoding of the "
            "search space, in the unit cube, and not on natural values.",
            "encode each parameterization with SearchSpaceEncoder.encodeParameters",
        )
    return points


def _fault(field: str, message: str, suggestion: str) -> AcquisitionContextError:
    """Builds the error for one malformed field of a build context.

    Args:
        field: The name of the field that is at fault.
        message: What is wrong.
        suggestion: How to fix it.

    Returns:
        The error, naming the context, the field and the way to fix it.
    """
    return AcquisitionContextError(
        message, component=_COMPONENT, field_path=field, suggestion=suggestion
    )
