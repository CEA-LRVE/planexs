"""One suggested or evaluated point of a study, with its lifecycle state."""

from __future__ import annotations

import math
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    field_serializer,
    field_validator,
    model_validator,
)

from planexs.core.TrialStateError import TrialStateError
from planexs.domain.TrialState import TrialState

# `bool` first, `int` before `float`: see `Parameter.default`'s comment.
_ParameterValue = bool | int | float | str


class Trial(BaseModel):
    """One parameterization with an identity and a state (specification 4, 5.4; ADR-0012).

    A trial is an immutable record: no attribute can be assigned, and the mappings it holds
    are read-only copies, so neither the caller of the constructor nor a reader of the trial
    can rewrite the history through it. A move to another state is a method that returns a
    new trial with the same `id` and leaves this one untouched (`tell`, `fail`, `abandon`, or
    `transitionTo` for any move). The record knows nothing of a problem or a backend: checking
    the parameterization against a problem is the study's job.

    Creating a trial is not a transition, so a trial can be built in any state: the study
    creates pending trials, `attach` creates completed or failed ones, and a snapshot is read
    back into trials in every state. Only `reason` is tied to the state here: a failed trial
    must carry a non-blank reason, and no other trial may.

    A trial is built with keyword arguments, `Trial(id=0, parameters={"x": 1.0})`, and has no
    `__init__` of its own, so that a malformed document read back from a file is always
    reported as a pydantic `ValidationError` (a `ValueError`) with the path of the bad field,
    never as a `TypeError`.

    The `observation` slot is a JSON-compatible mapping for now. P2-08 types it as an
    `Observation` and ties it to the state (a completed trial has one, the others have none).

    Mappings are stored deeply read-only, and a JSON array inside one is read back as a
    tuple, so `trial.metadata == {"k": [1]}` is `False` while
    `trial.metadata == {"k": (1,)}` is `True`. A non-finite number in `parameters`,
    `observation` or `metadata` is rejected: JSON has none, and writing it as `null` would
    change the history on save.

    Attributes:
        id: The identifier assigned by the study, a non-negative integer never reused.
        parameters: Maps parameter name to the value this trial uses.
        state: The lifecycle state. A new trial is pending unless it is given another.
        observation: The observed values, once the trial is completed. `None` otherwise.
        metadata: Free-form JSON-compatible data the caller attaches to the trial.
        reason: Why the evaluation failed. Required when failed, `None` otherwise.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: int = Field(strict=True, ge=0)
    parameters: Mapping[str, _ParameterValue]
    state: TrialState = TrialState.pending
    observation: Mapping[str, JsonValue] | None = None
    metadata: Mapping[str, JsonValue] = Field(default_factory=dict, validate_default=True)
    reason: str | None = None

    @field_validator("parameters", "observation", "metadata", mode="before")
    @classmethod
    def _thawContainers(cls, value: Any) -> Any:
        """Hands pydantic plain dicts and lists, so a trial built from another trial's
        read-only mappings is validated like any other input.
        """
        return _thaw(value)

    @field_validator("parameters", "observation", "metadata", mode="after")
    @classmethod
    def _freezeContainers(cls, value: Any) -> Any:
        """Stores a validated mapping as a deeply read-only copy of itself."""
        return _freeze(value)

    @field_serializer("parameters", "observation", "metadata")
    def _dumpContainers(self, value: Any) -> Any:
        """Writes the read-only mappings back as plain JSON-compatible data."""
        return _thaw(value)

    @model_validator(mode="after")
    def _validateReason(self) -> Trial:
        """Ties the reason to the state: required when failed, forbidden otherwise.

        Returns:
            The trial itself.

        Raises:
            ValueError: for a failed trial without a non-blank reason, or a reason on a trial
                that is not failed.
        """
        if self.state is TrialState.failed:
            if self.reason is None or not self.reason.strip():
                raise ValueError(
                    f"Trial {self.id} is failed and needs a non-blank reason that says why "
                    "the evaluation did not produce an observation."
                )
        elif self.reason is not None:
            raise ValueError(
                f"Trial {self.id} is {self.state.value} but has a reason: only a failed trial "
                "carries one."
            )
        return self

    def transitionTo(
        self,
        target: TrialState,
        *,
        observation: Mapping[str, JsonValue] | None = None,
        reason: str | None = None,
    ) -> Trial:
        """Moves the trial to `target`, if the table of ADR-0012 allows it.

        This is the one place the table is enforced: `tell`, `fail` and `abandon` call it. The
        trial itself is not changed.

        Args:
            target: The state to move to.
            observation: The observed values, for a move to completed.
            reason: Why the evaluation failed, for a move to failed.

        Returns:
            A new trial in `target` with the same `id`, parameters and metadata.

        Raises:
            TrialStateError: if the move is not one of the three legal ones, which includes
                every move out of a terminal state, a move to the same state and any move to
                pending.
            ValueError: if the new trial is invalid (a failed trial with a blank reason).
        """
        if not self.state.canMoveTo(target):
            raise _illegalMove(self, target)
        return Trial(
            id=self.id,
            parameters=self.parameters,
            state=target,
            observation=observation,
            metadata=self.metadata,
            reason=reason,
        )

    def tell(self, observation: Mapping[str, JsonValue]) -> Trial:
        """Completes a pending trial with its observation.

        Args:
            observation: The observed values.

        Returns:
            A new completed trial with the same `id`.

        Raises:
            TrialStateError: if the trial is not pending.
        """
        return self.transitionTo(TrialState.completed, observation=observation)

    def fail(self, reason: str) -> Trial:
        """Marks a pending trial as failed, with the reason.

        Args:
            reason: Why the evaluation did not produce an observation. Must not be blank.

        Returns:
            A new failed trial with the same `id`.

        Raises:
            TrialStateError: if the trial is not pending.
            ValueError: if `reason` is blank.
        """
        return self.transitionTo(TrialState.failed, reason=reason)

    def abandon(self) -> Trial:
        """Withdraws a pending trial.

        Returns:
            A new abandoned trial with the same `id`.

        Raises:
            TrialStateError: if the trial is not pending.
        """
        return self.transitionTo(TrialState.abandoned)

    def __hash__(self) -> int:
        """Hashes by `id`: equal trials have equal ids, and the read-only mappings a trial
        holds cannot be hashed themselves.
        """
        return hash(self.id)


def _illegalMove(trial: Trial, target: TrialState) -> TrialStateError:
    """Builds the error for a move the table of ADR-0012 does not allow.

    Args:
        trial: The trial that was asked to move.
        target: The state it was asked to move to.

    Returns:
        An error that names the trial, says why the move is illegal and what to do instead.
    """
    if trial.state.is_terminal:
        why = (
            f"{trial.state.value} is a terminal state, "
            "and a trial that has ended never changes"
        )
        fix = "record a repeated evaluation, a retry or a late result as a new trial"
    else:
        why = "a pending trial stays pending until it is reported"
        fix = "tell, fail or abandon the trial"
    return TrialStateError(
        f"Trial {trial.id} is {trial.state.value} and cannot become {target.value}: {why}.",
        component=f"trial {trial.id}",
        field_path="state",
        suggestion=fix,
    )


def _freeze(value: Any) -> Any:
    """Copies JSON-like data into read-only form: mappings become read-only views, lists become
    tuples.

    Args:
        value: Any JSON-like value.

    Returns:
        The frozen copy. Scalars are returned as they are.

    Raises:
        ValueError: if a float is not finite.
    """
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list | tuple):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(
            f"{value!r} is not JSON-compatible: a non-finite number would be written as null."
        )
    return value


def _thaw(value: Any) -> Any:
    """The inverse of `_freeze`: read-only mappings become dicts and tuples become lists.

    Args:
        value: Any JSON-like value, frozen or not.

    Returns:
        A plain, mutable copy made of dicts and lists.
    """
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_thaw(item) for item in value]
    return value
