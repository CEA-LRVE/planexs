"""Base class for parameter-level constraints."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, ConfigDict


class Constraint(BaseModel, ABC):
    """Base class for parameter-level constraints (see `LegacyOutcomeConstraint`
    for metric-level ones, which are not a `Constraint` subclass since they
    apply to a different thing entirely: an observed value, not a decision
    variable).

    `LinearConstraint` (FR9) is enforced by Ax itself, except for its `"="`
    form. `NonlinearConstraint` (FR17) is only enforceable by `planexs`'s own
    custom acquisition layer (section 4.6, ADR-0006, ADR-0008);
    `SearchSpace.constraints: Sequence[Constraint]` accepts any future
    `Constraint` subclass without changes.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    def __init__(self, **data: Any) -> None:
        """Validates and stores the fields of a concrete constraint.

        Declared explicitly so that type checkers accept the keyword arguments each
        concrete subclass forwards from its own constructor (`coefficients`,
        `expression`, ...): without it they only know this base class, which has no
        fields. Pydantic still validates everything at runtime.

        Args:
            **data: The fields of the concrete subclass, as keyword arguments.
        """
        super().__init__(**data)

    @property
    def requires_custom_acquisition_layer(self) -> bool:
        """Whether Ax's default generation strategy cannot enforce this
        constraint, so that only `planexs`'s own acquisition layer can
        (`BayesianOptimizer.ask()` switches to it, with a warning, when the
        caller leaves `alpha` unset). `False` unless a subclass says otherwise.
        Cheap (no work), snake_case.
        """
        return False

    @abstractmethod
    def toDict(self) -> dict[str, Any]:
        """Explicit, type-tagged, JSON-compatible form of this constraint,
        rebuilt by `constraintFromDict`. Same reason as `Parameter.toDict`:
        `SearchSpace` holds a list of the abstract base type.
        """
