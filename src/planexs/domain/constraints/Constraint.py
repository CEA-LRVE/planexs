"""Base class for parameter-level constraints."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Self

from pydantic import BaseModel, ConfigDict

from planexs.core.ComponentConfig import ComponentConfig


class Constraint(BaseModel, ABC):
    """Base class for parameter-level constraints (see `LegacyOutcomeConstraint`
    for metric-level ones, which are not a `Constraint` subclass since they
    apply to a different thing entirely: an observed value, not a decision
    variable).

    `LinearConstraint` (FR9) is enforced by the optimization backend itself,
    except for its `"="` form. `NonlinearConstraint` (FR17) is only
    enforceable by `planexs`'s own custom acquisition layer (section 4.6,
    ADR-0006, ADR-0008); `SearchSpace.constraints: Sequence[Constraint]`
    accepts any future `Constraint` subclass without changes.

    It is also the contract of the `constraint` kind of components (specification 5.3, 6). A
    constraint type is a concrete subclass that is registered under its type name with
    `registerConstraint`, together with the configuration schema that is its stored form:

    - `toDict` writes that form, with the type name under the key `type`;
    - `fromConfig` is the builder: it makes the constraint from the validated configuration;
    - `constraintFromDict` reads the form back through the registry, so a type that someone
      registers (a plugin, for example) is saved and reloaded like the built-in ones.
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
        """Whether the optimization backend's default generation strategy cannot
        enforce this constraint, so that only `planexs`'s own acquisition layer can
        (`BayesianOptimizer.ask()` switches to it, with a warning, when the
        caller leaves `alpha` unset). `False` unless a subclass says otherwise.
        Cheap (no work), snake_case.
        """
        return False

    @classmethod
    def fromConfig(cls, config: ComponentConfig) -> Self:
        """The builder of a registered constraint type: makes the constraint that a validated
        configuration describes.

        This default passes every field of the configuration, except `type`, to the
        constructor as a keyword argument, which is right when the fields of the
        configuration schema are those of the constructor. A type whose stored form differs
        overrides it. The constructor runs the validation that a schema cannot express alone,
        such as a grammar or a non-empty list.

        Args:
            config: The configuration, an instance of the schema that the type is registered
                with.

        Returns:
            The constraint, an instance of the class this is called on.
        """
        return cls(**config.model_dump(exclude={"type"}))

    @abstractmethod
    def toDict(self) -> dict[str, Any]:
        """Explicit, type-tagged, JSON-compatible form of this constraint.

        The `"type"` entry (`"linear"` or `"nonlinear"`) is the name that the constraint type
        is registered under: it lets `constraintFromDict` rebuild the right class on reload, by
        looking it up in the registry of the `constraint` kind. The rest of the form is what
        the configuration schema of that type accepts, and nothing else.

        Same reason as `Parameter.toDict` for being explicit: `SearchSpace` holds a list of
        the abstract base type.
        """
