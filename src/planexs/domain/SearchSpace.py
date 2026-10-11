"""The set of parameters and parameter-level constraints an optimizer
searches over.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, field_serializer, field_validator, model_validator

from planexs.domain.constraints.Constraint import Constraint
from planexs.domain.constraints.constraintFromDict import constraintFromDict
from planexs.domain.parameters.Parameter import Parameter
from planexs.domain.parameters.parameterFromDict import parameterFromDict


class SearchSpace(BaseModel):
    """The set of parameters and parameter-level constraints an optimizer
    searches over.

    Mutable by design (`addParameter`/`addConstraint`): a caller building up
    a study incrementally (e.g. from a config file processed one section at a
    time) does not have to collect every `Parameter` before constructing a
    `SearchSpace`. `BayesianOptimizer.__init__`'s "common case" builds one of
    these implicitly from a plain `Sequence[Parameter]`; see section 5.7.

    Serialization: `parameters` and `constraints` hold *abstract* base types,
    so they are (de)serialized explicitly through `toDict` /
    `parameterFromDict` / `constraintFromDict` (type-tagged dicts) rather
    than by Pydantic's declared-type handling, which would lose the subclass
    fields and could not rebuild an abstract class.
    """

    model_config = ConfigDict(extra="forbid")

    parameters: list[Parameter] = []
    constraints: list[Constraint] = []

    def __init__(
        self,
        parameters: Sequence[Parameter],
        constraints: Sequence[Constraint] | None = None,
    ) -> None:
        """Creates a search space from parameters and optional constraints.

        Args:
            parameters: The parameters. Names must be unique.
            constraints: Optional parameter-level constraints.

        Raises:
            ValueError: if two parameters share a name.
        """
        super().__init__(
            parameters=list(parameters),
            constraints=list(constraints) if constraints is not None else [],
        )

    @field_validator("parameters", mode="before")
    @classmethod
    def _rebuildParameters(cls, value: Any) -> Any:
        """Turns tagged dicts (a reloaded study) back into `Parameter`
        objects; already-built `Parameter` instances pass through untouched.
        """
        if isinstance(value, list):
            return [
                parameterFromDict(item) if isinstance(item, dict) else item for item in value
            ]
        return value

    @field_validator("constraints", mode="before")
    @classmethod
    def _rebuildConstraints(cls, value: Any) -> Any:
        """Turns tagged dicts (a reloaded study) back into `Constraint` objects.

        Args:
            value: The raw `constraints` field.

        Returns:
            The list with each dict rebuilt; built constraints pass through untouched.
        """
        if isinstance(value, list):
            return [
                constraintFromDict(item) if isinstance(item, dict) else item for item in value
            ]
        return value

    @field_serializer("parameters")
    def _dumpParameters(self, parameters: list[Parameter]) -> list[dict[str, Any]]:
        """Serializes the parameters through their own type-tagged `toDict`.

        Args:
            parameters: The parameters.

        Returns:
            One dict per parameter.
        """
        return [parameter.toDict() for parameter in parameters]

    @field_serializer("constraints")
    def _dumpConstraints(self, constraints: list[Constraint]) -> list[dict[str, Any]]:
        """Serializes the constraints through their own type-tagged `toDict`.

        Args:
            constraints: The constraints.

        Returns:
            One dict per constraint.
        """
        return [constraint.toDict() for constraint in constraints]

    @model_validator(mode="after")
    def _validateUniqueNames(self) -> SearchSpace:
        """Checks that no two parameters share a name.

        Returns:
            The search space itself.

        Raises:
            ValueError: on a duplicate name.
        """
        names = [parameter.name for parameter in self.parameters]
        duplicates = {name for name in names if names.count(name) > 1}
        if duplicates:
            raise ValueError(
                f"SearchSpace has duplicate parameter names: {sorted(duplicates)!r}."
            )
        return self

    def addParameter(self, parameter: Parameter) -> None:
        """Appends `parameter`, raising `ValueError` if its name collides
        with an existing one.
        """
        if parameter.name in self.parameter_names:
            raise ValueError(f"SearchSpace already has a parameter named {parameter.name!r}.")
        self.parameters.append(parameter)

    def addConstraint(self, constraint: Constraint) -> None:
        """Appends `constraint` to this search space's constraint list."""
        self.constraints.append(constraint)

    @property
    def parameter_names(self) -> list[str]:
        """Cheap (list comprehension over an already-held list): snake_case."""
        return [parameter.name for parameter in self.parameters]
