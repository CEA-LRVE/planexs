"""A discrete (ordinal or categorical) dimension of the search space."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

from planexs.domain.parameters.Parameter import Parameter

# `bool` first, `int` before `float`: see Parameter.default's comment.
ChoiceValue = bool | int | float | str


class Choice(Parameter):
    """A discrete dimension: ordinal or categorical, controlled by `is_ordered`.

    Field names and shape mirror `ax.api.configs.ChoiceParameterConfig`
    exactly, `dependent_parameters` included: a chosen value mapped to the
    names of other `Parameter`s that only become part of the active search
    space when that value is picked. This is Ax's own mechanism for
    conditional/hierarchical search spaces (FR14); planexs exposes it as-is
    rather than inventing a parallel one. See section 4.5.

    Attributes:
        values: The set of legal values for this dimension.
        parameter_type: The Python type of `values`' elements.
        is_ordered: Whether the values have a meaningful order (ordinal, e.g.
            "low"/"medium"/"high") or not (categorical). `None` leaves the
            choice to Ax's own default heuristic.
        dependent_parameters: Maps a value in `values` to the names of other
            `Parameter`s that only enter the active search space when this
            parameter takes that value (FR14).
    """

    values: list[bool] | list[int] | list[float] | list[str]
    parameter_type: Literal["float", "int", "str", "bool"]
    is_ordered: bool | None = None
    dependent_parameters: Mapping[ChoiceValue, Sequence[str]] | None = None

    def __init__(
        self,
        name: str,
        values: list[bool] | list[int] | list[float] | list[str],
        parameter_type: Literal["float", "int", "str", "bool"],
        is_ordered: bool | None = None,
        dependent_parameters: Mapping[ChoiceValue, Sequence[str]] | None = None,
        default: ChoiceValue | None = None,
    ) -> None:
        """Creates a discrete parameter.

        Args:
            name: The parameter's name.
            values: The legal values.
            parameter_type: The Python type of `values`' elements.
            is_ordered: Whether the values have a meaningful order; `None` leaves it to Ax.
            dependent_parameters: Optional map from a value to the names of parameters
                that only exist when it is chosen.
            default: Optional default value.
        """
        super().__init__(
            name=name,
            values=values,
            parameter_type=parameter_type,
            is_ordered=is_ordered,
            dependent_parameters=dependent_parameters,
            default=default,
        )

    def _validate_default(self) -> None:
        """Checks the values are usable and the default, if any, is one of them.

        Raises:
            ValueError: if `values` is empty or has duplicates, `dependent_parameters` is keyed
                on a value not in `values`, or the default is not one of the values.
        """
        if not self.values:
            raise ValueError(f"Choice {self.name!r} was given an empty `values` list.")
        if len(set(self.values)) != len(self.values):
            raise ValueError(f"Choice {self.name!r} has duplicate entries in `values`.")
        if self.dependent_parameters is not None:
            unknown = set(self.dependent_parameters.keys()) - set(self.values)
            if unknown:
                raise ValueError(
                    f"Choice {self.name!r} has dependent_parameters keyed on "
                    f"{sorted(unknown, key=str)!r}, which are not among its own "
                    f"values {self.values!r}."
                )
        if self.default is not None and self.default not in self.values:
            raise ValueError(
                f"Choice {self.name!r} has default={self.default!r}, which is "
                f"not one of its own values {self.values!r}."
            )

    def toAxKwargs(self) -> dict[str, Any]:
        """Field values ready to pass straight to
        `ax.api.configs.ChoiceParameterConfig(**kwargs)`. Deliberately excludes
        `default`, which Ax's own config has no field for.
        """
        return {
            "name": self.name,
            "values": self.values,
            "parameter_type": self.parameter_type,
            "is_ordered": self.is_ordered,
            "dependent_parameters": self.dependent_parameters,
        }

    def toDict(self) -> dict[str, Any]:
        """`dependent_parameters` is written as `[value, [names...]]` pairs,
        not a JSON object: JSON object keys are always strings, which would
        turn an `int`/`bool`/`float` key into text and break the "keys must
        be among `values`" invariant on reload.
        """
        dependent = (
            [[value, list(names)] for value, names in self.dependent_parameters.items()]
            if self.dependent_parameters is not None
            else None
        )
        return {
            "kind": "choice",
            "name": self.name,
            "values": list(self.values),
            "parameter_type": self.parameter_type,
            "is_ordered": self.is_ordered,
            "dependent_parameters": dependent,
            "default": self.default,
        }
