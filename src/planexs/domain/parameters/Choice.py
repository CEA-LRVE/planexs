"""A discrete (ordinal or categorical) dimension of the search space."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal, Self

from planexs.core.ComponentConfig import ComponentConfig
from planexs.domain.parameters.ChoiceConfig import ChoiceConfig
from planexs.domain.parameters.Parameter import Parameter
from planexs.domain.parameters.ParameterValueError import ParameterValueError
from planexs.domain.parameters.registerParameter import registerParameter

if TYPE_CHECKING:
    from planexs.domain.ParameterizationCheckConfig import ParameterizationCheckConfig

# `bool` first, `int` before `float`: see Parameter.default's comment.
ChoiceValue = bool | int | float | str


@registerParameter(
    "choice",
    config=ChoiceConfig,
    description="A discrete, ordinal or categorical, dimension of the search space.",
)
class Choice(Parameter):
    """A discrete dimension: ordinal or categorical, controlled by `is_ordered`.

    `dependent_parameters` maps a chosen value to the names of other
    `Parameter`s that only become part of the active search space when that
    value is picked: this is how conditional (hierarchical) search spaces are
    declared (FR14). See section 4.5.

    Attributes:
        values: The set of legal values for this dimension.
        parameter_type: The Python type of `values`' elements.
        is_ordered: Whether the values have a meaningful order (ordinal, e.g.
            "low"/"medium"/"high") or not (categorical). `None` leaves the
            decision to whatever consumes the parameter (the encoder's own
            rule, or a backend's).
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
            is_ordered: Whether the values have a meaningful order; `None` leaves the
                decision to whatever consumes the parameter.
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

    def validateValue(
        self, value: object, options: ParameterizationCheckConfig | None = None
    ) -> None:
        """Checks that `value` is one of the values of this choice.

        A value matches a level if it is equal to it and of the same family: a boolean for a
        boolean level, a string for a string level, a number for a numeric level. Python
        makes `True` equal to `1`, so a plain membership test would accept `1` for a boolean
        choice and `True` for an integer one, which are mistakes and are refused. `3.0`
        matches the level `3`: both are the number three.

        Args:
            value: The value as the caller gave it.
            options: Not used: the check is exact. Accepted so that every parameter type is
                checked through the same call.

        Raises:
            ParameterValueError: if the value matches none of the levels. The message lists
                them.
        """
        if not any(_isLevel(value, level) for level in self.values):
            raise ParameterValueError(
                f"The value {value!r} of the choice {self.name!r} is not one of its values "
                f"{list(self.values)!r}.",
                component=f"parameter {self.name!r}",
                field_path=self.name,
                suggestion=f"give one of {list(self.values)!r}",
            )

    def activations(self) -> Mapping[Any, Sequence[str]]:
        """Which parameters each value of this choice switches on (FR14).

        Returns:
            The `dependent_parameters` of the choice: value to the names of the parameters
            that only exist when it is chosen. Empty if it has none.
        """
        return self.dependent_parameters or {}

    @classmethod
    def fromConfig(cls, config: ComponentConfig) -> Self:
        """The builder of the `choice` type: makes the choice that a validated
        `ChoiceConfig` describes.

        The stored `[value, [names...]]` pairs of `dependent_parameters` become the mapping
        that the constructor takes; every other field is passed on as it is.

        Args:
            config: The validated configuration, a `ChoiceConfig`.

        Returns:
            The choice.
        """
        fields = config.model_dump(exclude={"type"})
        pairs = fields["dependent_parameters"]
        fields["dependent_parameters"] = (
            {value: list(names) for value, names in pairs} if pairs is not None else None
        )
        return cls(**fields)

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
            "type": "choice",
            "name": self.name,
            "values": list(self.values),
            "parameter_type": self.parameter_type,
            "is_ordered": self.is_ordered,
            "dependent_parameters": dependent,
            "default": self.default,
        }


def _family(value: object) -> str | None:
    """The family of a parameter value: boolean, string or number (`None` for anything else).

    `bool` is tested first because it is an `int` in Python.
    """
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, str):
        return "str"
    if isinstance(value, (int, float)):
        return "number"
    return None


def _isLevel(value: object, level: object) -> bool:
    """Whether `value` is the level `level`: of the same family, and equal."""
    family = _family(value)
    return family is not None and family == _family(level) and bool(value == level)
