"""A read-only quantity computed from other parameters."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from planexs.domain.parameters.DerivedConfig import DerivedConfig
from planexs.domain.parameters.isNumber import isNumber
from planexs.domain.parameters.Parameter import Parameter
from planexs.domain.parameters.ParameterValueError import ParameterValueError
from planexs.domain.parameters.registerParameter import registerParameter

if TYPE_CHECKING:
    from planexs.domain.ParameterizationCheckConfig import ParameterizationCheckConfig


@registerParameter(
    "derived",
    config=DerivedConfig,
    description="A read-only quantity computed from other parameters, never searched.",
)
class Derived(Parameter):
    """A read-only quantity computed from other parameters through an
    expression string (e.g. `"a + b"`), not itself searched over.

    Useful for referencing a computed quantity in a `LinearConstraint` or
    reading it back from a `LegacyTrial` without recomputing it outside planexs.

    A `Derived` parameter never has a `default`: it is computed, not chosen,
    so `default` always stays `None` (enforced by `_validate_default`).

    Attributes:
        expression: The expression computing this parameter's value from
            other parameters' names.
        parameter_type: The Python type this expression evaluates to.
    """

    expression: str
    parameter_type: Literal["float", "int", "str", "bool"]

    def __init__(
        self,
        name: str,
        expression: str,
        parameter_type: Literal["float", "int", "str", "bool"],
        default: None = None,
    ) -> None:
        """Creates a derived parameter.

        Args:
            name: The parameter's name.
            expression: The expression computing this parameter's value from others.
            parameter_type: The type the expression evaluates to.
            default: Always `None`: a derived parameter is computed, never chosen.
        """
        # `default` exists only so `Derived.model_validate(d.model_dump())`
        # works (a dump contains `default=None`); anything but None is
        # rejected by `_validate_default`.
        super().__init__(
            name=name,
            expression=expression,
            parameter_type=parameter_type,
            default=default,
        )

    def _validate_default(self) -> None:
        """Checks there is no default and the expression is not empty.

        Raises:
            ValueError: if a default is set or the expression is blank.
        """
        if self.default is not None:
            raise ValueError(
                f"Derived parameter {self.name!r} cannot have a default value: "
                "it is computed from other parameters, not chosen."
            )
        if not self.expression.strip():
            raise ValueError(f"Derived parameter {self.name!r} has an empty expression.")

    @property
    def is_computed(self) -> bool:
        """`True`: the value of a derived parameter is computed from the others, so a caller
        who reports a point does not have to give it. Cheap (no work), snake_case.
        """
        return True

    def validateValue(
        self, value: object, options: ParameterizationCheckConfig | None = None
    ) -> None:
        """Checks that `value` has the type that this parameter's expression evaluates to.

        Only the type is checked: a `float` or an `int` is a finite number (an `int` a whole
        one), a `bool` is a `bool`, a `str` is a string. The value is not recomputed from the
        other parameters to see that it is the right one: that would need a tolerance of its
        own, and the study computes it where it is needed.

        Args:
            value: The value as the caller gave it.
            options: Not used: the check is exact. Accepted so that every parameter type is
                checked through the same call.

        Raises:
            ParameterValueError: if the value is not of the type.
        """
        legal: bool
        if self.parameter_type == "bool":
            legal = isinstance(value, bool)
        elif self.parameter_type == "str":
            legal = isinstance(value, str)
        else:
            legal = isNumber(value) and (
                self.parameter_type == "float" or float(value).is_integer()
            )
        if not legal:
            raise ParameterValueError(
                f"The value {value!r} of the derived parameter {self.name!r} is not "
                f"{_describeType(self.parameter_type)}.",
                component=f"parameter {self.name!r}",
                field_path=self.name,
                suggestion="leave the parameter out: it is computed from the others",
            )

    def toDict(self) -> dict[str, Any]:
        """Serializes the parameter.

        Returns:
            A type-tagged, JSON-compatible form, rebuilt by `parameterFromDict`.
        """
        return {
            "type": "derived",
            "name": self.name,
            "expression": self.expression,
            "parameter_type": self.parameter_type,
        }


def _describeType(parameter_type: str) -> str:
    """The type of a derived parameter as a phrase for a message."""
    return {
        "float": "a finite number",
        "int": "a whole number",
        "bool": "a bool",
        "str": "a string",
    }[parameter_type]
