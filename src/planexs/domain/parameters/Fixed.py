"""A constant, non-optimized parameter, modeled as a single-value `Choice`."""

from __future__ import annotations

from typing import Literal

from planexs.domain.parameters.Choice import Choice, ChoiceValue


class Fixed(Choice):
    """A constant, non-optimized value that stays part of the declared
    parameterization (e.g. a setting recorded on every trial but never
    varied), matching Keras Tuner's `Fixed`.

    `ax.api.configs` has no separate `FixedParameterConfig` in the current
    `Client` API (only `Range`, `Choice`, `Derived`), though Ax's lower-level
    `parameter_from_config()` utility still names `FixedParameter` as a
    concept internally. The closest native equivalent, and what this maps to
    in `backends/ax`, is a single-value `Choice`; that mapping, not a
    reimplemented "fixed" concept, is what this class is.

    `default` is set equal to `value` automatically: since there is only one
    legal value, it is trivially also the default.
    """

    def __init__(self, name: str, value: ChoiceValue) -> None:
        """Creates a fixed, non-optimized parameter.

        Args:
            name: The parameter's name.
            value: The constant value. Its type decides the parameter type.

        Raises:
            TypeError: if the value is not a `bool`, `int`, `float` or `str`.
        """
        # `bool` is a subclass of `int` in Python, so it must be checked
        # before `int` or every boolean fixed value would be misclassified.
        # Branching with `isinstance` here (rather than a shared helper) lets
        # mypy narrow `value`'s type in each branch, so `[value]` matches the
        # exact `list[...]` member `values` expects.
        values: list[float] | list[int] | list[str] | list[bool]
        parameter_type: Literal["float", "int", "str", "bool"]
        if isinstance(value, bool):
            values, parameter_type = [value], "bool"
        elif isinstance(value, int):
            values, parameter_type = [value], "int"
        elif isinstance(value, float):
            values, parameter_type = [value], "float"
        elif isinstance(value, str):
            values, parameter_type = [value], "str"
        else:
            raise TypeError(f"Unsupported Fixed parameter value type: {type(value)!r}")

        super().__init__(
            name=name,
            values=values,
            parameter_type=parameter_type,
            is_ordered=None,
            dependent_parameters=None,
            default=value,
        )
