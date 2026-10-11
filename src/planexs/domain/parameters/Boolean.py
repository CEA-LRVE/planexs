"""Sugar for a `Choice` restricted to `[True, False]`."""

from __future__ import annotations

from planexs.domain.parameters.Choice import Choice


class Boolean(Choice):
    """`Boolean(name)` is sugar for
    `Choice(name, [True, False], parameter_type="bool", is_ordered=False)`.
    """

    def __init__(self, name: str, default: bool | None = None) -> None:
        """Creates a boolean parameter.

        Args:
            name: The parameter's name.
            default: Optional default value.
        """
        super().__init__(
            name=name,
            values=[True, False],
            parameter_type="bool",
            is_ordered=False,
            dependent_parameters=None,
            default=default,
        )
