"""Sugar for a `Range` fixed to `parameter_type="int"`."""

from __future__ import annotations

from typing import Literal

from planexs.domain.parameters.Range import Range


class Integer(Range):
    """`Integer(name, min_value, max_value)` is sugar for
    `Range(name, (min_value, max_value), parameter_type="int")`.
    """

    def __init__(
        self,
        name: str,
        min_value: int,
        max_value: int,
        step_size: int | None = None,
        scaling: Literal["linear", "log"] | None = None,
        default: int | None = None,
    ) -> None:
        """Creates an integer range.

        Args:
            name: The parameter's name.
            min_value: Lower bound, inclusive.
            max_value: Upper bound, inclusive.
            step_size: Optional spacing of the legal values.
            scaling: `"linear"` (default) or `"log"`.
            default: Optional default value.
        """
        super().__init__(
            name=name,
            bounds=(float(min_value), float(max_value)),
            parameter_type="int",
            step_size=float(step_size) if step_size is not None else None,
            scaling=scaling,
            default=default,
        )
