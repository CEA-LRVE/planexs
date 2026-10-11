"""How one searched parameter is laid out in the encoded (unit-cube) space."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

#: A `Choice` value as it appears in `Choice.values`.
LevelValue = bool | int | float | str


@dataclass(frozen=True)
class ParameterEncoding:
    """Describes how one searched parameter maps to columns of the encoded
    tensor `SearchSpaceEncoder` produces.

    Every encoded column lives in `[0, 1]`.

    Attributes:
        name: The parameter's name.
        kind: `"range"` for a `Range` (one column, linear or log scaled, with an
            optional grid), `"ordered"` for a `Choice` whose values have an
            order (one column holding the rank of the value), `"categorical"`
            for an unordered `Choice` (one 0/1 column for two values, a one-hot
            block of `len(levels)` columns otherwise).
        column: Index of this parameter's first encoded column.
        n_columns: Number of encoded columns it occupies.
        lower: For a range, its lower bound (integer-adjusted for an `int`
            range). Unused otherwise.
        upper: For a range, its upper bound. Unused otherwise.
        log_scale: Whether a range is encoded on a log scale.
        step: For a range, the spacing of its grid (`1` for an `int` range
            without an explicit step), or `None` for a continuous range.
        is_int: Whether a range's values are Python `int`.
        levels: For a choice, its values in rank order (`"ordered"`) or in the
            order given (`"categorical"`).
    """

    name: str
    kind: Literal["range", "ordered", "categorical"]
    column: int
    n_columns: int
    lower: float = 0.0
    upper: float = 1.0
    log_scale: bool = False
    step: float | None = None
    is_int: bool = False
    levels: tuple[LevelValue, ...] = ()

    @property
    def is_discrete_ordinal(self) -> bool:
        """Whether this parameter is ordered and discrete: an ordered choice or
        a range with a grid. These are optimized as continuous relaxations and
        then snapped (cheap: two attribute reads, snake_case).
        """
        return self.kind == "ordered" or (self.kind == "range" and self.step is not None)
