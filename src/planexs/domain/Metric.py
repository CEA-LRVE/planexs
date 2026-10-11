"""One measured quantity and its optimization direction."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class Metric(BaseModel):
    """One measured quantity and its optimization direction.

    Attributes:
        name: The metric's name, matching the key used in `tell()`'s `y`
            dictionary.
        minimize: `True` to minimize this metric, `False` to maximize it.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    minimize: bool

    def __init__(self, name: str, minimize: bool) -> None:
        """Creates a metric.

        Args:
            name: The metric's name.
            minimize: `True` to minimize this metric, `False` to maximize it.
        """
        super().__init__(name=name, minimize=minimize)
