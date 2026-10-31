"""One measured quantity and its optimization direction."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class LegacyMetric(BaseModel):
    """One measured quantity and its optimization direction.

    Legacy type of the 0.1.0 architecture, kept only until roadmap task P11-03 deletes it.
    New code must not use it.

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
