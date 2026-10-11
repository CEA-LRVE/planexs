"""A constraint on an observed metric, rather than on a decision variable."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict


class OutcomeConstraint(BaseModel):
    """A constraint on an observed metric rather than on a parameter (e.g.
    `"qps >= 100"`), matching what `Client.configure_optimization`'s own
    `outcome_constraints` expects (FR15). Lives alongside `Objective`, not
    inside `SearchSpace`, because it constrains an observed outcome, not a
    decision variable.

    Attributes:
        metric_name: The name of the constrained metric.
        bound: The right-hand side of the (in)equality.
        comparator: `"<="` or `">="`.
        relative: When `True`, `bound` expresses a multiple of a baseline
            trial rather than an absolute value (e.g. `bound=0.95` with
            `comparator=">="` means "at least 95% of the baseline"). Wiring
            this to `Client.attach_baseline` is listed as deferred in section
            4.7; this field just reserves the shape for it. Using
            `relative=True` requires the caller to have attached a baseline
            trial themselves, e.g. via the `axClient` escape hatch
            (`axClient.attach_baseline(...)`), since planexs does not call
            `attach_baseline` on the caller's behalf in Phase 1.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    metric_name: str
    bound: float
    comparator: Literal["<=", ">="]
    relative: bool = False

    def __init__(
        self,
        metric_name: str,
        bound: float,
        comparator: Literal["<=", ">="],
        relative: bool = False,
    ) -> None:
        """Creates a constraint on an observed metric.

        Args:
            metric_name: The name of the constrained metric.
            bound: The right-hand side of the (in)equality.
            comparator: `"<="` or `">="`.
            relative: Whether `bound` is a multiple of a baseline trial rather than absolute.
        """
        super().__init__(
            metric_name=metric_name, bound=bound, comparator=comparator, relative=relative
        )

    def toAxOutcomeConstraintString(self) -> str:
        """Renders this constraint as one entry of the `outcome_constraints`
        sequence `Client.configure_optimization` expects, e.g.
        `"qps >= 100"` or, when `relative` is set, `"qps >= 0.95 * baseline"`.
        """
        if self.relative:
            return f"{self.metric_name} {self.comparator} {self.bound} * baseline"
        return f"{self.metric_name} {self.comparator} {self.bound}"
