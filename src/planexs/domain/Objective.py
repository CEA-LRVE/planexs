"""One or more metrics with optional relative importance weights."""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, model_validator

from planexs.domain.Metric import Metric
from planexs.domain.OutcomeConstraint import OutcomeConstraint


class Objective(BaseModel):
    """One or more metrics with optional relative importance weights and
    optional constraints on the metrics themselves.

    `weights=None` with a single metric is the plain single-objective case.
    `weights=None` with several metrics means no preference between them (a
    standard Pareto multi-objective problem). Explicit weights scalarize the
    preference between objectives; they do not control exploration versus
    exploitation, which is `BayesianOptimizer.ask`'s `alpha` (section 4.3).

    Single-objective is the N = 1 case of multi-objective, not a separate
    code path (design philosophy #3): there is exactly one `Objective`
    concept in the codebase, used identically for N = 1 and N > 1.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    metrics: list[Metric]
    weights: list[float] | None = None
    outcome_constraints: list[OutcomeConstraint] = []

    def __init__(
        self,
        metrics: Sequence[Metric],
        weights: Sequence[float] | None = None,
        outcome_constraints: Sequence[OutcomeConstraint] | None = None,
    ) -> None:
        """Creates an objective from metrics, optional weights and outcome constraints.

        Args:
            metrics: The metrics, at least one.
            weights: Optional preference weights, one per metric.
            outcome_constraints: Optional constraints on the metrics.
        """
        super().__init__(
            metrics=list(metrics),
            weights=list(weights) if weights is not None else None,
            outcome_constraints=(
                list(outcome_constraints) if outcome_constraints is not None else []
            ),
        )

    @model_validator(mode="after")
    def _validateShape(self) -> Objective:
        """Checks the objective is coherent.

        Returns:
            The objective itself.

        Raises:
            ValueError: if there is no metric, metric names repeat, the weight count
                differs from the metric count, or a constraint names a metric the
                objective does not have.
        """
        if not self.metrics:
            raise ValueError("Objective requires at least one Metric.")
        names = [metric.name for metric in self.metrics]
        duplicates = {name for name in names if names.count(name) > 1}
        if duplicates:
            raise ValueError(f"Objective has duplicate metric names: {sorted(duplicates)!r}.")
        if self.weights is not None and len(self.weights) != len(self.metrics):
            raise ValueError(
                f"Objective has {len(self.metrics)} metric(s) but "
                f"{len(self.weights)} weight(s); these must match one-to-one."
            )
        constraint_names = {constraint.metric_name for constraint in self.outcome_constraints}
        unknown = constraint_names - set(names)
        if unknown:
            raise ValueError(
                f"Objective has outcome_constraints referencing metric(s) not "
                f"among its own metrics: {sorted(unknown)!r}."
            )
        return self

    @property
    def is_multi_objective(self) -> bool:
        """Cheap (len check): snake_case."""
        return len(self.metrics) > 1
