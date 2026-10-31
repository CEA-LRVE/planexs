"""One or more metrics with optional relative importance weights."""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, model_validator

from planexs.domain.LegacyMetric import LegacyMetric
from planexs.domain.LegacyOutcomeConstraint import LegacyOutcomeConstraint


class LegacyObjective(BaseModel):
    """One or more metrics with optional relative importance weights and
    optional constraints on the metrics themselves.

    `weights=None` with a single metric is the plain single-objective case.
    `weights=None` with several metrics means no preference between them (a
    standard Pareto multi-objective problem). Explicit weights scalarize the
    preference between objectives; they do not control exploration versus
    exploitation, which is `BayesianOptimizer.ask`'s `alpha` (section 4.3).

    Single-objective is the N = 1 case of multi-objective, not a separate
    code path (design philosophy #3): there is exactly one `LegacyObjective`
    concept in the codebase, used identically for N = 1 and N > 1.

    Legacy type of the 0.1.0 architecture, kept only until roadmap task P11-03 deletes it.
    New code must not use it.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    metrics: list[LegacyMetric]
    weights: list[float] | None = None
    outcome_constraints: list[LegacyOutcomeConstraint] = []

    def __init__(
        self,
        metrics: Sequence[LegacyMetric],
        weights: Sequence[float] | None = None,
        outcome_constraints: Sequence[LegacyOutcomeConstraint] | None = None,
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
    def _validateShape(self) -> LegacyObjective:
        """Checks the objective is coherent.

        Returns:
            The objective itself.

        Raises:
            ValueError: if there is no metric, metric names repeat, the weight count
                differs from the metric count, or a constraint names a metric the
                objective does not have.
        """
        if not self.metrics:
            raise ValueError("LegacyObjective requires at least one LegacyMetric.")
        names = [metric.name for metric in self.metrics]
        duplicates = {name for name in names if names.count(name) > 1}
        if duplicates:
            raise ValueError(
                f"LegacyObjective has duplicate metric names: {sorted(duplicates)!r}."
            )
        if self.weights is not None and len(self.weights) != len(self.metrics):
            raise ValueError(
                f"LegacyObjective has {len(self.metrics)} metric(s) but "
                f"{len(self.weights)} weight(s); these must match one-to-one."
            )
        constraint_names = {constraint.metric_name for constraint in self.outcome_constraints}
        unknown = constraint_names - set(names)
        if unknown:
            raise ValueError(
                f"LegacyObjective has outcome_constraints referencing metric(s) not "
                f"among its own metrics: {sorted(unknown)!r}."
            )
        return self

    @property
    def is_multi_objective(self) -> bool:
        """Cheap (len check): snake_case."""
        return len(self.metrics) > 1
