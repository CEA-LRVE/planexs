"""Parity data for the goals: legacy objectives, and the `optimize_v1` goal each one becomes
(roadmap task P4-03).

The goal replaces the roles of `LegacyObjective`, `LegacyMetric` and `LegacyOutcomeConstraint`.
Later tasks compare what the legacy layer computed with what the registered components compute
(the acquisitions of Phase 5, for instance), and they need the same problem stated both ways.
This module keeps that data in one place: the legacy objectives that cover every shape the
legacy layer handled, and the conversion of each into the configuration of an `optimize_v1`
goal. Kept in a module of its own and not in `conftest.py`, so that the test files can import
what is not a fixture. It is the one place of the new tests that imports the legacy types,
which is allowed to tests (`scripts/layers.toml`) and which no source module may do.
"""

from __future__ import annotations

from planexs import Real, SearchSpace
from planexs.domain.LegacyMetric import LegacyMetric
from planexs.domain.LegacyObjective import LegacyObjective
from planexs.domain.LegacyOutcomeConstraint import LegacyOutcomeConstraint
from planexs.domain.Outcome import Outcome
from planexs.domain.Problem import Problem
from planexs.goals.OptimizeGoalConfig import OptimizeGoalConfig
from planexs.goals.OutcomeConstraint import OutcomeConstraint

#: One legacy objective for each shape that the legacy layer handled, by a readable name. The
#: tags a goal reports for each are the same ones that the legacy layer's behaviour implies:
#: `AlphaAcquisitionStrategy` used its multi-objective acquisition for several metrics with no
#: weights and its single-objective one for everything else.
LEGACY_OBJECTIVES: dict[str, LegacyObjective] = {
    "single_minimize": LegacyObjective([LegacyMetric("loss", minimize=True)]),
    "single_maximize": LegacyObjective([LegacyMetric("score", minimize=False)]),
    "pareto": LegacyObjective(
        [LegacyMetric("cost", minimize=True), LegacyMetric("utility", minimize=False)]
    ),
    "weighted": LegacyObjective(
        [LegacyMetric("cost", minimize=True), LegacyMetric("utility", minimize=False)],
        weights=[2.0, 1.0],
    ),
    "single_constrained": LegacyObjective(
        [LegacyMetric("score", minimize=False), LegacyMetric("latency", minimize=True)],
        weights=[1.0, 3.0],
        outcome_constraints=[LegacyOutcomeConstraint("latency", 50.0, "<=")],
    ),
    "pareto_constrained": LegacyObjective(
        [LegacyMetric("cost", minimize=True), LegacyMetric("utility", minimize=False)],
        outcome_constraints=[
            LegacyOutcomeConstraint("utility", 0.5, ">="),
            LegacyOutcomeConstraint("cost", 10.0, "<="),
        ],
    ),
}


def goalConfigFromLegacy(objective: LegacyObjective) -> OptimizeGoalConfig:
    """The `optimize_v1` configuration that states what a legacy objective stated.

    A metric becomes an outcome with a direction, the weights become a mapping from outcome
    name to weight (the legacy list was in the order of the metrics), and each outcome
    constraint becomes an `OutcomeConstraint`.

    Args:
        objective: The legacy objective.

    Returns:
        The goal configuration.

    Raises:
        ValueError: if a constraint is relative to a baseline, which the goal does not
            support (the legacy layer only reserved the shape for it).
    """
    metrics = objective.metrics
    constraints: list[OutcomeConstraint] = []
    for constraint in objective.outcome_constraints:
        if constraint.relative:
            raise ValueError(
                f"The constraint on {constraint.metric_name!r} is relative to a baseline "
                "trial, which an optimize goal does not support."
            )
        constraints.append(
            OutcomeConstraint(
                outcome=constraint.metric_name,
                comparator=constraint.comparator,
                bound=constraint.bound,
            )
        )
    weights = (
        None
        if objective.weights is None
        else {
            metric.name: weight
            for metric, weight in zip(metrics, objective.weights, strict=True)
        }
    )
    return OptimizeGoalConfig(
        outcomes={
            metric.name: "minimize" if metric.minimize else "maximize" for metric in metrics
        },
        weights=weights,
        outcome_constraints=constraints,
    )


def problemFor(objective: LegacyObjective) -> Problem:
    """A problem that has a continuous outcome for each metric of a legacy objective.

    Args:
        objective: The legacy objective.

    Returns:
        A problem with one real parameter and the outcomes of the metrics, in their order.
    """
    return Problem(
        search_space=SearchSpace([Real("x", 0.0, 1.0)]),
        outcomes=[
            Outcome(name=metric.name, type="continuous") for metric in objective.metrics
        ],
    )
