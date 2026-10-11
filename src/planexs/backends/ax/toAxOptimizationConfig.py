"""Maps a planexs `Objective` to Ax's `objective`/`outcome_constraints` strings."""

from __future__ import annotations

from planexs.domain.Metric import Metric
from planexs.domain.Objective import Objective


def toAxOptimizationConfig(objective: Objective) -> tuple[str, list[str]]:
    """Maps a planexs `Objective` to the `(objective, outcome_constraints)`
    strings/string-list `Client.configure_optimization` expects, weighted or
    unweighted as appropriate.

    Ax's own convention (confirmed against
    https://ax.readthedocs.io/en/stable/api.html): a metric is maximized by
    default, and minimized by prepending its name with `"-"`.

    Three cases, matching `Objective`'s own docstring (design philosophy #3:
    single-objective is the N = 1 case of multi-objective, not a separate
    code path):

    - A single metric: `"-loss"` (minimize) or `"loss"` (maximize).
    - Several metrics, `weights=None`: a comma-separated multi-objective
      string, e.g. `"-cost, utility"`, for Pareto optimization with no
      preference between objectives.
    - Several metrics, `weights` set: a single scalarized (weighted-sum)
      objective string, e.g. `"-2.0 * cost + 1.0 * utility"`, where each
      metric's own `minimize` flips the sign of its coefficient.
    """
    if len(objective.metrics) == 1:
        objective_string = _signedMetricName(objective.metrics[0])
    elif objective.weights is None:
        objective_string = ", ".join(_signedMetricName(metric) for metric in objective.metrics)
    else:
        terms = [
            f"{-weight if metric.minimize else weight} * {metric.name}"
            for metric, weight in zip(objective.metrics, objective.weights, strict=True)
        ]
        objective_string = " + ".join(terms)

    outcome_constraint_strings = [
        outcome_constraint.toAxOutcomeConstraintString()
        for outcome_constraint in objective.outcome_constraints
    ]
    return objective_string, outcome_constraint_strings


def _signedMetricName(metric: Metric) -> str:
    """Names a metric the way Ax's objective strings do: a leading `-` means minimize.

    Args:
        metric: The metric.

    Returns:
        `"-name"` if it is minimized, `"name"` otherwise.
    """
    return f"-{metric.name}" if metric.minimize else metric.name
