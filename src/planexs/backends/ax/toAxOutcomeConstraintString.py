"""Renders a planexs `LegacyOutcomeConstraint` as one of Ax's `outcome_constraints` strings."""

from __future__ import annotations

from planexs.domain.LegacyOutcomeConstraint import LegacyOutcomeConstraint


def toAxOutcomeConstraintString(constraint: LegacyOutcomeConstraint) -> str:
    """Renders `constraint` as one entry of the `outcome_constraints` sequence that
    `Client.configure_optimization` expects, for example `"qps >= 100"` or, when
    `relative` is set, `"qps >= 0.95 * baseline"`.

    Args:
        constraint: The constraint on an observed metric.

    Returns:
        The constraint as the string-expression mini-language of Ax.
    """
    if constraint.relative:
        return (
            f"{constraint.metric_name} {constraint.comparator} {constraint.bound} * baseline"
        )
    return f"{constraint.metric_name} {constraint.comparator} {constraint.bound}"
