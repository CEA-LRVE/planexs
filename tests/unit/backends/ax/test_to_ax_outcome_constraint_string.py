"""Unit tests for `toAxOutcomeConstraintString`.

Pure string rendering over planexs' own `LegacyOutcomeConstraint`: does not import `ax.*`, so
these tests run without a real Ax install. `toAxOptimizationConfig` is tested separately, on
whole objectives.
"""

from __future__ import annotations

from planexs import LegacyOutcomeConstraint
from planexs.backends.ax.toAxOutcomeConstraintString import toAxOutcomeConstraintString


class TestToAxOutcomeConstraintString:
    def test_absolute_lower_bound(self) -> None:
        constraint = LegacyOutcomeConstraint("qps", 100.0, ">=")
        assert toAxOutcomeConstraintString(constraint) == "qps >= 100.0"

    def test_absolute_upper_bound(self) -> None:
        constraint = LegacyOutcomeConstraint("latency", 0.25, "<=")
        assert toAxOutcomeConstraintString(constraint) == "latency <= 0.25"

    def test_relative_bound_is_a_multiple_of_the_baseline(self) -> None:
        constraint = LegacyOutcomeConstraint("qps", 0.95, ">=", relative=True)
        assert toAxOutcomeConstraintString(constraint) == "qps >= 0.95 * baseline"
