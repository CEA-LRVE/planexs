"""Unit tests for `toAxOptimizationConfig`.

Pure string-rendering logic over planexs' own `Metric`/`Objective` domain
objects: does not import `ax.*`, so these tests run without a real Ax
install.
"""

from __future__ import annotations

from planexs import Metric, Objective, OutcomeConstraint
from planexs.backends.ax.toAxOptimizationConfig import toAxOptimizationConfig


class TestSingleObjective:
    def test_minimize_is_prefixed_with_a_minus_sign(self) -> None:
        objective = Objective(metrics=[Metric(name="loss", minimize=True)])
        objective_string, _ = toAxOptimizationConfig(objective)
        assert objective_string == "-loss"

    def test_maximize_has_no_prefix(self) -> None:
        objective = Objective(metrics=[Metric(name="accuracy", minimize=False)])
        objective_string, _ = toAxOptimizationConfig(objective)
        assert objective_string == "accuracy"


class TestUnweightedMultiObjective:
    def test_renders_a_comma_separated_pareto_objective(self) -> None:
        objective = Objective(
            metrics=[
                Metric(name="cost", minimize=True),
                Metric(name="quality", minimize=False),
            ]
        )
        objective_string, _ = toAxOptimizationConfig(objective)
        assert objective_string == "-cost, quality"


class TestWeightedMultiObjective:
    def test_renders_a_scalarized_weighted_sum(self) -> None:
        objective = Objective(
            metrics=[
                Metric(name="cost", minimize=True),
                Metric(name="quality", minimize=False),
            ],
            weights=[2.0, 1.0],
        )
        objective_string, _ = toAxOptimizationConfig(objective)
        assert objective_string == "-2.0 * cost + 1.0 * quality"


class TestOutcomeConstraints:
    def test_absolute_bound_is_rendered_plainly(self) -> None:
        objective = Objective(
            metrics=[Metric(name="loss", minimize=True), Metric(name="qps", minimize=False)],
            outcome_constraints=[OutcomeConstraint("qps", 100.0, ">=")],
        )
        _, outcome_constraint_strings = toAxOptimizationConfig(objective)
        assert outcome_constraint_strings == ["qps >= 100.0"]

    def test_relative_bound_is_rendered_as_a_multiple_of_baseline(self) -> None:
        objective = Objective(
            metrics=[Metric(name="loss", minimize=True), Metric(name="qps", minimize=False)],
            outcome_constraints=[OutcomeConstraint("qps", 0.95, ">=", relative=True)],
        )
        _, outcome_constraint_strings = toAxOptimizationConfig(objective)
        assert outcome_constraint_strings == ["qps >= 0.95 * baseline"]

    def test_no_outcome_constraints_renders_an_empty_list(self) -> None:
        objective = Objective(metrics=[Metric(name="loss", minimize=True)])
        _, outcome_constraint_strings = toAxOptimizationConfig(objective)
        assert outcome_constraint_strings == []
