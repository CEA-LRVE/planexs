"""Unit tests for `Objective`."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from planexs import Metric, Objective, OutcomeConstraint


class TestObjective:
    def test_single_metric_is_not_multi_objective(self) -> None:
        objective = Objective(metrics=[Metric(name="loss", minimize=True)])
        assert objective.is_multi_objective is False

    def test_several_metrics_is_multi_objective(self) -> None:
        objective = Objective(
            metrics=[
                Metric(name="cost", minimize=True),
                Metric(name="quality", minimize=False),
            ]
        )
        assert objective.is_multi_objective is True

    def test_requires_at_least_one_metric(self) -> None:
        with pytest.raises(ValidationError, match="at least one"):
            Objective(metrics=[])

    def test_rejects_duplicate_metric_names(self) -> None:
        with pytest.raises(ValidationError, match="duplicate"):
            Objective(
                metrics=[
                    Metric(name="loss", minimize=True),
                    Metric(name="loss", minimize=False),
                ]
            )

    def test_weights_must_match_metric_count(self) -> None:
        with pytest.raises(ValidationError, match="weight"):
            Objective(
                metrics=[
                    Metric(name="cost", minimize=True),
                    Metric(name="quality", minimize=False),
                ],
                weights=[1.0],
            )

    def test_matching_weights_are_accepted(self) -> None:
        objective = Objective(
            metrics=[
                Metric(name="cost", minimize=True),
                Metric(name="quality", minimize=False),
            ],
            weights=[2.0, 1.0],
        )
        assert objective.weights == [2.0, 1.0]

    def test_outcome_constraint_on_unknown_metric_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="not among its own metrics"):
            Objective(
                metrics=[Metric(name="loss", minimize=True)],
                outcome_constraints=[OutcomeConstraint("qps", 100.0, ">=")],
            )

    def test_outcome_constraint_on_known_metric_is_accepted(self) -> None:
        objective = Objective(
            metrics=[Metric(name="loss", minimize=True), Metric(name="qps", minimize=False)],
            outcome_constraints=[OutcomeConstraint("qps", 100.0, ">=")],
        )
        assert len(objective.outcome_constraints) == 1
