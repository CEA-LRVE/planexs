"""Unit tests for `LegacyObjective`."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from planexs import LegacyMetric, LegacyObjective, LegacyOutcomeConstraint


class TestObjective:
    def test_single_metric_is_not_multi_objective(self) -> None:
        objective = LegacyObjective(metrics=[LegacyMetric(name="loss", minimize=True)])
        assert objective.is_multi_objective is False

    def test_several_metrics_is_multi_objective(self) -> None:
        objective = LegacyObjective(
            metrics=[
                LegacyMetric(name="cost", minimize=True),
                LegacyMetric(name="quality", minimize=False),
            ]
        )
        assert objective.is_multi_objective is True

    def test_requires_at_least_one_metric(self) -> None:
        with pytest.raises(ValidationError, match="at least one"):
            LegacyObjective(metrics=[])

    def test_rejects_duplicate_metric_names(self) -> None:
        with pytest.raises(ValidationError, match="duplicate"):
            LegacyObjective(
                metrics=[
                    LegacyMetric(name="loss", minimize=True),
                    LegacyMetric(name="loss", minimize=False),
                ]
            )

    def test_weights_must_match_metric_count(self) -> None:
        with pytest.raises(ValidationError, match="weight"):
            LegacyObjective(
                metrics=[
                    LegacyMetric(name="cost", minimize=True),
                    LegacyMetric(name="quality", minimize=False),
                ],
                weights=[1.0],
            )

    def test_matching_weights_are_accepted(self) -> None:
        objective = LegacyObjective(
            metrics=[
                LegacyMetric(name="cost", minimize=True),
                LegacyMetric(name="quality", minimize=False),
            ],
            weights=[2.0, 1.0],
        )
        assert objective.weights == [2.0, 1.0]

    def test_outcome_constraint_on_unknown_metric_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="not among its own metrics"):
            LegacyObjective(
                metrics=[LegacyMetric(name="loss", minimize=True)],
                outcome_constraints=[LegacyOutcomeConstraint("qps", 100.0, ">=")],
            )

    def test_outcome_constraint_on_known_metric_is_accepted(self) -> None:
        objective = LegacyObjective(
            metrics=[
                LegacyMetric(name="loss", minimize=True),
                LegacyMetric(name="qps", minimize=False),
            ],
            outcome_constraints=[LegacyOutcomeConstraint("qps", 100.0, ">=")],
        )
        assert len(objective.outcome_constraints) == 1
