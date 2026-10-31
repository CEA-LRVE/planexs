"""Regression tests: a reloaded study must be as usable as the one that was saved.

`AxBackend.importState` used to look metric names up through an attribute Ax 1.x does
not have, so after every `load()` the backend knew no metrics: it logged a warning and
`parameterImportance()` silently returned an empty result.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from planexs import (
    BayesianOptimizer,
    LegacyMetric,
    LegacyObjective,
    LegacyOutcomeConstraint,
    Real,
    SearchSpace,
)

_OBJECTIVES = {
    "single": (LegacyObjective([LegacyMetric("a", minimize=True)]), ["a"]),
    "multi": (
        LegacyObjective([LegacyMetric("a", minimize=True), LegacyMetric("b", minimize=False)]),
        ["a", "b"],
    ),
    "weighted": (
        LegacyObjective(
            [LegacyMetric("a", minimize=True), LegacyMetric("b", minimize=False)],
            weights=[2.0, 1.0],
        ),
        ["a", "b"],
    ),
    "outcome-constrained": (
        LegacyObjective(
            [LegacyMetric("a", minimize=True), LegacyMetric("b", minimize=False)],
            outcome_constraints=[LegacyOutcomeConstraint("b", 0.1, ">=")],
        ),
        ["a", "b"],
    ),
}


@pytest.mark.parametrize("shape", list(_OBJECTIVES))
def test_a_reloaded_study_still_knows_its_metrics(
    shape: str, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    objective, expected = _OBJECTIVES[shape]
    bo = BayesianOptimizer(
        parameters=SearchSpace([Real("x", 0.0, 1.0)]), objective=objective, random_seed=0
    )
    for point in bo.ask(n_points=3):
        bo.tell(
            point,
            {
                m.name: point["x"] if m.minimize else 1.0 - point["x"]
                for m in objective.metrics
            },
        )  # type: ignore[operator]
    path = tmp_path / "study.json"
    bo.save(path)

    with caplog.at_level(logging.WARNING, logger="planexs"):
        reloaded = BayesianOptimizer.load(path)
        importance = reloaded.parameterImportance()

    assert "Could not resolve metric names" not in caplog.text
    assert reloaded._backend._metric_names == expected  # type: ignore[attr-defined]
    assert set(importance) == set(expected)
