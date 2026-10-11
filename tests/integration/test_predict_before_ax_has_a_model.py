"""Regression tests: `predict()` must work from the start of a study, not only after
Ax's generation strategy has left its initial space-filling phase.

It used to raise Ax's `UnsupportedError` there, i.e. exactly in the small-budget regime
the library is meant for (FR7, FR8).
"""

from __future__ import annotations

import pytest

from planexs import BayesianOptimizer, PredictionUnavailableError, Real


def _bowl(x: float, y: float) -> float:
    return (x - 0.7) ** 2 + (y - 0.2) ** 2


def _optimizer() -> BayesianOptimizer:
    return BayesianOptimizer(
        parameters=[Real("x", 0.0, 1.0), Real("y", 0.0, 1.0)],
        objective="minimize",
        random_seed=0,
    )


def test_ax_reports_that_it_has_no_model_yet_with_planexss_own_error() -> None:
    bo = _optimizer()
    bo.tell({"x": 0.1, "y": 0.9}, {"objective": _bowl(0.1, 0.9)})
    bo.tell({"x": 0.9, "y": 0.1}, {"objective": _bowl(0.9, 0.1)})

    with pytest.raises(PredictionUnavailableError, match="no predictive model yet"):
        bo._backend.predict({"x": 0.5, "y": 0.5})  # type: ignore[attr-defined]


def test_predict_works_from_the_second_trial_on() -> None:
    bo = _optimizer()
    bo.tell({"x": 0.1, "y": 0.9}, {"objective": _bowl(0.1, 0.9)})
    bo.tell({"x": 0.9, "y": 0.1}, {"objective": _bowl(0.9, 0.1)})

    near = bo.predict({"x": 0.9, "y": 0.1})
    far = bo.predict({"x": 0.5, "y": 0.5})

    assert near.mean["objective"] == pytest.approx(_bowl(0.9, 0.1), abs=0.3)
    assert far.sem["objective"] > near.sem["objective"] >= 0.0


def test_one_trial_is_not_enough_and_says_so() -> None:
    bo = _optimizer()
    bo.tell({"x": 0.1, "y": 0.9}, {"objective": _bowl(0.1, 0.9)})
    with pytest.raises(PredictionUnavailableError, match="at least 2"):
        bo.predict({"x": 0.5, "y": 0.5})


def test_predict_still_works_once_ax_has_a_model_of_its_own() -> None:
    bo = _optimizer()
    for _ in range(8):  # past Ax's initial space-filling budget
        (point,) = bo.ask()
        bo.tell(point, {"objective": _bowl(point["x"], point["y"])})  # type: ignore[arg-type]

    prediction = bo.predict({"x": 0.7, "y": 0.2})

    assert prediction.mean["objective"] == pytest.approx(0.0, abs=0.2)
    assert prediction.sem["objective"] >= 0.0
