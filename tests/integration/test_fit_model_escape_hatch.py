"""Regression tests for the `fitModel()` escape hatch (FR16, ADR-0005).

It used to fail on every call with Ax 1.x: it looked the model up through an attribute
(`GenerationStrategy.model`) that Ax renamed, and a broad `except` reported that as "not
enough trials", which hid the cause. Static type checking is what found it.
"""

from __future__ import annotations

import pytest
from botorch.models.model import Model

from planexs import BayesianOptimizer, Real


def _optimizer() -> BayesianOptimizer:
    return BayesianOptimizer(
        parameters=[Real("x", 0.0, 1.0), Real("y", 0.0, 1.0)],
        objective="minimize",
        random_seed=0,
    )


def test_returns_the_botorch_model_once_ax_has_fit_one() -> None:
    bo = _optimizer()
    for _ in range(8):  # past Ax's initial space-filling budget
        (point,) = bo.ask()
        bo.tell(point, {"objective": (point["x"] - 0.7) ** 2 + (point["y"] - 0.2) ** 2})  # type: ignore[operator]

    model = bo.fitModel()

    assert isinstance(model, Model)
    assert model.num_outputs == 1


def test_says_what_is_missing_while_ax_is_still_space_filling() -> None:
    bo = _optimizer()
    (point,) = bo.ask()
    bo.tell(point, {"objective": 1.0})

    with pytest.raises(RuntimeError, match=r"adapter is a.*space-filling phase"):
        bo.fitModel()
