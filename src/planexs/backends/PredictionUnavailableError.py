"""Raised when no model is available to predict with yet."""

from __future__ import annotations


class PredictionUnavailableError(RuntimeError):
    """There is no surrogate model to predict with yet.

    An `OptimizationBackend.predict` implementation raises this when it cannot
    predict *at this moment* (for `AxBackend`: while Ax's generation strategy
    is still in its initial space-filling phase and holds no model), as opposed
    to being given an invalid point. `BayesianOptimizer.predict` catches it and
    falls back to planexs' own surrogate model; if that cannot be fit either
    (fewer than two completed trials), it raises this error itself, so a caller
    only ever has one exception type to handle.
    """
