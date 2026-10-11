"""An in-memory `OptimizationBackend` test double.

Lets `tests/unit/api/test_bayesian_optimizer.py` exercise
`BayesianOptimizer`'s own facade logic (case resolution, trial bookkeeping,
property behavior, error handling) without depending on a real Ax install:
`BayesianOptimizer(..., backend=FakeBackend())` bypasses `AxBackend` entirely.

Named `fake_backend.py` (snake_case), not `FakeBackend.py`: section 10's
file-naming rule applies to `src/planexs/`'s public API surface, not test
helper modules, and pytest's own collection conventions expect
`test_*.py`/plain snake_case module names under `tests/`.
"""

from __future__ import annotations

from typing import Any

from planexs.backends.OptimizationBackend import OptimizationBackend
from planexs.domain.Objective import Objective
from planexs.domain.SearchSpace import SearchSpace
from planexs.domain.Trial import Trial


class FakeBackend(OptimizationBackend):
    """Records what it is asked to do; returns simple, fixed data back."""

    def __init__(self) -> None:
        self.created_search_space: SearchSpace | None = None
        self.created_objective: Objective | None = None
        self.attached_trials: list[Trial] = []
        self.exported_state_calls = 0
        self._next_trial_index = 0

    def createExperiment(self, search_space: SearchSpace, objective: Objective) -> None:
        self.created_search_space = search_space
        self.created_objective = objective

    def attachTrial(self, trial: Trial) -> int:
        index = self._next_trial_index
        self._next_trial_index += 1
        self.attached_trials.append(trial)
        return index

    def suggestDefault(self, n_points: int) -> list[dict[str, float | int | str | bool]]:
        return [{"x": float(i)} for i in range(n_points)]

    def fitModel(self) -> Any:
        raise RuntimeError("FakeBackend has no real surrogate model to fit.")

    def predict(
        self, x: dict[str, float | int | str | bool]
    ) -> dict[str, tuple[float, float]]:
        return {"objective": (0.5, 0.1)}

    def computeSensitivity(self) -> dict[str, dict[str, float]]:
        return {"objective": {"x": 1.0}}

    def getParetoFrontier(self) -> list[Trial]:
        return list(self.attached_trials)

    def getBestTrial(self) -> Trial | None:
        return self.attached_trials[-1] if self.attached_trials else None

    def exportState(self) -> dict[str, Any]:
        self.exported_state_calls += 1
        return {"n_attached_trials": len(self.attached_trials)}

    @classmethod
    def importState(cls, state: dict[str, Any]) -> FakeBackend:
        raise NotImplementedError("FakeBackend.importState is not exercised by these tests.")
