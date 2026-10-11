"""ABC for an optimization backend: the object `BayesianOptimizer` delegates
every non-trivial operation to.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from planexs.domain.Objective import Objective
from planexs.domain.SearchSpace import SearchSpace
from planexs.domain.Trial import Trial

if TYPE_CHECKING:
    # Only needed for the `fitModel` type hint below. `from __future__ import
    # annotations` (PEP 563) already makes every annotation in this file lazy;
    # gating the import itself behind TYPE_CHECKING means this ABC, and
    # api/BayesianOptimizer.py which imports it, do not force a BoTorch
    # install just to be imported. Only `backends/ax/AxBackend.py` (which
    # genuinely calls into BoTorch at runtime, e.g. `isinstance(model, Model)`
    # in `fitModel()`) needs the real import; see its own docstring.
    from botorch.models.model import Model


class OptimizationBackend(ABC):
    """The interface every optimization backend implements.

    `AxBackend` is the only implementation in Phase 1 (ADR-0001). Any future
    backend swap only requires a new `OptimizationBackend` implementation:
    `domain/` and `api/BayesianOptimizer.py` do not change (ADR-0001's
    consequences).
    """

    @abstractmethod
    def createExperiment(self, search_space: SearchSpace, objective: Objective) -> None:
        """Configures the backend for a new study: its search space and its
        optimization goal. Called once, from `BayesianOptimizer.__init__`.
        """

    @abstractmethod
    def attachTrial(self, trial: Trial) -> int:
        """Registers an already-evaluated trial, returns its backend-assigned
        index.
        """

    @abstractmethod
    def suggestDefault(self, n_points: int) -> list[dict[str, float | int | str | bool]]:
        """The no-manual-tuning default path (FR4), delegated entirely to the
        backend.
        """

    @abstractmethod
    def fitModel(self) -> Model:
        """Fits and returns the current surrogate model. Used by the
        acquisition layer, and directly exposed to callers as
        `BayesianOptimizer.fitModel()` (FR16, ADR-0005).
        """

    @abstractmethod
    def predict(
        self, x: dict[str, float | int | str | bool]
    ) -> dict[str, tuple[float, float]]:
        """Returns `{metric_name: (mean, sem)}`. Matches `Client.predict`'s
        own return shape (predicted mean and standard error of the mean, not
        variance) exactly, rather than converting to a different uncertainty
        representation.

        Raises:
            PredictionUnavailableError: if the backend has no model to predict
                with yet (e.g. it is still in an initial space-filling phase).
                `BayesianOptimizer.predict` falls back to planexs' own surrogate
                in that case.
        """

    @abstractmethod
    def computeSensitivity(self) -> dict[str, dict[str, float]]:
        """Returns `{metric_name: {parameter_name: importance}}`."""

    @abstractmethod
    def getParetoFrontier(self) -> list[Trial]:
        """Delegates to `Client.get_pareto_frontier(use_model_predictions=True)`,
        not a hand-rolled non-domination scan over raw trial data.
        """

    @abstractmethod
    def getBestTrial(self) -> Trial | None:
        """Delegates to `Client.get_best_parameterization()` for the
        single-objective case.
        """

    @abstractmethod
    def exportState(self) -> dict[str, Any]:
        """Returns this backend's complete state as a JSON-compatible `dict`,
        opaque to the domain layer, for `StudySnapshot.backend_state`
        (FR11, FR12).

        Not part of section 5.3's original `OptimizationBackend` listing:
        added because `persistence/JsonStudyRepository.py` needs *some*
        backend-agnostic hook to obtain the backend's own serialized state
        without either (a) `JsonStudyRepository` importing `ax.*` directly
        (breaking ADR-0001's "only `backends/ax` imports Ax" rule) or (b)
        `StudySnapshot` holding a live backend reference (breaking its own
        JSON-serializability, FR11). Flagged here for the project owner to
        fold back into section 5.3 once reviewed, per section 12.
        """

    @classmethod
    @abstractmethod
    def importState(cls, state: dict[str, Any]) -> OptimizationBackend:
        """The inverse of `exportState`: rebuilds a live backend instance
        from a previously exported state. See `exportState`'s docstring for
        why this exists beyond section 5.3's original listing.
        """
