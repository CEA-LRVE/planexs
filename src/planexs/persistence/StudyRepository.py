"""ABC for saving and loading a `StudySnapshot`."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from planexs.domain.StudySnapshot import StudySnapshot


class StudyRepository(ABC):
    """Saves and loads a `StudySnapshot` to/from some storage medium.

    `JsonStudyRepository` (a single `.json` file) is the only implementation
    in Phase 1. Section 4.7 lists a SQL-backed implementation
    (`ax.api.configs.StorageConfig`) as a deferred, additive option: this ABC
    exists precisely so that adding one later does not require touching
    `BayesianOptimizer.save`/`.load`.
    """

    @abstractmethod
    def save(self, snapshot: StudySnapshot, path: str | Path) -> None:
        """Persists `snapshot` to `path`."""

    @abstractmethod
    def load(self, path: str | Path) -> StudySnapshot:
        """Reconstructs a `StudySnapshot` previously saved to `path`."""
