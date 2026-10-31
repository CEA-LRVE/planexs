"""Persists a `LegacyStudySnapshot` as a single JSON file."""

from __future__ import annotations

from pathlib import Path

from planexs.persistence.LegacyStudySnapshot import LegacyStudySnapshot
from planexs.persistence.StudyRepository import StudyRepository


class JsonStudyRepository(StudyRepository):
    """Persists a `LegacyStudySnapshot` as a single, human-inspectable JSON file.

    The Ax-backed portion of the state (`LegacyStudySnapshot.backend_state`) is
    itself the very JSON object `Client.save_to_json_file`/
    `load_from_json_file` produces and consumes: `AxBackend.exportState`
    obtains it by delegating to `Client.save_to_json_file` against a
    temporary file and reading the result back, and `AxBackend.importState`
    is its inverse. `JsonStudyRepository` therefore does not reimplement
    Ax's own serialization (design philosophy #5); it only adds the one JSON
    file planexs' own domain objects (`SearchSpace` with its `default`/
    `dependent_parameters` metadata, `LegacyObjective` with its weights, the full
    `LegacyTrial` history, `ReproducibilityMetadata`) and the opaque Ax blob live
    in side by side, so a caller gets exactly one file per study (FR11).
    """

    def save(self, snapshot: LegacyStudySnapshot, path: str | Path) -> None:
        """Writes `snapshot` to `path` as pretty-printed JSON."""
        Path(path).write_text(snapshot.model_dump_json(indent=2), encoding="utf-8")

    def load(self, path: str | Path) -> LegacyStudySnapshot:
        """Reads a `LegacyStudySnapshot` previously written by `save`."""
        return LegacyStudySnapshot.model_validate_json(Path(path).read_text(encoding="utf-8"))
