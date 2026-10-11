"""Unit tests for `JsonStudyRepository`.

Round-trips a `StudySnapshot` built directly (not via a real `AxBackend`), so
these tests exercise the JSON save/load of planexs' own domain objects and
`ReproducibilityMetadata` without needing a real Ax install. `backend_state`
here is a small, arbitrary JSON-compatible dict standing in for whatever a
real `AxBackend.exportState()` would have produced.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from planexs import Categorical, LinearConstraint, Metric, Objective, Real, SearchSpace, Trial
from planexs.domain.StudySnapshot import StudySnapshot
from planexs.persistence.JsonStudyRepository import JsonStudyRepository
from planexs.persistence.ReproducibilityMetadata import ReproducibilityMetadata


def _snapshot() -> StudySnapshot:
    return StudySnapshot(
        name="round-trip-study",
        search_space=SearchSpace(
            parameters=[
                Real("x", 0.0, 1.0, default=0.5),
                Categorical("m", ["a", "b"], dependent_parameters={"a": ["x"]}),
            ],
            constraints=[LinearConstraint({"x": 1.0}, bound=1.0, comparator="<=")],
        ),
        objective=Objective(metrics=[Metric(name="objective", minimize=True)]),
        trials=[
            Trial(parameters={"x": 0.3}, results={"objective": 1.2}, trial_index=0),
            Trial(
                parameters={"x": 0.7},
                results={"objective": 0.4},
                result_std={"objective": 0.05},
                trial_index=1,
            ),
        ],
        reproducibility=ReproducibilityMetadata(
            random_seed=42,
            library_versions={"planexs": "0.1.0"},
            created_at=datetime(2026, 9, 25, tzinfo=UTC),
            planexs_version="0.1.0",
        ),
        backend_state={"fake": "ax-client-json-goes-here"},
    )


class TestJsonStudyRepository:
    def test_round_trips_name_and_search_space(self, tmp_path: Path) -> None:
        snapshot = _snapshot()
        path = tmp_path / "study.json"

        JsonStudyRepository().save(snapshot, path)
        reloaded = JsonStudyRepository().load(path)

        assert reloaded.name == snapshot.name
        assert reloaded.search_space.parameter_names == ["x", "m"]
        assert reloaded.search_space.parameters[0].default == 0.5

    def test_round_trips_dependent_parameters_and_constraints(self, tmp_path: Path) -> None:
        path = tmp_path / "study.json"

        JsonStudyRepository().save(_snapshot(), path)
        reloaded = JsonStudyRepository().load(path)

        assert reloaded.search_space.parameters[1].dependent_parameters == {"a": ["x"]}
        assert reloaded.search_space.constraints == _snapshot().search_space.constraints

    def test_round_trips_full_trial_history(self, tmp_path: Path) -> None:
        snapshot = _snapshot()
        path = tmp_path / "study.json"

        JsonStudyRepository().save(snapshot, path)
        reloaded = JsonStudyRepository().load(path)

        assert len(reloaded.trials) == 2
        assert reloaded.trials[1].result_std == {"objective": 0.05}

    def test_round_trips_reproducibility_metadata(self, tmp_path: Path) -> None:
        snapshot = _snapshot()
        path = tmp_path / "study.json"

        JsonStudyRepository().save(snapshot, path)
        reloaded = JsonStudyRepository().load(path)

        assert reloaded.reproducibility.random_seed == 42
        assert reloaded.reproducibility.planexs_version == "0.1.0"

    def test_round_trips_opaque_backend_state(self, tmp_path: Path) -> None:
        snapshot = _snapshot()
        path = tmp_path / "study.json"

        JsonStudyRepository().save(snapshot, path)
        reloaded = JsonStudyRepository().load(path)

        assert reloaded.backend_state == {"fake": "ax-client-json-goes-here"}
        assert reloaded.backend_kind == "ax"

    def test_saved_file_is_pretty_printed_json(self, tmp_path: Path) -> None:
        path = tmp_path / "study.json"
        JsonStudyRepository().save(_snapshot(), path)

        text = path.read_text(encoding="utf-8")
        assert "\n" in text  # indent=2 -> not minified onto a single line
