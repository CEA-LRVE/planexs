"""Everything needed to fully reconstruct a `BayesianOptimizer`."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from planexs.domain.Objective import Objective
from planexs.domain.SearchSpace import SearchSpace
from planexs.domain.Trial import Trial

# ReproducibilityMetadata lives under persistence/ (section 5.1's package
# layout), not domain/, since it is a persistence-format concern (FR12)
# rather than a modeling concept an optimization backend needs. Importing it
# here is safe (no import cycle): ReproducibilityMetadata itself has zero
# planexs dependencies, so domain's "zero ML dependencies" rule (section 4.1)
# is preserved in spirit; it is the one deliberate exception to domain/ never
# importing from a sibling top-level package.
from planexs.persistence.ReproducibilityMetadata import ReproducibilityMetadata


class StudySnapshot(BaseModel):
    """Everything needed to fully reconstruct a `BayesianOptimizer`: its
    search space, objective, constraints, trial history, and reproducibility
    metadata. The unit that `persistence/` reads and writes. Wraps, rather
    than duplicates, whatever Ax's own `Client.save_to_json_file` already
    captures; see section 5.6.

    Attributes:
        name: The study's name (mirrors `BayesianOptimizer.__init__`'s own
            `name` argument).
        search_space: The full `SearchSpace`, including any `Constraint`s.
            Kept as planexs' own domain object (not re-derived from
            `backend_state`) so that `default`/`dependent_parameters`
            metadata that Ax's own snapshot does not carry a field for
            round-trips exactly (FR11, FR12).
        objective: The full `Objective`, including weights and
            `OutcomeConstraint`s.
        trials: The complete trial history, in the order they were told to
            the optimizer (FR12: "full trial history ... not just the final
            state").
        reproducibility: The random seed, library versions, and creation
            timestamp recorded alongside the study (FR12).
        backend_state: The backend's own serialized state, opaque to the
            domain layer. For `AxBackend`, this is the JSON object Ax's own
            `Client.save_to_json_file` produces (obtained by writing to a
            temporary file and reading it back, since `Client` only offers a
            file-based save/load API), so that `JsonStudyRepository` can
            round-trip a single planexs file while still delegating the
            Ax-backed portion of persistence to Ax itself (design
            philosophy #5).
        backend_kind: Identifies which backend `backend_state` belongs to
            (`"ax"` in Phase 1). Lets `JsonStudyRepository.load` pick the
            right backend class to restore into.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    search_space: SearchSpace
    objective: Objective
    trials: list[Trial]
    reproducibility: ReproducibilityMetadata
    backend_state: dict[str, Any]
    backend_kind: str = "ax"

    def __init__(
        self,
        name: str,
        search_space: SearchSpace,
        objective: Objective,
        trials: list[Trial],
        reproducibility: ReproducibilityMetadata,
        backend_state: dict[str, Any],
        backend_kind: str = "ax",
    ) -> None:
        """Creates a snapshot of a whole study.

        Args:
            name: The study's name.
            search_space: The full search space.
            objective: The full objective.
            trials: The complete trial history.
            reproducibility: Seed, library versions and creation time.
            backend_state: The backend's own serialized state.
            backend_kind: Which backend `backend_state` belongs to.
        """
        super().__init__(
            name=name,
            search_space=search_space,
            objective=objective,
            trials=trials,
            reproducibility=reproducibility,
            backend_state=backend_state,
            backend_kind=backend_kind,
        )
