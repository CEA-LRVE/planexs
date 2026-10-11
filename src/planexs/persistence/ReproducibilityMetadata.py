"""Random seed, library versions, and creation time recorded with a study."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version

#: Packages whose installed version is recorded in `library_versions` so a
#: reloaded study can flag a mismatch against the environment it was created
#: in (FR12). Kept as a module-level constant, not buried in a function body,
#: so it is easy to extend if a future phase adds a dependency worth pinning
#: for reproducibility (e.g. `botorch`/`gpytorch` become relevant once the
#: acquisition layer, Phase 2, is in the loop).
TRACKED_PACKAGES: tuple[str, ...] = (
    "planexs",
    "ax-platform",
    "botorch",
    "gpytorch",
    "torch",
    "pydantic",
)


@dataclass(frozen=True)
class ReproducibilityMetadata:
    """Everything needed to judge whether a reloaded study is running in an
    environment consistent with the one that created it (FR12): not just the
    final state, but the random seed, library versions, and creation time.

    Attributes:
        random_seed: The random seed used for this study. Always a concrete
            `int`, even when the caller left `BayesianOptimizer`'s own
            `random_seed` argument as `None`: a seed is generated once and
            recorded here, so a reload is reproducible even for a study that
            never asked for a specific seed.
        library_versions: Maps package name (see `TRACKED_PACKAGES`) to the
            installed version string at the time this study was created, or
            `"not installed"` when a package is absent (e.g. `botorch`/
            `gpytorch` before Phase 2 is wired in).
        created_at: When this study (or, after a reload, this particular
            snapshot) was created, in UTC.
        planexs_version: The installed `planexs` version, duplicated from
            `library_versions` as its own field since it is the single most
            important version to check at a glance.
    """

    random_seed: int
    library_versions: dict[str, str]
    created_at: datetime
    planexs_version: str

    @staticmethod
    def captureCurrentEnvironment(random_seed: int) -> ReproducibilityMetadata:
        """Builds a `ReproducibilityMetadata` for *right now*: looks up the
        installed version of every package in `TRACKED_PACKAGES` (recording
        `"not installed"` for one that is absent, e.g. `botorch` before
        Phase 2), timestamps it in UTC, and pairs it with `random_seed`
        (already resolved to a concrete value by the caller; see
        `AxBackend.__init__`, which generates one when the user leaves it as
        `None`, precisely so it can be recorded here instead of staying an
        unreproducible `None`).
        """
        library_versions: dict[str, str] = {}
        for package_name in TRACKED_PACKAGES:
            try:
                library_versions[package_name] = version(package_name)
            except PackageNotFoundError:
                library_versions[package_name] = "not installed"
        return ReproducibilityMetadata(
            random_seed=random_seed,
            library_versions=library_versions,
            created_at=datetime.now(UTC),
            planexs_version=library_versions["planexs"],
        )
