"""planexs: a general-purpose, domain-agnostic toolkit for sequential experimental design.

This module re-exports planexs' public surface, so application code should import from here
(`from planexs import BayesianOptimizer, Real, ...`) rather than reaching into
`planexs.domain.parameters.Real`, etc. directly. Exempt from section 10's file-naming rule,
like every Python dunder file.

The re-exports are **lazy** (specification 8, 9.1). A name is imported the first time it is
asked for, so that `import planexs`, `import planexs.core` and `import planexs.domain` load no
machine-learning library: the core and the domain are pure (specification 3.9, 5.1), and one
of them must not pull in torch, BoTorch, GPyTorch and Ax because a sibling name is exported
here. `planexs.Real` loads the module of `Real` and nothing else; `planexs.BayesianOptimizer`
loads the engine it needs. A name that has been asked for once is kept in this module, so
that the lookup is not repeated.

To export a name, add it in three places, which a test checks against each other: the import
under `TYPE_CHECKING` (what type checkers and editors read), `_EXPORTS` (where the name is
found at run time) and `__all__`.
"""

from __future__ import annotations

import importlib
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from planexs.acquisition.AcquisitionStrategy import AcquisitionStrategy
    from planexs.acquisition.AlphaAcquisitionStrategy import AlphaAcquisitionStrategy
    from planexs.acquisition.ExplorationExploitationAcquisition import (
        ExplorationExploitationAcquisition,
    )
    from planexs.acquisition.MultiObjectiveExplorationExploitationAcquisition import (
        MultiObjectiveExplorationExploitationAcquisition,
    )
    from planexs.analysis.PredictionResult import PredictionResult
    from planexs.api.BayesianOptimizer import BayesianOptimizer
    from planexs.backends.ax.AxBackend import AxBackend
    from planexs.backends.OptimizationBackend import OptimizationBackend
    from planexs.backends.PredictionUnavailableError import PredictionUnavailableError
    from planexs.domain.constraints.Constraint import Constraint
    from planexs.domain.constraints.LinearConstraint import LinearConstraint
    from planexs.domain.constraints.NonlinearConstraint import NonlinearConstraint
    from planexs.domain.LegacyMetric import LegacyMetric
    from planexs.domain.LegacyObjective import LegacyObjective
    from planexs.domain.LegacyOutcomeConstraint import LegacyOutcomeConstraint
    from planexs.domain.LegacyTrial import LegacyTrial
    from planexs.domain.parameters.Boolean import Boolean
    from planexs.domain.parameters.Categorical import Categorical
    from planexs.domain.parameters.Choice import Choice
    from planexs.domain.parameters.Derived import Derived
    from planexs.domain.parameters.Fixed import Fixed
    from planexs.domain.parameters.Integer import Integer
    from planexs.domain.parameters.Parameter import Parameter
    from planexs.domain.parameters.Range import Range
    from planexs.domain.parameters.Real import Real
    from planexs.domain.SearchSpace import SearchSpace
    from planexs.persistence.JsonStudyRepository import JsonStudyRepository
    from planexs.persistence.LegacyStudySnapshot import LegacyStudySnapshot
    from planexs.persistence.ReproducibilityMetadata import ReproducibilityMetadata
    from planexs.persistence.StudyRepository import StudyRepository

#: Where each exported name is defined: the module that `__getattr__` imports to find it.
_EXPORTS: dict[str, str] = {
    "AcquisitionStrategy": "planexs.acquisition.AcquisitionStrategy",
    "AlphaAcquisitionStrategy": "planexs.acquisition.AlphaAcquisitionStrategy",
    "AxBackend": "planexs.backends.ax.AxBackend",
    "BayesianOptimizer": "planexs.api.BayesianOptimizer",
    "Boolean": "planexs.domain.parameters.Boolean",
    "Categorical": "planexs.domain.parameters.Categorical",
    "Choice": "planexs.domain.parameters.Choice",
    "Constraint": "planexs.domain.constraints.Constraint",
    "Derived": "planexs.domain.parameters.Derived",
    "ExplorationExploitationAcquisition": (
        "planexs.acquisition.ExplorationExploitationAcquisition"
    ),
    "Fixed": "planexs.domain.parameters.Fixed",
    "Integer": "planexs.domain.parameters.Integer",
    "JsonStudyRepository": "planexs.persistence.JsonStudyRepository",
    "LegacyMetric": "planexs.domain.LegacyMetric",
    "LegacyObjective": "planexs.domain.LegacyObjective",
    "LegacyOutcomeConstraint": "planexs.domain.LegacyOutcomeConstraint",
    "LegacyStudySnapshot": "planexs.persistence.LegacyStudySnapshot",
    "LegacyTrial": "planexs.domain.LegacyTrial",
    "LinearConstraint": "planexs.domain.constraints.LinearConstraint",
    "MultiObjectiveExplorationExploitationAcquisition": (
        "planexs.acquisition.MultiObjectiveExplorationExploitationAcquisition"
    ),
    "NonlinearConstraint": "planexs.domain.constraints.NonlinearConstraint",
    "OptimizationBackend": "planexs.backends.OptimizationBackend",
    "Parameter": "planexs.domain.parameters.Parameter",
    "PredictionResult": "planexs.analysis.PredictionResult",
    "PredictionUnavailableError": "planexs.backends.PredictionUnavailableError",
    "Range": "planexs.domain.parameters.Range",
    "Real": "planexs.domain.parameters.Real",
    "ReproducibilityMetadata": "planexs.persistence.ReproducibilityMetadata",
    "SearchSpace": "planexs.domain.SearchSpace",
    "StudyRepository": "planexs.persistence.StudyRepository",
}

# Single source of truth is pyproject.toml (bumped by scripts/cut_release.py);
# reading it back from the installed metadata avoids a second hand-edited copy
# drifting out of sync.
try:
    __version__ = version("planexs")
except PackageNotFoundError:  # running from a source checkout without install
    __version__ = "0.0.0+unknown"

__all__ = [
    "AcquisitionStrategy",
    "AlphaAcquisitionStrategy",
    "AxBackend",
    "BayesianOptimizer",
    "Boolean",
    "Categorical",
    "Choice",
    "Constraint",
    "Derived",
    "ExplorationExploitationAcquisition",
    "Fixed",
    "Integer",
    "JsonStudyRepository",
    "LegacyMetric",
    "LegacyObjective",
    "LegacyOutcomeConstraint",
    "LegacyStudySnapshot",
    "LegacyTrial",
    "LinearConstraint",
    "MultiObjectiveExplorationExploitationAcquisition",
    "NonlinearConstraint",
    "OptimizationBackend",
    "Parameter",
    "PredictionResult",
    "PredictionUnavailableError",
    "Range",
    "Real",
    "ReproducibilityMetadata",
    "SearchSpace",
    "StudyRepository",
    "__version__",
]

# Hidden from type checkers: a module-level `__getattr__` makes them accept every attribute of
# the module, as `Any`, so a misspelled `planexs.Realy` would pass. The imports above give
# them exactly the names that exist.
if not TYPE_CHECKING:

    def __getattr__(name: str) -> Any:
        """Imports an exported name when it is first asked for (PEP 562).

        Args:
            name: The attribute being looked up on the module.

        Returns:
            The exported object. It is also stored in the module, so that the next lookup
            does not come here.

        Raises:
            AttributeError: if `name` is not an exported name.
            ImportError: if the module of the name, or something it needs, cannot be imported.
                It is not turned into an `AttributeError`: a missing optional dependency
                should say so.
        """
        module_name = _EXPORTS.get(name)
        if module_name is None:
            raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
        value = getattr(importlib.import_module(module_name), name)
        globals()[name] = value
        return value

    def __dir__() -> list[str]:
        """The names of the module, the exported ones included, though not yet imported."""
        return sorted({*globals(), *_EXPORTS})
