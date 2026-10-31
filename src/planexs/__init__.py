"""planexs: a general-purpose, domain-agnostic Bayesian Optimization library.

This module re-exports planexs' public surface, so application code should
import from here (`from planexs import BayesianOptimizer, Real, ...`) rather
than reaching into `planexs.domain.parameters.Real`, etc. directly. Exempt
from section 10's file-naming rule, like every Python dunder file.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

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
