"""Configuration schemas with nested components, shared by the tests of nested resolution.

They are shaped like the ones real components will register. A study-level model, which is
not a component itself, holds a policy: the policy holds a surrogate, which holds a kernel.

    StudyConfig.policy        -> ComposedConfig (kind 'policy')
      .design                 -> SobolConfig / RandomConfig (kind 'design')
      .surrogate              -> GpConfig (kind 'surrogate')
        .kernel               -> MaternConfig / RbfConfig (kind 'kernel')
      .analyses[i]            -> LevelEstimateConfig / ImportanceConfig (kind 'analysis')
      .fallback               -> a design, or None
    StudyConfig.stopping[i]   -> MaxTrialsConfig (kind 'stopping')
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from planexs.core.ComponentConfig import ComponentConfig
from planexs.core.ComponentOf import ComponentOf


class MaternConfig(ComponentConfig):
    """A kernel."""

    type: Literal["matern_v1"] = "matern_v1"
    nu: float = Field(default=2.5, description="Smoothness of the kernel.")
    ard: bool = Field(default=True, description="Whether each dimension has its own length.")


class RbfConfig(ComponentConfig):
    """Another kernel."""

    type: Literal["rbf_v1"] = "rbf_v1"
    ard: bool = Field(default=False, description="Whether each dimension has its own length.")


class SobolConfig(ComponentConfig):
    """A design."""

    type: Literal["sobol_v1"] = "sobol_v1"
    n_points: int = Field(default=8, description="Number of points.")


class RandomConfig(ComponentConfig):
    """Another design."""

    type: Literal["random_v1"] = "random_v1"
    n_points: int = Field(default=4, description="Number of points.")


class GpConfig(ComponentConfig):
    """A surrogate that holds a kernel."""

    type: Literal["single_task_gp_v1"] = "single_task_gp_v1"
    min_observations: int = Field(default=2, description="Fewest observations to fit on.")
    kernel: Annotated[ComponentConfig, ComponentOf("kernel")] = Field(
        default_factory=lambda: {"type": "matern_v1"},  # type: ignore[arg-type, return-value]
        validate_default=True,
        description="The kernel of the GP.",
    )


class LevelEstimateConfig(ComponentConfig):
    """An analysis."""

    type: Literal["level_estimate_v1"] = "level_estimate_v1"
    level: float = Field(default=0.5, description="The level to estimate.")


class ImportanceConfig(ComponentConfig):
    """Another analysis."""

    type: Literal["importance_v1"] = "importance_v1"
    n_samples: int = Field(default=64, description="Number of samples.")


class ComposedConfig(ComponentConfig):
    """A policy that holds a design, a surrogate, a list of analyses and an optional design."""

    type: Literal["composed_v1"] = "composed_v1"
    design: Annotated[ComponentConfig, ComponentOf("design")] = Field(
        default_factory=lambda: {"type": "sobol_v1"},  # type: ignore[arg-type, return-value]
        validate_default=True,
        description="The design of the first phase.",
    )
    surrogate: Annotated[ComponentConfig, ComponentOf("surrogate")] = Field(
        default_factory=lambda: {"type": "single_task_gp_v1"},  # type: ignore[arg-type, return-value]
        validate_default=True,
        description="The surrogate of the model phase.",
    )
    analyses: list[Annotated[ComponentConfig, ComponentOf("analysis")]] = Field(
        default_factory=list, description="The analyses to run."
    )
    fallback: Annotated[ComponentConfig, ComponentOf("design")] | None = Field(
        default=None, description="A design to fall back on, if any."
    )
    bounds: tuple[float, float] = Field(default=(0.0, 1.0), description="A pair of bounds.")


class MaxTrialsConfig(ComponentConfig):
    """A stopping criterion."""

    type: Literal["max_trials_v1"] = "max_trials_v1"
    n_trials: int = Field(default=50, description="Number of trials after which to stop.")


class StudyConfig(BaseModel):
    """A model that holds components without being one: it has no `type`."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(default="study", description="The name of the study.")
    policy: Annotated[ComponentConfig, ComponentOf("policy")] = Field(
        description="The policy that chooses the next points."
    )
    stopping: list[Annotated[ComponentConfig, ComponentOf("stopping")]] = Field(
        default_factory=list, description="The stopping criteria."
    )


#: The components of each kind, which the `nestedKinds` fixture registers.
NESTED_COMPONENTS: dict[str, tuple[type[ComponentConfig], ...]] = {
    "kernel": (MaternConfig, RbfConfig),
    "design": (SobolConfig, RandomConfig),
    "surrogate": (GpConfig,),
    "analysis": (LevelEstimateConfig, ImportanceConfig),
    "policy": (ComposedConfig,),
    "stopping": (MaxTrialsConfig,),
}
