"""The `optimize_v1` goal: optimize one or several outcomes."""

from __future__ import annotations

from planexs.domain.Problem import Problem
from planexs.goals.AbstractGoal import AbstractGoal
from planexs.goals.OptimizeGoalConfig import OptimizeGoalConfig
from planexs.goals.registerGoal import registerGoal

# The capability tags of this goal. They are written once here, for the registry entry and for
# the narrowing of an instance below to agree, and are registered in the vocabulary by
# `goalRegistry` (specification 5.12).
_OPTIMIZE = "goal:optimize"
_SINGLE = "objectives:single"
_MULTI = "objectives:multi"
_OUTCOME_CONSTRAINTS = "outcome_constraints:supported"


@registerGoal(
    "optimize_v1",
    config=OptimizeGoalConfig,
    description="Optimize one or several outcomes, each with a direction.",
    needs=frozenset({_OUTCOME_CONSTRAINTS}),
    supports=frozenset({_OPTIMIZE, _SINGLE, _MULTI}),
)
class OptimizeGoal(AbstractGoal[OptimizeGoalConfig]):
    """Optimize one or several outcomes, each with a direction (specification 5.7, FR2, FR15).

    The goal says what is wanted and computes nothing (ADR-0013). Acquisitions and analyses
    read its configuration (`OptimizeGoalConfig`): the outcomes and their directions, the
    scalarizing weights and the limits on observed outcomes. It takes over the roles that
    `LegacyObjective`, `LegacyMetric` and `LegacyOutcomeConstraint` had in the legacy
    layer.

    **Capability tags.** The registry entry declares the union of what any configuration can
    state, and an instance narrows it to its own configuration:

    - it always supports `goal:optimize`;
    - it supports `objectives:single` for one outcome, and for several outcomes that `weights`
      turns into one scalar preference, as `toAxOptimizationConfig` made a weighted sum of
      them for Ax and `AlphaAcquisitionStrategy` handled them with its single-objective
      acquisition;
    - it supports `objectives:multi` for several outcomes with no weights, a Pareto problem;
    - it needs `outcome_constraints:supported`, which an acquisition supports, if and only if
      it has at least one outcome constraint, so that an acquisition that cannot enforce one is
      refused when the study is built and not discovered in the middle of a run (specification
      3.6). The legacy layer only logged a warning that it did not enforce them.

    It proposes no recipe and no default analysis yet: the recipes are written with the
    policies (roadmap task P6-07) and the analyses that answer it with the analyses (Phase 7),
    and until then the contract's defaults, which propose nothing, apply.
    """

    @property
    def supports(self) -> frozenset[str]:
        """The tags this goal supports, for its own configuration: `goal:optimize` and the
        objectives tag that fits it (a lookup and a count: cheap, snake_case).

        Returns:
            `goal:optimize` and `objectives:single` or `objectives:multi`, never both. Only
            tags that the registry entry declares.
        """
        objectives = _SINGLE if self._isScalar() else _MULTI
        return super().supports & frozenset({_OPTIMIZE, objectives})

    @property
    def needs(self) -> frozenset[str]:
        """The tags this goal needs some participant to support, for its own configuration:
        `outcome_constraints:supported` only when it has outcome constraints (a lookup and a
        truth test: cheap, snake_case).

        Returns:
            The tag, or an empty set for a goal with no outcome constraint. Only tags that the
            registry entry declares.
        """
        if not self.config.outcome_constraints:
            return frozenset()
        return super().needs & frozenset({_OUTCOME_CONSTRAINTS})

    def validateAgainst(self, problem: Problem) -> None:
        """Checks that every outcome the goal names exists in `problem` and is of an accepted
        kind.

        The outcomes to optimize are checked first, in the order written, and then the
        outcomes of the constraints, in the order written. The kinds accepted are the
        `outcome_kinds` of the configuration. That the weights match the outcomes, and that a
        bound is finite, are checked when the configuration is built and not here.

        Args:
            problem: What can be varied and what is measured.

        Raises:
            ConfigError: if an outcome is not one of the problem's, or its kind is not accepted
                (a binary label has no direction). The message names the goal, the field of its
                configuration that is at fault (`outcomes.<name>` or
                `outcome_constraints[<position>].outcome`) and a way to fix it.
        """
        kinds = self.config.outcome_kinds
        for name in self.config.outcomes:
            self.requireOutcome(problem, name, field_path=f"outcomes.{name}", kinds=kinds)
        for position, constraint in enumerate(self.config.outcome_constraints):
            self.requireOutcome(
                problem,
                constraint.outcome,
                field_path=f"outcome_constraints[{position}].outcome",
                kinds=kinds,
            )

    def _isScalar(self) -> bool:
        """Whether the goal has one scalar objective: one outcome, or several that `weights`
        combines into one preference.

        Returns:
            `True` for one outcome or for weighted outcomes, `False` for a Pareto problem.
        """
        return len(self.config.outcomes) == 1 or self.config.weights is not None
