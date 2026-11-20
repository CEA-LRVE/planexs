"""The goal of learning outcomes as well as possible, with no direction."""

from __future__ import annotations

from planexs.domain.Problem import Problem
from planexs.goals.AbstractGoal import AbstractGoal
from planexs.goals.ExploreGoalConfig import ExploreGoalConfig
from planexs.goals.registerGoal import registerGoal


@registerGoal(
    "explore_v1",
    config=ExploreGoalConfig,
    description="Learn the outcomes as well as possible, with no direction (FR21).",
    supports=frozenset({"goal:explore"}),
)
class ExploreGoal(AbstractGoal[ExploreGoalConfig]):
    """Learn the named outcomes as well as possible, with no direction (specification 5.7).

    Nothing is maximized, minimized or searched for: the aim is a surrogate that is accurate
    over the whole search space, which is active learning (UC-6). The goal names the outcomes
    to learn and, by default, names all of them.

    Like every goal it computes nothing (ADR-0013): it holds the names, which the acquisitions
    and the analyses of an exploration read through `effectiveOutcomes`, and it declares
    `goal:explore`, which an exploration acquisition needs. It proposes no recipe and no
    default analysis yet: the recipes come with the roadmap task P6-08.

    What the configuration cannot say, because it depends on the problem, is checked in
    `validateAgainst`: every outcome that is named must be an outcome of the problem. Any
    kind of outcome can be learned, so the kind is not checked.
    """

    def validateAgainst(self, problem: Problem) -> None:
        """Checks that every outcome the goal names is an outcome of the problem.

        Args:
            problem: What can be varied and what is measured.

        Raises:
            ConfigError: if a name is not an outcome of the problem. The error names the goal,
                the field at fault (`outcomes[i]`, the position of the name) and the outcomes
                the problem does have.
        """
        self.effectiveOutcomes(problem)

    def effectiveOutcomes(self, problem: Problem) -> tuple[str, ...]:
        """The names of the outcomes to learn, once the goal is checked against `problem`.

        They are the `outcomes` of the configuration in the order given, and when there are
        none, which means all, the outcomes of the problem in the order of the problem. It is
        a method and not a property because the default depends on the problem
        (specification 10.2).

        Args:
            problem: The problem the goal is used with.

        Returns:
            The names, at least one, each once.

        Raises:
            ConfigError: for any of the reasons `validateAgainst` lists.
        """
        names = self.config.outcomes
        if names is None:
            return tuple(outcome.name for outcome in problem.outcomes)
        for position, name in enumerate(names):
            self.requireOutcome(problem, name, field_path=f"outcomes[{position}]")
        return tuple(names)
