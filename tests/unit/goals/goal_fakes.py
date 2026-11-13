"""Fakes shared by the tests of `planexs.goals`: problems and goals.

A fake goal stands for a real one (roadmap tasks P4-03 to P4-05) that does not exist yet, so
that the contract of the kind (registration, tags, validation against a problem, recipes) is
tested on its own. Kept in a module of its own and not in `conftest.py`, so that the test files
can import what is not a fixture.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, create_model

from planexs import Real, SearchSpace
from planexs.core.ComponentConfig import ComponentConfig
from planexs.domain.Outcome import Outcome
from planexs.domain.Problem import Problem
from planexs.goals.AbstractGoal import AbstractGoal
from planexs.goals.registerGoal import registerGoal

#: The outcomes of `makeProblem`: two continuous ones and one binary one.
CONTINUOUS = ("yield", "cost")
BINARY = "phase"


def makeProblem(n_parameters: int = 2) -> Problem:
    """A problem with `n_parameters` real parameters, two continuous outcomes and a binary one.

    Args:
        n_parameters: How many parameters the search space has, so that a recipe that depends
            on the size of the problem can be tested.
    """
    return Problem(
        search_space=SearchSpace(
            [Real(f"x{index}", 0.0, 1.0) for index in range(n_parameters)]
        ),
        outcomes=[
            *(Outcome(name=name, type="continuous") for name in CONTINUOUS),
            Outcome(name=BINARY, type="binary"),
        ],
    )


def fakeConfig(name: str) -> type[ComponentConfig]:
    """A configuration schema as real goals will write theirs: every field described, the
    `type` field pinned to `name`.

    Args:
        name: The registered name the schema is for.
    """
    schema: type[ComponentConfig] = create_model(
        f"{name.title().replace('_', '')}Config",
        __base__=ComponentConfig,
        type=(Literal[name], Field(default=name, description=f"The registered name, {name}.")),  # type: ignore[valid-type]
        outcomes=(
            list[str],
            Field(min_length=1, description="The names of the outcomes the goal is about."),
        ),
        constrained=(
            bool,
            Field(default=False, description="Whether the goal has constraints on outcomes."),
        ),
    )
    return schema


class _FakeBehaviour(AbstractGoal[Any]):
    """What the fake goals share: the checks, the recipe and the analysis of a real goal."""

    def validateAgainst(self, problem: Problem) -> None:
        """Every outcome named must exist and be continuous."""
        for position, name in enumerate(self.config.outcomes):
            self.requireOutcome(
                problem, name, field_path=f"outcomes[{position}]", kinds=("continuous",)
            )

    def recipes(self, problem: Problem) -> list[dict[str, Any]]:
        """Two recipes, the first of which depends on the size of the problem."""
        return [
            {"type": "fake_policy_v1", "n_points": 3 * len(problem.search_space.parameters)},
            {"type": "fake_policy_v1", "n_points": 1},
        ]

    def defaultAnalyses(self) -> list[dict[str, Any]]:
        """One analysis."""
        return [{"type": "fake_analysis_v1"}]


def defineFake(
    name: str = "fake_goal_v1",
    *,
    needs: frozenset[str] = frozenset(),
    supports: frozenset[str] = frozenset({"goal:optimize"}),
    description: str = "A fake goal for tests.",
) -> type[AbstractGoal[Any]]:
    """Registers a fake goal in the kind that is current, and returns its class.

    Its tags are the ones the registry entry declares, unchanged. Call it inside a test that
    uses the `isolatedGoalKind` fixture: a registered name is permanent, and must not reach the
    real registry.

    Args:
        name: The registered name.
        needs: The capability tags it declares as needs.
        supports: The capability tags it declares as supports.
        description: Its one-line description.

    Returns:
        The registered class.
    """

    @registerGoal(
        name, config=fakeConfig(name), description=description, needs=needs, supports=supports
    )
    class FakeGoal(_FakeBehaviour):
        """A goal that checks its outcomes and proposes a recipe."""

    return FakeGoal


def defineRefining(name: str = "refining_v1") -> type[AbstractGoal[Any]]:
    """Registers a fake goal whose tags depend on its configuration, and returns its class.

    The entry declares the union of what any configuration can state: it supports
    `goal:optimize` and both `objectives:` tags, and it needs `toy:constrained` (a tag that the
    test registers in its own vocabulary). An instance supports `objectives:single` for one
    outcome and `objectives:multi` for several, and needs `toy:constrained` only when its
    configuration says `constrained`.

    Args:
        name: The registered name.

    Returns:
        The registered class.
    """

    @registerGoal(
        name,
        config=fakeConfig(name),
        description="A fake goal whose tags depend on its configuration.",
        needs=frozenset({"toy:constrained"}),
        supports=frozenset({"goal:optimize", "objectives:single", "objectives:multi"}),
    )
    class RefiningGoal(_FakeBehaviour):
        """A goal that narrows the tags of its entry to its own configuration."""

        @property
        def supports(self) -> frozenset[str]:
            """`goal:optimize` and the objectives tag that fits the number of outcomes."""
            kind = "single" if len(self.config.outcomes) == 1 else "multi"
            return frozenset({"goal:optimize", f"objectives:{kind}"})

        @property
        def needs(self) -> frozenset[str]:
            """`toy:constrained` if the goal has constraints, nothing otherwise."""
            return frozenset({"toy:constrained"}) if self.config.constrained else frozenset()

    return RefiningGoal
