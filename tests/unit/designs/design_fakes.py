"""Fakes and builders shared by the tests of `planexs.designs`.

Spaces with the constraints of the acceptance test of roadmap task P4-06 (linear and nonlinear,
and a mixture that sums to one), problems, trials in every state, build contexts, and a fake
design that stands for a plugin's, so that the contract of the kind is tested on its own.
"""

from __future__ import annotations

from typing import Any, Literal

import torch
from pydantic import Field, create_model
from torch import Tensor

from planexs import (
    Categorical,
    Integer,
    LinearConstraint,
    NonlinearConstraint,
    Real,
    SearchSpace,
)
from planexs.core.ComponentConfig import ComponentConfig
from planexs.core.RuntimeConfig import RuntimeConfig
from planexs.designs.AbstractDesign import AbstractDesign
from planexs.designs.DesignContext import DesignContext
from planexs.designs.registerDesign import registerDesign
from planexs.domain.Observation import Observation
from planexs.domain.Outcome import Outcome
from planexs.domain.Problem import Problem
from planexs.domain.Trial import Trial
from planexs.domain.TrialState import TrialState
from planexs.encoding.EncodedConstraints import EncodedConstraints
from planexs.encoding.FeasibleSamplingConfig import FeasibleSamplingConfig
from planexs.encoding.SearchSpaceEncoder import SearchSpaceEncoder
from planexs.encoding.torch_runtime import torchGenerator

#: The parameterization every fake trial carries: the record does not check it.
PARAMETERS: dict[str, float | int | str | bool] = {"x": 5.0, "y": 5.0, "n": 3, "kind": "a"}


def makeSpace() -> SearchSpace:
    """A mixed space with a linear and a nonlinear constraint: `x + y <= 12` and `x * y >= 4`.

    Two reals, an integer on a grid and an unordered categorical, so that snapping, one-hot
    columns and both kinds of constraint are all exercised.
    """
    return SearchSpace(
        [
            Real("x", 0.0, 10.0),
            Real("y", 0.0, 10.0),
            Integer("n", 1, 9),
            Categorical("kind", ["a", "b", "c"]),
        ],
        [
            LinearConstraint({"x": 1.0, "y": 1.0}, 12.0, "<="),
            NonlinearConstraint("x * y", ">=", 4.0),
        ],
    )


def makeNonlinearSpace(bound: float) -> SearchSpace:
    """Two reals in `[0, 10]` with the single constraint `x * y >= bound`.

    A large bound leaves a small feasible region, so that the sampler needs several rounds.
    """
    return SearchSpace(
        [Real("x", 0.0, 10.0), Real("y", 0.0, 10.0)],
        [NonlinearConstraint("x * y", ">=", bound)],
    )


def makeMixtureSpace() -> SearchSpace:
    """Three reals in `[0, 1]` that sum to one: an equality, drawn by the hit-and-run chain."""
    return SearchSpace(
        [Real("a", 0.0, 1.0), Real("b", 0.0, 1.0), Real("c", 0.0, 1.0)],
        [LinearConstraint({"a": 1.0, "b": 1.0, "c": 1.0}, 1.0, "=")],
    )


def makeTrial(trial_id: int, state: TrialState) -> Trial:
    """A trial in the given state, with what that state needs: an observation or a reason."""
    if state is TrialState.completed:
        observation = Observation(values={"o": float(trial_id)})
        return Trial(id=trial_id, parameters=PARAMETERS, state=state, observation=observation)
    if state is TrialState.failed:
        return Trial(id=trial_id, parameters=PARAMETERS, state=state, reason="it broke")
    return Trial(id=trial_id, parameters=PARAMETERS, state=state)


def makeTrials(**counts: int) -> list[Trial]:
    """Trials with the given number in each state, for example `makeTrials(completed=3)`.

    Args:
        **counts: The number of trials per state name (`pending`, `completed`, `failed`,
            `abandoned`). A state that is not given has none.

    Returns:
        The trials, with consecutive ids, in the order of the states as given.
    """
    trials: list[Trial] = []
    for name, number in counts.items():
        for _ in range(number):
            trials.append(makeTrial(len(trials), TrialState(name)))
    return trials


def makeContext(
    space: SearchSpace | None = None,
    trials: list[Trial] | None = None,
    seed: int = 7,
    runtime: RuntimeConfig | None = None,
    sampling: FeasibleSamplingConfig | None = None,
) -> DesignContext:
    """A design context over `space`, with real constraints and a seeded generator.

    Args:
        space: The search space. `None` means `makeSpace()`.
        trials: The history. `None` means none.
        seed: The seed of the generator.
        runtime: The runtime settings. `None` means `RuntimeConfig()`.
        sampling: The settings of the feasible sampler that the constraints carry. `None`
            means `FeasibleSamplingConfig()`.

    Returns:
        The context.
    """
    settings = RuntimeConfig() if runtime is None else runtime
    chosen = makeSpace() if space is None else space
    problem = Problem(search_space=chosen, outcomes=[Outcome(name="o", type="continuous")])
    encoder = SearchSpaceEncoder(chosen, settings)
    return DesignContext(
        problem=problem,
        encoder=encoder,
        constraints=EncodedConstraints(chosen, encoder, sampling),
        trials=[] if trials is None else trials,
        generator=torchGenerator(seed, settings),
    )


def assertLegal(context: DesignContext, points: Tensor) -> None:
    """Checks that every point decodes to a legal parameterization of the problem.

    Args:
        context: The context the points were proposed for.
        points: The encoded points, shape `(n, d)`.

    Raises:
        ParameterizationError: if a decoded point is not legal (a value out of range, a
            constraint that does not hold).
    """
    for row in points:
        context.problem.validateParameterization(context.encoder.decode(row))


def fakeConfig(name: str) -> type[ComponentConfig]:
    """A configuration schema as real designs will write theirs: every field described, the
    `type` field pinned to `name`.

    Args:
        name: The registered name the schema is for.
    """
    schema: type[ComponentConfig] = create_model(
        f"{name.title().replace('_', '')}Config",
        __base__=ComponentConfig,
        type=(Literal[name], Field(default=name, description=f"The registered name, {name}.")),  # type: ignore[valid-type]
        n_points=(
            int,
            Field(default=4, ge=0, description="The size of the design, in points."),
        ),
    )
    return schema


class RecordingDesign(AbstractDesign[Any]):
    """The least a design defines: its size, and a draw. It records what it was asked.

    Its size is `n_points` less the number of trials, whatever their state, and it draws
    points that all sit at the midpoint of the unit cube.
    """

    drawn: list[int]

    def __init__(self, config: Any) -> None:
        """Keeps the configuration and starts an empty record of the draws."""
        super().__init__(config)
        self.drawn = []

    def pointsNeeded(self, trials: Any) -> int:
        """`n_points` less the number of trials, never less than zero."""
        return max(0, self.config.n_points - len(trials))

    def drawPoints(self, context: DesignContext, n: int) -> Tensor:
        """Records `n` and returns that many points at the midpoint of the cube."""
        self.drawn.append(n)
        return torch.full((n, context.encoder.n_columns), 0.5, dtype=context.encoder.dtype)


def defineFake(
    name: str = "fake_design_v1",
    *,
    needs: frozenset[str] = frozenset(),
    supports: frozenset[str] = frozenset(),
    description: str = "A fake design for tests.",
) -> type[AbstractDesign[Any]]:
    """Registers a fake design in the kind that is current, and returns its class.

    Call it inside a test that uses the `isolatedDesignKind` fixture: a registered name is
    permanent, and must not reach the real registry.

    Args:
        name: The registered name.
        needs: The capability tags it needs.
        supports: The capability tags it supports.
        description: Its one-line description.

    Returns:
        The registered class.
    """

    @registerDesign(
        name, config=fakeConfig(name), description=description, needs=needs, supports=supports
    )
    class FakeDesign(RecordingDesign):
        """A design that draws midpoints."""

    return FakeDesign
