"""Fakes shared by the tests of `planexs.policies`: problems, views, trials and policies.

A fake policy stands for a real one (roadmap tasks P6-03, P6-06 and P10-07) that does not exist
yet, so that the contract of the kind (registration, tags, suggestions, notifications, state)
is tested on its own. Kept in a module of its own and not in `conftest.py`, so that the test
files can import what is not a fixture.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

from pydantic import Field, JsonValue, create_model

from planexs import Real, SearchSpace
from planexs.core.ComponentConfig import ComponentConfig
from planexs.core.ConfigError import ConfigError
from planexs.core.KindTable import KindTable
from planexs.core.resolveComponent import resolveComponent
from planexs.core.RngTree import RngTree
from planexs.core.RuntimeConfig import RuntimeConfig
from planexs.domain.Observation import Observation
from planexs.domain.Outcome import Outcome
from planexs.domain.Problem import Problem
from planexs.domain.Trial import Trial
from planexs.domain.TrialState import TrialState
from planexs.goals.AbstractGoal import AbstractGoal
from planexs.goals.ExploreGoal import ExploreGoal
from planexs.goals.ExploreGoalConfig import ExploreGoalConfig
from planexs.policies.AbstractPolicy import AbstractPolicy
from planexs.policies.policyRegistry import policyRegistry
from planexs.policies.registerPolicy import registerPolicy
from planexs.policies.StudyStateView import StudyStateView
from planexs.policies.Suggestion import Suggestion

#: The parameters of `makeProblem`, in order.
PARAMETERS = ("x0", "x1")

#: The path of the stream that the fake policy draws its numbers from.
FAKE_STREAM = "policy"


def makeProblem() -> Problem:
    """A problem with two real parameters in `[0, 1]` and one continuous outcome."""
    return Problem(
        search_space=SearchSpace([Real(name, 0.0, 1.0) for name in PARAMETERS]),
        outcomes=[Outcome(name="yield", type="continuous")],
    )


def makeGoal() -> AbstractGoal[Any]:
    """The explore goal, which names every outcome of the problem."""
    return ExploreGoal.fromConfig(ExploreGoalConfig())


def makeTrial(trial_id: int, state: TrialState = TrialState.pending) -> Trial:
    """A trial in the given state, with what that state needs: an observation or a reason."""
    parameters = {name: 0.1 * (trial_id + 1) for name in PARAMETERS}
    if state is TrialState.completed:
        observation = Observation(values={"yield": float(trial_id)})
        return Trial(id=trial_id, parameters=parameters, state=state, observation=observation)
    if state is TrialState.failed:
        return Trial(id=trial_id, parameters=parameters, state=state, reason="it broke")
    return Trial(id=trial_id, parameters=parameters, state=state)


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


def makeView(
    trials: Sequence[Trial] | None = None,
    *,
    history_version: int = 0,
    ask_index: int = 0,
    seed: int = 7,
) -> StudyStateView:
    """A study-state view over `makeProblem` and the explore goal.

    Args:
        trials: The history. `None` means none.
        history_version: The version of the history.
        ask_index: The index of the ask the view is for.
        seed: The seed of the random sources.
    """
    return StudyStateView(
        problem=makeProblem(),
        goal=makeGoal(),
        trials=[] if trials is None else trials,
        history_version=history_version,
        ask_index=ask_index,
        random_sources=RngTree(seed),
    )


def fakeConfig(name: str) -> type[ComponentConfig]:
    """A configuration schema as real policies will write theirs: every field described, the
    `type` field pinned to `name`.

    Args:
        name: The registered name the schema is for.
    """
    schema: type[ComponentConfig] = create_model(
        f"{name.title().replace('_', '')}Config",
        __base__=ComponentConfig,
        type=(Literal[name], Field(default=name, description=f"The registered name, {name}.")),  # type: ignore[valid-type]
        stateful=(
            bool,
            Field(default=True, description="Whether the policy keeps state of its own."),
        ),
    )
    return schema


class FakeBehaviour(AbstractPolicy[Any]):
    """What the fake policies share: they draw one number per parameter from their own stream.

    The policy counts what it was asked for and writes down every change it is told of, which
    is its state. The count is what a round trip must keep, and the notes are what the tests
    read to see that the policy was notified.
    """

    def __init__(self, config: Any) -> None:
        """Starts with nothing asked and nothing noted."""
        super().__init__(config)
        self.asked = 0
        self.notes: list[tuple[int, str | None, str]] = []

    def suggest(self, view: StudyStateView, n: int) -> list[Suggestion]:
        """`n` suggestions, each with one number per parameter, tagged with the ask.

        The metadata says how many suggestions the policy had made before this call (`served`,
        which is its state, so a policy that lost its state shows it) and how many trials
        were pending in the view (`pending`).
        """
        generator = view.generatorFor(FAKE_STREAM)
        served = self.asked
        self.asked += n
        return [
            Suggestion(
                parameters={name: float(generator.random()) for name in PARAMETERS},
                metadata={
                    "phase": "fake",
                    "ask": view.ask_index,
                    "pending": len(view.trialsIn(TrialState.pending)),
                    "served": served,
                },
            )
            for _ in range(n)
        ]

    def trialChanged(self, trial: Trial, previous_state: TrialState | None) -> None:
        """Writes the change down."""
        before = None if previous_state is None else previous_state.value
        self.notes.append((trial.id, before, trial.state.value))

    def exportState(self) -> dict[str, JsonValue]:
        """The count and the notes, as JSON-compatible data."""
        if not self.config.stateful:
            return {}
        return {"asked": self.asked, "notes": [list(note) for note in self.notes]}

    def importState(self, state: Mapping[str, JsonValue]) -> None:
        """Takes back what `exportState` gave; refuses anything else."""
        if not self.config.stateful:
            if state:
                raise ConfigError(
                    f"The {self.label} keeps no state but was given one.",
                    component=self.label,
                    field_path="state",
                    suggestion="import only a state that the same policy exported",
                )
            return
        if set(state) != {"asked", "notes"}:
            raise ConfigError(
                f"The {self.label} cannot read this state: it has the keys {sorted(state)!r}.",
                component=self.label,
                field_path="state",
                suggestion="import only a state that the same policy exported",
            )
        asked = state["asked"]
        notes = state["notes"]
        if not isinstance(asked, int) or not isinstance(notes, list):
            raise ConfigError(
                f"The {self.label} cannot read this state: 'asked' must be an int and "
                "'notes' a list.",
                component=self.label,
                field_path="state",
                suggestion="import only a state that the same policy exported",
            )
        self.asked = asked
        self.notes = [(int(a), None if b is None else str(b), str(c)) for a, b, c in notes]  # type: ignore[misc]


def defineFake(
    name: str = "fake_policy_v1",
    *,
    needs: frozenset[str] = frozenset(),
    supports: frozenset[str] = frozenset(),
    description: str = "A fake policy for tests.",
) -> type[AbstractPolicy[Any]]:
    """Registers a fake policy in the kind that is current, and returns its class.

    Its tags are the ones the registry entry declares, unchanged. Call it inside a test that
    uses the `isolatedPolicyKind` fixture: a registered name is permanent, and must not reach
    the real registry.

    Args:
        name: The registered name.
        needs: The capability tags it declares as needs.
        supports: The capability tags it declares as supports.
        description: Its one-line description.

    Returns:
        The registered class.
    """

    @registerPolicy(
        name, config=fakeConfig(name), description=description, needs=needs, supports=supports
    )
    class FakePolicy(FakeBehaviour):
        """A policy that draws numbers, notes what it is told and keeps a count."""

    return FakePolicy


def defineRefining(name: str = "refining_v1") -> type[AbstractPolicy[Any]]:
    """Registers a fake policy whose tags depend on its configuration, and returns its class.

    The entry declares the union of what any configuration can state: it supports `toy:batch`
    and `toy:resumable`, and it needs `toy:model`. An instance supports `toy:batch` always and
    `toy:resumable` only if it keeps state (`stateful`), and needs `toy:model` only if it keeps
    state too. The test registers the three tags in its own vocabulary first, with
    `registerToyTags`.

    Args:
        name: The registered name.

    Returns:
        The registered class.
    """

    @registerPolicy(
        name,
        config=fakeConfig(name),
        description="A fake policy whose tags depend on its configuration.",
        needs=frozenset({"toy:model"}),
        supports=frozenset({"toy:batch", "toy:resumable"}),
    )
    class RefiningPolicy(FakeBehaviour):
        """A policy that narrows the tags of its entry to its own configuration."""

        @property
        def supports(self) -> frozenset[str]:
            """`toy:batch`, and `toy:resumable` if the policy keeps state."""
            tags = {"toy:batch"}
            if self.config.stateful:
                tags.add("toy:resumable")
            return frozenset(tags)

        @property
        def needs(self) -> frozenset[str]:
            """`toy:model` if the policy keeps state, nothing otherwise."""
            return frozenset({"toy:model"}) if self.config.stateful else frozenset()

    return RefiningPolicy


def registerToyTags(table: KindTable) -> None:
    """Registers the capability tags of the fakes in the vocabulary of a swapped table.

    Args:
        table: The table that the `isolatedPolicyKind` fixture returned, whose vocabulary is a
            copy: a tag is permanent, and must not reach the real one.
    """
    table.vocabulary.register("toy:batch", "The policy can propose several points at once.")
    table.vocabulary.register("toy:model", "A participant provides a model of the outcomes.")
    table.vocabulary.register("toy:resumable", "The policy continues from an exported state.")


def buildFake(name: str = "fake_policy_v1", **fields: Any) -> AbstractPolicy[Any]:
    """Builds a registered fake the way the study does: from a resolved configuration.

    Args:
        name: The registered name of the fake, which must be defined in the current kind.
        **fields: Settings of its configuration other than `type`.
    """
    resolved = resolveComponent(policyRegistry(), {"type": name, **fields})
    policy: AbstractPolicy[Any] = resolved.entry.builder(
        resolved.config, problem=makeProblem(), goal=makeGoal(), runtime=RuntimeConfig()
    )
    return policy
