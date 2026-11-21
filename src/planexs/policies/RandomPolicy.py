"""Random search: a policy that draws its points from a design and ignores the observations."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Self

from pydantic import JsonValue

from planexs.core.ConfigError import ConfigError
from planexs.core.RuntimeConfig import RuntimeConfig
from planexs.designs.AbstractDesign import AbstractDesign
from planexs.designs.DesignContext import DesignContext
from planexs.designs.designRegistry import designRegistry
from planexs.domain.Problem import Problem
from planexs.domain.TrialState import TrialState
from planexs.encoding.EncodedConstraints import EncodedConstraints
from planexs.encoding.SearchSpaceEncoder import SearchSpaceEncoder
from planexs.encoding.torch_runtime import torchGenerator
from planexs.goals.AbstractGoal import AbstractGoal
from planexs.policies.AbstractPolicy import AbstractPolicy
from planexs.policies.RandomPolicyConfig import RandomPolicyConfig
from planexs.policies.registerPolicy import registerPolicy
from planexs.policies.StudyStateView import StudyStateView
from planexs.policies.Suggestion import Suggestion

#: The path of the stream that the nested design draws from, as the specification names a
#: component (specification 5.11): the design of the policy.
_DESIGN_PATH = "policy.design"


@registerPolicy(
    "random_policy_v1",
    config=RandomPolicyConfig,
    description="Random search: draws points from a design, ignoring the observations.",
)
class RandomPolicy(AbstractPolicy[RandomPolicyConfig]):
    """Random search: every suggestion is a point that a design draws, whatever the history
    says (specification 5.9, FR27).

    It is an *opaque* policy: it honours the contract of `AbstractPolicy` with no surrogate, no
    acquisition and no optimizer, and a study treats it like any other policy. It is the
    baseline that a model-based policy is measured against, and the second implementation of
    the policy seam (specification 3.10) after the composed one.

    **Where the points come from.** The design is a component, selected by name in the
    configuration (`design`) and obtained through the `design` registry, so that the policy
    never names one implementation: `random_v1` gives independent uniform points, `sobol_v1`
    a space-filling sequence, and a plugin's design works as well. The policy hands the design
    a context with the encoder of the problem, the constraints rewritten on the encoded
    columns and a generator, and decodes what it proposes. The design guarantees that every
    point satisfies the constraints and decodes to a legal parameterization (specification
    5.8), so every suggestion is legal and feasible. The policy always asks for as many
    points as it was asked for (`n`), and not for the design's own size.

    **What it ignores.** The observations: it reads neither the outcomes nor the number of
    completed trials, and it keeps no state, so `exportState` gives an empty dictionary and
    `importState` accepts one. Whatever the history, a suggestion depends on the study seed,
    the ask index and the pending trials, and on nothing else.

    **Pending points.** It honours them (specification 5.2): a point that repeats the
    parameterization of a pending trial is not suggested, and neither is one that repeats
    another point of the same batch, so that a worker that asks while others are still
    evaluating, or a batch, never receives a point that is already being evaluated. Whether
    two points repeat each other is decided on the decoded parameterization, because that is
    what the caller evaluates: in a finite space two draws often land on the same one. A
    repeated point is replaced by a new draw, from a new stream, up to `max_redraws` times.
    The parameterizations of completed, failed and abandoned trials are not avoided: random
    search ignores what it has learnt, and a point that has been evaluated may be drawn again.

    **Randomness.** Every number comes from the sources of the view (specification 3.7): the
    first draw uses the stream of `policy.design` for the ask, and the redraws use streams
    of their own, `policy.design.redraw.1` and so on, so that a redraw never offers again the
    points that were repeats. The same study seed, ask index and history give the same
    suggestions.

    **Capability tags.** The entry declares none. The policy needs nothing from the other
    participants, and it supports every kind of search space and constraint, since the design
    honours them; those tags are not in the vocabulary yet (they are registered with the
    optimizers, roadmap task P5-11), and a declaration of support for them is added to the
    entry when they are.

    Args:
        config: The validated configuration of the policy.
        runtime: The precision, the device and the determinism of the study.
        encoder: The encoder of the problem's search space, built with the settings of the
            configuration.
        constraints: The constraints of the search space, rewritten on the encoded columns.
        design: The design that the points are drawn from.
    """

    def __init__(
        self,
        config: RandomPolicyConfig,
        *,
        runtime: RuntimeConfig,
        encoder: SearchSpaceEncoder,
        constraints: EncodedConstraints,
        design: AbstractDesign[Any],
    ) -> None:
        """Keeps the configuration and what the draws are made with."""
        super().__init__(config)
        self._runtime = runtime
        self._encoder = encoder
        self._constraints = constraints
        self._design = design

    @classmethod
    def fromConfig(
        cls,
        config: RandomPolicyConfig,
        *,
        problem: Problem,
        goal: AbstractGoal[Any],
        runtime: RuntimeConfig,
    ) -> Self:
        """Builds the policy for a problem: its encoder, its constraints and its design.

        The design is looked up in the `design` registry by the `type` of the nested
        configuration and built from it. Nothing is drawn here.

        Args:
            config: The validated configuration.
            problem: What can be varied and what is measured: its search space is what the
                points are drawn in.
            goal: What the study is for. Not used: random search does not look at the goal.
            runtime: The precision, the device and the determinism of the study.

        Returns:
            The policy.

        Raises:
            RegistryError: if the nested design is not registered.
        """
        del goal  # random search does not look at what the study is for
        space = problem.search_space
        encoder = SearchSpaceEncoder(space, runtime, config.encoder)
        constraints = EncodedConstraints(space, encoder, config.sampling)
        design = designRegistry().lookup(config.design.type).builder(config.design)
        return cls(
            config, runtime=runtime, encoder=encoder, constraints=constraints, design=design
        )

    def suggest(self, view: StudyStateView, n: int) -> list[Suggestion]:
        """Draws `n` parameterizations from the design, avoiding the pending ones.

        Args:
            view: The read-only state of the study. Only its problem, its pending trials, its
                ask index and its random sources are read.
            n: The number of suggestions, one or more.

        Returns:
            Exactly `n` suggestions, each legal and feasible, none equal to a pending
            parameterization or to another suggestion of the batch, each tagged with the
            phase `design` and the registered name of the design.

        Raises:
            ConfigError: if `n` is not a whole number of one or more.
            RuntimeError: if the design leaves no feasible point to propose, or if after
                `max_redraws` redraws fewer than `n` parameterizations that are new were
                found (the space is too small for the batch, or nearly taken by pending
                trials).
        """
        self._requireCount(n)
        taken = {_keyOf(trial.parameters) for trial in view.trialsIn(TrialState.pending)}
        metadata = {"phase": "design", "design": self.config.design.type}
        chosen: list[Suggestion] = []
        for redraw in range(self.config.max_redraws + 1):
            path = _DESIGN_PATH if redraw == 0 else f"{_DESIGN_PATH}.redraw.{redraw}"
            context = DesignContext(
                problem=view.problem,
                encoder=self._encoder,
                constraints=self._constraints,
                trials=view.trials,
                generator=torchGenerator(view.seedFor(path), self._runtime),
            )
            for row in self._design.propose(context, n - len(chosen)):
                parameters = self._encoder.decode(row)
                key = _keyOf(parameters)
                if key not in taken:
                    taken.add(key)
                    chosen.append(Suggestion(parameters=parameters, metadata=metadata))
            if len(chosen) == n:
                return chosen
        raise RuntimeError(
            f"The {self.label} found only {len(chosen)} of the {n} parameterizations asked "
            f"for that differ from the pending ones and from each other, after "
            f"{self.config.max_redraws} redraws of the design {self.config.design.type!r}. "
            f"The search space may be too small for the batch, or nearly all of it is being "
            f"evaluated. Wait for pending trials to be told, ask for fewer points, or raise "
            f"'max_redraws'."
        )

    def exportState(self) -> dict[str, JsonValue]:
        """The state of the policy: none, so an empty dictionary.

        Returns:
            A new empty dictionary.
        """
        return {}

    def importState(self, state: Mapping[str, JsonValue]) -> None:
        """Takes back the state that `exportState` gave, which is empty.

        Args:
            state: The state to take back.

        Raises:
            ConfigError: if the state is not empty: it was not made by this policy.
        """
        if state:
            raise ConfigError(
                f"The {self.label} keeps no state, but was given one with the keys "
                f"{sorted(state)!r}.",
                component=self.label,
                field_path="state",
                suggestion="import only a state that the same policy exported",
            )

    def _requireCount(self, n: object) -> None:
        """Checks that the number of suggestions asked for is a whole number of one or more.

        Args:
            n: The number to check.

        Raises:
            ConfigError: if it is not.
        """
        if isinstance(n, bool) or not isinstance(n, int) or n < 1:
            raise ConfigError(
                f"The {self.label} needs the number of suggestions to be a whole number of "
                f"one or more, but got {n!r}.",
                component=self.label,
                field_path="n",
                suggestion="ask for one suggestion or more",
            )


def _keyOf(parameters: Mapping[str, Any]) -> tuple[tuple[str, Any], ...]:
    """A hashable form of a parameterization, so that two of them can be compared and kept.

    Args:
        parameters: The parameterization: a mapping from parameter name to value.

    Returns:
        The pairs of name and value, in the order of the names.
    """
    return tuple(sorted(parameters.items()))
