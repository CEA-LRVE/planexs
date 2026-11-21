"""The contract of the `policy` kind of components."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, Generic, Self, TypeVar

from pydantic import JsonValue

from planexs.core.ComponentConfig import ComponentConfig
from planexs.core.RuntimeConfig import RuntimeConfig
from planexs.domain.Problem import Problem
from planexs.domain.Trial import Trial
from planexs.domain.TrialState import TrialState
from planexs.goals.AbstractGoal import AbstractGoal
from planexs.policies.policyRegistry import policyRegistry
from planexs.policies.StudyStateView import StudyStateView
from planexs.policies.Suggestion import Suggestion

ConfigT = TypeVar("ConfigT", bound=ComponentConfig)


class AbstractPolicy(ABC, Generic[ConfigT]):
    """The part of a study that decides which parameterizations to evaluate next
    (specification 5.9, 6; ADR-0013).

    A policy answers one question: given the state of the study, which parameterizations
    should be evaluated next. It has three operations, and a concrete policy is a subclass of
    this class, registered under a permanent name with `registerPolicy` together with the
    configuration schema that holds its settings:

    - `suggest` answers the question. It receives a read-only `StudyStateView` and returns
      suggestions.
    - `trialChanged` tells it of every change of a trial's state, so that a policy that keeps
      a picture of the history of its own (an adapter that mirrors the trials into a client)
      can keep it current.
    - `exportState` and `importState` give its internal state to persistence and take it back,
      so that a reloaded study makes the same next suggestions as the uninterrupted one would
      have (specification 5.11).

    There is one contract and two forms of policy, and a study treats them identically. A
    *composed* policy is made of a design, a surrogate, an acquisition and an acquisition
    optimizer. An *opaque* policy is any object that honours this contract: random search, a
    grid, an adapter to another engine, or the caller's own code. Neither form is given more
    than the contract offers, and neither is trusted with more than it: the study validates
    every suggestion against the problem before it records a pending trial, so no policy can
    leave the history in an illegal state.

    **Construction.** The registry entry's builder is `fromConfig`. A policy is built for a
    problem and a goal, and with the runtime settings of the study, because what it builds from
    its configuration can depend on them: an adapter creates its client from the problem and
    the goal, and a composed policy builds the encoder of the problem's search space, which
    holds the precision and the device. The default keeps only the configuration, so a policy
    that needs none of the three overrides nothing. A builder is free of side effects at
    import and registration time, and deterministic given its configuration and its inputs
    (specification 6); a policy never draws from a global random state, only from the sources
    that a view gives it.

    **Capability tags.** A policy is a participant of the compatibility check (`Participant`):
    it has a `label`, and the tags it `needs` some participant to support and `supports`
    itself (specification 5.12). The default is the tags of the registry entry, which describe
    the policy *type*; a policy whose tags depend on its configuration overrides the two
    properties, as a goal does. A composed policy reports the tags of its components.

    Args:
        config: The validated configuration of the policy, an instance of the schema it is
            registered with.

    Attributes:
        config: The configuration the policy was made with. It is a read-only attribute.
    """

    def __init__(self, config: ConfigT) -> None:
        """Keeps the validated configuration."""
        self._config = config

    @property
    def config(self) -> ConfigT:
        """The configuration this policy was made with (a stored value: cheap, snake_case)."""
        return self._config

    @classmethod
    def fromConfig(
        cls,
        config: ConfigT,
        *,
        problem: Problem,
        goal: AbstractGoal[Any],
        runtime: RuntimeConfig,
    ) -> Self:
        """The builder of a registered policy: makes the policy object that a validated
        configuration describes, for a problem and a goal.

        This is what the registry entry stores as its builder, so
        `resolveComponent(registry, {...})` followed by
        `entry.builder(config, problem=..., goal=..., runtime=...)` gives the policy. The study
        calls it once, when it is built, after the goal has been validated against the problem
        (ADR-0013).

        The default keeps the configuration and ignores the rest. A policy that needs the
        problem, the goal or the runtime settings to be built overrides this method, and must
        not touch the engine it adapts before it is called (specification 6: a builder has no
        side effect at import or registration time).

        Args:
            config: The configuration, an instance of the schema that the policy is registered
                with.
            problem: What can be varied and what is measured. It has been validated against
                the goal.
            goal: What the study is for.
            runtime: The precision, the device and the determinism of the study.

        Returns:
            The policy, an instance of the class this is called on.
        """
        del problem, goal, runtime  # for the policies that override this: see the docstring
        return cls(config)

    @property
    def label(self) -> str:
        """Who this policy is, as an error message of the compatibility check names it, for
        example `policy 'composed_v1'` (a string built from the stored name: cheap,
        snake_case).
        """
        return f"policy {self._config.type!r}"

    @property
    def needs(self) -> frozenset[str]:
        """The capability tags this policy needs some participant to support, for its own
        configuration (a lookup of the entry's tags: cheap, snake_case).

        The default is what the registry entry declares for the type. A policy whose needs
        depend on its configuration overrides this, and reports only tags that the entry
        declares.
        """
        return policyRegistry().lookup(self._config.type).needs

    @property
    def supports(self) -> frozenset[str]:
        """The capability tags this policy supports, for its own configuration (a lookup of
        the entry's tags: cheap, snake_case).

        The default is what the registry entry declares for the type. A policy whose tags
        depend on its configuration overrides this, and reports only tags that the entry
        declares. The declaration must be truthful: a tag the policy does not honour makes the
        compatibility check accept a study that cannot run (specification 5.12).
        """
        return policyRegistry().lookup(self._config.type).supports

    @abstractmethod
    def suggest(self, view: StudyStateView, n: int) -> list[Suggestion]:
        """Suggests the parameterizations to evaluate next.

        Asking never requires the previous suggestions to have been reported (specification
        5.2): the pending trials of the view are part of the state, and a policy takes them
        into account, so that a batch, or a worker that asks while others are still
        evaluating, does not receive a point that is already being evaluated.

        Every random number comes from the sources of the view (`StudyStateView.seedFor` and
        `generatorFor`), and from nowhere else: the same configuration, history and ask index
        give the same suggestions (specification 3.7). The policy does not change the view, and
        does not keep a reference to its trials beyond the call: the study owns the history.

        The suggestions are not recorded here. The study checks each against the problem, and
        raises an error that names the policy for an illegal one, then records the trials and
        notifies the policy (`trialChanged`).

        Args:
            view: The read-only state of the study: the problem, the goal, every trial, the
                version of the history, the index of this ask and the random sources.
            n: The number of suggestions, one or more.

        Returns:
            Exactly `n` suggestions, each a parameterization that satisfies the constraints of
            the problem, with the policy's metadata (a composed policy tags the phase that
            produced each one).

        Raises:
            RuntimeError: if the policy cannot produce suggestions: its constraints leave no
                point to propose, say. The error says why.
        """

    def trialChanged(self, trial: Trial, previous_state: TrialState | None) -> None:
        """Is told that a trial of the study has changed its state, or has just appeared.

        The study calls it for every change, in the order they happen: after an ask has
        recorded its pending trials, and after `tell`, `fail`, `abandon` and `attach`. A policy
        that works from the view alone, as a composed policy does, needs nothing here, and the
        default does nothing. A policy that keeps a picture of the history of its own uses it:
        an adapter mirrors each change into the client of the engine it adapts.

        It is a notification, not a question: the policy cannot refuse the change, and it must
        not raise for a change that is legal. The trial is an immutable record, and the new
        state of the study is in the next view.

        Args:
            trial: The trial in its new state. A completed trial carries its observation, and a
                failed one its reason.
            previous_state: The state the trial was in, or `None` when the trial has just been
                created: suggested by an ask (it is then pending), or attached from outside
                (it is then completed or failed). A move between states is one of the three
                that ADR-0012 allows, from pending to completed, failed or abandoned.
        """

    @abstractmethod
    def exportState(self) -> dict[str, JsonValue]:
        """Exports the internal state of the policy, so that persistence can keep it.

        The state is opaque to the study and to the snapshot: it is kept and given back, never
        read. It is plain JSON-compatible data (dicts, lists, strings, numbers, booleans and
        `None`, with finite numbers), because a snapshot is data and loading one never
        executes code (specification 5.11). A policy with no state of its own returns an empty
        dictionary. State that can be rebuilt from the history, which a composed policy reads
        from the view, is not exported: the state is what the history does not say.

        Returns:
            A new dictionary that the policy does not keep a reference to, so that a caller who
            changes it does not change the policy.
        """

    @abstractmethod
    def importState(self, state: Mapping[str, JsonValue]) -> None:
        """Takes back a state that `exportState` gave, on a policy built from the same
        configuration.

        The study calls it when it loads a snapshot, on a policy that was just built, so that
        the policy continues as the exported one would have. It replaces whatever state the
        policy holds. Exporting after importing gives an equal state: the round trip is exact
        (`importState(p.exportState())` leaves `p` as it was).

        Args:
            state: A state that `exportState` returned, in the form it was stored in: the
                mapping may be read-only and may have gone through JSON, so a number that was
                an int is an int and a tuple has become a list.

        Raises:
            ConfigError: if the state is not one that this policy exports, for example one
                from another type of policy or from a version that it cannot read. The error
                names the policy and the part of the state at fault.
        """
