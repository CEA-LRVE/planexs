"""Unit tests for `AbstractPolicy`, the contract of the `policy` kind (roadmap task P6-01,
specification 5.9, 6, ADR-0013).

What is tested: the contract itself (abstract, configured once, built from a configuration, a
problem, a goal and the runtime settings), the capability tags a policy reports for its own
configuration, what `suggest` is given and returns, the notification of trial changes, and the
export and import of state that persistence relies on. A fake policy stands for a real one: the
contract is tested before any policy exists.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, Literal

import pytest
from policy_fakes import (
    FAKE_STREAM,
    PARAMETERS,
    FakeBehaviour,
    buildFake,
    defineFake,
    defineRefining,
    fakeConfig,
    makeGoal,
    makeProblem,
    makeTrial,
    makeTrials,
    makeView,
    registerToyTags,
)
from pydantic import Field, JsonValue

import planexs.policies  # noqa: F401
from planexs.core.ComponentConfig import ComponentConfig
from planexs.core.ConfigError import ConfigError
from planexs.core.KindTable import KindTable
from planexs.core.RegistryError import RegistryError
from planexs.core.resolveComponent import resolveComponent
from planexs.core.RuntimeConfig import RuntimeConfig
from planexs.domain.Trial import Trial
from planexs.domain.TrialState import TrialState
from planexs.policies.AbstractPolicy import AbstractPolicy
from planexs.policies.policyRegistry import policyRegistry
from planexs.policies.registerPolicy import registerPolicy
from planexs.policies.StudyStateView import StudyStateView
from planexs.policies.Suggestion import Suggestion


class MinimalConfig(ComponentConfig):
    """A schema for a policy that is built directly, without being registered."""

    type: Literal["minimal_v1"] = Field(default="minimal_v1", description="The name.")


class MinimalPolicy(AbstractPolicy[MinimalConfig]):
    """The least a policy has to define: how it suggests, and what it exports and imports."""

    def suggest(self, view: StudyStateView, n: int) -> list[Suggestion]:
        """`n` suggestions at the centre of the space."""
        return [Suggestion(parameters={name: 0.5 for name in PARAMETERS}) for _ in range(n)]

    def exportState(self) -> dict[str, JsonValue]:
        """A policy with no state of its own."""
        return {}

    def importState(self, state: Mapping[str, JsonValue]) -> None:
        """Nothing to take back."""


def buildMinimal() -> MinimalPolicy:
    """A policy of `MinimalPolicy`, made the way a study makes one."""
    return MinimalPolicy.fromConfig(
        MinimalConfig(), problem=makeProblem(), goal=makeGoal(), runtime=RuntimeConfig()
    )


class TestTheContract:
    def test_the_contract_cannot_be_instantiated(self) -> None:
        with pytest.raises(TypeError, match="abstract"):
            AbstractPolicy(MinimalConfig())  # type: ignore[abstract]

    def test_a_policy_has_three_abstract_operations(self) -> None:
        # What the contract cannot say for a policy: how it suggests and what its state is.
        # Notification has a default, because a policy that reads the view needs none.
        assert AbstractPolicy.__abstractmethods__ == {"suggest", "exportState", "importState"}

    @pytest.mark.parametrize("missing", ["suggest", "exportState", "importState"])
    def test_a_policy_that_leaves_one_out_cannot_be_instantiated(self, missing: str) -> None:
        operations: dict[str, Any] = {
            "suggest": lambda self, view, n: [],
            "exportState": lambda self: {},
            "importState": lambda self, state: None,
        }
        del operations[missing]
        Incomplete = type("Incomplete", (AbstractPolicy,), operations)

        with pytest.raises(TypeError, match=missing):
            Incomplete(MinimalConfig())

    def test_a_policy_that_defines_the_three_is_complete(self) -> None:
        assert isinstance(MinimalPolicy(MinimalConfig()), AbstractPolicy)

    def test_the_configuration_is_kept_as_it_was_given(self) -> None:
        config = MinimalConfig()

        policy = MinimalPolicy(config)

        assert policy.config is config

    def test_the_configuration_cannot_be_assigned(self) -> None:
        policy = MinimalPolicy(MinimalConfig())

        with pytest.raises(AttributeError):
            policy.config = MinimalConfig()  # type: ignore[misc]

    def test_a_policy_is_labelled_with_its_kind_and_name(self) -> None:
        assert buildMinimal().label == "policy 'minimal_v1'"


class TestBuildingFromAConfiguration:
    """`fromConfig` is the builder of the registry entry: what a study calls once."""

    def test_by_default_the_builder_keeps_the_configuration_and_ignores_the_rest(self) -> None:
        config = MinimalConfig()

        policy = MinimalPolicy.fromConfig(
            config, problem=makeProblem(), goal=makeGoal(), runtime=RuntimeConfig()
        )

        assert type(policy) is MinimalPolicy
        assert policy.config is config

    @pytest.mark.parametrize("name", ["problem", "goal", "runtime"])
    def test_the_problem_the_goal_and_the_runtime_are_given_by_keyword(
        self, name: str
    ) -> None:
        # Three inputs of the same shape are too easy to swap: the builder cannot be called
        # with them by position.
        inputs = {"problem": makeProblem(), "goal": makeGoal(), "runtime": RuntimeConfig()}
        positional = [MinimalConfig(), *inputs.values()]

        with pytest.raises(TypeError, match="positional"):
            MinimalPolicy.fromConfig(*positional)  # type: ignore[misc]
        del inputs[name]
        with pytest.raises(TypeError, match=name):
            MinimalPolicy.fromConfig(MinimalConfig(), **inputs)  # type: ignore[call-arg]

    def test_a_policy_that_needs_the_inputs_overrides_the_builder_and_receives_them(
        self, isolatedPolicyKind: KindTable
    ) -> None:
        # An adapter creates its client from the problem and the goal; a composed policy builds
        # the encoder of the search space with the runtime settings. The registry entry holds
        # the override, and the study calls it with the very objects it has.
        received: dict[str, Any] = {}
        problem, goal, runtime = makeProblem(), makeGoal(), RuntimeConfig()

        @registerPolicy(
            "adapting_v1",
            config=fakeConfig("adapting_v1"),
            description="An adapter for tests.",
        )
        class Adapting(FakeBehaviour):
            @classmethod
            def fromConfig(cls, config: Any, *, problem: Any, goal: Any, runtime: Any) -> Any:
                received.update(problem=problem, goal=goal, runtime=runtime)
                return super().fromConfig(config, problem=problem, goal=goal, runtime=runtime)

        resolved = resolveComponent(policyRegistry(), {"type": "adapting_v1"})

        policy = resolved.entry.builder(
            resolved.config, problem=problem, goal=goal, runtime=runtime
        )

        assert received["problem"] is problem
        assert received["goal"] is goal
        assert received["runtime"] is runtime
        assert isinstance(policy, Adapting)
        assert policy.config is resolved.config

    def test_building_does_not_suggest_and_does_not_notify(
        self, isolatedPolicyKind: KindTable
    ) -> None:
        defineFake()

        policy = buildFake()

        assert policy.exportState() == {"asked": 0, "notes": []}


class TestCapabilityTags:
    """A policy is a participant of the compatibility check: its tags are read from it."""

    def test_by_default_the_tags_are_the_ones_of_the_registry_entry(
        self, isolatedPolicyKind: KindTable
    ) -> None:
        registerToyTags(isolatedPolicyKind)
        defineFake(
            "plain_v1",
            needs=frozenset({"toy:model"}),
            supports=frozenset({"toy:batch"}),
        )

        policy = buildFake("plain_v1")

        assert policy.needs == {"toy:model"}
        assert policy.supports == {"toy:batch"}
        assert isinstance(policy.needs, frozenset)
        assert isinstance(policy.supports, frozenset)

    def test_a_policy_that_declares_nothing_reports_nothing(
        self, isolatedPolicyKind: KindTable
    ) -> None:
        defineFake("bare_v1")

        policy = buildFake("bare_v1")

        assert policy.needs == frozenset()
        assert policy.supports == frozenset()

    def test_a_policy_can_narrow_the_tags_of_its_entry_to_its_configuration(
        self, isolatedPolicyKind: KindTable
    ) -> None:
        registerToyTags(isolatedPolicyKind)
        defineRefining()
        entry = policyRegistry().lookup("refining_v1")

        stateful = buildFake("refining_v1")
        stateless = buildFake("refining_v1", stateful=False)

        # The entry declares the union of what any configuration can state ...
        assert entry.supports == {"toy:batch", "toy:resumable"}
        assert entry.needs == {"toy:model"}
        # ... and each policy reports what holds for its own.
        assert stateful.supports == {"toy:batch", "toy:resumable"}
        assert stateful.needs == {"toy:model"}
        assert stateless.supports == {"toy:batch"}
        assert stateless.needs == frozenset()

    def test_a_policy_that_is_not_registered_cannot_say_what_its_entry_declares(self) -> None:
        # Built by hand, under a name nobody registered. The default tags are those of the
        # registry entry, and there is no entry: the error is the registry's, naming the kind.
        policy = buildMinimal()

        with pytest.raises(RegistryError, match="Unknown policy 'minimal_v1'"):
            policy.needs  # noqa: B018
        with pytest.raises(RegistryError, match="Unknown policy 'minimal_v1'"):
            policy.supports  # noqa: B018


class TestSuggest:
    def test_it_returns_exactly_the_number_asked_for(
        self, isolatedPolicyKind: KindTable
    ) -> None:
        defineFake()
        policy = buildFake()

        for n in (1, 2, 5):
            suggestions = policy.suggest(makeView(), n)

            assert len(suggestions) == n
            assert all(isinstance(suggestion, Suggestion) for suggestion in suggestions)

    def test_each_suggestion_names_every_parameter_and_the_phase_that_made_it(
        self, isolatedPolicyKind: KindTable
    ) -> None:
        defineFake()

        (suggestion,) = buildFake().suggest(makeView(), 1)

        assert set(suggestion.parameters) == set(PARAMETERS)
        assert suggestion.phase == "fake"

    def test_the_same_policy_state_history_and_ask_index_give_the_same_suggestions(
        self, isolatedPolicyKind: KindTable
    ) -> None:
        # Determinism (specification 3.7): the numbers come from the sources of the view.
        defineFake()
        trials = makeTrials(completed=2, pending=1)

        first = buildFake().suggest(makeView(trials, ask_index=4, seed=11), 3)
        second = buildFake().suggest(makeView(trials, ask_index=4, seed=11), 3)

        assert first == second

    def test_another_ask_index_or_another_seed_gives_other_suggestions(
        self, isolatedPolicyKind: KindTable
    ) -> None:
        defineFake()
        reference = buildFake().suggest(makeView(ask_index=0, seed=11), 1)[0].parameters

        later_ask = buildFake().suggest(makeView(ask_index=1, seed=11), 1)[0].parameters
        other_seed = buildFake().suggest(makeView(ask_index=0, seed=12), 1)[0].parameters

        assert later_ask != reference
        assert other_seed != reference

    def test_the_stream_of_the_view_is_the_only_source_of_the_numbers(
        self, isolatedPolicyKind: KindTable
    ) -> None:
        defineFake()
        view = makeView(ask_index=3, seed=5)

        (suggestion,) = buildFake().suggest(view, 1)

        expected = view.generatorFor(FAKE_STREAM)
        assert [suggestion.parameters[name] for name in PARAMETERS] == [
            float(expected.random()) for _ in PARAMETERS
        ]

    def test_the_pending_trials_of_the_view_are_part_of_what_it_sees(
        self, isolatedPolicyKind: KindTable
    ) -> None:
        # Asking never requires the earlier suggestions to have been reported (5.2).
        defineFake()
        policy = buildFake()

        quiet = policy.suggest(makeView(makeTrials(completed=2)), 1)[0]
        busy = policy.suggest(makeView(makeTrials(completed=2, pending=3)), 1)[0]

        assert quiet.metadata["pending"] == 0
        assert busy.metadata["pending"] == 3

    def test_asking_changes_neither_the_view_nor_the_history_it_holds(
        self, isolatedPolicyKind: KindTable
    ) -> None:
        defineFake()
        trials = makeTrials(completed=2, pending=1)
        view = makeView(trials, history_version=6, ask_index=2)

        buildFake().suggest(view, 4)

        assert view.trials == tuple(trials)
        assert view.history_version == 6
        assert view.ask_index == 2

    def test_asking_does_not_record_anything_as_a_trial(
        self, isolatedPolicyKind: KindTable
    ) -> None:
        # Recording is the study's: it checks each suggestion against the problem first.
        defineFake()
        policy = buildFake()
        view = makeView(makeTrials(pending=1))

        policy.suggest(view, 3)

        assert len(view.trials) == 1
        assert policy.exportState()["notes"] == []  # type: ignore[index]

    def test_suggestions_are_values_that_a_trial_can_be_made_from(
        self, isolatedPolicyKind: KindTable
    ) -> None:
        defineFake()

        (suggestion,) = buildFake().suggest(makeView(), 1)
        trial = Trial(id=0, parameters=suggestion.parameters, metadata=suggestion.metadata)

        assert trial.state is TrialState.pending
        assert dict(trial.parameters) == dict(suggestion.parameters)

    def test_a_policy_that_cannot_suggest_says_why(self) -> None:
        class Exhausted(MinimalPolicy):
            def suggest(self, view: StudyStateView, n: int) -> list[Suggestion]:
                raise RuntimeError("the constraints leave no point to propose")

        with pytest.raises(RuntimeError, match="no point to propose"):
            Exhausted(MinimalConfig()).suggest(makeView(), 1)


class TestNotification:
    def test_by_default_a_policy_ignores_every_change(self) -> None:
        # A policy that reads the view needs nothing here, and says nothing.
        policy = buildMinimal()
        changes = [
            (makeTrial(0, TrialState.pending), None),
            (makeTrial(0, TrialState.completed), TrialState.pending),
            (makeTrial(1, TrialState.failed), TrialState.pending),
            (makeTrial(2, TrialState.abandoned), TrialState.pending),
            (makeTrial(3, TrialState.completed), None),
        ]

        results = [policy.trialChanged(trial, before) for trial, before in changes]

        assert results == [None] * len(changes)
        assert policy.exportState() == {}

    def test_a_policy_that_overrides_it_is_told_the_trial_and_the_state_it_left(self) -> None:
        seen: list[tuple[Trial, TrialState | None]] = []

        class Spy(MinimalPolicy):
            def trialChanged(self, trial: Trial, previous_state: TrialState | None) -> None:
                seen.append((trial, previous_state))

        policy = Spy(MinimalConfig())
        created = makeTrial(0, TrialState.pending)
        completed = makeTrial(0, TrialState.completed)

        policy.trialChanged(created, None)
        policy.trialChanged(completed, TrialState.pending)

        assert seen[0][0] is created and seen[0][1] is None
        assert seen[1][0] is completed and seen[1][1] is TrialState.pending

    def test_every_legal_change_is_told_in_the_order_it_happened(
        self, isolatedPolicyKind: KindTable
    ) -> None:
        # The moves of ADR-0012: created pending (an ask), attached completed or failed, and
        # from pending to completed, failed or abandoned.
        defineFake()
        policy = buildFake()

        policy.trialChanged(makeTrial(0, TrialState.pending), None)
        policy.trialChanged(makeTrial(1, TrialState.pending), None)
        policy.trialChanged(makeTrial(2, TrialState.completed), None)
        policy.trialChanged(makeTrial(3, TrialState.failed), None)
        policy.trialChanged(makeTrial(0, TrialState.completed), TrialState.pending)
        policy.trialChanged(makeTrial(1, TrialState.failed), TrialState.pending)
        policy.trialChanged(makeTrial(4, TrialState.abandoned), TrialState.pending)

        assert policy.exportState()["notes"] == [
            [0, None, "pending"],
            [1, None, "pending"],
            [2, None, "completed"],
            [3, None, "failed"],
            [0, "pending", "completed"],
            [1, "pending", "failed"],
            [4, "pending", "abandoned"],
        ]

    def test_a_notification_changes_the_state_that_is_exported(
        self, isolatedPolicyKind: KindTable
    ) -> None:
        # That is why the state is exported: what a policy learnt from the notifications is
        # not in the history it reads from a view.
        defineFake()
        policy = buildFake()
        before = policy.exportState()

        policy.trialChanged(makeTrial(0, TrialState.pending), None)

        assert policy.exportState() != before


class TestStateRoundTrip:
    """Export and import of opaque state (specification 5.11, ADR-0013)."""

    def policyWithAHistory(self) -> AbstractPolicy[Any]:
        policy = buildFake()
        policy.suggest(makeView(ask_index=0), 2)
        policy.trialChanged(makeTrial(0, TrialState.pending), None)
        policy.trialChanged(makeTrial(1, TrialState.pending), None)
        policy.trialChanged(makeTrial(0, TrialState.completed), TrialState.pending)
        return policy

    def test_the_state_is_plain_data_that_survives_json(
        self, isolatedPolicyKind: KindTable
    ) -> None:
        defineFake()

        state = self.policyWithAHistory().exportState()

        assert isinstance(state, dict)
        assert json.loads(json.dumps(state, allow_nan=False)) == state

    def test_a_policy_built_again_and_given_the_state_continues_as_the_first_would(
        self, isolatedPolicyKind: KindTable
    ) -> None:
        defineFake()
        uninterrupted = self.policyWithAHistory()
        reloaded = buildFake()
        reloaded.importState(uninterrupted.exportState())
        trials = makeTrials(completed=1, pending=1)

        after_uninterrupted = uninterrupted.suggest(makeView(trials, ask_index=1), 2)
        after_reloaded = reloaded.suggest(makeView(trials, ask_index=1), 2)

        assert after_reloaded == after_uninterrupted
        assert [s.metadata["served"] for s in after_reloaded] == [2, 2]

    def test_without_the_state_a_policy_built_again_is_not_the_same(
        self, isolatedPolicyKind: KindTable
    ) -> None:
        # The witness of the test above: what is imported is what makes the difference.
        defineFake()
        uninterrupted = self.policyWithAHistory()
        fresh = buildFake()

        after_uninterrupted = uninterrupted.suggest(makeView(ask_index=1), 1)
        after_fresh = fresh.suggest(makeView(ask_index=1), 1)

        assert after_fresh != after_uninterrupted

    def test_the_state_goes_through_json_and_comes_back_the_same(
        self, isolatedPolicyKind: KindTable
    ) -> None:
        defineFake()
        original = self.policyWithAHistory()
        stored = json.loads(json.dumps(original.exportState(), allow_nan=False))
        reloaded = buildFake()

        reloaded.importState(stored)

        assert reloaded.exportState() == original.exportState()

    def test_exporting_after_importing_gives_an_equal_state(
        self, isolatedPolicyKind: KindTable
    ) -> None:
        defineFake()
        policy = self.policyWithAHistory()
        state = policy.exportState()

        policy.importState(state)

        assert policy.exportState() == state

    def test_importing_replaces_the_state_the_policy_holds(
        self, isolatedPolicyKind: KindTable
    ) -> None:
        defineFake()
        busy = self.policyWithAHistory()
        busy.suggest(makeView(ask_index=2), 5)
        empty = buildFake().exportState()

        busy.importState(empty)

        assert busy.exportState() == empty

    def test_a_read_only_mapping_can_be_imported(self, isolatedPolicyKind: KindTable) -> None:
        defineFake()
        original = self.policyWithAHistory()
        reloaded = buildFake()

        reloaded.importState(MappingProxyType(original.exportState()))

        assert reloaded.exportState() == original.exportState()

    def test_changing_the_exported_state_does_not_change_the_policy(
        self, isolatedPolicyKind: KindTable
    ) -> None:
        defineFake()
        policy = self.policyWithAHistory()
        state = policy.exportState()
        expected = json.loads(json.dumps(state))

        state["asked"] = 99
        state["notes"].append(["oops"])  # type: ignore[union-attr]
        state["extra"] = True

        assert policy.exportState() == expected

    def test_each_export_is_a_new_dictionary(self, isolatedPolicyKind: KindTable) -> None:
        defineFake()
        policy = self.policyWithAHistory()

        first, second = policy.exportState(), policy.exportState()

        assert first == second
        assert first is not second
        assert first["notes"] is not second["notes"]

    def test_changing_the_state_that_was_imported_does_not_change_the_policy(
        self, isolatedPolicyKind: KindTable
    ) -> None:
        defineFake()
        state = self.policyWithAHistory().exportState()
        policy = buildFake()
        policy.importState(state)
        expected = policy.exportState()

        state["notes"].append([9, None, "pending"])  # type: ignore[union-attr]
        state["asked"] = 99

        assert policy.exportState() == expected

    @pytest.mark.parametrize(
        "state",
        [
            {},
            {"asked": 1},
            {"asked": "many", "notes": []},
            {"asked": 1, "notes": "none"},
            {"asked": 1, "notes": [], "extra": True},
            {"samples": [1, 2, 3], "version": 2},
        ],
    )
    def test_a_state_that_the_policy_does_not_export_is_refused_with_its_name(
        self, isolatedPolicyKind: KindTable, state: dict[str, JsonValue]
    ) -> None:
        defineFake()
        policy = buildFake()

        with pytest.raises(ConfigError, match="policy 'fake_policy_v1'") as caught:
            policy.importState(state)

        assert caught.value.component == "policy 'fake_policy_v1'"
        assert caught.value.field_path == "state"
        assert "exported" in caught.value.suggestion

    def test_a_policy_with_no_state_exports_an_empty_dictionary_and_round_trips_it(
        self, isolatedPolicyKind: KindTable
    ) -> None:
        defineFake()
        policy = buildFake(stateful=False)
        policy.suggest(makeView(), 3)
        policy.trialChanged(makeTrial(0, TrialState.pending), None)

        state = policy.exportState()
        restored = buildFake(stateful=False)
        restored.importState(state)

        assert state == {}
        assert restored.exportState() == {}

    def test_a_policy_with_no_state_refuses_one(self, isolatedPolicyKind: KindTable) -> None:
        defineFake()
        policy = buildFake(stateful=False)

        with pytest.raises(ConfigError, match="keeps no state"):
            policy.importState({"asked": 1, "notes": []})
