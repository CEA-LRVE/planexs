"""Unit tests for `registerPolicy` (roadmap task P6-01, specification 5.12, 6, 10.6).

A fake policy stands for a real one: the contract is tested before any policy exists. Every
test that registers one uses `isolatedPolicyKind`, because a registered name is permanent and
must never reach the real registry.
"""

from __future__ import annotations

from typing import Any, Literal

import pytest
from policy_fakes import (
    FakeBehaviour,
    buildFake,
    defineFake,
    defineRefining,
    fakeConfig,
    makeGoal,
    makeProblem,
    makeView,
    registerToyTags,
)
from pydantic import Field

import planexs.policies  # noqa: F401
from planexs.core.checkCompatibility import checkCompatibility
from planexs.core.CompatibilityError import CompatibilityError
from planexs.core.ComponentConfig import ComponentConfig
from planexs.core.ConfigError import ConfigError
from planexs.core.DeclaredParticipant import DeclaredParticipant
from planexs.core.KindTable import KindTable
from planexs.core.RegistryError import RegistryError
from planexs.core.resolveComponent import resolveComponent
from planexs.core.RuntimeConfig import RuntimeConfig
from planexs.core.toResolvedConfig import toResolvedConfig
from planexs.policies.AbstractPolicy import AbstractPolicy
from planexs.policies.policyRegistry import policyRegistry
from planexs.policies.registerPolicy import registerPolicy


class TestRegistering:
    def test_the_entry_holds_everything_the_decorator_was_given(
        self, isolatedPolicyKind: KindTable
    ) -> None:
        registerToyTags(isolatedPolicyKind)
        policy = defineFake(
            "batch_like_v1",
            needs=frozenset({"toy:model"}),
            supports=frozenset({"toy:batch", "toy:resumable"}),
            description="A batch-like policy for tests.",
        )

        entry = policyRegistry().lookup("batch_like_v1")

        assert entry.name == "batch_like_v1"
        assert entry.description == "A batch-like policy for tests."
        assert entry.needs == {"toy:model"}
        assert entry.supports == {"toy:batch", "toy:resumable"}
        assert entry.config_schema.model_fields["type"].default == "batch_like_v1"
        assert entry.builder == policy.fromConfig

    def test_the_class_is_returned_unchanged(self, isolatedPolicyKind: KindTable) -> None:
        policy = defineFake()

        assert issubclass(policy, AbstractPolicy)
        assert policy.__name__ == "FakePolicy"

    def test_a_policy_that_declares_nothing_is_legal(
        self, isolatedPolicyKind: KindTable
    ) -> None:
        defineFake()

        entry = policyRegistry().lookup("fake_policy_v1")

        assert entry.needs == frozenset()
        assert entry.supports == frozenset()

    def test_several_policies_live_side_by_side(self, isolatedPolicyKind: KindTable) -> None:
        defineFake("a_v1")
        defineFake("b_v1")

        assert policyRegistry().names() == ["a_v1", "b_v1"]

    def test_registering_builds_nothing(self, isolatedPolicyKind: KindTable) -> None:
        # A builder is free of side effects at import and registration time (specification 6).
        calls: list[str] = []

        @registerPolicy("lazy_v1", config=fakeConfig("lazy_v1"), description="Lazy.")
        class Lazy(FakeBehaviour):
            def __init__(self, config: Any) -> None:
                calls.append("init")
                super().__init__(config)

            @classmethod
            def fromConfig(cls, config: Any, **inputs: Any) -> Any:
                calls.append("fromConfig")
                return super().fromConfig(config, **inputs)

        assert calls == []

    def test_the_registry_entry_passes_the_contract_of_specification_6(
        self, isolatedPolicyKind: KindTable
    ) -> None:
        registerToyTags(isolatedPolicyKind)
        defineFake("policy_v1", supports=frozenset({"toy:batch"}))
        entry = policyRegistry().lookup("policy_v1")
        vocabulary = isolatedPolicyKind.vocabulary

        assert policyRegistry().isValidName(entry.name)
        assert issubclass(entry.config_schema, ComponentConfig)
        assert all(
            field.description and field.description.strip()
            for field in entry.config_schema.model_fields.values()
        )
        assert all(tag in vocabulary for tag in entry.needs | entry.supports)
        assert callable(entry.builder)
        assert entry.description.splitlines() == [entry.description]


class TestFromAConfigurationToAPolicy:
    """The path that a study will take: a mapping, the registry, a policy, a suggestion."""

    def test_a_mapping_resolves_to_the_entry_and_a_validated_configuration(
        self, isolatedPolicyKind: KindTable
    ) -> None:
        defineFake("policy_v1")

        resolved = resolveComponent(policyRegistry(), {"type": "policy_v1", "stateful": False})

        assert resolved.entry.name == "policy_v1"
        assert resolved.config.model_dump() == {"type": "policy_v1", "stateful": False}

    def test_the_entry_builder_makes_a_policy_that_suggests_notifies_and_exports(
        self, isolatedPolicyKind: KindTable
    ) -> None:
        defineFake("policy_v1")
        resolved = resolveComponent(policyRegistry(), {"type": "policy_v1"})

        policy = resolved.entry.builder(
            resolved.config, problem=makeProblem(), goal=makeGoal(), runtime=RuntimeConfig()
        )

        assert isinstance(policy, AbstractPolicy)
        assert policy.config is resolved.config
        assert len(policy.suggest(makeView(), 2)) == 2
        assert policy.exportState()["asked"] == 2

    def test_the_builder_takes_the_problem_the_goal_and_the_runtime_by_keyword(
        self, isolatedPolicyKind: KindTable
    ) -> None:
        defineFake("policy_v1")
        resolved = resolveComponent(policyRegistry(), {"type": "policy_v1"})

        with pytest.raises(TypeError):
            resolved.entry.builder(resolved.config)
        with pytest.raises(TypeError):
            resolved.entry.builder(resolved.config, makeProblem(), makeGoal(), RuntimeConfig())

    def test_each_build_is_a_new_policy_with_a_state_of_its_own(
        self, isolatedPolicyKind: KindTable
    ) -> None:
        defineFake("policy_v1")

        first = buildFake("policy_v1")
        second = buildFake("policy_v1")
        first.suggest(makeView(), 3)

        assert first is not second
        assert first.exportState() != second.exportState()
        assert second.exportState()["asked"] == 0

    def test_the_resolved_configuration_has_every_default_written_out(
        self, isolatedPolicyKind: KindTable
    ) -> None:
        defineFake("policy_v1")

        resolved = resolveComponent(policyRegistry(), {"type": "policy_v1"})

        assert toResolvedConfig(resolved.config) == {"type": "policy_v1", "stateful": True}

    def test_the_resolved_configuration_resolves_to_the_same_policy_again(
        self, isolatedPolicyKind: KindTable
    ) -> None:
        # What a study stores is what it reads back (specification 5.11): a policy built from
        # the stored configuration is the one that was built from the first.
        defineFake("policy_v1")
        first = resolveComponent(policyRegistry(), {"type": "policy_v1", "stateful": False})

        again = resolveComponent(policyRegistry(), toResolvedConfig(first.config))

        assert again.config == first.config
        assert again.entry is first.entry

    def test_an_unknown_field_is_refused_with_its_name(
        self, isolatedPolicyKind: KindTable
    ) -> None:
        defineFake("policy_v1")

        with pytest.raises(ConfigError, match="statefull") as caught:
            resolveComponent(policyRegistry(), {"type": "policy_v1", "statefull": True})

        assert "policy 'policy_v1'" in caught.value.message

    def test_a_value_of_the_wrong_type_is_refused_with_the_field(
        self, isolatedPolicyKind: KindTable
    ) -> None:
        defineFake("policy_v1")

        with pytest.raises(ConfigError, match="stateful") as caught:
            resolveComponent(policyRegistry(), {"type": "policy_v1", "stateful": "maybe"})

        assert caught.value.field_path == "stateful"

    def test_an_unknown_policy_lists_the_registered_ones(
        self, isolatedPolicyKind: KindTable
    ) -> None:
        defineFake("a_v1")
        defineFake("b_v1")

        with pytest.raises(ConfigError, match="Registered names: a_v1, b_v1") as caught:
            resolveComponent(policyRegistry(), {"type": "c_v1"})

        assert caught.value.field_path == "type"


class TestARegistrationThatIsRefused:
    def assertNothingWasRegistered(self, table: KindTable) -> None:
        assert table.registry("policy").names() == []

    @pytest.mark.parametrize(
        "name", ["Upper", "1st_v1", "with space", "trailing_", "", "a__b"]
    )
    def test_an_invalid_name(self, isolatedPolicyKind: KindTable, name: str) -> None:
        with pytest.raises(RegistryError, match="not a valid component name"):
            defineFake(name)

        self.assertNothingWasRegistered(isolatedPolicyKind)

    def test_a_name_that_is_taken_is_never_replaced(
        self, isolatedPolicyKind: KindTable
    ) -> None:
        defineFake("policy_v1", description="The first one.")

        with pytest.raises(RegistryError, match="already registered") as caught:
            defineFake("policy_v1", description="The second one.")

        assert "permanent identifier" in str(caught.value)
        assert policyRegistry().lookup("policy_v1").description == "The first one."

    def test_a_class_that_is_not_a_policy(self, isolatedPolicyKind: KindTable) -> None:
        class NotAPolicy:
            """Not a subclass of the contract."""

        with pytest.raises(RegistryError, match="not a subclass of AbstractPolicy") as caught:
            registerPolicy("plain_v1", config=fakeConfig("plain_v1"), description="Plain.")(
                NotAPolicy  # type: ignore[type-var]
            )

        assert caught.value.component == "policy 'plain_v1'"
        self.assertNothingWasRegistered(isolatedPolicyKind)

    def test_a_goal_is_not_a_policy(self, isolatedPolicyKind: KindTable) -> None:
        # The kinds have contracts of their own: a class of another kind is refused here.
        from planexs.goals.ExploreGoal import ExploreGoal

        with pytest.raises(RegistryError, match="not a subclass of AbstractPolicy"):
            registerPolicy("goal_v1", config=fakeConfig("goal_v1"), description="A goal.")(
                ExploreGoal  # type: ignore[type-var]
            )

        self.assertNothingWasRegistered(isolatedPolicyKind)

    def test_a_schema_whose_type_is_not_pinned_to_the_name(
        self, isolatedPolicyKind: KindTable
    ) -> None:
        class OtherConfig(ComponentConfig):
            """A schema pinned to another name."""

            type: Literal["other_v1"] = Field(default="other_v1", description="The name.")

        with pytest.raises(RegistryError, match="pinned to the registered name") as caught:

            @registerPolicy("policy_v1", config=OtherConfig, description="Mismatch.")
            class Mismatch(FakeBehaviour):
                pass

        assert caught.value.field_path == "type"
        self.assertNothingWasRegistered(isolatedPolicyKind)

    @pytest.mark.parametrize("description", ["", "   ", "Two\nlines."])
    def test_a_description_that_is_not_one_line(
        self, isolatedPolicyKind: KindTable, description: str
    ) -> None:
        with pytest.raises(RegistryError, match="description"):
            defineFake(description=description)

        self.assertNothingWasRegistered(isolatedPolicyKind)

    def test_a_needs_that_is_a_single_string_and_not_a_collection(
        self, isolatedPolicyKind: KindTable
    ) -> None:
        with pytest.raises(RegistryError, match="needs"):
            defineFake(needs="toy:model")  # type: ignore[arg-type]

        self.assertNothingWasRegistered(isolatedPolicyKind)

    @pytest.mark.parametrize("side", ["needs", "supports"])
    def test_a_tag_that_is_not_in_the_vocabulary(
        self, isolatedPolicyKind: KindTable, side: str
    ) -> None:
        # A typo in a declaration is an error where it is made, not a need nobody can meet.
        with pytest.raises(RegistryError, match="toy:batsh"):
            defineFake(**{side: frozenset({"toy:batsh"})})

        self.assertNothingWasRegistered(isolatedPolicyKind)


class TestTheCompatibilityCheck:
    """A policy is a participant: its tags are what the compatibility check reads (5.12)."""

    def study(self, *needs: str) -> DeclaredParticipant:
        return DeclaredParticipant(label="study", needs=frozenset(needs))

    def test_a_policy_is_a_participant_with_a_label_and_tags(
        self, isolatedPolicyKind: KindTable
    ) -> None:
        registerToyTags(isolatedPolicyKind)
        defineFake(
            "policy_v1", needs=frozenset({"toy:model"}), supports=frozenset({"toy:batch"})
        )

        policy = buildFake("policy_v1")

        assert policy.label == "policy 'policy_v1'"
        assert policy.needs == {"toy:model"}
        assert policy.supports == {"toy:batch"}

    def test_a_need_that_the_policy_supports_is_met(
        self, isolatedPolicyKind: KindTable
    ) -> None:
        registerToyTags(isolatedPolicyKind)
        defineFake("policy_v1", supports=frozenset({"toy:batch"}))
        policy = buildFake("policy_v1")

        checkCompatibility([policy, self.study("toy:batch")], isolatedPolicyKind)

    def test_a_need_that_no_participant_supports_names_the_policies_that_would(
        self, isolatedPolicyKind: KindTable
    ) -> None:
        registerToyTags(isolatedPolicyKind)
        defineFake("serial_v1", supports=frozenset({"toy:resumable"}))
        defineFake("batch_v1", supports=frozenset({"toy:batch"}))
        policy = buildFake("serial_v1")

        with pytest.raises(CompatibilityError) as caught:
            checkCompatibility([policy, self.study("toy:batch")], isolatedPolicyKind)

        message = caught.value.message
        assert "Need 'toy:batch'" in message
        assert "needed by: study" in message
        assert "supported by: policy 'batch_v1'" in message
        assert "serial_v1" not in message.split("supported by:")[1]

    def test_the_needs_of_a_policy_are_met_by_another_participant(
        self, isolatedPolicyKind: KindTable
    ) -> None:
        registerToyTags(isolatedPolicyKind)
        defineFake("policy_v1", needs=frozenset({"toy:model"}))
        policy = buildFake("policy_v1")

        with pytest.raises(CompatibilityError, match="Need 'toy:model'"):
            checkCompatibility([policy], isolatedPolicyKind)

        provider = DeclaredParticipant(
            label="surrogate 'toy_v1'", supports=frozenset({"toy:model"})
        )
        checkCompatibility([policy, provider], isolatedPolicyKind)

    def test_the_tags_narrowed_to_a_configuration_decide_the_outcome_of_the_check(
        self, isolatedPolicyKind: KindTable
    ) -> None:
        # The entry supports both tags, but an instance without state supports only
        # `toy:batch`: a study that needs `toy:resumable` fails, and the message still names
        # the policy that could be configured to meet it.
        registerToyTags(isolatedPolicyKind)
        defineRefining()
        provider = DeclaredParticipant(label="surrogate", supports=frozenset({"toy:model"}))
        stateful = buildFake("refining_v1")
        stateless = buildFake("refining_v1", stateful=False)

        checkCompatibility(
            [stateful, provider, self.study("toy:resumable")], isolatedPolicyKind
        )
        checkCompatibility([stateless, self.study("toy:batch")], isolatedPolicyKind)
        with pytest.raises(CompatibilityError) as caught:
            checkCompatibility([stateless, self.study("toy:resumable")], isolatedPolicyKind)

        assert "supported by: policy 'refining_v1'" in caught.value.message

    def test_a_need_that_depends_on_the_configuration_appears_only_when_it_applies(
        self, isolatedPolicyKind: KindTable
    ) -> None:
        registerToyTags(isolatedPolicyKind)
        defineRefining()
        stateless = buildFake("refining_v1", stateful=False)
        stateful = buildFake("refining_v1")

        checkCompatibility([stateless], isolatedPolicyKind)
        with pytest.raises(CompatibilityError, match="needed by: policy 'refining_v1'"):
            checkCompatibility([stateful], isolatedPolicyKind)

    def test_a_policy_whose_tag_is_not_in_the_vocabulary_is_reported_as_a_typo(
        self, isolatedPolicyKind: KindTable
    ) -> None:
        # The check reads the tags of the policy object, and a typo there is not an unmet need
        # that nobody could ever meet: it is an error that names the participant.
        defineFake("policy_v1")
        policy = buildFake("policy_v1")

        class Typo(type(policy)):  # type: ignore[misc]
            @property
            def supports(self) -> frozenset[str]:
                return frozenset({"toy:batsh"})

        with pytest.raises(RegistryError, match="toy:batsh") as caught:
            checkCompatibility([Typo(policy.config)], isolatedPolicyKind)

        assert caught.value.component == "policy 'policy_v1'"
        assert caught.value.field_path == "supports"
