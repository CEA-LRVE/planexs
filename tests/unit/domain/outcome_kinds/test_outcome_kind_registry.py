"""Unit tests for the `outcome_kind` registry and for the two built-in kinds as components."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from planexs.core.CapabilityVocabulary import CapabilityVocabulary
from planexs.core.ComponentConfig import ComponentConfig
from planexs.core.KindTable import KindTable
from planexs.domain.outcome_kinds.BinaryKind import BinaryKind
from planexs.domain.outcome_kinds.BinaryKindConfig import BinaryKindConfig
from planexs.domain.outcome_kinds.ContinuousKind import ContinuousKind
from planexs.domain.outcome_kinds.ContinuousKindConfig import ContinuousKindConfig
from planexs.domain.outcome_kinds.OutcomeKind import OutcomeKind
from planexs.domain.outcome_kinds.outcomeKindRegistry import outcomeKindRegistry

REGISTERED: dict[str, tuple[type[OutcomeKind], type[ComponentConfig], bool]] = {
    "continuous": (ContinuousKind, ContinuousKindConfig, True),
    "binary": (BinaryKind, BinaryKindConfig, False),
}


class TestTheRegisteredKinds:
    def test_the_kind_holds_the_two_kinds_of_decision_d7(self) -> None:
        assert outcomeKindRegistry().names() == ["binary", "continuous"]

    def test_the_kind_is_in_the_process_wide_table(self) -> None:
        assert "outcome_kind" in KindTable.default().kinds()
        assert KindTable.default().registry("outcome_kind") is outcomeKindRegistry()

    @pytest.mark.parametrize("name", sorted(REGISTERED))
    def test_an_entry_names_its_schema_and_its_builder(self, name: str) -> None:
        cls, schema, _ = REGISTERED[name]
        entry = outcomeKindRegistry().lookup(name)

        assert entry.name == name
        assert entry.config_schema is schema
        assert entry.builder == cls.fromConfig
        assert entry.description.strip() and "\n" not in entry.description

    @pytest.mark.parametrize("name", sorted(REGISTERED))
    def test_the_builder_makes_the_kind_from_its_configuration(self, name: str) -> None:
        cls, schema, _ = REGISTERED[name]
        entry = outcomeKindRegistry().lookup(name)

        built = entry.builder(schema())

        assert type(built) is cls

    @pytest.mark.parametrize("name", sorted(REGISTERED))
    def test_a_schema_is_pinned_to_its_name_and_has_no_setting(self, name: str) -> None:
        _, schema, _ = REGISTERED[name]

        assert schema.model_fields["type"].default == name
        assert set(schema.model_fields) == {"type"}
        assert schema.model_fields["type"].description
        with pytest.raises(ValidationError):
            schema.model_validate({"type": "other"})
        with pytest.raises(ValidationError):
            schema.model_validate({"type": name, "setting": 1})

    def test_an_entry_declares_no_capability_tag_of_its_own(self) -> None:
        # The tag of a kind goes in the vocabulary, not in the `supports` of the entry: a tag
        # there would make the compatibility check list the outcome kind, among the registered
        # components, as something that meets the need for it, which it does not.
        for name in REGISTERED:
            entry = outcomeKindRegistry().lookup(name)
            assert entry.needs == frozenset()
            assert entry.supports == frozenset()


class TestTheNoiseRule:
    @pytest.mark.parametrize(("name", "allows"), [("continuous", True), ("binary", False)])
    def test_a_continuous_outcome_may_have_a_noise_and_a_binary_one_may_not(
        self, name: str, allows: bool
    ) -> None:
        cls, _, expected = REGISTERED[name]

        assert expected is allows
        assert cls.allows_noise is allows

    def test_the_rule_is_read_from_the_kind_a_registry_builds(self) -> None:
        for name, (_, schema, expected) in REGISTERED.items():
            built = outcomeKindRegistry().lookup(name).builder(schema())
            assert built.allows_noise is expected


class TestTheCapabilityTags:
    @pytest.mark.parametrize("name", sorted(REGISTERED))
    def test_a_kind_names_the_tag_of_its_name(self, name: str) -> None:
        cls, _, _ = REGISTERED[name]

        assert cls.capability_tag == f"outcome_kind:{name}"

    @pytest.mark.parametrize("name", sorted(REGISTERED))
    def test_the_tag_is_registered_in_the_vocabulary_with_the_description_of_the_kind(
        self, name: str
    ) -> None:
        cls, _, _ = REGISTERED[name]
        vocabulary = CapabilityVocabulary.default()

        assert cls.capability_tag in vocabulary
        assert (
            vocabulary.describe(cls.capability_tag)
            == outcomeKindRegistry().lookup(name).description
        )

    def test_the_vocabulary_has_the_two_tags_in_their_group_and_no_other(self) -> None:
        vocabulary = CapabilityVocabulary.default()

        assert vocabulary.tags("outcome_kind") == [
            "outcome_kind:binary",
            "outcome_kind:continuous",
        ]
        assert "outcome_kind" in vocabulary.groups()

    def test_the_registry_checks_its_tags_against_that_vocabulary(self) -> None:
        assert outcomeKindRegistry().vocabulary is CapabilityVocabulary.default()

    def test_a_tag_that_nobody_registered_is_still_refused_with_the_group_listed(self) -> None:
        with pytest.raises(Exception, match=r"outcome_kind:binary, outcome_kind:continuous"):
            CapabilityVocabulary.default().check("outcome_kind:count")
