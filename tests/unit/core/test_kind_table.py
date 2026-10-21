"""Unit tests for `KindTable` and `defineKind` (specification 5.12, 10.6)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from planexs.core.CapabilityVocabulary import CapabilityVocabulary
from planexs.core.ComponentEntry import ComponentEntry
from planexs.core.defineKind import defineKind
from planexs.core.KindTable import KindTable
from planexs.core.Registry import Registry
from planexs.core.RegistryError import RegistryError

MakeEntry = Callable[..., ComponentEntry]

KINDS = ["surrogate", "kernel", "acquisition_optimizer", "outcome_kind", "stopping"]

INVALID_KINDS = [
    pytest.param("", id="empty"),
    pytest.param("Surrogate", id="uppercase"),
    pytest.param("two words", id="space"),
    pytest.param("1kind", id="leading-digit"),
    pytest.param("_kind", id="leading-underscore"),
    pytest.param("kind_", id="trailing-underscore"),
    pytest.param("kind-name", id="hyphen"),
    pytest.param("kind\n", id="trailing-newline"),
]


@pytest.fixture
def table(vocabulary: CapabilityVocabulary) -> KindTable:
    return KindTable(vocabulary)


class TestDefine:
    def test_a_defined_kind_has_an_empty_registry_of_that_kind(self, table: KindTable) -> None:
        registry = table.define("surrogate")
        assert isinstance(registry, Registry)
        assert registry.kind == "surrogate"
        assert registry.names() == []

    def test_the_registry_is_the_one_the_table_returns_afterwards(
        self, table: KindTable
    ) -> None:
        registry = table.define("surrogate")
        assert table.registry("surrogate") is registry
        assert table.registry("surrogate") is table.registry("surrogate")

    def test_a_defined_kind_is_a_member(self, table: KindTable) -> None:
        table.define("surrogate")
        assert "surrogate" in table
        assert "kernel" not in table

    @pytest.mark.parametrize("kind", [3, None, ["surrogate"]])
    def test_something_that_is_not_a_string_is_not_a_member_and_does_not_raise(
        self, table: KindTable, kind: object
    ) -> None:
        table.define("surrogate")
        assert kind not in table

    def test_components_registered_in_the_returned_registry_are_found_through_the_table(
        self, table: KindTable, makeEntry: MakeEntry
    ) -> None:
        entry = makeEntry("single_task_gp_v1")
        table.define("surrogate").register(entry)
        assert table.registry("surrogate").lookup("single_task_gp_v1") is entry


class TestDuplicateKinds:
    def test_defining_a_kind_twice_raises(self, table: KindTable) -> None:
        table.define("surrogate")
        with pytest.raises(RegistryError) as caught:
            table.define("surrogate")
        assert "'surrogate'" in caught.value.message
        assert "already defined" in caught.value.message

    def test_the_first_registry_and_its_components_survive_a_duplicate(
        self, table: KindTable, makeEntry: MakeEntry
    ) -> None:
        first = table.define("surrogate")
        first.register(makeEntry("a"))
        with pytest.raises(RegistryError):
            table.define("surrogate")
        assert table.registry("surrogate") is first
        assert first.names() == ["a"]
        assert table.kinds() == ["surrogate"]

    def test_the_error_lists_the_defined_kinds(self, table: KindTable) -> None:
        for kind in ["kernel", "surrogate"]:
            table.define(kind)
        with pytest.raises(RegistryError) as caught:
            table.define("kernel")
        assert "kernel, surrogate" in caught.value.message
        assert caught.value.component == "kernel"
        assert caught.value.suggestion is not None


class TestInvalidKinds:
    @pytest.mark.parametrize("kind", INVALID_KINDS)
    def test_an_invalid_kind_name_is_rejected(self, table: KindTable, kind: str) -> None:
        with pytest.raises(RegistryError, match="not a valid name"):
            table.define(kind)

    @pytest.mark.parametrize("kind", INVALID_KINDS)
    def test_a_refused_kind_leaves_the_table_unchanged(
        self, table: KindTable, kind: str
    ) -> None:
        table.define("kept")
        with pytest.raises(RegistryError):
            table.define(kind)
        assert table.kinds() == ["kept"]

    @pytest.mark.parametrize("kind", [3, None])
    def test_a_kind_that_is_not_a_string_raises_registry_error(
        self, table: KindTable, kind: Any
    ) -> None:
        with pytest.raises(RegistryError):
            table.define(kind)


class TestUnknownKinds:
    def test_asking_for_an_undefined_kind_raises(self, table: KindTable) -> None:
        with pytest.raises(RegistryError):
            table.registry("surrogate")

    def test_the_message_lists_the_defined_kinds_sorted(self, table: KindTable) -> None:
        for kind in ["surrogate", "acquisition", "kernel"]:
            table.define(kind)
        with pytest.raises(RegistryError) as caught:
            table.registry("surogate")
        assert "'surogate'" in caught.value.message
        assert "acquisition, kernel, surrogate" in caught.value.message
        assert "acquisition, kernel, surrogate" in str(caught.value)

    def test_the_message_says_so_when_no_kind_is_defined(self, table: KindTable) -> None:
        with pytest.raises(RegistryError, match=r"Defined kinds: none\."):
            table.registry("surrogate")

    def test_a_kind_that_is_not_a_string_is_unknown_not_a_crash(
        self, table: KindTable
    ) -> None:
        with pytest.raises(RegistryError):
            table.registry(["surrogate"])  # type: ignore[arg-type]


class TestEnumeration:
    def test_an_empty_table_enumerates_nothing(self, table: KindTable) -> None:
        assert table.kinds() == []

    def test_kinds_are_sorted_whatever_the_definition_order(self, table: KindTable) -> None:
        for kind in KINDS:
            table.define(kind)
        assert table.kinds() == sorted(KINDS)

    def test_the_listing_is_a_copy(self, table: KindTable) -> None:
        table.define("surrogate")
        listing = table.kinds()
        listing.clear()
        assert table.kinds() == ["surrogate"]

    def test_every_enumerated_kind_leads_to_its_registry(
        self, table: KindTable, makeEntry: MakeEntry
    ) -> None:
        for kind in KINDS:
            table.define(kind).register(makeEntry("a_v1", description=f"A {kind}."))
        visited = {
            kind: table.registry(kind).lookup("a_v1").description for kind in table.kinds()
        }
        assert visited == {kind: f"A {kind}." for kind in KINDS}


class TestReset:
    def test_reset_removes_every_kind(self, table: KindTable) -> None:
        for kind in KINDS:
            table.define(kind)
        table.reset()
        assert table.kinds() == []
        assert "surrogate" not in table

    def test_a_reset_kind_is_unknown(self, table: KindTable) -> None:
        table.define("surrogate")
        table.reset()
        with pytest.raises(RegistryError):
            table.registry("surrogate")

    def test_a_kind_can_be_defined_again_after_a_reset_with_a_fresh_registry(
        self, table: KindTable, makeEntry: MakeEntry
    ) -> None:
        old = table.define("surrogate")
        old.register(makeEntry("a"))
        table.reset()
        new = table.define("surrogate")
        assert new is not old
        assert new.names() == []
        assert old.names() == ["a"]

    def test_a_reset_does_not_touch_the_vocabulary(
        self, table: KindTable, vocabulary: CapabilityVocabulary
    ) -> None:
        vocabulary.register("posterior:gaussian", "The posterior is jointly Gaussian.")
        table.define("surrogate")
        table.reset()
        assert vocabulary.tags() == ["posterior:gaussian"]
        assert table.vocabulary is vocabulary

    def test_resetting_an_empty_table_is_harmless(self, table: KindTable) -> None:
        table.reset()
        table.reset()
        assert table.kinds() == []


class TestIndependence:
    def test_two_tables_share_no_state(self) -> None:
        first, second = KindTable(CapabilityVocabulary()), KindTable(CapabilityVocabulary())
        first.define("surrogate")
        assert second.kinds() == []
        assert "surrogate" not in second
        second.define("surrogate")
        first.reset()
        assert second.kinds() == ["surrogate"]

    def test_the_registries_of_two_kinds_share_no_state(
        self, table: KindTable, makeEntry: MakeEntry
    ) -> None:
        first, second = makeEntry("matern_v1"), makeEntry("matern_v1")
        table.define("surrogate").register(first)
        assert table.define("kernel").names() == []
        table.registry("kernel").register(second)
        assert table.registry("surrogate").lookup("matern_v1") is first
        assert table.registry("kernel").lookup("matern_v1") is second

    def test_the_registries_of_two_tables_share_no_state(self, makeEntry: MakeEntry) -> None:
        first, second = KindTable(CapabilityVocabulary()), KindTable(CapabilityVocabulary())
        first.define("surrogate").register(makeEntry("a"))
        assert second.define("surrogate").names() == []

    def test_two_tables_may_share_a_vocabulary_without_sharing_kinds(
        self, vocabulary: CapabilityVocabulary
    ) -> None:
        first, second = KindTable(vocabulary), KindTable(vocabulary)
        first.define("surrogate")
        assert second.kinds() == []
        assert first.vocabulary is second.vocabulary


class TestVocabulary:
    def test_the_table_keeps_the_vocabulary_it_was_given(
        self, table: KindTable, vocabulary: CapabilityVocabulary
    ) -> None:
        assert table.vocabulary is vocabulary

    def test_every_registry_of_the_table_uses_the_tables_vocabulary(
        self, table: KindTable, vocabulary: CapabilityVocabulary
    ) -> None:
        for kind in KINDS:
            assert table.define(kind).vocabulary is vocabulary

    def test_a_registry_from_the_table_checks_the_tags_of_the_entries_it_gets(
        self, table: KindTable, makeEntry: MakeEntry
    ) -> None:
        registry = table.define("surrogate")
        with pytest.raises(RegistryError, match="Unknown capability tag"):
            registry.register(makeEntry("gp_v1", supports={"posterior:gaussian"}))

    def test_a_tag_registered_after_the_kind_was_defined_is_accepted(
        self, table: KindTable, vocabulary: CapabilityVocabulary, makeEntry: MakeEntry
    ) -> None:
        registry = table.define("surrogate")
        vocabulary.register("posterior:gaussian", "The posterior is jointly Gaussian.")
        entry = makeEntry("gp_v1", supports={"posterior:gaussian"})
        registry.register(entry)
        assert registry.lookup("gp_v1") is entry

    def test_two_tables_with_different_vocabularies_judge_tags_differently(
        self, makeEntry: MakeEntry
    ) -> None:
        known, unknown = CapabilityVocabulary(), CapabilityVocabulary()
        known.register("posterior:gaussian", "The posterior is jointly Gaussian.")
        entry = makeEntry("gp_v1", supports={"posterior:gaussian"})
        KindTable(known).define("surrogate").register(entry)
        with pytest.raises(RegistryError):
            KindTable(unknown).define("surrogate").register(entry)


class TestDefaultTable:
    def test_the_default_table_is_one_object(self) -> None:
        assert KindTable.default() is KindTable.default()

    def test_the_default_table_is_not_a_table_made_for_a_test(self) -> None:
        assert KindTable.default() is not KindTable(CapabilityVocabulary())

    def test_the_default_table_checks_tags_against_the_default_vocabulary(self) -> None:
        assert KindTable.default().vocabulary is CapabilityVocabulary.default()

    def test_define_kind_records_the_kind_in_the_default_table(
        self, isolatedKindTable: KindTable
    ) -> None:
        registry = defineKind("surrogate")
        assert KindTable.default() is isolatedKindTable
        assert isolatedKindTable.registry("surrogate") is registry
        assert registry.kind == "surrogate"

    def test_the_default_table_can_be_enumerated(self, isolatedKindTable: KindTable) -> None:
        for kind in KINDS:
            defineKind(kind)
        assert KindTable.default().kinds() == sorted(KINDS)
        assert [
            KindTable.default().registry(kind).kind for kind in KindTable.default().kinds()
        ] == (sorted(KINDS))

    def test_the_default_table_can_be_reset(self, isolatedKindTable: KindTable) -> None:
        defineKind("surrogate")
        KindTable.default().reset()
        assert KindTable.default().kinds() == []
        defineKind("surrogate")
        assert KindTable.default().kinds() == ["surrogate"]

    def test_defining_a_kind_twice_through_define_kind_raises(
        self, isolatedKindTable: KindTable
    ) -> None:
        defineKind("surrogate")
        with pytest.raises(RegistryError, match="already defined"):
            defineKind("surrogate")

    @pytest.mark.parametrize("kind", INVALID_KINDS)
    def test_define_kind_rejects_an_invalid_name(
        self, isolatedKindTable: KindTable, kind: str
    ) -> None:
        with pytest.raises(RegistryError):
            defineKind(kind)
        assert isolatedKindTable.kinds() == []

    def test_a_registry_from_define_kind_registers_and_looks_up(
        self, isolatedKindTable: KindTable, makeEntry: MakeEntry
    ) -> None:
        entry = makeEntry("single_task_gp_v1")
        defineKind("surrogate").register(entry)
        found = KindTable.default().registry("surrogate").lookup("single_task_gp_v1")
        assert found is entry

    def test_the_isolation_does_not_touch_the_real_table(
        self, isolatedKindTable: KindTable
    ) -> None:
        # `isolatedKindTable` is what `default()` returns inside the test; the real table is
        # held by the module again as soon as `monkeypatch` undoes its patch.
        defineKind("only_in_this_test")
        assert KindTable.default().kinds() == ["only_in_this_test"]
