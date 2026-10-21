"""Unit tests for `Registry` (specification 5.12, 10.6)."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any

import pytest

from planexs.core.CapabilityVocabulary import CapabilityVocabulary
from planexs.core.ComponentEntry import ComponentEntry
from planexs.core.PlanexsError import PlanexsError
from planexs.core.Registry import Registry
from planexs.core.RegistryError import RegistryError

MakeEntry = Callable[..., ComponentEntry]

VALID_NAMES = [
    "a",
    "gp",
    "single_task_gp_v1",
    "botorch_mixed_v1",
    "exploration_exploitation_multi_v1",
    "gp2",
    "v1",
    "x1_y2_z3",
    "a_1",
]

INVALID_NAMES = [
    pytest.param("", id="empty"),
    pytest.param("Upper", id="uppercase-initial"),
    pytest.param("mixedCase", id="camel-case"),
    pytest.param("SHOUT", id="all-uppercase"),
    pytest.param("has space", id="space"),
    pytest.param(" padded", id="leading-space"),
    pytest.param("padded ", id="trailing-space"),
    pytest.param("1leading", id="leading-digit"),
    pytest.param("_leading", id="leading-underscore"),
    pytest.param("trailing_", id="trailing-underscore"),
    pytest.param("double__underscore", id="double-underscore"),
    pytest.param("kebab-case", id="hyphen"),
    pytest.param("dotted.name", id="dot"),
    pytest.param("group:name", id="colon"),
    pytest.param("slash/name", id="slash"),
    pytest.param("caf\u00e9", id="non-ascii-letter"),
    pytest.param("\u0661\u0662", id="non-ascii-digits"),
    pytest.param("name\n", id="trailing-newline"),
    pytest.param("\nname", id="leading-newline"),
    pytest.param("two\nlines", id="embedded-newline"),
]

NOT_STRINGS = [
    pytest.param(3, id="int"),
    pytest.param(None, id="none"),
    pytest.param(["a"], id="list"),
]


@dataclasses.dataclass(frozen=True)
class RichEntry(ComponentEntry):
    """An entry with a field of its own, as a kind with special needs might define."""

    extra: int = 0


@pytest.fixture
def registry(vocabulary: CapabilityVocabulary) -> Registry[ComponentEntry]:
    return Registry("surrogate", vocabulary)


@pytest.fixture
def tagged(vocabulary: CapabilityVocabulary) -> CapabilityVocabulary:
    """A vocabulary with two groups, so that errors have neighbours to list."""
    vocabulary.register("posterior:marginal_normal", "The posterior has a normal marginal.")
    vocabulary.register("posterior:gaussian", "The posterior is jointly Gaussian.")
    vocabulary.register("outputs:single", "One outcome.")
    vocabulary.register("outputs:multi", "Several outcomes.")
    vocabulary.register("space:categorical", "Unordered choices.")
    return vocabulary


class TestNames:
    @pytest.mark.parametrize("name", VALID_NAMES)
    def test_a_valid_name_is_accepted(self, name: str) -> None:
        assert Registry.isValidName(name)

    @pytest.mark.parametrize("name", INVALID_NAMES)
    def test_an_invalid_name_is_rejected(self, name: str) -> None:
        assert not Registry.isValidName(name)

    @pytest.mark.parametrize("name", NOT_STRINGS)
    def test_something_that_is_not_a_string_is_not_a_name(self, name: object) -> None:
        assert not Registry.isValidName(name)


class TestRegisterAndLookup:
    def test_a_registered_entry_is_found_by_its_own_name(
        self, registry: Registry[ComponentEntry], makeEntry: MakeEntry
    ) -> None:
        entry = makeEntry("single_task_gp_v1")
        registry.register(entry)
        assert registry.lookup("single_task_gp_v1") is entry

    def test_every_valid_name_can_be_registered_and_found(
        self, registry: Registry[ComponentEntry], makeEntry: MakeEntry
    ) -> None:
        entries = [makeEntry(name) for name in VALID_NAMES]
        for entry in entries:
            registry.register(entry)
        assert [registry.lookup(name) for name in VALID_NAMES] == entries
        assert all(
            registry.lookup(name) is entry
            for name, entry in zip(VALID_NAMES, entries, strict=True)
        )

    def test_a_subclass_of_the_entry_can_be_stored_and_comes_back_as_it_was(
        self, vocabulary: CapabilityVocabulary, makeEntry: MakeEntry
    ) -> None:
        registry: Registry[RichEntry] = Registry("surrogate", vocabulary)
        base = makeEntry("a_v1")
        rich = RichEntry(
            name="rich_v1",
            builder=base.builder,
            config_schema=base.config_schema,
            description="An entry with an extra field.",
            extra=7,
        )
        registry.register(rich)
        assert registry.lookup("rich_v1") is rich
        assert registry.lookup("rich_v1").extra == 7

    def test_the_kind_and_the_vocabulary_are_kept(
        self, vocabulary: CapabilityVocabulary
    ) -> None:
        registry: Registry[ComponentEntry] = Registry("acquisition_optimizer", vocabulary)
        assert registry.kind == "acquisition_optimizer"
        assert registry.vocabulary is vocabulary

    def test_the_old_two_argument_form_is_gone(
        self, registry: Registry[ComponentEntry], makeEntry: MakeEntry
    ) -> None:
        with pytest.raises(TypeError):
            registry.register("a_v1", makeEntry("a_v1"))  # type: ignore[call-arg]


class TestEntryType:
    @pytest.mark.parametrize(
        "thing",
        [
            pytest.param(object(), id="object"),
            pytest.param("single_task_gp_v1", id="a-name"),
            pytest.param({"name": "a_v1"}, id="dict"),
            pytest.param(None, id="none"),
        ],
    )
    def test_something_that_is_not_an_entry_is_rejected(
        self, registry: Registry[ComponentEntry], thing: Any
    ) -> None:
        with pytest.raises(RegistryError, match="stores ComponentEntry records"):
            registry.register(thing)
        assert registry.names() == []

    def test_the_error_names_the_kind_and_the_type_it_got(
        self, registry: Registry[ComponentEntry]
    ) -> None:
        with pytest.raises(RegistryError) as caught:
            registry.register(3)  # type: ignore[arg-type]
        assert "surrogate" in caught.value.message
        assert "int" in caught.value.message
        assert caught.value.suggestion is not None


class TestListing:
    def test_an_empty_registry_lists_nothing(self, registry: Registry[ComponentEntry]) -> None:
        assert registry.names() == []

    def test_names_are_sorted_whatever_the_registration_order(
        self, registry: Registry[ComponentEntry], makeEntry: MakeEntry
    ) -> None:
        for name in ["zeta_v1", "alpha_v1", "mid_v2", "mid_v1"]:
            registry.register(makeEntry(name))
        assert registry.names() == ["alpha_v1", "mid_v1", "mid_v2", "zeta_v1"]

    def test_the_listing_is_a_copy(
        self, registry: Registry[ComponentEntry], makeEntry: MakeEntry
    ) -> None:
        registry.register(makeEntry("a"))
        listing = registry.names()
        listing.append("intruder")
        listing.clear()
        assert registry.names() == ["a"]


class TestMembership:
    def test_a_registered_name_is_a_member(
        self, registry: Registry[ComponentEntry], makeEntry: MakeEntry
    ) -> None:
        registry.register(makeEntry("a"))
        assert "a" in registry

    def test_an_unregistered_name_is_not_a_member(
        self, registry: Registry[ComponentEntry], makeEntry: MakeEntry
    ) -> None:
        registry.register(makeEntry("a"))
        assert "b" not in registry

    def test_an_invalid_name_is_not_a_member(self, registry: Registry[ComponentEntry]) -> None:
        assert "Not A Name" not in registry

    @pytest.mark.parametrize("name", NOT_STRINGS)
    def test_something_that_is_not_a_string_is_not_a_member_and_does_not_raise(
        self, registry: Registry[ComponentEntry], makeEntry: MakeEntry, name: object
    ) -> None:
        registry.register(makeEntry("a"))
        assert name not in registry


class TestDuplicates:
    def test_registering_a_name_twice_raises(
        self, registry: Registry[ComponentEntry], makeEntry: MakeEntry
    ) -> None:
        registry.register(makeEntry("single_task_gp_v1"))
        with pytest.raises(RegistryError) as caught:
            registry.register(makeEntry("single_task_gp_v1"))
        assert "'single_task_gp_v1'" in caught.value.message
        assert "surrogate" in caught.value.message

    def test_the_first_entry_survives_a_duplicate(
        self, registry: Registry[ComponentEntry], makeEntry: MakeEntry
    ) -> None:
        first = makeEntry("a", description="First.")
        registry.register(first)
        with pytest.raises(RegistryError):
            registry.register(makeEntry("a", description="Second."))
        assert registry.lookup("a") is first
        assert registry.names() == ["a"]

    def test_even_the_same_entry_cannot_be_registered_twice(
        self, registry: Registry[ComponentEntry], makeEntry: MakeEntry
    ) -> None:
        entry = makeEntry("a")
        registry.register(entry)
        with pytest.raises(RegistryError):
            registry.register(entry)

    def test_the_error_says_how_to_fix_it(
        self, registry: Registry[ComponentEntry], makeEntry: MakeEntry
    ) -> None:
        registry.register(makeEntry("a"))
        with pytest.raises(RegistryError) as caught:
            registry.register(makeEntry("a"))
        assert caught.value.component == "surrogate"
        assert caught.value.suggestion is not None
        assert "_v2" in caught.value.suggestion


class TestUnknownNames:
    def test_looking_up_an_unknown_name_raises(
        self, registry: Registry[ComponentEntry]
    ) -> None:
        with pytest.raises(RegistryError):
            registry.lookup("missing")

    def test_the_message_names_the_kind_and_the_unknown_name(
        self, registry: Registry[ComponentEntry]
    ) -> None:
        with pytest.raises(RegistryError) as caught:
            registry.lookup("missing")
        assert "surrogate" in caught.value.message
        assert "'missing'" in caught.value.message

    def test_the_message_lists_every_registered_name_sorted(
        self, registry: Registry[ComponentEntry], makeEntry: MakeEntry
    ) -> None:
        for name in ["gamma_v1", "alpha_v1", "beta_v1"]:
            registry.register(makeEntry(name))
        with pytest.raises(RegistryError) as caught:
            registry.lookup("alpah_v1")
        message = caught.value.message
        assert "alpha_v1, beta_v1, gamma_v1" in message
        # The same text reaches the user, who sees `str()` and not `.message`.
        assert "alpha_v1, beta_v1, gamma_v1" in str(caught.value)

    def test_the_message_says_so_when_nothing_is_registered(
        self, registry: Registry[ComponentEntry]
    ) -> None:
        with pytest.raises(RegistryError, match=r"Registered names: none\."):
            registry.lookup("missing")

    def test_a_name_that_is_not_a_string_is_unknown_not_a_crash(
        self, registry: Registry[ComponentEntry]
    ) -> None:
        with pytest.raises(RegistryError):
            registry.lookup(["a"])  # type: ignore[arg-type]

    def test_the_error_carries_the_kind_and_a_suggestion(
        self, registry: Registry[ComponentEntry]
    ) -> None:
        with pytest.raises(RegistryError) as caught:
            registry.lookup("missing")
        assert caught.value.component == "surrogate"
        assert caught.value.suggestion is not None


class TestInvalidNames:
    @pytest.mark.parametrize("name", INVALID_NAMES)
    def test_registering_an_invalid_name_raises(
        self, registry: Registry[ComponentEntry], makeEntry: MakeEntry, name: str
    ) -> None:
        with pytest.raises(RegistryError, match="not a valid component name"):
            registry.register(makeEntry(name))

    @pytest.mark.parametrize("name", INVALID_NAMES)
    def test_a_refused_name_leaves_the_registry_unchanged(
        self, registry: Registry[ComponentEntry], makeEntry: MakeEntry, name: str
    ) -> None:
        registry.register(makeEntry("kept"))
        with pytest.raises(RegistryError):
            registry.register(makeEntry(name))
        assert registry.names() == ["kept"]
        assert name not in registry

    @pytest.mark.parametrize("name", NOT_STRINGS)
    def test_registering_an_entry_whose_name_is_not_a_string_raises_registry_error(
        self, registry: Registry[ComponentEntry], makeEntry: MakeEntry, name: Any
    ) -> None:
        with pytest.raises(RegistryError):
            registry.register(makeEntry(name))

    def test_the_message_states_the_rule_and_the_error_gives_an_example(
        self, registry: Registry[ComponentEntry], makeEntry: MakeEntry
    ) -> None:
        with pytest.raises(RegistryError) as caught:
            registry.register(makeEntry("Bad Name"))
        assert "lowercase snake_case" in caught.value.message
        assert "'Bad Name'" in caught.value.message
        assert caught.value.suggestion is not None
        assert "single_task_gp_v1" in caught.value.suggestion

    def test_every_registry_error_is_a_planexs_error(
        self, registry: Registry[ComponentEntry], makeEntry: MakeEntry
    ) -> None:
        with pytest.raises(PlanexsError):
            registry.register(makeEntry("Bad Name"))


class TestTags:
    def test_an_entry_with_known_needs_and_supports_is_registered(
        self, tagged: CapabilityVocabulary, makeEntry: MakeEntry
    ) -> None:
        registry: Registry[ComponentEntry] = Registry("surrogate", tagged)
        entry = makeEntry(
            "gp_v1",
            needs={"space:categorical"},
            supports={"posterior:gaussian", "outputs:single"},
        )
        registry.register(entry)
        assert registry.lookup("gp_v1") is entry

    def test_an_entry_with_no_tags_needs_no_vocabulary(
        self, registry: Registry[ComponentEntry], makeEntry: MakeEntry
    ) -> None:
        registry.register(makeEntry("a"))
        assert registry.names() == ["a"]

    def test_an_unknown_need_is_rejected_and_the_message_lists_the_tags_of_its_group(
        self, tagged: CapabilityVocabulary, makeEntry: MakeEntry
    ) -> None:
        registry: Registry[ComponentEntry] = Registry("surrogate", tagged)
        with pytest.raises(RegistryError) as caught:
            registry.register(makeEntry("gp_v1", needs={"posterior:gausian"}))
        message = caught.value.message
        assert "'posterior:gausian'" in message
        assert "posterior:gaussian, posterior:marginal_normal" in message
        assert "outputs" not in message
        assert "space:categorical" not in message

    def test_an_unknown_support_is_rejected(
        self, tagged: CapabilityVocabulary, makeEntry: MakeEntry
    ) -> None:
        registry: Registry[ComponentEntry] = Registry("surrogate", tagged)
        with pytest.raises(RegistryError, match="outputs:multi, outputs:single"):
            registry.register(makeEntry("gp_v1", supports={"outputs:several"}))

    def test_the_error_says_which_component_and_which_field(
        self, tagged: CapabilityVocabulary, makeEntry: MakeEntry
    ) -> None:
        registry: Registry[ComponentEntry] = Registry("surrogate", tagged)
        with pytest.raises(RegistryError) as needs_error:
            registry.register(makeEntry("gp_v1", needs={"posterior:gausian"}))
        assert needs_error.value.component == "surrogate 'gp_v1'"
        assert needs_error.value.field_path == "needs"
        with pytest.raises(RegistryError) as supports_error:
            registry.register(makeEntry("gp_v1", supports={"posterior:gausian"}))
        assert supports_error.value.field_path == "supports"
        assert "Component: surrogate 'gp_v1'" in str(supports_error.value)
        assert "Field: supports" in str(supports_error.value)

    def test_a_tag_of_an_unknown_group_lists_the_known_groups(
        self, tagged: CapabilityVocabulary, makeEntry: MakeEntry
    ) -> None:
        registry: Registry[ComponentEntry] = Registry("surrogate", tagged)
        with pytest.raises(RegistryError, match="Known groups: outputs, posterior, space"):
            registry.register(makeEntry("gp_v1", needs={"posteriors:gaussian"}))

    @pytest.mark.parametrize("field", ["needs", "supports"])
    @pytest.mark.parametrize(
        "tag", ["Not A Tag", "nogroup", "posterior:", "Posterior:gaussian"]
    )
    def test_a_malformed_tag_is_rejected(
        self, tagged: CapabilityVocabulary, makeEntry: MakeEntry, field: str, tag: str
    ) -> None:
        registry: Registry[ComponentEntry] = Registry("surrogate", tagged)
        with pytest.raises(RegistryError, match="not a well-formed capability tag") as caught:
            registry.register(makeEntry("gp_v1", **{field: {tag}}))
        assert caught.value.field_path == field

    def test_an_empty_vocabulary_rejects_every_tag(
        self, registry: Registry[ComponentEntry], makeEntry: MakeEntry
    ) -> None:
        with pytest.raises(RegistryError, match=r"Known groups: none\."):
            registry.register(makeEntry("gp_v1", supports={"posterior:gaussian"}))

    def test_a_rejected_entry_is_not_stored_and_its_name_stays_free(
        self, tagged: CapabilityVocabulary, makeEntry: MakeEntry
    ) -> None:
        registry: Registry[ComponentEntry] = Registry("surrogate", tagged)
        with pytest.raises(RegistryError):
            registry.register(makeEntry("gp_v1", needs={"posterior:gausian"}))
        assert registry.names() == []
        assert "gp_v1" not in registry
        fixed = makeEntry("gp_v1", needs={"posterior:gaussian"})
        registry.register(fixed)
        assert registry.lookup("gp_v1") is fixed

    def test_needs_are_checked_before_supports_and_each_in_sorted_order(
        self, tagged: CapabilityVocabulary, makeEntry: MakeEntry
    ) -> None:
        registry: Registry[ComponentEntry] = Registry("surrogate", tagged)
        entry = makeEntry(
            "gp_v1",
            needs={"zzz:late", "aaa:early"},
            supports={"bbb:other"},
        )
        with pytest.raises(RegistryError) as caught:
            registry.register(entry)
        assert caught.value.field_path == "needs"
        assert "'aaa:early'" in caught.value.message

    def test_the_vocabulary_is_read_at_registration_not_at_creation(
        self, vocabulary: CapabilityVocabulary, makeEntry: MakeEntry
    ) -> None:
        registry: Registry[ComponentEntry] = Registry("surrogate", vocabulary)
        entry = makeEntry("gp_v1", supports={"posterior:gaussian"})
        with pytest.raises(RegistryError):
            registry.register(entry)
        vocabulary.register("posterior:gaussian", "The posterior is jointly Gaussian.")
        registry.register(entry)
        assert registry.lookup("gp_v1") is entry

    def test_each_registry_uses_its_own_vocabulary(self, makeEntry: MakeEntry) -> None:
        strict, lenient = CapabilityVocabulary(), CapabilityVocabulary()
        lenient.register("posterior:gaussian", "The posterior is jointly Gaussian.")
        entry = makeEntry("gp_v1", supports={"posterior:gaussian"})
        Registry("surrogate", lenient).register(entry)
        with pytest.raises(RegistryError):
            Registry("surrogate", strict).register(entry)

    def test_a_bad_name_is_reported_before_a_bad_tag(
        self, tagged: CapabilityVocabulary, makeEntry: MakeEntry
    ) -> None:
        registry: Registry[ComponentEntry] = Registry("surrogate", tagged)
        with pytest.raises(RegistryError, match="not a valid component name"):
            registry.register(makeEntry("Bad Name", needs={"posterior:gausian"}))

    def test_a_duplicate_is_reported_before_a_bad_tag(
        self, tagged: CapabilityVocabulary, makeEntry: MakeEntry
    ) -> None:
        registry: Registry[ComponentEntry] = Registry("surrogate", tagged)
        registry.register(makeEntry("gp_v1"))
        with pytest.raises(RegistryError, match="already registered"):
            registry.register(makeEntry("gp_v1", needs={"posterior:gausian"}))


class TestIndependence:
    def test_two_registries_share_no_state(
        self, vocabulary: CapabilityVocabulary, makeEntry: MakeEntry
    ) -> None:
        first: Registry[ComponentEntry] = Registry("surrogate", vocabulary)
        second: Registry[ComponentEntry] = Registry("kernel", vocabulary)
        first.register(makeEntry("matern_v1"))
        assert second.names() == []
        assert "matern_v1" not in second
        with pytest.raises(RegistryError):
            second.lookup("matern_v1")

    def test_two_registries_of_the_same_kind_share_no_state(
        self, vocabulary: CapabilityVocabulary, makeEntry: MakeEntry
    ) -> None:
        first: Registry[ComponentEntry] = Registry("surrogate", vocabulary)
        second: Registry[ComponentEntry] = Registry("surrogate", vocabulary)
        first.register(makeEntry("a"))
        assert second.names() == []

    def test_the_same_name_can_be_registered_in_two_registries(
        self, vocabulary: CapabilityVocabulary, makeEntry: MakeEntry
    ) -> None:
        first: Registry[ComponentEntry] = Registry("surrogate", vocabulary)
        second: Registry[ComponentEntry] = Registry("kernel", vocabulary)
        one, two = makeEntry("matern_v1"), makeEntry("matern_v1")
        first.register(one)
        second.register(two)
        assert first.lookup("matern_v1") is one
        assert second.lookup("matern_v1") is two

    def test_registries_may_share_a_vocabulary_without_sharing_entries(
        self, tagged: CapabilityVocabulary, makeEntry: MakeEntry
    ) -> None:
        first: Registry[ComponentEntry] = Registry("surrogate", tagged)
        second: Registry[ComponentEntry] = Registry("kernel", tagged)
        first.register(makeEntry("a", supports={"posterior:gaussian"}))
        assert first.vocabulary is second.vocabulary
        assert second.names() == []
