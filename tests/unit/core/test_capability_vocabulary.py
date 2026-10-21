"""Unit tests for `CapabilityVocabulary` (specification 5.12)."""

from __future__ import annotations

from typing import Any

import pytest

from planexs.core.CapabilityVocabulary import CapabilityVocabulary
from planexs.core.PlanexsError import PlanexsError
from planexs.core.RegistryError import RegistryError

VALID_TAGS = [
    "posterior:gaussian",
    "posterior:marginal_normal",
    "outcome_kind:binary",
    "outcome_constraints:supported",
    "objectives:single",
    "link:probit",
    "a:b",
    "g1:n1",
    "x_y:z_w2",
]

MALFORMED_TAGS = [
    pytest.param("", id="empty"),
    pytest.param("nogroup", id="no-colon"),
    pytest.param(":name", id="empty-group"),
    pytest.param("group:", id="empty-name"),
    pytest.param(":", id="only-a-colon"),
    pytest.param("Group:name", id="uppercase-group"),
    pytest.param("group:Name", id="uppercase-name"),
    pytest.param("GROUP:NAME", id="all-uppercase"),
    pytest.param("group: name", id="space-after-colon"),
    pytest.param("group :name", id="space-before-colon"),
    pytest.param("gr oup:name", id="space-in-group"),
    pytest.param("group:na me", id="space-in-name"),
    pytest.param("1group:name", id="group-leading-digit"),
    pytest.param("group:1name", id="name-leading-digit"),
    pytest.param("_group:name", id="group-leading-underscore"),
    pytest.param("group_:name", id="group-trailing-underscore"),
    pytest.param("group:_name", id="name-leading-underscore"),
    pytest.param("group:name_", id="name-trailing-underscore"),
    pytest.param("gr__oup:name", id="group-double-underscore"),
    pytest.param("group:na__me", id="name-double-underscore"),
    pytest.param("a:b:c", id="two-colons"),
    pytest.param("group::name", id="double-colon"),
    pytest.param("group-x:name", id="hyphen-in-group"),
    pytest.param("group:name-x", id="hyphen-in-name"),
    pytest.param("group.name", id="dot-instead-of-colon"),
    pytest.param("group/name", id="slash-instead-of-colon"),
    pytest.param("group:name\n", id="trailing-newline"),
    pytest.param("\ngroup:name", id="leading-newline"),
    pytest.param("group:caf\u00e9", id="non-ascii"),
]

NOT_STRINGS = [
    pytest.param(3, id="int"),
    pytest.param(None, id="none"),
    pytest.param(["a:b"], id="list"),
    pytest.param(("a", "b"), id="tuple"),
]

BAD_DESCRIPTIONS = [
    pytest.param("", id="empty"),
    pytest.param("   ", id="blank"),
    pytest.param("two\nlines", id="two-lines"),
    pytest.param("trailing newline\n", id="trailing-newline"),
    pytest.param(None, id="none"),
]


@pytest.fixture
def filled(vocabulary: CapabilityVocabulary) -> CapabilityVocabulary:
    """A vocabulary with two groups, registered out of alphabetical order on purpose."""
    vocabulary.register("posterior:marginal_normal", "The posterior has a normal marginal.")
    vocabulary.register("outputs:single", "One outcome.")
    vocabulary.register("posterior:gaussian", "The posterior is jointly Gaussian.")
    vocabulary.register("outputs:multi", "Several outcomes.")
    return vocabulary


class TestWellFormedness:
    @pytest.mark.parametrize("tag", VALID_TAGS)
    def test_a_well_formed_tag_is_accepted(self, tag: str) -> None:
        assert CapabilityVocabulary.isValidTag(tag)

    @pytest.mark.parametrize("tag", MALFORMED_TAGS)
    def test_a_malformed_tag_is_rejected(self, tag: str) -> None:
        assert not CapabilityVocabulary.isValidTag(tag)

    @pytest.mark.parametrize("tag", NOT_STRINGS)
    def test_something_that_is_not_a_string_is_not_a_tag(self, tag: object) -> None:
        assert not CapabilityVocabulary.isValidTag(tag)


class TestRegister:
    def test_a_new_vocabulary_is_empty(self, vocabulary: CapabilityVocabulary) -> None:
        assert vocabulary.tags() == []
        assert vocabulary.groups() == []

    @pytest.mark.parametrize("tag", VALID_TAGS)
    def test_every_well_formed_tag_can_be_registered(
        self, vocabulary: CapabilityVocabulary, tag: str
    ) -> None:
        vocabulary.register(tag, f"Means {tag}.")
        assert tag in vocabulary
        assert vocabulary.describe(tag) == f"Means {tag}."

    def test_a_registered_tag_is_a_member_and_an_unregistered_one_is_not(
        self, filled: CapabilityVocabulary
    ) -> None:
        assert "posterior:gaussian" in filled
        assert "posterior:laplace" not in filled

    @pytest.mark.parametrize("tag", NOT_STRINGS)
    def test_something_that_is_not_a_string_is_not_a_member_and_does_not_raise(
        self, filled: CapabilityVocabulary, tag: object
    ) -> None:
        assert tag not in filled


class TestDuplicates:
    def test_registering_a_tag_twice_raises(self, vocabulary: CapabilityVocabulary) -> None:
        vocabulary.register("posterior:gaussian", "First.")
        with pytest.raises(RegistryError, match="already registered") as caught:
            vocabulary.register("posterior:gaussian", "Second.")
        assert "'posterior:gaussian'" in caught.value.message
        assert caught.value.suggestion is not None

    def test_the_first_description_survives_a_duplicate(
        self, vocabulary: CapabilityVocabulary
    ) -> None:
        vocabulary.register("posterior:gaussian", "First.")
        with pytest.raises(RegistryError):
            vocabulary.register("posterior:gaussian", "Second.")
        assert vocabulary.describe("posterior:gaussian") == "First."
        assert vocabulary.tags() == ["posterior:gaussian"]

    def test_the_error_shows_the_existing_description(
        self, vocabulary: CapabilityVocabulary
    ) -> None:
        vocabulary.register("posterior:gaussian", "The posterior is jointly Gaussian.")
        with pytest.raises(RegistryError, match="jointly Gaussian"):
            vocabulary.register("posterior:gaussian", "Something else.")


class TestMalformedTags:
    @pytest.mark.parametrize("tag", MALFORMED_TAGS)
    def test_registering_a_malformed_tag_raises(
        self, vocabulary: CapabilityVocabulary, tag: str
    ) -> None:
        with pytest.raises(RegistryError, match="not a well-formed capability tag"):
            vocabulary.register(tag, "A description.")

    @pytest.mark.parametrize("tag", MALFORMED_TAGS)
    def test_a_refused_tag_leaves_the_vocabulary_unchanged(
        self, filled: CapabilityVocabulary, tag: str
    ) -> None:
        before = filled.tags()
        with pytest.raises(RegistryError):
            filled.register(tag, "A description.")
        assert filled.tags() == before

    @pytest.mark.parametrize("tag", NOT_STRINGS)
    def test_registering_something_that_is_not_a_string_raises_registry_error(
        self, vocabulary: CapabilityVocabulary, tag: Any
    ) -> None:
        with pytest.raises(RegistryError):
            vocabulary.register(tag, "A description.")

    @pytest.mark.parametrize("tag", MALFORMED_TAGS)
    def test_checking_a_malformed_tag_raises(
        self, filled: CapabilityVocabulary, tag: str
    ) -> None:
        with pytest.raises(RegistryError, match="not a well-formed capability tag"):
            filled.check(tag)

    def test_the_message_states_the_rule_and_the_error_gives_a_suggestion(
        self, vocabulary: CapabilityVocabulary
    ) -> None:
        with pytest.raises(RegistryError) as caught:
            vocabulary.register("Not A Tag", "A description.")
        assert "'Not A Tag'" in caught.value.message
        assert "group:name" in caught.value.message
        assert "'posterior:gaussian'" in caught.value.message
        assert caught.value.suggestion is not None


class TestDescriptions:
    @pytest.mark.parametrize("description", BAD_DESCRIPTIONS)
    def test_a_tag_needs_a_description_of_one_non_blank_line(
        self, vocabulary: CapabilityVocabulary, description: Any
    ) -> None:
        with pytest.raises(RegistryError, match="without a description") as caught:
            vocabulary.register("posterior:gaussian", description)
        assert caught.value.field_path == "description"
        assert "posterior:gaussian" not in vocabulary

    def test_the_description_is_required(self, vocabulary: CapabilityVocabulary) -> None:
        with pytest.raises(TypeError):
            vocabulary.register("posterior:gaussian")  # type: ignore[call-arg]

    def test_describe_returns_what_was_registered(self, filled: CapabilityVocabulary) -> None:
        assert filled.describe("outputs:single") == "One outcome."

    def test_describing_an_unknown_tag_raises_like_a_check(
        self, filled: CapabilityVocabulary
    ) -> None:
        with pytest.raises(RegistryError, match="Unknown capability tag"):
            filled.describe("outputs:several")


class TestCheck:
    def test_a_registered_tag_passes(self, filled: CapabilityVocabulary) -> None:
        assert filled.check("posterior:gaussian") is None

    def test_an_unknown_tag_in_a_known_group_lists_the_tags_of_that_group(
        self, filled: CapabilityVocabulary
    ) -> None:
        with pytest.raises(RegistryError) as caught:
            filled.check("posterior:gausian")
        assert caught.value.message == (
            "Unknown capability tag 'posterior:gausian'. "
            "Known tags of the group 'posterior': "
            "posterior:gaussian, posterior:marginal_normal."
        )

    def test_the_listing_leaves_out_the_tags_of_other_groups(
        self, filled: CapabilityVocabulary
    ) -> None:
        with pytest.raises(RegistryError) as caught:
            filled.check("outputs:several")
        assert "outputs:multi, outputs:single" in caught.value.message
        assert "posterior" not in caught.value.message

    def test_the_listing_reaches_the_user_through_str(
        self, filled: CapabilityVocabulary
    ) -> None:
        with pytest.raises(RegistryError) as caught:
            filled.check("posterior:gausian")
        assert "posterior:gaussian, posterior:marginal_normal" in str(caught.value)

    def test_an_unknown_group_lists_the_known_groups(
        self, filled: CapabilityVocabulary
    ) -> None:
        with pytest.raises(RegistryError) as caught:
            filled.check("posteriors:gaussian")
        assert caught.value.message == (
            "Unknown capability tag 'posteriors:gaussian': no tag of the group 'posteriors' "
            "is registered. Known groups: outputs, posterior."
        )

    def test_an_empty_vocabulary_says_so(self, vocabulary: CapabilityVocabulary) -> None:
        with pytest.raises(RegistryError, match=r"Known groups: none\."):
            vocabulary.check("posterior:gaussian")

    def test_the_error_carries_the_hints_it_was_given(
        self, filled: CapabilityVocabulary
    ) -> None:
        with pytest.raises(RegistryError) as caught:
            filled.check(
                "posterior:gausian", component="surrogate 'gp_v1'", field_path="needs"
            )
        assert caught.value.component == "surrogate 'gp_v1'"
        assert caught.value.field_path == "needs"
        assert caught.value.suggestion is not None
        assert "register the tag" in caught.value.suggestion

    def test_a_malformed_tag_error_carries_the_hints_too(
        self, filled: CapabilityVocabulary
    ) -> None:
        with pytest.raises(RegistryError) as caught:
            filled.check("Bad Tag", component="surrogate 'gp_v1'", field_path="supports")
        assert caught.value.component == "surrogate 'gp_v1'"
        assert caught.value.field_path == "supports"

    def test_the_hints_are_keyword_only(self, filled: CapabilityVocabulary) -> None:
        with pytest.raises(TypeError):
            filled.check("posterior:gausian", "surrogate")  # type: ignore[misc]

    def test_every_check_error_is_a_planexs_error(self, filled: CapabilityVocabulary) -> None:
        with pytest.raises(PlanexsError):
            filled.check("posterior:gausian")

    def test_a_tag_registered_later_passes(self, filled: CapabilityVocabulary) -> None:
        with pytest.raises(RegistryError):
            filled.check("posterior:laplace")
        filled.register("posterior:laplace", "The posterior is Laplace.")
        filled.check("posterior:laplace")

    def test_the_listing_in_the_message_includes_a_tag_added_later(
        self, filled: CapabilityVocabulary
    ) -> None:
        filled.register("posterior:laplace", "The posterior is Laplace.")
        with pytest.raises(RegistryError) as caught:
            filled.check("posterior:gausian")
        assert "posterior:gaussian, posterior:laplace, posterior:marginal_normal" in (
            caught.value.message
        )


class TestSortedListing:
    def test_tags_are_sorted_whatever_the_registration_order(
        self, filled: CapabilityVocabulary
    ) -> None:
        assert filled.tags() == [
            "outputs:multi",
            "outputs:single",
            "posterior:gaussian",
            "posterior:marginal_normal",
        ]

    def test_tags_can_be_listed_by_group(self, filled: CapabilityVocabulary) -> None:
        assert filled.tags("posterior") == ["posterior:gaussian", "posterior:marginal_normal"]
        assert filled.tags("outputs") == ["outputs:multi", "outputs:single"]

    def test_a_group_with_no_tag_lists_nothing(self, filled: CapabilityVocabulary) -> None:
        assert filled.tags("nothing") == []

    def test_a_group_name_is_matched_whole_not_as_a_prefix(
        self, filled: CapabilityVocabulary
    ) -> None:
        assert filled.tags("post") == []
        assert filled.tags("posterior:gaussian") == []

    def test_groups_are_sorted_and_listed_once(self, filled: CapabilityVocabulary) -> None:
        assert filled.groups() == ["outputs", "posterior"]

    def test_groups_are_sorted_whatever_the_registration_order(
        self, vocabulary: CapabilityVocabulary
    ) -> None:
        # Many groups, registered in reverse: a set's own iteration order would not pass.
        names = ["zeta", "yankee", "mike", "kilo", "bravo", "alpha"]
        for name in names:
            vocabulary.register(f"{name}:x", "A description.")
            vocabulary.register(f"{name}:y", "A description.")
        assert vocabulary.groups() == sorted(names)
        assert vocabulary.tags() == sorted(f"{n}:{w}" for n in names for w in "xy")

    def test_the_listings_are_copies(self, filled: CapabilityVocabulary) -> None:
        tags, groups = filled.tags(), filled.groups()
        tags.clear()
        groups.clear()
        assert len(filled.tags()) == 4
        assert filled.groups() == ["outputs", "posterior"]

    def test_the_listing_is_sorted_by_tag_not_by_description(
        self, vocabulary: CapabilityVocabulary
    ) -> None:
        vocabulary.register("a:late", "Zzz.")
        vocabulary.register("a:early", "Aaa.")
        assert vocabulary.tags() == ["a:early", "a:late"]


class TestIndependence:
    def test_two_vocabularies_share_no_state(self) -> None:
        first, second = CapabilityVocabulary(), CapabilityVocabulary()
        first.register("posterior:gaussian", "A description.")
        assert second.tags() == []
        assert "posterior:gaussian" not in second
        second.register("posterior:gaussian", "Another description.")
        assert first.describe("posterior:gaussian") == "A description."

    def test_the_default_vocabulary_is_one_object(self) -> None:
        assert CapabilityVocabulary.default() is CapabilityVocabulary.default()

    def test_the_default_vocabulary_is_not_a_vocabulary_made_for_a_test(self) -> None:
        assert CapabilityVocabulary.default() is not CapabilityVocabulary()

    def test_tags_registered_in_the_default_vocabulary_are_read_through_default(
        self, isolatedVocabulary: CapabilityVocabulary
    ) -> None:
        CapabilityVocabulary.default().register("posterior:gaussian", "A description.")
        assert CapabilityVocabulary.default() is isolatedVocabulary
        assert isolatedVocabulary.tags() == ["posterior:gaussian"]
