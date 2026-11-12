"""Unit tests for the capability tags of the surrogate layer (roadmap task P3-07,
specification 5.12).

The layer introduces seven tags, and registers them next to its kind. They are what a surrogate
passes as `needs` and `supports`, and what acquisitions and analyses will need from one.
"""

from __future__ import annotations

import pytest
from surrogate_fakes import defineFake

import planexs.surrogates  # noqa: F401
from planexs.core.CapabilityVocabulary import CapabilityVocabulary
from planexs.core.KindTable import KindTable
from planexs.core.RegistryError import RegistryError
from planexs.surrogates.surrogateRegistry import surrogateRegistry

#: The tags of roadmap task P3-07, by group.
TAGS = {
    "posterior": [
        "posterior:gaussian",
        "posterior:latent_gaussian",
        "posterior:marginal_normal",
    ],
    "outputs": ["outputs:multi", "outputs:single"],
    "noise": ["noise:inferred", "noise:known"],
}
ALL_TAGS = [tag for group in TAGS.values() for tag in group]


class TestTheTagsAreInTheVocabulary:
    @pytest.mark.parametrize("tag", ALL_TAGS)
    def test_each_tag_is_registered_with_a_one_line_description(self, tag: str) -> None:
        vocabulary = CapabilityVocabulary.default()

        assert tag in vocabulary
        description = vocabulary.describe(tag)
        assert description.strip()
        assert description.splitlines() == [description]
        assert description.endswith(".")

    @pytest.mark.parametrize("group", list(TAGS))
    def test_every_tag_of_each_group_is_listed_under_it(self, group: str) -> None:
        listed = CapabilityVocabulary.default().tags(group)

        assert set(TAGS[group]) <= set(listed)

    def test_the_groups_exist(self) -> None:
        assert set(TAGS) <= set(CapabilityVocabulary.default().groups())

    def test_the_descriptions_tell_the_tags_apart(self) -> None:
        vocabulary = CapabilityVocabulary.default()

        descriptions = [vocabulary.describe(tag) for tag in ALL_TAGS]

        assert len(set(descriptions)) == len(ALL_TAGS)

    def test_the_tags_are_registered_once_and_cannot_be_registered_again(self) -> None:
        with pytest.raises(RegistryError, match="already registered"):
            CapabilityVocabulary.default().register("posterior:gaussian", "Another meaning.")


class TestASurrogateCanDeclareThem:
    @pytest.mark.parametrize("tag", ALL_TAGS)
    def test_as_a_support(self, isolatedSurrogateKind: KindTable, tag: str) -> None:
        defineFake(supports=frozenset({tag}))

        assert surrogateRegistry().lookup("fake_v1").supports == {tag}

    @pytest.mark.parametrize("tag", ALL_TAGS)
    def test_as_a_need(self, isolatedSurrogateKind: KindTable, tag: str) -> None:
        defineFake(needs=frozenset({tag}), supports=frozenset())

        assert surrogateRegistry().lookup("fake_v1").needs == {tag}

    def test_all_of_them_at_once(self, isolatedSurrogateKind: KindTable) -> None:
        defineFake(supports=frozenset(ALL_TAGS))

        assert surrogateRegistry().lookup("fake_v1").supports == set(ALL_TAGS)


class TestATypoIsCaught:
    def test_a_misspelled_tag_names_the_tags_of_its_group(
        self, isolatedSurrogateKind: KindTable
    ) -> None:
        with pytest.raises(
            RegistryError, match="Unknown capability tag 'posterior:gausian'"
        ) as caught:
            defineFake(supports=frozenset({"posterior:gausian"}))

        for tag in TAGS["posterior"]:
            assert tag in caught.value.message
        assert caught.value.component == "surrogate 'fake_v1'"
        assert caught.value.field_path == "supports"
        assert "fake_v1" not in surrogateRegistry()

    def test_a_misspelled_need_is_caught_the_same_way(
        self, isolatedSurrogateKind: KindTable
    ) -> None:
        with pytest.raises(RegistryError, match="'noise:unknown'") as caught:
            defineFake(needs=frozenset({"noise:unknown"}))

        assert caught.value.field_path == "needs"
        assert "noise:known" in caught.value.message

    def test_a_tag_of_a_group_that_does_not_exist_names_the_groups(
        self, isolatedSurrogateKind: KindTable
    ) -> None:
        with pytest.raises(RegistryError, match="no tag of the group 'posteriors'") as caught:
            defineFake(supports=frozenset({"posteriors:gaussian"}))

        assert "posterior" in caught.value.message

    @pytest.mark.parametrize("tag", ["Posterior:gaussian", "gaussian", "posterior:", "a b:c"])
    def test_a_malformed_tag_is_refused(
        self, isolatedSurrogateKind: KindTable, tag: str
    ) -> None:
        with pytest.raises(RegistryError, match="not a well-formed capability tag"):
            defineFake(supports=frozenset({tag}))
