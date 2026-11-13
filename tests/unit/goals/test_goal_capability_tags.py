"""Unit tests for the capability tags of the goal layer (roadmap task P4-02, specification
5.12).

The layer introduces five tags, and registers them next to its kind. A goal supports them, and
acquisitions, analyses and stopping criteria need them from it.
"""

from __future__ import annotations

import pytest
from goal_fakes import defineFake

import planexs.goals  # importing the package registers the tags
from planexs.core.CapabilityVocabulary import CapabilityVocabulary
from planexs.core.KindTable import KindTable
from planexs.core.RegistryError import RegistryError
from planexs.goals.goalRegistry import goalRegistry

#: The tags of roadmap task P4-02, by group.
TAGS = {
    "goal": ["goal:explore", "goal:level_set", "goal:optimize"],
    "objectives": ["objectives:multi", "objectives:single"],
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
            CapabilityVocabulary.default().register("goal:optimize", "Another meaning.")

    def test_the_registry_of_the_kind_shares_the_vocabulary_they_are_in(self) -> None:
        assert planexs.goals.goalRegistry.goalRegistry().vocabulary is (
            CapabilityVocabulary.default()
        )


class TestAGoalCanDeclareThem:
    @pytest.mark.parametrize("tag", ALL_TAGS)
    def test_as_a_support(self, isolatedGoalKind: KindTable, tag: str) -> None:
        defineFake(supports=frozenset({tag}))

        assert goalRegistry().lookup("fake_goal_v1").supports == {tag}

    @pytest.mark.parametrize("tag", ALL_TAGS)
    def test_as_a_need(self, isolatedGoalKind: KindTable, tag: str) -> None:
        defineFake(needs=frozenset({tag}), supports=frozenset())

        assert goalRegistry().lookup("fake_goal_v1").needs == {tag}

    def test_all_of_them_at_once(self, isolatedGoalKind: KindTable) -> None:
        defineFake(supports=frozenset(ALL_TAGS))

        assert goalRegistry().lookup("fake_goal_v1").supports == set(ALL_TAGS)


class TestATypoIsCaught:
    def test_a_misspelled_tag_names_the_tags_of_its_group(
        self, isolatedGoalKind: KindTable
    ) -> None:
        with pytest.raises(
            RegistryError, match="Unknown capability tag 'goal:optimise'"
        ) as caught:
            defineFake(supports=frozenset({"goal:optimise"}))

        for tag in TAGS["goal"]:
            assert tag in caught.value.message
        assert caught.value.component == "goal 'fake_goal_v1'"
        assert caught.value.field_path == "supports"
        assert "fake_goal_v1" not in goalRegistry()

    def test_a_misspelled_need_is_caught_the_same_way(
        self, isolatedGoalKind: KindTable
    ) -> None:
        with pytest.raises(RegistryError, match="'objectives:several'") as caught:
            defineFake(needs=frozenset({"objectives:several"}))

        assert caught.value.field_path == "needs"
        assert "objectives:multi" in caught.value.message

    def test_a_tag_of_a_group_that_does_not_exist_names_the_groups(
        self, isolatedGoalKind: KindTable
    ) -> None:
        with pytest.raises(RegistryError, match="no tag of the group 'goals'") as caught:
            defineFake(supports=frozenset({"goals:optimize"}))

        assert "goal" in caught.value.message

    @pytest.mark.parametrize("tag", ["Goal:optimize", "optimize", "goal:", "a b:c"])
    def test_a_malformed_tag_is_refused(self, isolatedGoalKind: KindTable, tag: str) -> None:
        with pytest.raises(RegistryError, match="not a well-formed capability tag"):
            defineFake(supports=frozenset({tag}))
