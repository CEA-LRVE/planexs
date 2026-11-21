"""Unit tests for the capability tags of the acquisition layer (roadmap task P5-01,
specification 5.12).

The layer's three tags are what an acquisition passes as `supports`, and what the optimizers
and the policies will need from one. Two are registered next to the kind, and the third,
`outcome_constraints:supported`, with the goal tags (roadmap task P4-03), because a goal is
what needs it and a registry rejects a tag that is not in the vocabulary.
"""

from __future__ import annotations

import subprocess
import sys

import pytest
from acquisition_fakes import defineFake

import planexs.acquisition  # noqa: F401
from planexs.acquisition.acquisitionRegistry import acquisitionRegistry
from planexs.core.CapabilityVocabulary import CapabilityVocabulary
from planexs.core.KindTable import KindTable
from planexs.core.RegistryError import RegistryError

#: The tags registered with the kind, by group.
OWN_TAGS = {
    "acquisition": ["acquisition:differentiable"],
    "pending": ["pending:supported"],
}

#: The tag of the layer that the goal layer registers.
GOAL_LAYER_TAG = "outcome_constraints:supported"

ALL_TAGS = [tag for group in OWN_TAGS.values() for tag in group] + [GOAL_LAYER_TAG]


class TestTheTagsAreInTheVocabulary:
    @pytest.mark.parametrize("tag", ALL_TAGS)
    def test_each_tag_is_registered_with_a_one_line_description(self, tag: str) -> None:
        vocabulary = CapabilityVocabulary.default()

        assert tag in vocabulary
        description = vocabulary.describe(tag)
        assert description.strip()
        assert description.splitlines() == [description]
        assert description.endswith(".")

    @pytest.mark.parametrize("group", list(OWN_TAGS))
    def test_every_tag_of_each_group_is_listed_under_it(self, group: str) -> None:
        listed = CapabilityVocabulary.default().tags(group)

        assert set(OWN_TAGS[group]) <= set(listed)

    def test_the_groups_exist(self) -> None:
        groups = set(CapabilityVocabulary.default().groups())

        assert set(OWN_TAGS) | {"outcome_constraints"} <= groups

    def test_the_descriptions_tell_the_tags_apart(self) -> None:
        vocabulary = CapabilityVocabulary.default()

        descriptions = [vocabulary.describe(tag) for tag in ALL_TAGS]

        assert len(set(descriptions)) == len(ALL_TAGS)

    @pytest.mark.parametrize("tag", ALL_TAGS)
    def test_a_tag_is_registered_once_and_cannot_be_registered_again(self, tag: str) -> None:
        # Why the acquisition layer does not register `outcome_constraints:supported` itself.
        with pytest.raises(RegistryError, match="already registered"):
            CapabilityVocabulary.default().register(tag, "Another meaning.")

    def test_the_tag_of_the_goal_layer_exists_as_soon_as_the_goal_layer_is_imported(
        self,
    ) -> None:
        # It is the goal layer that registers it, so that the entry of `optimize_v1` can
        # declare it before any acquisition exists: importing `planexs.goals` alone has it,
        # and has not the two tags that belong to this layer.
        program = (
            "import planexs.goals\n"
            "from planexs.core.CapabilityVocabulary import CapabilityVocabulary\n"
            "vocabulary = CapabilityVocabulary.default()\n"
            "print('outcome_constraints:supported' in vocabulary)\n"
            "print('acquisition:differentiable' in vocabulary)\n"
            "print('pending:supported' in vocabulary)\n"
        )

        result = subprocess.run(
            [sys.executable, "-c", program], capture_output=True, text=True, check=False
        )

        assert result.returncode == 0, result.stderr
        assert result.stdout.split() == ["True", "False", "False"]


class TestTheRegistryValidatesTheTagsOfAnEntry:
    @pytest.mark.parametrize("tag", ALL_TAGS)
    def test_each_tag_is_accepted_as_a_support(
        self, tag: str, isolatedAcquisitionKind: KindTable
    ) -> None:
        defineFake("supports_it_v1", supports=frozenset({tag}))

        assert acquisitionRegistry().lookup("supports_it_v1").supports == {tag}

    @pytest.mark.parametrize("tag", ALL_TAGS)
    def test_each_tag_is_accepted_as_a_need(
        self, tag: str, isolatedAcquisitionKind: KindTable
    ) -> None:
        defineFake("needs_it_v1", needs=frozenset({tag}), supports=frozenset())

        assert acquisitionRegistry().lookup("needs_it_v1").needs == {tag}

    def test_an_acquisition_may_declare_all_three_at_once(
        self, isolatedAcquisitionKind: KindTable
    ) -> None:
        defineFake("everything_v1", supports=frozenset(ALL_TAGS))

        assert acquisitionRegistry().lookup("everything_v1").supports == set(ALL_TAGS)

    def test_a_misspelled_tag_is_refused_and_the_tags_of_its_group_are_listed(
        self, isolatedAcquisitionKind: KindTable
    ) -> None:
        with pytest.raises(RegistryError) as caught:
            defineFake("typo_v1", supports=frozenset({"pending:suported"}))

        assert "Unknown capability tag 'pending:suported'" in caught.value.message
        assert "pending:supported" in caught.value.message  # the right spelling is listed
        assert caught.value.field_path == "supports"
        assert caught.value.component == "acquisition 'typo_v1'"
        assert "typo_v1" not in acquisitionRegistry()  # nothing was registered

    def test_a_misspelled_need_is_refused_too(
        self, isolatedAcquisitionKind: KindTable
    ) -> None:
        with pytest.raises(RegistryError) as caught:
            defineFake("typo_v1", needs=frozenset({"acquisition:differentiabl"}))

        assert caught.value.field_path == "needs"
        assert "acquisition:differentiable" in caught.value.message

    def test_a_tag_of_an_unknown_group_lists_the_groups(
        self, isolatedAcquisitionKind: KindTable
    ) -> None:
        with pytest.raises(RegistryError, match="Known groups"):
            defineFake("typo_v1", supports=frozenset({"batches:supported"}))

    def test_a_malformed_tag_is_refused(self, isolatedAcquisitionKind: KindTable) -> None:
        with pytest.raises(RegistryError, match="not a well-formed capability tag"):
            defineFake("typo_v1", supports=frozenset({"Pending-Supported"}))
