"""Unit tests for `checkCompatibility` (specification 3.6, 5.12)."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel

from planexs.core.CapabilityVocabulary import CapabilityVocabulary
from planexs.core.checkCompatibility import checkCompatibility
from planexs.core.CompatibilityError import CompatibilityError
from planexs.core.ComponentEntry import ComponentEntry
from planexs.core.DeclaredParticipant import DeclaredParticipant
from planexs.core.KindTable import KindTable
from planexs.core.PlanexsError import PlanexsError
from planexs.core.RegistryError import RegistryError

MakeEntry = Callable[..., ComponentEntry]

TAGS = {
    "goal:optimize": "The goal is to optimize outcomes.",
    "outputs:multi": "Several outcomes.",
    "outputs:single": "One outcome.",
    "pending:supported": "Points still being evaluated are taken into account.",
    "posterior:gaussian": "The posterior is jointly Gaussian.",
    "space:categorical": "The search space has unordered choices.",
    "space:conditional": "The search space has conditional parameters.",
}


@pytest.fixture
def table(vocabulary: CapabilityVocabulary, makeEntry: MakeEntry) -> KindTable:
    """A kind table with the tags above and two kinds of registered component."""
    for tag, description in TAGS.items():
        vocabulary.register(tag, description)
    kinds = KindTable(vocabulary)
    surrogates = kinds.define("surrogate")
    surrogates.register(
        makeEntry("single_task_gp_v1", supports={"posterior:gaussian", "outputs:single"})
    )
    surrogates.register(
        makeEntry("multi_gp_v1", supports={"posterior:gaussian", "outputs:multi"})
    )
    optimizers = kinds.define("acquisition_optimizer")
    optimizers.register(makeEntry("relaxed_v1", supports={"space:conditional"}))
    optimizers.register(
        makeEntry("mixed_v1", supports={"space:conditional", "space:categorical"})
    )
    optimizers.register(
        makeEntry(
            "random_v1",
            supports={"space:conditional", "space:categorical", "pending:supported"},
            needs={"outputs:single"},
        )
    )
    return kinds


def declared(label: str, needs: Iterable[str] = (), supports: Iterable[str] = ()) -> Any:
    return DeclaredParticipant(label, needs=frozenset(needs), supports=frozenset(supports))


def failure(participants: Iterable[Any], table: KindTable) -> CompatibilityError:
    with pytest.raises(CompatibilityError) as caught:
        checkCompatibility(participants, table)
    return caught.value


class TestAllMet:
    def test_no_participant_is_compatible(self, table: KindTable) -> None:
        assert checkCompatibility([], table) is None

    def test_participants_without_needs_are_compatible(self, table: KindTable) -> None:
        participants = [declared("problem"), declared("goal", supports={"goal:optimize"})]
        assert checkCompatibility(participants, table) is None

    def test_a_need_met_by_another_participant_passes(self, table: KindTable) -> None:
        participants = [
            declared("optimizer", needs={"space:categorical"}),
            declared("other", supports={"space:categorical"}),
        ]
        assert checkCompatibility(participants, table) is None

    def test_a_need_met_by_the_goal_participant_passes(self, table: KindTable) -> None:
        participants = [
            declared("problem"),
            declared("goal 'optimize_v1'", supports={"goal:optimize", "outputs:single"}),
            declared("acquisition 'ucb_v1'", needs={"goal:optimize"}),
        ]
        assert checkCompatibility(participants, table) is None

    def test_a_need_met_by_a_component_taken_from_its_entry_passes(
        self, table: KindTable
    ) -> None:
        surrogate = DeclaredParticipant.fromEntry(
            "surrogate", table.registry("surrogate").lookup("single_task_gp_v1")
        )
        participants = [declared("acquisition", needs={"posterior:gaussian"}), surrogate]
        assert checkCompatibility(participants, table) is None

    def test_several_needs_met_by_different_participants_pass(self, table: KindTable) -> None:
        participants = [
            declared("problem", needs={"space:categorical", "outputs:multi"}),
            declared("optimizer", supports={"space:categorical"}),
            declared("surrogate", supports={"outputs:multi"}),
        ]
        assert checkCompatibility(participants, table) is None

    def test_one_support_can_meet_the_needs_of_many_participants(
        self, table: KindTable
    ) -> None:
        participants = [
            declared("a", needs={"goal:optimize"}),
            declared("b", needs={"goal:optimize"}),
            declared("goal", supports={"goal:optimize"}),
        ]
        assert checkCompatibility(participants, table) is None

    def test_a_support_nobody_needs_is_harmless(self, table: KindTable) -> None:
        assert checkCompatibility([declared("x", supports={"outputs:multi"})], table) is None

    def test_a_participant_that_supports_its_own_need_meets_it(self, table: KindTable) -> None:
        # "Some participant" in specification 5.12 includes the one that has the need.
        participants = [
            declared("self-sufficient", needs={"goal:optimize"}, supports={"goal:optimize"})
        ]
        assert checkCompatibility(participants, table) is None

    def test_the_order_of_participants_does_not_matter_when_all_are_met(
        self, table: KindTable
    ) -> None:
        participants = [
            declared("a", needs={"goal:optimize"}),
            declared("goal", supports={"goal:optimize"}),
        ]
        assert checkCompatibility(participants[::-1], table) is None

    def test_participants_may_be_given_as_a_generator(self, table: KindTable) -> None:
        participants = (
            declared(label, needs=needs, supports=supports)
            for label, needs, supports in [
                ("a", {"goal:optimize"}, set()),
                ("goal", set(), {"goal:optimize"}),
            ]
        )
        assert checkCompatibility(participants, table) is None

    def test_a_generator_is_read_in_full_so_its_unmet_needs_are_seen(
        self, table: KindTable
    ) -> None:
        # A consumed or half-read generator would look like "no participants": compatible.
        participants = (
            declared(label, needs=needs) for label, needs in [("problem", {"outputs:multi"})]
        )
        message = failure(participants, table).message
        assert "needed by: problem" in message


class TestUnmet:
    def test_one_unmet_need_raises_a_compatibility_error(self, table: KindTable) -> None:
        error = failure([declared("problem", needs={"outputs:multi"})], table)
        assert isinstance(error, CompatibilityError)
        assert isinstance(error, PlanexsError)
        assert "'outputs:multi'" in error.message

    def test_the_count_is_singular_for_one_need(self, table: KindTable) -> None:
        error = failure([declared("problem", needs={"outputs:multi"})], table)
        assert error.message.splitlines()[0] == (
            "The chosen components are not compatible: 1 need is not met."
        )

    def test_several_unmet_needs_are_reported_together(self, table: KindTable) -> None:
        error = failure(
            [
                declared(
                    "problem", needs={"outputs:multi", "space:categorical", "goal:optimize"}
                )
            ],
            table,
        )
        assert error.message.splitlines()[0].endswith("3 needs are not met.")
        for tag in ("outputs:multi", "space:categorical", "goal:optimize"):
            assert f"Need '{tag}'" in error.message

    def test_unmet_needs_from_different_participants_are_reported_together(
        self, table: KindTable
    ) -> None:
        error = failure(
            [
                declared("problem", needs={"outputs:multi"}),
                declared("goal", needs={"goal:optimize"}),
            ],
            table,
        )
        assert "2 needs are not met" in error.message
        assert "Need 'outputs:multi'" in error.message
        assert "Need 'goal:optimize'" in error.message

    def test_only_the_unmet_needs_are_listed(self, table: KindTable) -> None:
        error = failure(
            [
                declared("problem", needs={"outputs:multi", "goal:optimize"}),
                declared("goal", supports={"goal:optimize"}),
            ],
            table,
        )
        assert "1 need is not met" in error.message
        assert "Need 'outputs:multi'" in error.message
        assert "Need 'goal:optimize'" not in error.message

    def test_unmet_needs_are_listed_in_sorted_order_whatever_the_participant_order(
        self, table: KindTable
    ) -> None:
        participants = [
            declared("late", needs={"space:categorical"}),
            declared("early", needs={"goal:optimize"}),
            declared("middle", needs={"outputs:multi"}),
        ]
        for ordered in (participants, participants[::-1]):
            message = failure(ordered, table).message
            positions = [
                message.index(f"Need '{tag}'")
                for tag in ("goal:optimize", "outputs:multi", "space:categorical")
            ]
            assert positions == sorted(positions)

    def test_everyone_who_needs_a_tag_is_listed_in_participant_order(
        self, table: KindTable
    ) -> None:
        participants = [
            declared("problem", needs={"outputs:multi"}),
            declared("surrogate 'x_v1'", needs={"outputs:multi"}),
            declared("goal", needs={"outputs:multi"}),
        ]
        assert (
            "  needed by: problem, surrogate 'x_v1', goal"
            in failure(participants, table).message
        )
        reversed_message = failure(participants[::-1], table).message
        assert "  needed by: goal, surrogate 'x_v1', problem" in reversed_message

    def test_two_participants_with_the_same_label_are_both_listed(
        self, table: KindTable
    ) -> None:
        # Two nested kernels of the same name are two participants: both are involved.
        participants = [
            declared("kernel 'rbf_v1'", needs={"outputs:multi"}),
            declared("kernel 'rbf_v1'", needs={"outputs:multi"}),
        ]
        message = failure(participants, table).message
        assert "  needed by: kernel 'rbf_v1', kernel 'rbf_v1'" in message

    def test_a_participant_with_two_unmet_needs_appears_under_each(
        self, table: KindTable
    ) -> None:
        message = failure(
            [declared("problem", needs={"outputs:multi", "goal:optimize"})], table
        ).message
        assert message.count("needed by: problem") == 2

    def test_a_tag_needed_twice_is_one_unmet_need(self, table: KindTable) -> None:
        message = failure(
            [declared("a", needs={"outputs:multi"}), declared("b", needs={"outputs:multi"})],
            table,
        ).message
        assert "1 need is not met" in message
        assert message.count("Need 'outputs:multi'") == 1

    def test_a_support_of_another_tag_does_not_meet_a_need(self, table: KindTable) -> None:
        participants = [
            declared("problem", needs={"outputs:multi"}),
            declared("surrogate", supports={"outputs:single"}),
        ]
        assert "Need 'outputs:multi'" in failure(participants, table).message

    def test_the_description_of_each_unmet_tag_comes_from_the_vocabulary(
        self, table: KindTable
    ) -> None:
        message = failure([declared("p", needs={"space:categorical"})], table).message
        assert "Need 'space:categorical' (The search space has unordered choices.)" in message

    def test_the_error_carries_a_suggestion(self, table: KindTable) -> None:
        error = failure([declared("p", needs={"outputs:multi"})], table)
        assert error.suggestion is not None
        assert "Suggestion: choose components" in str(error)

    def test_a_failed_check_changes_nothing(self, table: KindTable) -> None:
        before = (table.kinds(), table.registry("surrogate").names(), table.vocabulary.tags())
        failure([declared("p", needs={"outputs:multi"})], table)
        after = (table.kinds(), table.registry("surrogate").names(), table.vocabulary.tags())
        assert before == after


class TestAlternatives:
    def test_registered_components_that_support_the_need_are_listed(
        self, table: KindTable
    ) -> None:
        message = failure([declared("problem", needs={"space:categorical"})], table).message
        assert (
            "  supported by: acquisition_optimizer 'mixed_v1', "
            "acquisition_optimizer 'random_v1'"
        ) in message

    def test_components_that_do_not_support_the_need_are_not_listed(
        self, table: KindTable
    ) -> None:
        message = failure([declared("problem", needs={"space:categorical"})], table).message
        assert "relaxed_v1" not in message
        assert "single_task_gp_v1" not in message

    def test_alternatives_are_found_across_kinds(self, table: KindTable) -> None:
        message = failure([declared("problem", needs={"posterior:gaussian"})], table).message
        assert "surrogate 'multi_gp_v1', surrogate 'single_task_gp_v1'" in message

    def test_alternatives_are_sorted_by_kind_and_then_by_name(
        self, table: KindTable, makeEntry: MakeEntry
    ) -> None:
        table.define("kernel").register(makeEntry("zz_v1", supports={"space:categorical"}))
        table.define("aaa_kind").register(makeEntry("zz_v1", supports={"space:categorical"}))
        table.registry("aaa_kind").register(makeEntry("aa_v1", supports={"space:categorical"}))
        message = failure([declared("p", needs={"space:categorical"})], table).message
        line = next(line for line in message.splitlines() if "supported by" in line)
        assert line == (
            "  supported by: aaa_kind 'aa_v1', aaa_kind 'zz_v1', "
            "acquisition_optimizer 'mixed_v1', acquisition_optimizer 'random_v1', "
            "kernel 'zz_v1'"
        )

    def test_each_unmet_need_gets_its_own_alternatives(self, table: KindTable) -> None:
        message = failure(
            [declared("p", needs={"space:categorical", "outputs:multi", "pending:supported"})],
            table,
        ).message
        lines = message.splitlines()
        by_need = {
            line.split("'")[1]: lines[position + 2]
            for position, line in enumerate(lines)
            if line.startswith("Need ")
        }
        assert by_need["outputs:multi"] == "  supported by: surrogate 'multi_gp_v1'"
        assert by_need["pending:supported"] == (
            "  supported by: acquisition_optimizer 'random_v1'"
        )
        assert "mixed_v1" in by_need["space:categorical"]

    def test_a_component_that_only_needs_the_tag_is_not_an_alternative(
        self, table: KindTable
    ) -> None:
        # `random_v1` needs `outputs:single`; the only supporter of it is `single_task_gp_v1`.
        message = failure([declared("p", needs={"outputs:single"})], table).message
        assert "  supported by: surrogate 'single_task_gp_v1'" in message
        assert "random_v1" not in message

    def test_a_need_nobody_supports_says_so(self, table: KindTable) -> None:
        message = failure([declared("p", needs={"goal:optimize"})], table).message
        assert "  supported by: no registered component" in message

    def test_a_kind_with_no_component_adds_nothing(self, table: KindTable) -> None:
        table.define("empty_kind")
        message = failure([declared("p", needs={"goal:optimize"})], table).message
        assert "no registered component" in message
        assert "empty_kind" not in message

    def test_an_empty_table_still_reports_the_unmet_need(
        self, vocabulary: CapabilityVocabulary
    ) -> None:
        vocabulary.register("goal:optimize", "The goal is to optimize outcomes.")
        message = failure(
            [declared("p", needs={"goal:optimize"})], KindTable(vocabulary)
        ).message
        assert "  supported by: no registered component" in message

    def test_a_component_registered_later_is_found(
        self, table: KindTable, makeEntry: MakeEntry
    ) -> None:
        participants = [declared("p", needs={"goal:optimize"})]
        assert "no registered component" in failure(participants, table).message
        table.registry("surrogate").register(makeEntry("late_v1", supports={"goal:optimize"}))
        assert "surrogate 'late_v1'" in failure(participants, table).message

    def test_the_default_kind_table_is_used_when_none_is_given(
        self, isolatedKindTable: KindTable, makeEntry: MakeEntry
    ) -> None:
        vocabulary = isolatedKindTable.vocabulary
        vocabulary.register("goal:optimize", "The goal is to optimize outcomes.")
        isolatedKindTable.define("goal").register(
            makeEntry("optimize_v1", supports={"goal:optimize"})
        )
        with pytest.raises(CompatibilityError) as caught:
            checkCompatibility([declared("acquisition", needs={"goal:optimize"})])
        assert "supported by: goal 'optimize_v1'" in caught.value.message
        assert checkCompatibility([declared("a")]) is None


class TestVocabulary:
    @pytest.mark.parametrize("field", ["needs", "supports"])
    def test_an_unknown_tag_in_a_group_lists_the_group_and_names_the_participant(
        self, table: KindTable, field: str
    ) -> None:
        participant = declared("problem", **{field: {"space:categorcal"}})
        with pytest.raises(RegistryError) as caught:
            checkCompatibility([participant], table)
        assert "'space:categorcal'" in caught.value.message
        assert "space:categorical, space:conditional" in caught.value.message
        assert caught.value.component == "problem"
        assert caught.value.field_path == field

    def test_an_unknown_group_lists_the_known_groups(self, table: KindTable) -> None:
        with pytest.raises(RegistryError, match="Known groups: goal, outputs, pending"):
            checkCompatibility([declared("problem", needs={"spaces:categorical"})], table)

    @pytest.mark.parametrize("field", ["needs", "supports"])
    def test_a_malformed_tag_is_reported(self, table: KindTable, field: str) -> None:
        with pytest.raises(RegistryError, match="not a well-formed capability tag") as caught:
            checkCompatibility([declared("problem", **{field: {"Not A Tag"}})], table)
        assert caught.value.field_path == field

    def test_a_tag_that_is_not_a_string_is_reported(self, table: KindTable) -> None:
        participant = SimpleNamespace(label="odd", needs={3}, supports=set())
        with pytest.raises(RegistryError, match="not a well-formed capability tag"):
            checkCompatibility([participant], table)

    def test_a_typo_is_reported_rather_than_an_unmet_need_nobody_could_meet(
        self, table: KindTable
    ) -> None:
        participants = [
            declared("problem", needs={"outputs:multi", "space:categorcal"}),
        ]
        with pytest.raises(RegistryError):
            checkCompatibility(participants, table)

    def test_the_first_bad_tag_is_the_one_reported(self, table: KindTable) -> None:
        participants = [
            declared("first", needs={"zzz:late", "aaa:early"}),
            declared("second", needs={"bbb:other"}),
        ]
        with pytest.raises(RegistryError) as caught:
            checkCompatibility(participants, table)
        assert caught.value.component == "first"
        assert "'aaa:early'" in caught.value.message

    def test_needs_are_checked_before_supports(self, table: KindTable) -> None:
        participant = declared("p", needs={"bbb:bad"}, supports={"aaa:bad"})
        with pytest.raises(RegistryError) as caught:
            checkCompatibility([participant], table)
        assert caught.value.field_path == "needs"

    def test_the_tags_are_judged_by_the_vocabulary_of_the_table_given(
        self, table: KindTable
    ) -> None:
        other = KindTable(CapabilityVocabulary())
        participant = declared("p", supports={"goal:optimize"})
        assert checkCompatibility([participant], table) is None
        with pytest.raises(RegistryError, match="Unknown capability tag"):
            checkCompatibility([participant], other)

    def test_a_tag_added_to_the_vocabulary_later_is_accepted(self, table: KindTable) -> None:
        participant = declared("p", supports={"link:probit"})
        with pytest.raises(RegistryError):
            checkCompatibility([participant], table)
        table.vocabulary.register("link:probit", "The response is a probit link.")
        assert checkCompatibility([participant], table) is None


class Config(BaseModel):
    """The validated configuration of a fake configurable component."""

    multi_output: bool = False
    categorical: bool = False


class ConfiguredSurrogate:
    """A participant whose tags are computed from its own validated configuration."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.label = "surrogate 'configured_v1'"

    @property
    def needs(self) -> frozenset[str]:
        return frozenset({"space:categorical"}) if self.config.categorical else frozenset()

    @property
    def supports(self) -> frozenset[str]:
        return frozenset({"outputs:multi"}) if self.config.multi_output else frozenset()


class CountingParticipant:
    """A participant that counts how many times its tags are read."""

    label = "counting"

    def __init__(self) -> None:
        self.needs_reads = 0
        self.supports_reads = 0

    @property
    def needs(self) -> frozenset[str]:
        self.needs_reads += 1
        return frozenset({"goal:optimize"})

    @property
    def supports(self) -> frozenset[str]:
        self.supports_reads += 1
        return frozenset({"goal:optimize"})


class TestTagsRefinedFromConfiguration:
    def test_supports_computed_from_the_configuration_meet_a_need(
        self, table: KindTable
    ) -> None:
        participants = [
            declared("problem", needs={"outputs:multi"}),
            ConfiguredSurrogate(Config(multi_output=True)),
        ]
        assert checkCompatibility(participants, table) is None

    def test_the_same_participant_with_another_configuration_leaves_the_need_unmet(
        self, table: KindTable
    ) -> None:
        participants = [
            declared("problem", needs={"outputs:multi"}),
            ConfiguredSurrogate(Config(multi_output=False)),
        ]
        assert "Need 'outputs:multi'" in failure(participants, table).message

    def test_needs_computed_from_the_configuration_are_checked(self, table: KindTable) -> None:
        participants = [ConfiguredSurrogate(Config(categorical=True))]
        message = failure(participants, table).message
        assert "Need 'space:categorical'" in message
        assert "needed by: surrogate 'configured_v1'" in message

    def test_a_configuration_without_the_need_passes(self, table: KindTable) -> None:
        assert checkCompatibility([ConfiguredSurrogate(Config())], table) is None

    def test_the_tags_are_read_when_the_check_runs_not_when_the_participant_is_built(
        self, table: KindTable
    ) -> None:
        config = Config(multi_output=False)
        participants = [
            declared("problem", needs={"outputs:multi"}),
            ConfiguredSurrogate(config),
        ]
        with pytest.raises(CompatibilityError):
            checkCompatibility(participants, table)
        config.multi_output = True
        assert checkCompatibility(participants, table) is None
        config.multi_output = False
        with pytest.raises(CompatibilityError):
            checkCompatibility(participants, table)

    def test_each_participants_tags_are_read_once_per_check(self, table: KindTable) -> None:
        counting = CountingParticipant()
        checkCompatibility([counting], table)
        assert (counting.needs_reads, counting.supports_reads) == (1, 1)
        checkCompatibility([counting], table)
        assert (counting.needs_reads, counting.supports_reads) == (2, 2)

    def test_a_failed_check_also_reads_each_participants_tags_once(
        self, table: KindTable
    ) -> None:
        counting = CountingParticipant()
        with pytest.raises(CompatibilityError):
            checkCompatibility([counting, declared("x", needs={"outputs:multi"})], table)
        assert (counting.needs_reads, counting.supports_reads) == (1, 1)


class TestAnyObjectWithTheThreeAttributesIsAParticipant:
    def test_a_plain_namespace_works(self, table: KindTable) -> None:
        participants = [
            SimpleNamespace(
                label="a", needs=frozenset({"goal:optimize"}), supports=frozenset()
            ),
            SimpleNamespace(
                label="b", needs=frozenset(), supports=frozenset({"goal:optimize"})
            ),
        ]
        assert checkCompatibility(participants, table) is None

    @pytest.mark.parametrize("make", [list, tuple, set, frozenset])
    def test_tags_may_be_any_collection_of_strings(
        self, table: KindTable, make: Callable[[Iterable[str]], Any]
    ) -> None:
        participants = [
            SimpleNamespace(label="a", needs=make(["goal:optimize"]), supports=make([])),
            SimpleNamespace(label="b", needs=make([]), supports=make(["goal:optimize"])),
        ]
        assert checkCompatibility(participants, table) is None

    def test_repeated_tags_are_one_tag(self, table: KindTable) -> None:
        participant = SimpleNamespace(
            label="a", needs=["outputs:multi", "outputs:multi"], supports=[]
        )
        assert "1 need is not met" in failure([participant], table).message


class TestMessageSnapshot:
    def test_the_wording_of_a_failure_is_pinned(
        self, vocabulary: CapabilityVocabulary, makeEntry: MakeEntry
    ) -> None:
        vocabulary.register("goal:optimize", "The goal is to optimize outcomes.")
        vocabulary.register("outputs:multi", "Several outcomes.")
        vocabulary.register("space:categorical", "The search space has unordered choices.")
        kinds = KindTable(vocabulary)
        optimizers = kinds.define("acquisition_optimizer")
        optimizers.register(makeEntry("relaxed_v1"))
        optimizers.register(makeEntry("random_v1", supports={"space:categorical"}))
        optimizers.register(makeEntry("mixed_v1", supports={"space:categorical"}))
        participants = [
            declared("problem", needs={"space:categorical", "outputs:multi"}),
            declared("goal 'optimize_v1'", supports={"goal:optimize"}),
            declared("acquisition_optimizer 'relaxed_v1'", needs={"space:categorical"}),
        ]
        with pytest.raises(CompatibilityError) as caught:
            checkCompatibility(participants, kinds)
        assert str(caught.value) == (
            "The chosen components are not compatible: 2 needs are not met.\n"
            "Need 'outputs:multi' (Several outcomes.)\n"
            "  needed by: problem\n"
            "  supported by: no registered component\n"
            "Need 'space:categorical' (The search space has unordered choices.)\n"
            "  needed by: problem, acquisition_optimizer 'relaxed_v1'\n"
            "  supported by: acquisition_optimizer 'mixed_v1', "
            "acquisition_optimizer 'random_v1'\n"
            "Suggestion: choose components that support the needs above, or change the "
            "problem or the goal so that they are no longer needed"
        )
