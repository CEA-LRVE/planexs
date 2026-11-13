"""Unit tests for `registerGoal` (roadmap task P4-02, specification 5.12, 6, 10.6).

A fake goal stands for a real one: the contract is tested before any goal exists. Every test
that registers one uses `isolatedGoalKind`, because a registered name is permanent and must
never reach the real registry.
"""

from __future__ import annotations

from typing import Any, Literal

import pytest
from goal_fakes import defineFake, defineRefining, fakeConfig, makeProblem
from pydantic import Field

import planexs.goals  # noqa: F401
from planexs.core.checkCompatibility import checkCompatibility
from planexs.core.CompatibilityError import CompatibilityError
from planexs.core.ComponentConfig import ComponentConfig
from planexs.core.ConfigError import ConfigError
from planexs.core.DeclaredParticipant import DeclaredParticipant
from planexs.core.KindTable import KindTable
from planexs.core.RegistryError import RegistryError
from planexs.core.resolveComponent import resolveComponent
from planexs.core.toResolvedConfig import toResolvedConfig
from planexs.domain.Outcome import Outcome
from planexs.domain.Problem import Problem
from planexs.goals.AbstractGoal import AbstractGoal
from planexs.goals.goalRegistry import goalRegistry
from planexs.goals.registerGoal import registerGoal


class TestRegistering:
    def test_the_entry_holds_everything_the_decorator_was_given(
        self, isolatedGoalKind: KindTable
    ) -> None:
        goal = defineFake(
            "optimize_like_v1",
            needs=frozenset({"objectives:multi"}),
            supports=frozenset({"goal:optimize", "objectives:single"}),
            description="An optimize-like goal for tests.",
        )

        entry = goalRegistry().lookup("optimize_like_v1")

        assert entry.name == "optimize_like_v1"
        assert entry.description == "An optimize-like goal for tests."
        assert entry.needs == {"objectives:multi"}
        assert entry.supports == {"goal:optimize", "objectives:single"}
        assert entry.config_schema.model_fields["type"].default == "optimize_like_v1"
        assert entry.builder == goal.fromConfig

    def test_the_class_is_returned_unchanged(self, isolatedGoalKind: KindTable) -> None:
        goal = defineFake()

        assert issubclass(goal, AbstractGoal)
        assert goal.__name__ == "FakeGoal"

    def test_a_goal_that_declares_nothing_is_legal(self, isolatedGoalKind: KindTable) -> None:
        defineFake(supports=frozenset())

        entry = goalRegistry().lookup("fake_goal_v1")

        assert entry.needs == frozenset()
        assert entry.supports == frozenset()

    def test_several_goals_live_side_by_side(self, isolatedGoalKind: KindTable) -> None:
        defineFake("a_v1")
        defineFake("b_v1")

        assert goalRegistry().names() == ["a_v1", "b_v1"]

    def test_registering_builds_nothing(self, isolatedGoalKind: KindTable) -> None:
        # A builder is free of side effects at import and registration time (specification 6).
        calls: list[str] = []

        @registerGoal("lazy_v1", config=fakeConfig("lazy_v1"), description="Lazy.")
        class Lazy(AbstractGoal[Any]):
            def __init__(self, config: Any) -> None:
                calls.append("init")
                super().__init__(config)

            def validateAgainst(self, problem: Problem) -> None:
                calls.append("validate")

        assert calls == []

    def test_the_registry_entry_passes_the_contract_of_specification_6(
        self, isolatedGoalKind: KindTable
    ) -> None:
        defineFake("goal_v1", supports=frozenset({"goal:optimize"}))
        entry = goalRegistry().lookup("goal_v1")
        vocabulary = isolatedGoalKind.vocabulary

        assert goalRegistry().isValidName(entry.name)
        assert issubclass(entry.config_schema, ComponentConfig)
        assert all(
            field.description and field.description.strip()
            for field in entry.config_schema.model_fields.values()
        )
        assert all(tag in vocabulary for tag in entry.needs | entry.supports)
        assert callable(entry.builder)
        assert entry.description.splitlines() == [entry.description]


class TestFromAConfigurationToAGoal:
    """The path that a study will take: a mapping, the registry, a goal, a problem."""

    def test_a_mapping_resolves_to_the_entry_and_a_validated_configuration(
        self, isolatedGoalKind: KindTable
    ) -> None:
        defineFake("goal_v1")

        resolved = resolveComponent(
            goalRegistry(), {"type": "goal_v1", "outcomes": ["yield"], "constrained": True}
        )

        assert resolved.entry.name == "goal_v1"
        assert resolved.config.model_dump() == {
            "type": "goal_v1",
            "outcomes": ["yield"],
            "constrained": True,
        }

    def test_the_entry_builder_makes_the_goal_and_the_goal_validates_proposes_and_answers(
        self, isolatedGoalKind: KindTable
    ) -> None:
        defineFake("goal_v1")
        resolved = resolveComponent(goalRegistry(), {"type": "goal_v1", "outcomes": ["yield"]})

        goal = resolved.entry.builder(resolved.config)

        assert isinstance(goal, AbstractGoal)
        assert goal.config is resolved.config
        assert goal.validateAgainst(makeProblem()) is None
        assert [recipe["n_points"] for recipe in goal.recipes(makeProblem(2))] == [6, 1]
        assert goal.defaultAnalyses() == [{"type": "fake_analysis_v1"}]

    def test_the_same_configuration_is_judged_against_each_problem(
        self, isolatedGoalKind: KindTable
    ) -> None:
        defineFake("goal_v1")
        resolved = resolveComponent(
            goalRegistry(), {"type": "goal_v1", "outcomes": ["profit"]}
        )
        goal = resolved.entry.builder(resolved.config)
        fits = Problem(
            search_space=makeProblem().search_space,
            outcomes=[Outcome(name="profit", type="continuous")],
        )

        assert goal.validateAgainst(fits) is None
        with pytest.raises(ConfigError, match="'profit'"):
            goal.validateAgainst(makeProblem())

    def test_the_resolved_configuration_has_every_default_written_out(
        self, isolatedGoalKind: KindTable
    ) -> None:
        defineFake("goal_v1")

        resolved = resolveComponent(goalRegistry(), {"type": "goal_v1", "outcomes": ["yield"]})

        assert toResolvedConfig(resolved.config) == {
            "type": "goal_v1",
            "outcomes": ["yield"],
            "constrained": False,
        }

    def test_an_unknown_field_is_refused_with_its_name(
        self, isolatedGoalKind: KindTable
    ) -> None:
        defineFake("goal_v1")

        with pytest.raises(ConfigError, match="constrain") as caught:
            resolveComponent(
                goalRegistry(), {"type": "goal_v1", "outcomes": ["yield"], "constrain": True}
            )

        assert "goal 'goal_v1'" in caught.value.message

    def test_a_required_field_that_is_missing_is_named(
        self, isolatedGoalKind: KindTable
    ) -> None:
        defineFake("goal_v1")

        with pytest.raises(ConfigError, match="outcomes") as caught:
            resolveComponent(goalRegistry(), {"type": "goal_v1"})

        assert caught.value.field_path == "outcomes"

    def test_an_unknown_goal_lists_the_registered_ones(
        self, isolatedGoalKind: KindTable
    ) -> None:
        defineFake("a_v1")
        defineFake("b_v1")

        with pytest.raises(ConfigError, match="Registered names: a_v1, b_v1") as caught:
            resolveComponent(goalRegistry(), {"type": "c_v1", "outcomes": ["yield"]})

        assert caught.value.field_path == "type"


class TestARegistrationThatIsRefused:
    def assertNothingWasRegistered(self, table: KindTable) -> None:
        assert table.registry("goal").names() == []

    @pytest.mark.parametrize(
        "name", ["Upper", "1st_v1", "with space", "trailing_", "", "a__b"]
    )
    def test_an_invalid_name(self, isolatedGoalKind: KindTable, name: str) -> None:
        with pytest.raises(RegistryError, match="not a valid component name"):
            defineFake(name)

        self.assertNothingWasRegistered(isolatedGoalKind)

    def test_a_name_that_is_taken_is_never_replaced(self, isolatedGoalKind: KindTable) -> None:
        defineFake("goal_v1", description="The first one.")

        with pytest.raises(RegistryError, match="already registered") as caught:
            defineFake("goal_v1", description="The second one.")

        assert "permanent identifier" in str(caught.value)
        assert goalRegistry().lookup("goal_v1").description == "The first one."

    def test_a_class_that_is_not_a_goal(self, isolatedGoalKind: KindTable) -> None:
        class NotAGoal:
            """Not a subclass of the contract."""

        with pytest.raises(RegistryError, match="not a subclass of AbstractGoal") as caught:
            registerGoal("plain_v1", config=fakeConfig("plain_v1"), description="Plain.")(
                NotAGoal  # type: ignore[type-var]
            )

        assert caught.value.component == "goal 'plain_v1'"
        self.assertNothingWasRegistered(isolatedGoalKind)

    def test_a_schema_whose_type_is_not_pinned_to_the_name(
        self, isolatedGoalKind: KindTable
    ) -> None:
        class OtherConfig(ComponentConfig):
            """A schema pinned to another name."""

            type: Literal["other_v1"] = Field(default="other_v1", description="The name.")

        with pytest.raises(RegistryError, match="pinned to the registered name") as caught:

            @registerGoal("goal_v1", config=OtherConfig, description="Mismatch.")
            class Mismatch(AbstractGoal[OtherConfig]):
                def validateAgainst(self, problem: Problem) -> None:
                    return None

        assert caught.value.field_path == "type"
        self.assertNothingWasRegistered(isolatedGoalKind)

    @pytest.mark.parametrize("description", ["", "   ", "Two\nlines."])
    def test_a_description_that_is_not_one_line(
        self, isolatedGoalKind: KindTable, description: str
    ) -> None:
        with pytest.raises(RegistryError, match="description"):
            defineFake(description=description)

        self.assertNothingWasRegistered(isolatedGoalKind)

    def test_a_needs_that_is_a_single_string_and_not_a_collection(
        self, isolatedGoalKind: KindTable
    ) -> None:
        with pytest.raises(RegistryError, match="needs"):
            defineFake(needs="goal:optimize")  # type: ignore[arg-type]

        self.assertNothingWasRegistered(isolatedGoalKind)


class TestTheCompatibilityCheck:
    """A goal is a participant: its tags are what the compatibility check reads (5.12)."""

    def build(self, name: str, **fields: Any) -> AbstractGoal[Any]:
        entry = goalRegistry().lookup(name)
        return entry.builder(entry.config_schema(**fields))

    def acquisition(self, *needs: str) -> DeclaredParticipant:
        return DeclaredParticipant(label="acquisition 'toy_v1'", needs=frozenset(needs))

    def test_a_goal_is_a_participant_with_a_label_and_tags(
        self, isolatedGoalKind: KindTable
    ) -> None:
        defineFake("goal_v1", needs=frozenset({"objectives:multi"}))

        goal = self.build("goal_v1", outcomes=["yield"])

        assert goal.label == "goal 'goal_v1'"
        assert goal.needs == {"objectives:multi"}
        assert goal.supports == {"goal:optimize"}

    def test_a_need_that_the_goal_supports_is_met(self, isolatedGoalKind: KindTable) -> None:
        defineFake("goal_v1", supports=frozenset({"goal:optimize"}))
        goal = self.build("goal_v1", outcomes=["yield"])

        checkCompatibility([goal, self.acquisition("goal:optimize")], isolatedGoalKind)

    def test_a_need_that_no_participant_supports_names_the_goals_that_would(
        self, isolatedGoalKind: KindTable
    ) -> None:
        defineFake("explore_like_v1", supports=frozenset({"goal:explore"}))
        defineFake("optimize_like_v1", supports=frozenset({"goal:optimize"}))
        goal = self.build("explore_like_v1", outcomes=["yield"])

        with pytest.raises(CompatibilityError) as caught:
            checkCompatibility([goal, self.acquisition("goal:optimize")], isolatedGoalKind)

        message = caught.value.message
        assert "Need 'goal:optimize'" in message
        assert "needed by: acquisition 'toy_v1'" in message
        assert "supported by: goal 'optimize_like_v1'" in message
        assert "explore_like_v1" not in message.split("supported by:")[1]

    def test_the_needs_of_a_goal_are_met_by_another_participant(
        self, isolatedGoalKind: KindTable
    ) -> None:
        defineFake("goal_v1", needs=frozenset({"objectives:multi"}), supports=frozenset())
        goal = self.build("goal_v1", outcomes=["yield"])

        with pytest.raises(CompatibilityError, match="Need 'objectives:multi'"):
            checkCompatibility([goal], isolatedGoalKind)

        provider = DeclaredParticipant(
            label="surrogate", supports=frozenset({"objectives:multi"})
        )
        checkCompatibility([goal, provider], isolatedGoalKind)

    def test_the_tags_narrowed_to_a_configuration_decide_the_outcome_of_the_check(
        self, isolatedGoalKind: KindTable
    ) -> None:
        # The entry supports both objectives tags, but an instance with one outcome supports
        # only `objectives:single`: an acquisition that needs `objectives:multi` fails, and the
        # message still names the goal that could be configured to meet it.
        isolatedGoalKind.vocabulary.register("toy:constrained", "The goal has constraints.")
        defineRefining()
        one = self.build("refining_v1", outcomes=["yield"])
        several = self.build("refining_v1", outcomes=["yield", "cost"])

        checkCompatibility([one, self.acquisition("objectives:single")], isolatedGoalKind)
        checkCompatibility([several, self.acquisition("objectives:multi")], isolatedGoalKind)
        with pytest.raises(CompatibilityError) as caught:
            checkCompatibility([one, self.acquisition("objectives:multi")], isolatedGoalKind)

        assert "supported by: goal 'refining_v1'" in caught.value.message

    def test_a_need_that_depends_on_the_configuration_appears_only_when_it_applies(
        self, isolatedGoalKind: KindTable
    ) -> None:
        isolatedGoalKind.vocabulary.register("toy:constrained", "The goal has constraints.")
        defineRefining()
        free = self.build("refining_v1", outcomes=["yield"])
        constrained = self.build("refining_v1", outcomes=["yield"], constrained=True)

        checkCompatibility([free], isolatedGoalKind)
        with pytest.raises(CompatibilityError, match="needed by: goal 'refining_v1'"):
            checkCompatibility([constrained], isolatedGoalKind)

    def test_a_goal_whose_tag_is_not_in_the_vocabulary_is_reported_as_a_typo(
        self, isolatedGoalKind: KindTable
    ) -> None:
        # The check reads the tags of the goal object, and a typo there is not an unmet need
        # that nobody could ever meet: it is an error that names the participant.
        defineFake("goal_v1")
        goal = self.build("goal_v1", outcomes=["yield"])

        class Typo(type(goal)):  # type: ignore[misc]
            @property
            def supports(self) -> frozenset[str]:
                return frozenset({"goal:optimise"})

        with pytest.raises(RegistryError, match="goal:optimise") as caught:
            checkCompatibility([Typo(goal.config)], isolatedGoalKind)

        assert caught.value.component == "goal 'goal_v1'"
        assert caught.value.field_path == "supports"
