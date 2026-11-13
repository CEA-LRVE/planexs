"""Unit tests for `AbstractGoal`, the contract of the `goal` kind (roadmap task P4-02,
specification 5.7, 6, ADR-0013).

What is tested: the contract itself (abstract, configured once), the capability tags a goal
reports for its own configuration, the recipes and the default analyses, and the two checks
that every goal makes against a problem, in `requireOutcome`.
"""

from __future__ import annotations

from typing import Any, Literal

import pytest
from goal_fakes import BINARY, CONTINUOUS, defineFake, defineRefining, fakeConfig, makeProblem
from pydantic import Field

import planexs.goals  # noqa: F401
from planexs.core.ComponentConfig import ComponentConfig
from planexs.core.ConfigError import ConfigError
from planexs.core.KindTable import KindTable
from planexs.core.PlanexsError import PlanexsError
from planexs.core.RegistryError import RegistryError
from planexs.domain.Problem import Problem
from planexs.goals.AbstractGoal import AbstractGoal
from planexs.goals.goalRegistry import goalRegistry
from planexs.goals.registerGoal import registerGoal


class MinimalConfig(ComponentConfig):
    """A schema for a goal that is built directly, without being registered."""

    type: Literal["minimal_v1"] = Field(default="minimal_v1", description="The name.")
    outcome: str = Field(default="yield", description="The name of the outcome.")


class MinimalGoal(AbstractGoal[MinimalConfig]):
    """The least a goal has to define: what it checks against a problem."""

    def validateAgainst(self, problem: Problem) -> None:
        """Checks that the outcome exists."""
        self.requireOutcome(problem, self.config.outcome, field_path="outcome")


class TestTheContract:
    def test_the_abstract_class_cannot_be_instantiated(self) -> None:
        with pytest.raises(TypeError, match="abstract"):
            AbstractGoal(MinimalConfig())  # type: ignore[abstract]

    def test_a_subclass_without_validate_against_cannot_be_instantiated(self) -> None:
        class NoValidation(AbstractGoal[Any]):
            """Forgot `validateAgainst`."""

        with pytest.raises(TypeError, match="validateAgainst"):
            NoValidation(MinimalConfig())  # type: ignore[abstract]

    def test_validating_is_the_only_thing_a_goal_has_to_define(self) -> None:
        assert isinstance(MinimalGoal(MinimalConfig()), AbstractGoal)

    def test_validate_against_returns_nothing_and_raises_for_a_goal_that_does_not_fit(
        self,
    ) -> None:
        goal = MinimalGoal(MinimalConfig(outcome="yield"))

        assert goal.validateAgainst(makeProblem()) is None
        with pytest.raises(ConfigError):
            MinimalGoal(MinimalConfig(outcome="nothing")).validateAgainst(makeProblem())


class TestTheConfiguration:
    def test_a_goal_keeps_the_configuration_it_was_made_with(self) -> None:
        config = MinimalConfig(outcome="cost")

        assert MinimalGoal(config).config is config

    def test_the_configuration_is_read_only(self) -> None:
        goal = MinimalGoal(MinimalConfig())

        with pytest.raises(AttributeError):
            goal.config = MinimalConfig(outcome="cost")  # type: ignore[misc]

    def test_the_label_says_who_the_goal_is_as_the_compatibility_check_names_it(self) -> None:
        assert MinimalGoal(MinimalConfig()).label == "goal 'minimal_v1'"

    def test_from_config_makes_the_class_it_is_called_on(self) -> None:
        config = MinimalConfig()

        goal = MinimalGoal.fromConfig(config)

        assert type(goal) is MinimalGoal
        assert goal.config is config

    def test_from_config_makes_the_subclass_and_not_the_base(self) -> None:
        class Sub(MinimalGoal):
            """A subclass of a goal."""

        assert type(Sub.fromConfig(MinimalConfig())) is Sub

    def test_the_configuration_is_not_modified_by_what_the_goal_does(self) -> None:
        config = MinimalConfig(outcome="cost")
        before = config.model_dump()
        goal = MinimalGoal.fromConfig(config)

        goal.validateAgainst(makeProblem())
        goal.recipes(makeProblem())
        goal.defaultAnalyses()

        assert config.model_dump() == before


class TestTheCapabilityTags:
    """A goal is a participant of the compatibility check: a label, needs and supports."""

    def test_by_default_a_goal_reports_the_tags_its_entry_declares(
        self, isolatedGoalKind: KindTable
    ) -> None:
        defineFake(
            "plain_v1",
            needs=frozenset({"objectives:multi"}),
            supports=frozenset({"goal:optimize", "objectives:single"}),
        )
        entry = goalRegistry().lookup("plain_v1")

        goal = entry.builder(entry.config_schema(outcomes=["yield"]))

        assert goal.needs == {"objectives:multi"}
        assert goal.supports == {"goal:optimize", "objectives:single"}
        assert isinstance(goal.needs, frozenset)
        assert isinstance(goal.supports, frozenset)

    def test_a_goal_that_declares_nothing_reports_nothing(
        self, isolatedGoalKind: KindTable
    ) -> None:
        defineFake("bare_v1", supports=frozenset())
        entry = goalRegistry().lookup("bare_v1")

        goal = entry.builder(entry.config_schema(outcomes=["yield"]))

        assert goal.needs == frozenset()
        assert goal.supports == frozenset()

    def test_a_goal_can_narrow_the_tags_of_its_entry_to_its_configuration(
        self, isolatedGoalKind: KindTable
    ) -> None:
        isolatedGoalKind.vocabulary.register("toy:constrained", "The goal has constraints.")
        defineRefining()
        entry = goalRegistry().lookup("refining_v1")
        schema = entry.config_schema

        one = entry.builder(schema(outcomes=["yield"]))
        several = entry.builder(schema(outcomes=["yield", "cost"], constrained=True))

        # The entry declares the union of what any configuration can state ...
        assert entry.supports == {"goal:optimize", "objectives:single", "objectives:multi"}
        assert entry.needs == {"toy:constrained"}
        # ... and each goal reports what holds for its own.
        assert one.supports == {"goal:optimize", "objectives:single"}
        assert one.needs == frozenset()
        assert several.supports == {"goal:optimize", "objectives:multi"}
        assert several.needs == {"toy:constrained"}

    def test_a_goal_that_is_not_registered_cannot_say_what_its_entry_declares(self) -> None:
        # Built by hand, under a name nobody registered. The default tags are those of the
        # registry entry, and there is no entry: the error is the registry's, naming the goal.
        goal = MinimalGoal(MinimalConfig())

        with pytest.raises(RegistryError, match="Unknown goal 'minimal_v1'"):
            goal.needs  # noqa: B018
        with pytest.raises(RegistryError, match="Unknown goal 'minimal_v1'"):
            goal.supports  # noqa: B018


class TestRecipesAndAnalyses:
    def test_by_default_a_goal_proposes_no_recipe_and_no_analysis(self) -> None:
        goal = MinimalGoal(MinimalConfig())

        assert goal.recipes(makeProblem()) == []
        assert goal.defaultAnalyses() == []

    def test_each_call_returns_new_lists(self) -> None:
        goal = MinimalGoal(MinimalConfig())
        first = goal.recipes(makeProblem())
        first.append({"type": "oops"})
        analyses = goal.defaultAnalyses()
        analyses.append({"type": "oops"})

        assert goal.recipes(makeProblem()) == []
        assert goal.defaultAnalyses() == []

    def test_a_goal_proposes_recipes_in_order_of_preference_as_plain_mappings(
        self, isolatedGoalKind: KindTable
    ) -> None:
        defineFake("plain_v1")
        entry = goalRegistry().lookup("plain_v1")
        goal = entry.builder(entry.config_schema(outcomes=["yield"]))

        recipes = goal.recipes(makeProblem())

        assert [recipe["type"] for recipe in recipes] == ["fake_policy_v1", "fake_policy_v1"]
        assert recipes[0] != recipes[1]
        assert all(isinstance(recipe, dict) for recipe in recipes)

    def test_a_recipe_can_depend_on_the_size_of_the_problem(
        self, isolatedGoalKind: KindTable
    ) -> None:
        # The reason `recipes` is given the problem: "initial points per dimension".
        defineFake("plain_v1")
        entry = goalRegistry().lookup("plain_v1")
        goal = entry.builder(entry.config_schema(outcomes=["yield"]))

        small = goal.recipes(makeProblem(n_parameters=2))
        large = goal.recipes(makeProblem(n_parameters=5))

        assert small[0]["n_points"] == 6
        assert large[0]["n_points"] == 15

    def test_a_goal_names_the_analyses_that_answer_it(
        self, isolatedGoalKind: KindTable
    ) -> None:
        defineFake("plain_v1")
        entry = goalRegistry().lookup("plain_v1")
        goal = entry.builder(entry.config_schema(outcomes=["yield"]))

        assert goal.defaultAnalyses() == [{"type": "fake_analysis_v1"}]

    def test_a_goal_that_overrides_neither_still_registers_and_resolves(
        self, isolatedGoalKind: KindTable
    ) -> None:
        # A recipe is not part of the contract's abstract surface: roadmap task P4-03 says
        # "Recipes are added in P6-07", so a goal exists first and proposes later.
        @registerGoal("later_v1", config=fakeConfig("later_v1"), description="Recipes later.")
        class Later(MinimalGoal):
            """A goal with no recipe yet."""

        entry = goalRegistry().lookup("later_v1")
        goal = entry.builder(entry.config_schema(outcomes=["yield"]))

        assert goal.recipes(makeProblem()) == []


class TestRequireOutcome:
    """The two checks that every goal makes against a problem: the outcome exists, and it is of
    a kind the goal can use.
    """

    def goal(self) -> AbstractGoal[Any]:
        return MinimalGoal(MinimalConfig())

    def test_it_returns_the_outcome_of_the_problem(self) -> None:
        problem = makeProblem()

        outcome = self.goal().requireOutcome(problem, "cost", field_path="outcome")

        assert outcome is problem.outcomes[1]
        assert outcome.name == "cost"

    def test_any_kind_is_accepted_when_none_is_asked_for(self) -> None:
        problem = makeProblem()

        assert self.goal().requireOutcome(problem, BINARY, field_path="o").type == "binary"
        assert (
            self.goal().requireOutcome(problem, "yield", field_path="o").type == "continuous"
        )

    def test_a_missing_outcome_names_the_goal_the_field_and_the_outcomes_there_are(
        self,
    ) -> None:
        with pytest.raises(ConfigError) as caught:
            self.goal().requireOutcome(makeProblem(), "yeild", field_path="outcome")

        error = caught.value
        assert error.component == "goal 'minimal_v1'"
        assert error.field_path == "outcome"
        assert "The goal 'minimal_v1' names the outcome 'yeild'" in error.message
        assert "which the problem does not have" in error.message
        for name in (*CONTINUOUS, BINARY):
            assert repr(name) in error.message
        assert error.suggestion
        assert isinstance(error, PlanexsError)

    def test_the_path_of_a_field_in_a_list_is_kept_as_given(self) -> None:
        with pytest.raises(ConfigError) as caught:
            self.goal().requireOutcome(makeProblem(), "nope", field_path="outcomes[2]")

        assert caught.value.field_path == "outcomes[2]"

    def test_an_outcome_of_a_kind_the_goal_cannot_use_is_refused(self) -> None:
        with pytest.raises(ConfigError) as caught:
            self.goal().requireOutcome(
                makeProblem(), BINARY, field_path="outcome", kinds=("continuous",)
            )

        error = caught.value
        assert error.component == "goal 'minimal_v1'"
        assert error.field_path == "outcome"
        assert "cannot use the outcome 'phase'" in error.message
        assert "it is of the kind 'binary'" in error.message
        assert "one of the kinds: continuous" in error.message
        assert error.suggestion

    @pytest.mark.parametrize("kinds", [("continuous",), ["continuous"], {"continuous"}])
    def test_the_kinds_may_be_any_collection_of_names(self, kinds: Any) -> None:
        outcome = self.goal().requireOutcome(
            makeProblem(), "yield", field_path="o", kinds=kinds
        )

        assert outcome.name == "yield"

    def test_several_kinds_are_listed_in_sorted_order_and_all_of_them_are_accepted(
        self,
    ) -> None:
        problem = makeProblem()
        goal = self.goal()

        assert goal.requireOutcome(
            problem, BINARY, field_path="o", kinds=("continuous", "binary")
        )
        with pytest.raises(ConfigError, match="one of the kinds: count, ordinal"):
            goal.requireOutcome(problem, "yield", field_path="o", kinds=("ordinal", "count"))

    def test_the_existence_is_judged_before_the_kind(self) -> None:
        with pytest.raises(ConfigError, match="does not have"):
            self.goal().requireOutcome(
                makeProblem(), "nope", field_path="o", kinds=("binary",)
            )

    def test_an_empty_collection_of_kinds_accepts_none(self) -> None:
        with pytest.raises(ConfigError, match="one of the kinds: none"):
            self.goal().requireOutcome(makeProblem(), "yield", field_path="o", kinds=())

    def test_it_does_not_modify_the_problem(self) -> None:
        problem = makeProblem()
        before = problem.model_dump_json()

        self.goal().requireOutcome(problem, "yield", field_path="o", kinds=("continuous",))

        assert problem.model_dump_json() == before


class TestValidationAgainstAProblem:
    """What a goal's `validateAgainst` does, with the fake that stands for P4-03 to P4-05."""

    def build(self, table: KindTable, **fields: Any) -> AbstractGoal[Any]:
        defineFake("plain_v1")
        entry = goalRegistry().lookup("plain_v1")
        return entry.builder(entry.config_schema(**fields))

    def test_a_goal_that_fits_the_problem_is_accepted(
        self, isolatedGoalKind: KindTable
    ) -> None:
        goal = self.build(isolatedGoalKind, outcomes=["yield", "cost"])

        assert goal.validateAgainst(makeProblem()) is None

    def test_an_outcome_the_problem_does_not_have_is_refused_with_its_position(
        self, isolatedGoalKind: KindTable
    ) -> None:
        goal = self.build(isolatedGoalKind, outcomes=["yield", "profit"])

        with pytest.raises(ConfigError) as caught:
            goal.validateAgainst(makeProblem())

        assert caught.value.field_path == "outcomes[1]"
        assert "'profit'" in caught.value.message
        assert caught.value.component == "goal 'plain_v1'"

    def test_an_outcome_of_an_unsuitable_kind_is_refused_with_its_position(
        self, isolatedGoalKind: KindTable
    ) -> None:
        goal = self.build(isolatedGoalKind, outcomes=["cost", BINARY])

        with pytest.raises(ConfigError) as caught:
            goal.validateAgainst(makeProblem())

        assert caught.value.field_path == "outcomes[1]"
        assert "kind 'binary'" in caught.value.message

    def test_the_first_fault_is_the_one_reported(self, isolatedGoalKind: KindTable) -> None:
        goal = self.build(isolatedGoalKind, outcomes=["profit", BINARY])

        with pytest.raises(ConfigError) as caught:
            goal.validateAgainst(makeProblem())

        assert caught.value.field_path == "outcomes[0]"
