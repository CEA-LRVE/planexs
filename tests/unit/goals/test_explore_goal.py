"""Unit tests for the `explore_v1` goal (roadmap task P4-05, specification 5.7, FR21).

What is tested: the schema (what it accepts, rejects and defaults), the outcomes that are
learned by default (all of them) and when they are named, the names that are refused, the
registration and the capability tags, and the round trip of the stored configuration.
"""

from __future__ import annotations

import json
from typing import Any, ClassVar, Literal

import pytest
from goal_fakes import BINARY, CONTINUOUS, makeProblem
from pydantic import Field

import planexs.goals  # noqa: F401
from planexs import Real, SearchSpace
from planexs.core.ComponentConfig import ComponentConfig
from planexs.core.ConfigError import ConfigError
from planexs.core.KindTable import KindTable
from planexs.core.resolveComponent import resolveComponent
from planexs.core.toResolvedConfig import toResolvedConfig
from planexs.core.validateConfig import validateConfig
from planexs.domain.Outcome import Outcome
from planexs.domain.outcome_kinds.OutcomeKind import OutcomeKind
from planexs.domain.outcome_kinds.registerOutcomeKind import registerOutcomeKind
from planexs.domain.Problem import Problem
from planexs.goals.AbstractGoal import AbstractGoal
from planexs.goals.ExploreGoal import ExploreGoal
from planexs.goals.ExploreGoalConfig import ExploreGoalConfig
from planexs.goals.goalRegistry import goalRegistry

#: The outcomes of `makeProblem`, in the order of the problem.
ALL_OUTCOMES = (*CONTINUOUS, BINARY)


def makeGoal(**fields: Any) -> ExploreGoal:
    """An explore goal built the way the registry builds one, from a validated config."""
    return ExploreGoal.fromConfig(ExploreGoalConfig(**fields))


class TestTheSchema:
    def test_every_field_has_a_default_so_an_empty_configuration_is_a_goal(self) -> None:
        config = validateConfig(ExploreGoalConfig, {})

        assert config.type == "explore_v1"
        assert config.outcomes is None

    def test_the_type_is_pinned_to_the_registered_name(self) -> None:
        with pytest.raises(ConfigError) as caught:
            validateConfig(ExploreGoalConfig, {"type": "optimize_v1"})

        assert caught.value.field_path == "type"

    def test_an_unknown_field_is_refused(self) -> None:
        with pytest.raises(ConfigError) as caught:
            validateConfig(ExploreGoalConfig, {"outcome": "yield"})

        assert caught.value.field_path == "outcome"

    def test_a_list_of_names_is_kept_in_the_order_given(self) -> None:
        assert makeGoal(outcomes=["phase", "yield"]).config.outcomes == ["phase", "yield"]

    def test_an_empty_list_is_refused_and_is_not_read_as_all(self) -> None:
        with pytest.raises(ConfigError) as caught:
            validateConfig(ExploreGoalConfig, {"outcomes": []})

        assert caught.value.field_path == "outcomes"

    def test_a_single_name_is_not_a_list(self) -> None:
        with pytest.raises(ConfigError) as caught:
            validateConfig(ExploreGoalConfig, {"outcomes": "yield"})

        assert caught.value.field_path == "outcomes"

    def test_an_empty_name_is_refused(self) -> None:
        with pytest.raises(ConfigError, match="empty name") as caught:
            validateConfig(ExploreGoalConfig, {"outcomes": ["yield", ""]})

        assert caught.value.field_path == "outcomes"

    def test_a_name_that_is_not_a_string_is_refused(self) -> None:
        with pytest.raises(ConfigError) as caught:
            validateConfig(ExploreGoalConfig, {"outcomes": ["yield", 3]})

        assert caught.value.field_path.startswith("outcomes")

    def test_an_outcome_named_twice_is_refused(self) -> None:
        with pytest.raises(ConfigError, match="once") as caught:
            validateConfig(ExploreGoalConfig, {"outcomes": ["yield", "cost", "yield"]})

        assert caught.value.field_path == "outcomes"
        assert "yield" in str(caught.value)

    def test_every_field_is_described(self) -> None:
        for name, field in ExploreGoalConfig.model_fields.items():
            assert field.description, name
            assert field.description.strip() == field.description, name


class TestTheDefaultOutcomes:
    def test_all_the_outcomes_of_the_problem_are_learned_by_default(self) -> None:
        goal = makeGoal()

        goal.validateAgainst(makeProblem())

        assert goal.effectiveOutcomes(makeProblem()) == ALL_OUTCOMES

    def test_the_default_is_in_the_order_of_the_problem(self) -> None:
        problem = Problem(
            search_space=SearchSpace([Real("x0", 0.0, 1.0)]),
            outcomes=[
                Outcome(name="b", type="continuous"),
                Outcome(name="a", type="continuous"),
                Outcome(name="c", type="binary"),
            ],
        )

        assert makeGoal().effectiveOutcomes(problem) == ("b", "a", "c")

    def test_the_default_follows_the_problem_and_is_not_stored_in_the_goal(self) -> None:
        goal = makeGoal()
        one = Problem(
            search_space=SearchSpace([Real("x0", 0.0, 1.0)]),
            outcomes=[Outcome(name="loss", type="continuous")],
        )

        assert goal.effectiveOutcomes(one) == ("loss",)
        assert goal.effectiveOutcomes(makeProblem()) == ALL_OUTCOMES
        assert goal.config.outcomes is None

    def test_a_binary_outcome_is_learned_like_any_other(self) -> None:
        assert BINARY in makeGoal().effectiveOutcomes(makeProblem())

    @pytest.mark.parametrize(
        "names",
        [["cost"], ["phase"], ["cost", "yield"], ["phase", "yield"], list(ALL_OUTCOMES)],
    )
    def test_named_outcomes_are_learned_in_the_order_given(self, names: list[str]) -> None:
        goal = makeGoal(outcomes=names)

        goal.validateAgainst(makeProblem())

        assert goal.effectiveOutcomes(makeProblem()) == tuple(names)

    def test_naming_every_outcome_is_the_same_set_as_the_default(self) -> None:
        named = makeGoal(outcomes=list(ALL_OUTCOMES)).effectiveOutcomes(makeProblem())

        assert named == makeGoal().effectiveOutcomes(makeProblem())

    def test_it_returns_an_immutable_value_so_a_caller_cannot_change_the_goal(self) -> None:
        goal = makeGoal(outcomes=["yield"])

        assert isinstance(goal.effectiveOutcomes(makeProblem()), tuple)
        assert goal.config.outcomes == ["yield"]


class TestAnOutcomeThatIsRefused:
    def test_an_unknown_outcome_is_rejected(self) -> None:
        goal = makeGoal(outcomes=["nope"])

        with pytest.raises(ConfigError, match="does not have") as caught:
            goal.validateAgainst(makeProblem())

        assert caught.value.field_path == "outcomes[0]"
        assert caught.value.component == "goal 'explore_v1'"
        assert "'nope'" in str(caught.value)

    def test_the_error_points_at_the_position_of_the_name(self) -> None:
        goal = makeGoal(outcomes=["yield", "cost", "nope"])

        with pytest.raises(ConfigError) as caught:
            goal.validateAgainst(makeProblem())

        assert caught.value.field_path == "outcomes[2]"

    def test_the_error_lists_the_outcomes_of_the_problem(self) -> None:
        goal = makeGoal(outcomes=["nope"])

        with pytest.raises(ConfigError) as caught:
            goal.validateAgainst(makeProblem())

        for name in ALL_OUTCOMES:
            assert repr(name) in str(caught.value)
        assert caught.value.suggestion

    def test_the_first_unknown_name_is_the_one_reported(self) -> None:
        goal = makeGoal(outcomes=["yield", "nope", "other"])

        with pytest.raises(ConfigError) as caught:
            goal.validateAgainst(makeProblem())

        assert caught.value.field_path == "outcomes[1]"
        assert "'nope'" in str(caught.value)

    def test_names_are_case_sensitive(self) -> None:
        with pytest.raises(ConfigError, match="does not have") as caught:
            makeGoal(outcomes=["Yield"]).validateAgainst(makeProblem())

        assert caught.value.field_path == "outcomes[0]"

    def test_the_effective_outcomes_are_not_given_for_an_unknown_name_either(self) -> None:
        with pytest.raises(ConfigError, match="does not have"):
            makeGoal(outcomes=["nope"]).effectiveOutcomes(makeProblem())

    def test_an_outcome_of_a_kind_the_core_has_never_heard_of_can_be_learned(
        self, isolatedGoalKind: KindTable
    ) -> None:
        class OrdinalToyConfig(ComponentConfig):
            type: Literal["ordinal_toy"] = Field(
                default="ordinal_toy", description="Always so."
            )

        class OrdinalToy(OutcomeKind):
            capability_tag: ClassVar[str] = "outcome_kind:ordinal_toy"
            allows_noise: ClassVar[bool] = False

            def validateValue(self, value: object, outcome: str) -> float:
                return float(value)  # type: ignore[arg-type]

        registerOutcomeKind("ordinal_toy", config=OrdinalToyConfig, description="A toy.")(
            OrdinalToy
        )
        problem = Problem(
            search_space=SearchSpace([Real("x0", 0.0, 1.0)]),
            outcomes=[Outcome(name="grade", type="ordinal_toy")],
        )

        assert makeGoal().effectiveOutcomes(problem) == ("grade",)
        assert makeGoal(outcomes=["grade"]).effectiveOutcomes(problem) == ("grade",)

    def test_validating_does_not_modify_the_problem(self) -> None:
        problem = makeProblem()
        before = problem.model_dump_json()

        makeGoal(outcomes=["yield"]).validateAgainst(problem)
        makeGoal().validateAgainst(problem)

        assert problem.model_dump_json() == before


class TestTheGoalIsRegistered:
    def test_it_is_a_registered_goal_under_its_permanent_name(self) -> None:
        entry = goalRegistry().lookup("explore_v1")

        assert entry.name == "explore_v1"
        assert entry.config_schema is ExploreGoalConfig
        assert entry.description.strip()

    def test_it_supports_the_explore_tag_and_nothing_else(self) -> None:
        entry = goalRegistry().lookup("explore_v1")

        assert entry.supports == frozenset({"goal:explore"})
        assert entry.needs == frozenset()

    @pytest.mark.parametrize("fields", [{}, {"outcomes": ["yield"]}, {"outcomes": ["phase"]}])
    def test_an_instance_reports_the_tags_of_its_entry_whatever_it_learns(
        self, fields: dict[str, Any]
    ) -> None:
        goal = makeGoal(**fields)

        assert goal.supports == frozenset({"goal:explore"})
        assert goal.needs == frozenset()
        assert "goal:optimize" not in goal.supports
        assert "goal:level_set" not in goal.supports

    def test_it_has_no_direction_and_so_no_objective_tags(self) -> None:
        supported = makeGoal().supports

        assert "objectives:single" not in supported
        assert "objectives:multi" not in supported

    def test_the_registry_builds_an_explore_goal(self) -> None:
        resolved = resolveComponent(
            goalRegistry(), {"type": "explore_v1", "outcomes": ["cost", "phase"]}
        )

        goal = resolved.entry.builder(resolved.config)

        assert isinstance(goal, ExploreGoal)
        assert isinstance(goal, AbstractGoal)
        assert goal.effectiveOutcomes(makeProblem()) == ("cost", "phase")

    def test_the_registry_builds_one_from_nothing_but_its_type(self) -> None:
        resolved = resolveComponent(goalRegistry(), {"type": "explore_v1"})

        goal = resolved.entry.builder(resolved.config)

        assert goal.effectiveOutcomes(makeProblem()) == ALL_OUTCOMES

    def test_its_label_names_the_registered_type(self) -> None:
        assert makeGoal().label == "goal 'explore_v1'"

    def test_it_proposes_no_recipe_and_no_analysis_yet(self) -> None:
        # Recipes come with roadmap task P6-08.
        goal = makeGoal()

        assert goal.recipes(makeProblem()) == []
        assert goal.defaultAnalyses() == []

    def test_it_keeps_the_configuration_it_was_made_with(self) -> None:
        config = ExploreGoalConfig(outcomes=["yield"])

        assert ExploreGoal.fromConfig(config).config is config

    def test_it_is_registered_next_to_the_level_set_goal(self) -> None:
        assert {"explore_v1", "level_set_v1"} <= set(goalRegistry().names())


class TestTheRoundTrip:
    CONFIGURATIONS: ClassVar[list[dict[str, Any]]] = [
        {},
        {"outcomes": ["yield"]},
        {"outcomes": ["phase", "cost"]},
        {"outcomes": ["yield", "cost", "phase"]},
    ]

    @pytest.mark.parametrize("fields", CONFIGURATIONS)
    def test_the_resolved_configuration_reads_back_equal(self, fields: dict[str, Any]) -> None:
        config = ExploreGoalConfig(**fields)

        again = validateConfig(ExploreGoalConfig, toResolvedConfig(config))

        assert again == config

    @pytest.mark.parametrize("fields", CONFIGURATIONS)
    def test_it_goes_through_json_unchanged(self, fields: dict[str, Any]) -> None:
        config = ExploreGoalConfig(**fields)
        resolved = toResolvedConfig(config)

        assert json.loads(json.dumps(resolved)) == resolved
        assert validateConfig(ExploreGoalConfig, json.loads(json.dumps(resolved))) == config

    @pytest.mark.parametrize("fields", CONFIGURATIONS)
    def test_resolving_the_registry_entry_gives_the_same_dictionary_again(
        self, fields: dict[str, Any]
    ) -> None:
        first = resolveComponent(goalRegistry(), {"type": "explore_v1", **fields})

        second = resolveComponent(goalRegistry(), toResolvedConfig(first.config))

        assert toResolvedConfig(second.config) == toResolvedConfig(first.config)
        assert second.config == first.config

    def test_the_default_is_written_out_as_none_and_not_as_the_outcomes_it_stands_for(
        self,
    ) -> None:
        # The default means "all the outcomes of the problem", whatever they are when the
        # goal is used, so the stored form must not freeze today's outcomes.
        assert toResolvedConfig(ExploreGoalConfig()) == {
            "type": "explore_v1",
            "outcomes": None,
        }

    def test_the_goal_made_from_the_stored_form_behaves_like_the_first(self) -> None:
        problem = makeProblem()
        for fields in self.CONFIGURATIONS:
            first = makeGoal(**fields)
            stored = json.loads(json.dumps(toResolvedConfig(first.config)))
            second = ExploreGoal.fromConfig(validateConfig(ExploreGoalConfig, stored))

            assert second.effectiveOutcomes(problem) == first.effectiveOutcomes(problem)
