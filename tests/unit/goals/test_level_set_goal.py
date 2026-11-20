"""Unit tests for the `level_set_v1` goal (roadmap task P4-04, specification 5.7, FR20).

What is tested: the schema (what it requires, rejects and defaults), the level of a continuous
outcome (required, in the units of the outcome), the level of a binary outcome (a probability,
with a documented default), the outcomes that are refused, the registration and the capability
tags, and the round trip of the stored configuration.
"""

from __future__ import annotations

import json
import math
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
from planexs.goals.goalRegistry import goalRegistry
from planexs.goals.LevelSetGoal import LevelSetGoal
from planexs.goals.LevelSetGoalConfig import LevelSetGoalConfig

#: The documented default level of a binary outcome (see the description of `binary_level`).
DOCUMENTED_BINARY_DEFAULT = 0.5


def makeGoal(**fields: Any) -> LevelSetGoal:
    """A level-set goal built the way the registry builds one: from a validated configuration.

    The fields default to the first continuous outcome of `makeProblem` and no level.
    """
    fields.setdefault("outcome", CONTINUOUS[0])
    return LevelSetGoal.fromConfig(LevelSetGoalConfig(**fields))


class TestTheSchema:
    def test_the_outcome_is_required(self) -> None:
        with pytest.raises(ConfigError) as caught:
            validateConfig(LevelSetGoalConfig, {})

        assert caught.value.field_path == "outcome"

    def test_an_empty_outcome_name_is_refused(self) -> None:
        with pytest.raises(ConfigError) as caught:
            validateConfig(LevelSetGoalConfig, {"outcome": ""})

        assert caught.value.field_path == "outcome"

    def test_the_type_is_pinned_to_the_registered_name(self) -> None:
        assert LevelSetGoalConfig(outcome="yield").type == "level_set_v1"
        with pytest.raises(ConfigError) as caught:
            validateConfig(LevelSetGoalConfig, {"type": "optimize_v1", "outcome": "yield"})

        assert caught.value.field_path == "type"

    def test_an_unknown_field_is_refused(self) -> None:
        with pytest.raises(ConfigError) as caught:
            validateConfig(LevelSetGoalConfig, {"outcome": "yield", "levl": 0.5})

        assert caught.value.field_path == "levl"

    def test_a_goal_has_one_outcome_and_not_several(self) -> None:
        with pytest.raises(ConfigError) as caught:
            validateConfig(LevelSetGoalConfig, {"outcome": ["yield", "cost"], "level": 1.0})

        assert caught.value.field_path == "outcome"

    def test_the_level_has_no_default_value_of_its_own(self) -> None:
        # None means "not given": a continuous outcome needs it, a binary one defaults.
        assert LevelSetGoalConfig(outcome="yield").level is None

    def test_the_binary_default_is_the_documented_one(self) -> None:
        assert LevelSetGoalConfig(outcome="phase").binary_level == DOCUMENTED_BINARY_DEFAULT

    @pytest.mark.parametrize("level", [math.nan, math.inf, -math.inf])
    def test_a_level_that_is_not_finite_is_refused(self, level: float) -> None:
        with pytest.raises(ConfigError) as caught:
            validateConfig(LevelSetGoalConfig, {"outcome": "yield", "level": level})

        assert caught.value.field_path == "level"

    @pytest.mark.parametrize("level", [-3.5, 0, 0.0, 1, 42.5, 1e6])
    def test_any_finite_number_is_a_level(self, level: float) -> None:
        # The schema cannot know the units of the outcome, so it accepts any finite number.
        assert LevelSetGoalConfig(outcome="yield", level=level).level == level

    @pytest.mark.parametrize("level", [0.0, 1.0, -0.1, 1.5, math.nan])
    def test_the_binary_default_must_be_a_probability(self, level: float) -> None:
        with pytest.raises(ConfigError) as caught:
            validateConfig(LevelSetGoalConfig, {"outcome": "phase", "binary_level": level})

        assert caught.value.field_path == "binary_level"

    def test_every_field_is_described(self) -> None:
        for name, field in LevelSetGoalConfig.model_fields.items():
            assert field.description, name
            assert field.description.strip() == field.description, name


class TestTheLevelOfAContinuousOutcome:
    @pytest.mark.parametrize("name", CONTINUOUS)
    def test_it_is_required(self, name: str) -> None:
        goal = makeGoal(outcome=name)

        with pytest.raises(ConfigError, match="needs a level") as caught:
            goal.validateAgainst(makeProblem())

        assert caught.value.field_path == "level"
        assert caught.value.component == "goal 'level_set_v1'"
        assert repr(name) in str(caught.value)
        assert caught.value.suggestion

    def test_the_binary_default_does_not_stand_in_for_it(self) -> None:
        # A probability is not a value in the units of a measured quantity.
        goal = makeGoal(outcome="yield", binary_level=0.25)

        with pytest.raises(ConfigError, match="needs a level") as caught:
            goal.effectiveLevel(makeProblem())

        assert caught.value.field_path == "level"

    @pytest.mark.parametrize("level", [-3.5, 0.0, 0.5, 1.0, 42.5, 1e6])
    def test_it_is_in_the_units_of_the_outcome_and_can_be_any_finite_number(
        self, level: float
    ) -> None:
        goal = makeGoal(outcome="yield", level=level)

        goal.validateAgainst(makeProblem())

        assert goal.effectiveLevel(makeProblem()) == level

    def test_an_integer_level_is_read_as_a_number(self) -> None:
        goal = makeGoal(outcome="cost", level=40)

        assert goal.effectiveLevel(makeProblem()) == 40.0


class TestTheLevelOfABinaryOutcome:
    def test_it_is_defaulted_when_it_is_not_given(self) -> None:
        goal = makeGoal(outcome=BINARY)

        goal.validateAgainst(makeProblem())

        assert goal.effectiveLevel(makeProblem()) == DOCUMENTED_BINARY_DEFAULT

    def test_the_default_is_the_one_of_the_schema_and_not_a_value_of_the_goal(self) -> None:
        default = LevelSetGoalConfig.model_fields["binary_level"].default

        assert makeGoal(outcome=BINARY).effectiveLevel(makeProblem()) == default

    def test_the_default_can_be_changed_in_the_configuration(self) -> None:
        goal = makeGoal(outcome=BINARY, binary_level=0.9)

        assert goal.effectiveLevel(makeProblem()) == 0.9

    @pytest.mark.parametrize("level", [0.01, 0.25, 0.5, 0.99])
    def test_a_level_that_is_given_is_a_probability_and_is_used(self, level: float) -> None:
        goal = makeGoal(outcome=BINARY, level=level)

        goal.validateAgainst(makeProblem())

        assert goal.effectiveLevel(makeProblem()) == level

    def test_a_level_that_is_given_wins_over_the_default(self) -> None:
        goal = makeGoal(outcome=BINARY, level=0.2, binary_level=0.9)

        assert goal.effectiveLevel(makeProblem()) == 0.2

    @pytest.mark.parametrize("level", [0.0, 1.0, -0.5, 1.5, 42.5])
    def test_a_level_that_is_not_a_probability_is_refused(self, level: float) -> None:
        # The same number is a fine level for a continuous outcome: only the kind decides.
        goal = makeGoal(outcome=BINARY, level=level)

        with pytest.raises(ConfigError, match="strictly between 0 and 1") as caught:
            goal.validateAgainst(makeProblem())

        assert caught.value.field_path == "level"
        assert caught.value.component == "goal 'level_set_v1'"
        assert repr(BINARY) in str(caught.value)
        assert caught.value.suggestion

    def test_a_level_outside_the_range_is_a_good_level_for_a_continuous_outcome(self) -> None:
        assert makeGoal(outcome="yield", level=42.5).effectiveLevel(makeProblem()) == 42.5


class TestAnOutcomeThatIsRefused:
    def test_an_unknown_outcome_is_rejected(self) -> None:
        goal = makeGoal(outcome="nope", level=0.5)

        with pytest.raises(ConfigError, match="does not have") as caught:
            goal.validateAgainst(makeProblem())

        assert caught.value.field_path == "outcome"
        assert caught.value.component == "goal 'level_set_v1'"
        assert "'nope'" in str(caught.value)

    def test_the_error_lists_the_outcomes_of_the_problem(self) -> None:
        goal = makeGoal(outcome="nope", level=0.5)

        with pytest.raises(ConfigError) as caught:
            goal.validateAgainst(makeProblem())

        for name in (*CONTINUOUS, BINARY):
            assert repr(name) in str(caught.value)

    def test_an_unknown_outcome_is_rejected_even_when_there_is_no_level(self) -> None:
        # The outcome is judged first: the level cannot be judged without its kind.
        with pytest.raises(ConfigError, match="does not have") as caught:
            makeGoal(outcome="nope").validateAgainst(makeProblem())

        assert caught.value.field_path == "outcome"

    def test_the_effective_level_is_not_given_for_an_unknown_outcome_either(self) -> None:
        with pytest.raises(ConfigError, match="does not have"):
            makeGoal(outcome="nope", level=0.5).effectiveLevel(makeProblem())

    def test_an_outcome_of_a_kind_that_a_level_does_not_apply_to_is_rejected(
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
        goal = makeGoal(outcome="grade", level=2.0)

        with pytest.raises(ConfigError, match="cannot use the outcome 'grade'") as caught:
            goal.validateAgainst(problem)

        assert caught.value.field_path == "outcome"
        assert "binary, continuous" in str(caught.value)

    def test_validating_does_not_modify_the_problem(self) -> None:
        problem = makeProblem()
        before = problem.model_dump_json()

        makeGoal(outcome=BINARY).validateAgainst(problem)

        assert problem.model_dump_json() == before


class TestTheGoalIsRegistered:
    def test_it_is_a_registered_goal_under_its_permanent_name(self) -> None:
        entry = goalRegistry().lookup("level_set_v1")

        assert entry.name == "level_set_v1"
        assert entry.config_schema is LevelSetGoalConfig
        assert entry.description.strip()

    def test_it_supports_the_level_set_tag_and_nothing_else(self) -> None:
        entry = goalRegistry().lookup("level_set_v1")

        assert entry.supports == frozenset({"goal:level_set"})
        assert entry.needs == frozenset()

    def test_an_instance_reports_the_tags_of_its_entry(self) -> None:
        goal = makeGoal(outcome=BINARY)

        assert "goal:level_set" in goal.supports
        assert "goal:optimize" not in goal.supports
        assert "goal:explore" not in goal.supports
        assert goal.needs == frozenset()

    @pytest.mark.parametrize("outcome", [*CONTINUOUS, BINARY])
    def test_the_tags_do_not_depend_on_the_outcome(self, outcome: str) -> None:
        assert makeGoal(outcome=outcome).supports == frozenset({"goal:level_set"})

    def test_the_registry_builds_a_level_set_goal(self) -> None:
        resolved = resolveComponent(
            goalRegistry(), {"type": "level_set_v1", "outcome": BINARY, "level": 0.3}
        )

        goal = resolved.entry.builder(resolved.config)

        assert isinstance(goal, LevelSetGoal)
        assert isinstance(goal, AbstractGoal)
        assert goal.config.outcome == BINARY
        assert goal.effectiveLevel(makeProblem()) == 0.3

    def test_its_label_names_the_registered_type(self) -> None:
        assert makeGoal().label == "goal 'level_set_v1'"

    def test_it_proposes_no_recipe_and_no_analysis_yet(self) -> None:
        # Recipes come with roadmap task P6-08.
        goal = makeGoal(outcome=BINARY)

        assert goal.recipes(makeProblem()) == []
        assert goal.defaultAnalyses() == []

    def test_it_keeps_the_configuration_it_was_made_with(self) -> None:
        config = LevelSetGoalConfig(outcome=BINARY, level=0.4)

        assert LevelSetGoal.fromConfig(config).config is config


class TestTheRoundTrip:
    CONFIGURATIONS: ClassVar[list[dict[str, Any]]] = [
        {"outcome": "yield", "level": 42.5},
        {"outcome": "yield", "level": -3.0},
        {"outcome": "phase"},
        {"outcome": "phase", "level": 0.25},
        {"outcome": "phase", "binary_level": 0.8},
        {"outcome": "phase", "level": 0.25, "binary_level": 0.8},
    ]

    @pytest.mark.parametrize("fields", CONFIGURATIONS)
    def test_the_resolved_configuration_reads_back_equal(self, fields: dict[str, Any]) -> None:
        config = LevelSetGoalConfig(**fields)

        again = validateConfig(LevelSetGoalConfig, toResolvedConfig(config))

        assert again == config

    @pytest.mark.parametrize("fields", CONFIGURATIONS)
    def test_it_goes_through_json_unchanged(self, fields: dict[str, Any]) -> None:
        config = LevelSetGoalConfig(**fields)
        resolved = toResolvedConfig(config)

        assert json.loads(json.dumps(resolved)) == resolved
        assert validateConfig(LevelSetGoalConfig, json.loads(json.dumps(resolved))) == config

    @pytest.mark.parametrize("fields", CONFIGURATIONS)
    def test_resolving_the_registry_entry_gives_the_same_dictionary_again(
        self, fields: dict[str, Any]
    ) -> None:
        first = resolveComponent(goalRegistry(), {"type": "level_set_v1", **fields})

        second = resolveComponent(goalRegistry(), toResolvedConfig(first.config))

        assert toResolvedConfig(second.config) == toResolvedConfig(first.config)
        assert second.config == first.config

    def test_every_field_is_written_out_including_the_defaults(self) -> None:
        resolved = toResolvedConfig(LevelSetGoalConfig(outcome="phase"))

        assert resolved == {
            "type": "level_set_v1",
            "outcome": "phase",
            "level": None,
            "binary_level": DOCUMENTED_BINARY_DEFAULT,
        }

    def test_the_goal_made_from_the_stored_form_behaves_like_the_first(self) -> None:
        problem = makeProblem()
        for fields in self.CONFIGURATIONS:
            first = makeGoal(**fields)
            stored = json.loads(json.dumps(toResolvedConfig(first.config)))
            second = LevelSetGoal.fromConfig(validateConfig(LevelSetGoalConfig, stored))

            assert second.effectiveLevel(problem) == first.effectiveLevel(problem)
