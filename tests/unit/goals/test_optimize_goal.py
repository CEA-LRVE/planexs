"""Unit tests for the `optimize_v1` goal (roadmap task P4-03, specification 5.7, FR2, FR15).

What is tested: the registry entry and what makes it (the population rule, the tags), the goal
for one, several and weighted outcomes, the capability tags it reports in each case and how the
compatibility check uses them, the validation against a problem (missing outcome, unsuitable
kind, constraints on outcomes the goal does not optimize), the accepted outcome kinds being a
setting that a plugin's kind can use, and the round trip of the configuration through the
registry, JSON and YAML. The checks that a schema makes alone are in
`test_optimize_goal_config.py`, and the comparison with `LegacyObjective` is in
`test_optimize_goal_legacy_parity.py`.
"""

from __future__ import annotations

import json
import subprocess
import sys
from typing import Any, Literal

import pytest
from goal_fakes import BINARY, makeProblem
from pydantic import Field

import planexs.goals  # noqa: F401  importing the package registers `optimize_v1`
from planexs import Real, SearchSpace
from planexs.core.CapabilityVocabulary import CapabilityVocabulary
from planexs.core.checkCompatibility import checkCompatibility
from planexs.core.CompatibilityError import CompatibilityError
from planexs.core.ComponentConfig import ComponentConfig
from planexs.core.ConfigError import ConfigError
from planexs.core.configFiles import dumpConfig, parseConfig
from planexs.core.DeclaredParticipant import DeclaredParticipant
from planexs.core.KindTable import KindTable
from planexs.core.RegistryError import RegistryError
from planexs.core.resolveComponent import resolveComponent
from planexs.core.toResolvedConfig import toResolvedConfig
from planexs.core.validateConfig import validateConfig
from planexs.domain.Outcome import Outcome
from planexs.domain.outcome_kinds.OutcomeKind import OutcomeKind
from planexs.domain.outcome_kinds.OutcomeValueError import OutcomeValueError
from planexs.domain.outcome_kinds.registerOutcomeKind import registerOutcomeKind
from planexs.domain.Problem import Problem
from planexs.goals.AbstractGoal import AbstractGoal
from planexs.goals.goalRegistry import goalRegistry
from planexs.goals.OptimizeGoal import OptimizeGoal
from planexs.goals.OptimizeGoalConfig import OptimizeGoalConfig
from planexs.goals.OutcomeConstraint import OutcomeConstraint

NEEDS_CONSTRAINTS = "outcome_constraints:supported"


def makeGoal(**fields: Any) -> OptimizeGoal:
    """An `optimize_v1` goal built from the fields of its configuration."""
    return OptimizeGoal(OptimizeGoalConfig(**fields))


def limit(
    outcome: str = "cost", comparator: Literal["<=", ">="] = "<=", bound: float = 5.0
) -> OutcomeConstraint:
    """One `OutcomeConstraint`."""
    return OutcomeConstraint(outcome=outcome, comparator=comparator, bound=bound)


class TestTheRegisteredEntry:
    def test_the_goal_is_registered_under_its_permanent_name(self) -> None:
        assert "optimize_v1" in goalRegistry()

    def test_the_entry_holds_the_schema_the_builder_and_a_one_line_description(self) -> None:
        entry = goalRegistry().lookup("optimize_v1")

        assert entry.config_schema is OptimizeGoalConfig
        assert entry.builder(
            OptimizeGoalConfig(outcomes={"yield": "maximize"})
        ).config.type == ("optimize_v1")
        assert entry.description.splitlines() == [entry.description]

    def test_the_entry_declares_the_union_of_what_any_configuration_can_state(self) -> None:
        entry = goalRegistry().lookup("optimize_v1")

        assert entry.supports == {"goal:optimize", "objectives:single", "objectives:multi"}
        assert entry.needs == {NEEDS_CONSTRAINTS}

    def test_the_builder_makes_an_optimize_goal_from_a_mapping(self) -> None:
        resolved = resolveComponent(
            goalRegistry(), {"type": "optimize_v1", "outcomes": {"yield": "maximize"}}
        )

        goal = resolved.entry.builder(resolved.config)

        assert isinstance(goal, OptimizeGoal)
        assert isinstance(goal, AbstractGoal)
        assert goal.config is resolved.config

    def test_an_unknown_goal_lists_this_one_among_the_registered_names(self) -> None:
        with pytest.raises(ConfigError, match="optimize_v1"):
            resolveComponent(goalRegistry(), {"type": "optimise_v1", "outcomes": {}})

    def test_a_fresh_interpreter_has_the_goal_as_soon_as_the_package_is_imported(self) -> None:
        # The population rule (specification 5.12): the package `__init__` imports the module.
        program = (
            "import planexs.goals\n"
            "from planexs.goals.goalRegistry import goalRegistry\n"
            "print('optimize_v1' in goalRegistry())\n"
        )

        result = subprocess.run(
            [sys.executable, "-c", program], capture_output=True, text=True, check=False
        )

        assert result.returncode == 0, result.stderr
        assert result.stdout.split() == ["True"]


class TestTheTagThatTheGoalIntroduces:
    def test_it_is_in_the_vocabulary_with_a_one_line_description(self) -> None:
        vocabulary = CapabilityVocabulary.default()

        assert NEEDS_CONSTRAINTS in vocabulary
        description = vocabulary.describe(NEEDS_CONSTRAINTS)
        assert description.strip()
        assert description.splitlines() == [description]
        assert description.endswith(".")

    def test_it_cannot_be_registered_a_second_time(self) -> None:
        # The acquisition layer (roadmap task P5-01) must not register it again.
        with pytest.raises(RegistryError, match="already registered"):
            CapabilityVocabulary.default().register(NEEDS_CONSTRAINTS, "Another meaning.")


class TestOneSeveralAndWeightedOutcomes:
    def test_one_outcome_to_minimize(self) -> None:
        goal = makeGoal(outcomes={"cost": "minimize"})

        goal.validateAgainst(makeProblem())

        assert goal.config.outcomes == {"cost": "minimize"}
        assert goal.config.weights is None
        assert goal.config.outcome_constraints == []

    def test_one_outcome_to_maximize(self) -> None:
        goal = makeGoal(outcomes={"yield": "maximize"})

        goal.validateAgainst(makeProblem())

        assert goal.config.outcomes == {"yield": "maximize"}

    def test_several_outcomes_each_with_its_own_direction(self) -> None:
        goal = makeGoal(outcomes={"yield": "maximize", "cost": "minimize"})

        goal.validateAgainst(makeProblem())

        assert goal.config.outcomes == {"yield": "maximize", "cost": "minimize"}
        assert goal.config.weights is None

    def test_several_outcomes_with_weights_keyed_by_outcome(self) -> None:
        goal = makeGoal(
            outcomes={"yield": "maximize", "cost": "minimize"},
            weights={"yield": 2.0, "cost": 0.5},
        )

        goal.validateAgainst(makeProblem())

        assert goal.config.weights == {"yield": 2.0, "cost": 0.5}

    def test_weights_do_not_depend_on_the_order_they_are_written_in(self) -> None:
        in_order = makeGoal(
            outcomes={"yield": "maximize", "cost": "minimize"},
            weights={"yield": 2.0, "cost": 0.5},
        )
        reversed_weights = makeGoal(
            outcomes={"yield": "maximize", "cost": "minimize"},
            weights={"cost": 0.5, "yield": 2.0},
        )

        assert in_order.config == reversed_weights.config

    def test_an_integer_weight_is_read_as_a_float(self) -> None:
        goal = makeGoal(
            outcomes={"yield": "maximize", "cost": "minimize"}, weights={"yield": 2, "cost": 1}
        )

        assert goal.config.weights == {"yield": 2.0, "cost": 1.0}
        assert all(
            isinstance(weight, float) for weight in (goal.config.weights or {}).values()
        )

    def test_the_problem_may_have_more_outcomes_than_the_goal_optimizes(self) -> None:
        # `makeProblem` has three outcomes; the goal names one of them.
        makeGoal(outcomes={"yield": "maximize"}).validateAgainst(makeProblem())

    def test_the_order_of_the_problem_outcomes_does_not_matter(self) -> None:
        problem = Problem(
            search_space=SearchSpace([Real("x", 0.0, 1.0)]),
            outcomes=[
                Outcome(name="cost", type="continuous"),
                Outcome(name="yield", type="continuous"),
            ],
        )

        makeGoal(outcomes={"yield": "maximize", "cost": "minimize"}).validateAgainst(problem)


class TestTheTagsInEachCase:
    @pytest.mark.parametrize(
        ("fields", "supports"),
        [
            pytest.param(
                {"outcomes": {"yield": "maximize"}},
                {"goal:optimize", "objectives:single"},
                id="one-outcome",
            ),
            pytest.param(
                {"outcomes": {"yield": "maximize"}, "weights": {"yield": 3.0}},
                {"goal:optimize", "objectives:single"},
                id="one-outcome-with-a-weight",
            ),
            pytest.param(
                {"outcomes": {"yield": "maximize", "cost": "minimize"}},
                {"goal:optimize", "objectives:multi"},
                id="several-outcomes-no-weights",
            ),
            pytest.param(
                {
                    "outcomes": {"yield": "maximize", "cost": "minimize"},
                    "weights": {"yield": 1.0, "cost": 1.0},
                },
                {"goal:optimize", "objectives:single"},
                id="several-outcomes-with-weights",
            ),
        ],
    )
    def test_the_goal_supports_the_objectives_tag_that_fits_its_configuration(
        self, fields: dict[str, Any], supports: set[str]
    ) -> None:
        assert makeGoal(**fields).supports == supports

    @pytest.mark.parametrize(
        "fields",
        [
            pytest.param({"outcomes": {"yield": "maximize"}}, id="one-outcome"),
            pytest.param(
                {"outcomes": {"yield": "maximize", "cost": "minimize"}}, id="several-outcomes"
            ),
            pytest.param(
                {
                    "outcomes": {"yield": "maximize", "cost": "minimize"},
                    "weights": {"yield": 1.0, "cost": 1.0},
                },
                id="weighted",
            ),
        ],
    )
    def test_it_never_supports_both_objectives_tags(self, fields: dict[str, Any]) -> None:
        supports = makeGoal(**fields).supports

        assert {"objectives:single", "objectives:multi"} & supports in (
            {"objectives:single"},
            {"objectives:multi"},
        )

    def test_without_outcome_constraints_it_needs_nothing(self) -> None:
        assert makeGoal(outcomes={"yield": "maximize"}).needs == frozenset()
        assert makeGoal(outcomes={"yield": "maximize"}, outcome_constraints=[]).needs == (
            frozenset()
        )

    def test_with_an_outcome_constraint_it_needs_an_acquisition_that_enforces_it(self) -> None:
        goal = makeGoal(outcomes={"yield": "maximize"}, outcome_constraints=[limit()])

        assert goal.needs == {NEEDS_CONSTRAINTS}

    def test_several_constraints_need_the_tag_once(self) -> None:
        goal = makeGoal(
            outcomes={"yield": "maximize"},
            outcome_constraints=[limit(bound=5.0), limit(comparator=">=", bound=1.0)],
        )

        assert goal.needs == {NEEDS_CONSTRAINTS}

    def test_constraints_do_not_change_what_the_goal_supports(self) -> None:
        plain = makeGoal(outcomes={"yield": "maximize"})
        constrained = makeGoal(outcomes={"yield": "maximize"}, outcome_constraints=[limit()])

        assert plain.supports == constrained.supports

    @pytest.mark.parametrize(
        "fields",
        [
            {"outcomes": {"yield": "maximize"}},
            {"outcomes": {"yield": "maximize", "cost": "minimize"}},
            {"outcomes": {"yield": "maximize"}, "outcome_constraints": [limit()]},
        ],
    )
    def test_the_tags_of_an_instance_are_among_the_ones_its_entry_declares(
        self, fields: dict[str, Any]
    ) -> None:
        # Truthfulness (specification 10.6): an instance narrows its entry's tags, it never
        # adds one, and every tag is in the vocabulary.
        goal = makeGoal(**fields)
        entry = goalRegistry().lookup("optimize_v1")

        assert goal.supports <= entry.supports
        assert goal.needs <= entry.needs
        for tag in goal.supports | goal.needs:
            assert tag in CapabilityVocabulary.default()

    def test_the_label_names_the_goal_for_the_error_messages(self) -> None:
        assert makeGoal(outcomes={"yield": "maximize"}).label == "goal 'optimize_v1'"


class TestTheCompatibilityCheck:
    """The goal is a participant: the tags it reports are what the check works with."""

    def test_a_goal_with_constraints_is_refused_when_no_participant_enforces_them(
        self,
    ) -> None:
        goal = makeGoal(outcomes={"yield": "maximize"}, outcome_constraints=[limit()])

        with pytest.raises(CompatibilityError) as caught:
            checkCompatibility([goal])

        assert "outcome_constraints:supported" in caught.value.message
        assert "goal 'optimize_v1'" in caught.value.message

    def test_a_goal_with_constraints_is_accepted_when_a_participant_supports_them(
        self,
    ) -> None:
        goal = makeGoal(outcomes={"yield": "maximize"}, outcome_constraints=[limit()])
        acquisition = DeclaredParticipant(
            "acquisition 'toy'", supports=frozenset({NEEDS_CONSTRAINTS})
        )

        checkCompatibility([goal, acquisition])

    def test_a_goal_without_constraints_needs_no_such_participant(self) -> None:
        checkCompatibility([makeGoal(outcomes={"yield": "maximize"})])

    def test_an_acquisition_that_needs_a_single_objective_is_refused_for_a_pareto_goal(
        self,
    ) -> None:
        acquisition = DeclaredParticipant(
            "acquisition 'toy'", needs=frozenset({"objectives:single"})
        )

        checkCompatibility([makeGoal(outcomes={"yield": "maximize"}), acquisition])
        with pytest.raises(CompatibilityError, match="objectives:single"):
            checkCompatibility(
                [makeGoal(outcomes={"yield": "maximize", "cost": "minimize"}), acquisition]
            )

    def test_weights_make_a_pareto_goal_acceptable_to_a_single_objective_acquisition(
        self,
    ) -> None:
        acquisition = DeclaredParticipant(
            "acquisition 'toy'", needs=frozenset({"objectives:single"})
        )
        weighted = makeGoal(
            outcomes={"yield": "maximize", "cost": "minimize"},
            weights={"yield": 1.0, "cost": 1.0},
        )

        checkCompatibility([weighted, acquisition])

    def test_an_acquisition_that_needs_a_pareto_goal_is_refused_when_weights_are_given(
        self,
    ) -> None:
        acquisition = DeclaredParticipant(
            "acquisition 'toy'", needs=frozenset({"objectives:multi"})
        )
        weighted = makeGoal(
            outcomes={"yield": "maximize", "cost": "minimize"},
            weights={"yield": 1.0, "cost": 1.0},
        )

        with pytest.raises(CompatibilityError, match="objectives:multi"):
            checkCompatibility([weighted, acquisition])


class TestValidationAgainstAProblem:
    def test_an_outcome_that_the_problem_does_not_have_is_rejected(self) -> None:
        goal = makeGoal(outcomes={"yield": "maximize", "nothing": "minimize"})

        with pytest.raises(ConfigError, match="'nothing'") as caught:
            goal.validateAgainst(makeProblem())

        assert caught.value.field_path == "outcomes.nothing"
        assert caught.value.component == "goal 'optimize_v1'"
        assert "'yield'" in caught.value.message  # the outcomes of the problem are listed

    def test_a_binary_outcome_has_no_direction_and_is_rejected_by_default(self) -> None:
        goal = makeGoal(outcomes={BINARY: "maximize"})

        with pytest.raises(ConfigError, match="'binary'") as caught:
            goal.validateAgainst(makeProblem())

        assert caught.value.field_path == f"outcomes.{BINARY}"
        assert "continuous" in caught.value.message  # what the goal does take

    def test_a_constraint_on_an_unknown_outcome_is_rejected(self) -> None:
        goal = makeGoal(
            outcomes={"yield": "maximize"},
            outcome_constraints=[limit("cost"), limit("throughput", ">=", 100.0)],
        )

        with pytest.raises(ConfigError, match="'throughput'") as caught:
            goal.validateAgainst(makeProblem())

        assert caught.value.field_path == "outcome_constraints[1].outcome"
        assert caught.value.component == "goal 'optimize_v1'"

    def test_a_constraint_on_a_binary_outcome_is_rejected_by_default(self) -> None:
        goal = makeGoal(outcomes={"yield": "maximize"}, outcome_constraints=[limit(BINARY)])

        with pytest.raises(ConfigError, match="'binary'") as caught:
            goal.validateAgainst(makeProblem())

        assert caught.value.field_path == "outcome_constraints[0].outcome"

    def test_a_constraint_may_limit_an_outcome_that_the_goal_does_not_optimize(self) -> None:
        # The case the legacy layer could not state: `cost` is measured and limited, and only
        # `yield` is maximized.
        goal = makeGoal(
            outcomes={"yield": "maximize"}, outcome_constraints=[limit("cost", "<=", 3.0)]
        )

        goal.validateAgainst(makeProblem())

    def test_a_constraint_may_limit_an_outcome_that_the_goal_also_optimizes(self) -> None:
        goal = makeGoal(
            outcomes={"yield": "maximize"}, outcome_constraints=[limit("yield", ">=", 0.5)]
        )

        goal.validateAgainst(makeProblem())

    def test_two_constraints_on_one_outcome_state_a_band(self) -> None:
        goal = makeGoal(
            outcomes={"yield": "maximize"},
            outcome_constraints=[limit("cost", ">=", 1.0), limit("cost", "<=", 4.0)],
        )

        goal.validateAgainst(makeProblem())

        assert [c.comparator for c in goal.config.outcome_constraints] == [">=", "<="]

    def test_the_first_fault_found_is_the_one_reported_outcomes_before_constraints(
        self,
    ) -> None:
        goal = makeGoal(
            outcomes={"missing": "maximize"}, outcome_constraints=[limit("also_missing")]
        )

        with pytest.raises(ConfigError) as caught:
            goal.validateAgainst(makeProblem())

        assert caught.value.field_path == "outcomes.missing"

    def test_validating_returns_nothing_and_changes_nothing(self) -> None:
        goal = makeGoal(
            outcomes={"yield": "maximize"}, outcome_constraints=[limit("cost", "<=", 3.0)]
        )
        before = toResolvedConfig(goal.config)

        assert goal.validateAgainst(makeProblem()) is None
        assert toResolvedConfig(goal.config) == before


class TestTheAcceptedOutcomeKindsAreASetting:
    """`outcome_kinds` is a policy value, reachable through the configuration (specification
    3.4): a kind that a plugin registers can be optimized by naming it, with no change to
    planexs (specification 1.3).
    """

    def test_the_default_accepts_only_the_continuous_kind(self) -> None:
        assert OptimizeGoalConfig(outcomes={"yield": "maximize"}).outcome_kinds == (
            "continuous",
        )

    def test_a_binary_outcome_is_accepted_when_the_configuration_names_the_kind(self) -> None:
        goal = makeGoal(outcomes={BINARY: "maximize"}, outcome_kinds=("continuous", "binary"))

        goal.validateAgainst(makeProblem())

    def test_a_kind_that_a_plugin_registers_is_accepted_when_the_configuration_names_it(
        self, isolatedGoalKind: KindTable
    ) -> None:
        class CountKindConfig(ComponentConfig):
            """No setting."""

            type: Literal["count"] = Field(default="count", description="The name, count.")

        @registerOutcomeKind("count", config=CountKindConfig, description="A whole count.")
        class CountKind(OutcomeKind):
            """A non-negative whole number, as a plugin could register it."""

            capability_tag = "outcome_kind:count"
            allows_noise = False

            def validateValue(self, value: object, outcome: str) -> float:
                """Accepts a non-negative int."""
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise OutcomeValueError(f"{value!r} is not a count of {outcome!r}.")
                return float(value)

        problem = Problem(
            search_space=SearchSpace([Real("x", 0.0, 1.0)]),
            outcomes=[Outcome(name="defects", type="count")],
        )

        with pytest.raises(ConfigError, match="'count'"):
            makeGoal(outcomes={"defects": "minimize"}).validateAgainst(problem)
        makeGoal(
            outcomes={"defects": "minimize"}, outcome_kinds=("continuous", "count")
        ).validateAgainst(problem)
        assert CountKind.capability_tag == "outcome_kind:count"


#: A goal configuration that uses every field, as a mapping.
ROUND_TRIP_CONFIG: dict[str, Any] = {
    "outcomes": {"yield": "maximize", "cost": "minimize"},
    "weights": {"yield": 2.0, "cost": 0.5},
    "outcome_constraints": [
        {"outcome": "throughput", "comparator": ">=", "bound": 100.0},
        {"outcome": "latency", "comparator": "<=", "bound": 50.0},
    ],
    "outcome_kinds": ["continuous"],
}


class TestTheRoundTrip:
    def config(self) -> OptimizeGoalConfig:
        return OptimizeGoalConfig(**ROUND_TRIP_CONFIG)

    def test_the_resolved_configuration_survives_json(self) -> None:
        config = self.config()
        resolved = toResolvedConfig(config)

        reloaded = validateConfig(OptimizeGoalConfig, json.loads(json.dumps(resolved)))

        assert reloaded == config
        assert toResolvedConfig(reloaded) == resolved

    def test_the_resolved_configuration_writes_every_default_out(self) -> None:
        resolved = toResolvedConfig(OptimizeGoalConfig(outcomes={"yield": "maximize"}))

        assert resolved == {
            "type": "optimize_v1",
            "outcomes": {"yield": "maximize"},
            "weights": None,
            "outcome_constraints": [],
            "outcome_kinds": ["continuous"],
        }

    def test_the_outcomes_keep_the_order_they_were_written_in(self) -> None:
        resolved = toResolvedConfig(
            OptimizeGoalConfig(outcomes={"b": "maximize", "a": "minimize"})
        )

        assert list(resolved["outcomes"]) == ["b", "a"]

    def test_the_registry_resolves_a_mapping_to_the_same_configuration(self) -> None:
        resolved = resolveComponent(
            goalRegistry(), {"type": "optimize_v1", **ROUND_TRIP_CONFIG}
        )

        assert resolved.config == self.config()
        assert toResolvedConfig(resolved.config) == toResolvedConfig(self.config())

    def test_a_configuration_with_defaults_left_out_reads_the_same_as_the_resolved_one(
        self,
    ) -> None:
        short = resolveComponent(
            goalRegistry(), {"type": "optimize_v1", "outcomes": {"yield": "maximize"}}
        )
        full = resolveComponent(goalRegistry(), toResolvedConfig(short.config))

        assert short.config == full.config

    def test_yaml_and_json_mean_the_same(self) -> None:
        text = (
            "type: optimize_v1\n"
            "outcomes: {yield: maximize, cost: minimize}\n"
            "weights: {yield: 2, cost: 0.5}\n"
            "outcome_constraints:\n"
            "  - {outcome: throughput, comparator: '>=', bound: 100}\n"
            "  - {outcome: latency, comparator: '<=', bound: 50}\n"
        )

        from_yaml = resolveComponent(goalRegistry(), parseConfig(text, "yaml"))
        written = dumpConfig(toResolvedConfig(from_yaml.config), "json")
        from_json = resolveComponent(goalRegistry(), parseConfig(written, "json"))

        assert from_yaml.config == self.config()
        assert from_json.config == from_yaml.config

    def test_the_resolved_yaml_reads_back_to_the_same_goal(self) -> None:
        resolved = toResolvedConfig(self.config())

        reloaded = resolveComponent(
            goalRegistry(), parseConfig(dumpConfig(resolved, "yaml"), "yaml")
        )

        assert reloaded.config == self.config()
