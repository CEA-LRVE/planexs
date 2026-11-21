"""Unit tests for `OptimizeGoalConfig` and `OutcomeConstraint` (roadmap task P4-03,
specification 5.12, 10.8): what a configuration can check without a problem.

Every failure is read through `validateConfig`, which is how a configuration read from a file
reaches the schema, so that what is asserted is what a user sees: a `ConfigError` whose message
lists the problem with its field path.
"""

from __future__ import annotations

import math
from typing import Any

import pytest
from pydantic import ValidationError

import planexs.goals  # noqa: F401  importing the package registers the goal and its tags
from planexs.core.ConfigError import ConfigError
from planexs.core.validateConfig import validateConfig
from planexs.goals.OptimizeGoalConfig import OptimizeGoalConfig
from planexs.goals.OutcomeConstraint import OutcomeConstraint


def read(**fields: Any) -> OptimizeGoalConfig:
    """Reads a mapping through `validateConfig`, as a configuration file is."""
    return validateConfig(OptimizeGoalConfig, {"type": "optimize_v1", **fields})


def fault(**fields: Any) -> ConfigError:
    """The `ConfigError` that reading these fields raises."""
    with pytest.raises(ConfigError) as caught:
        read(**fields)
    return caught.value


TWO = {"yield": "maximize", "cost": "minimize"}


class TestTheOutcomes:
    def test_at_least_one_outcome_is_required(self) -> None:
        error = fault(outcomes={})

        assert error.field_path == "outcomes"
        assert "at least 1" in error.message

    def test_the_outcomes_are_required_and_have_no_default(self) -> None:
        error = fault()

        assert error.field_path == "outcomes"
        assert "Field required" in error.message

    @pytest.mark.parametrize("direction", ["up", "min", "Maximize", "", None, True, 1])
    def test_a_direction_is_one_of_minimize_and_maximize(self, direction: Any) -> None:
        error = fault(outcomes={"yield": direction})

        assert error.field_path == "outcomes.yield"
        assert "'minimize' or 'maximize'" in error.message

    def test_the_direction_that_is_wrong_is_found_among_good_ones(self) -> None:
        assert fault(outcomes={"yield": "maximize", "cost": "down"}).field_path == (
            "outcomes.cost"
        )

    def test_a_list_of_names_is_not_a_mapping(self) -> None:
        assert fault(outcomes=["yield"]).field_path == "outcomes"


class TestTheWeights:
    def test_a_weight_for_each_outcome_is_accepted(self) -> None:
        config = read(outcomes=TWO, weights={"yield": 1.0, "cost": 3.0})

        assert config.weights == {"yield": 1.0, "cost": 3.0}

    def test_fewer_weights_than_outcomes_are_rejected(self) -> None:
        error = fault(outcomes=TWO, weights={"yield": 1.0})

        assert error.field_path == "weights"
        assert "1 weight(s) for 2 outcome(s)" in error.message
        assert "no weight for ['cost']" in error.message
        assert "leave weights out" in error.message  # the way to fix it

    def test_more_weights_than_outcomes_are_rejected(self) -> None:
        error = fault(outcomes={"yield": "maximize"}, weights={"yield": 1.0, "cost": 2.0})

        assert error.field_path == "weights"
        assert "2 weight(s) for 1 outcome(s)" in error.message
        assert "a weight for ['cost'], which is not an outcome of the goal" in error.message

    def test_the_right_number_of_weights_for_the_wrong_outcomes_is_rejected(self) -> None:
        error = fault(outcomes=TWO, weights={"yield": 1.0, "price": 2.0})

        assert error.field_path == "weights"
        assert "2 weight(s) for 2 outcome(s)" in error.message
        assert "no weight for ['cost']" in error.message
        assert "a weight for ['price']" in error.message

    def test_an_empty_weights_mapping_is_a_mismatch_and_not_the_same_as_none(self) -> None:
        assert fault(outcomes=TWO, weights={}).field_path == "weights"

    def test_without_weights_nothing_is_checked(self) -> None:
        assert read(outcomes=TWO).weights is None
        assert read(outcomes=TWO, weights=None).weights is None

    @pytest.mark.parametrize("weight", [0.0, -1.0, -0.001])
    def test_a_weight_is_above_zero_because_the_direction_belongs_to_the_outcome(
        self, weight: float
    ) -> None:
        error = fault(outcomes={"yield": "maximize"}, weights={"yield": weight})

        assert error.field_path == "weights.yield"
        assert "greater than 0" in error.message

    @pytest.mark.parametrize("weight", [math.inf, math.nan])
    def test_a_weight_is_finite(self, weight: float) -> None:
        error = fault(outcomes={"yield": "maximize"}, weights={"yield": weight})

        assert error.field_path == "weights.yield"

    def test_a_weight_that_is_not_a_number_is_rejected(self) -> None:
        assert fault(
            outcomes={"yield": "maximize"}, weights={"yield": "heavy"}
        ).field_path == ("weights.yield")

    def test_a_weight_mismatch_is_not_reported_twice_when_the_outcomes_are_wrong(self) -> None:
        # `outcomes` fails its own validation, and a second error about the weights that
        # depends on it would only repeat the first.
        error = fault(outcomes={}, weights={"yield": 1.0})

        assert error.field_path == "outcomes"
        assert "weight(s)" not in error.message


class TestTheOutcomeConstraints:
    def test_the_default_is_no_constraint(self) -> None:
        assert read(outcomes=TWO).outcome_constraints == []

    def test_a_constraint_is_an_outcome_a_comparator_and_a_bound(self) -> None:
        config = read(
            outcomes={"yield": "maximize"},
            outcome_constraints=[{"outcome": "qps", "comparator": ">=", "bound": 100}],
        )

        assert config.outcome_constraints == [
            OutcomeConstraint(outcome="qps", comparator=">=", bound=100.0)
        ]

    def test_the_default_list_is_not_shared_between_configurations(self) -> None:
        first = read(outcomes=TWO)
        second = read(outcomes=TWO)

        first.outcome_constraints.append(
            OutcomeConstraint(outcome="qps", comparator=">=", bound=1.0)
        )

        assert second.outcome_constraints == []

    def test_a_bound_relative_to_a_baseline_is_not_supported(self) -> None:
        # The legacy layer reserved a `relative` flag: it is out of scope here, and an unknown
        # field is an error and not a silent no-op.
        error = fault(
            outcomes={"yield": "maximize"},
            outcome_constraints=[
                {"outcome": "qps", "comparator": ">=", "bound": 0.95, "relative": True}
            ],
        )

        assert error.field_path == "outcome_constraints[0].relative"
        assert "Unknown field" in error.message

    @pytest.mark.parametrize("comparator", ["=", "==", "<", ">", "!=", "", None])
    def test_a_comparator_is_at_most_or_at_least(self, comparator: Any) -> None:
        error = fault(
            outcomes={"yield": "maximize"},
            outcome_constraints=[{"outcome": "qps", "comparator": comparator, "bound": 1.0}],
        )

        assert error.field_path == "outcome_constraints[0].comparator"
        assert "'<=' or '>='" in error.message

    @pytest.mark.parametrize("bound", [math.inf, -math.inf, math.nan])
    def test_a_bound_is_finite(self, bound: float) -> None:
        error = fault(
            outcomes={"yield": "maximize"},
            outcome_constraints=[{"outcome": "qps", "comparator": ">=", "bound": bound}],
        )

        assert error.field_path == "outcome_constraints[0].bound"

    def test_every_field_of_a_constraint_is_required(self) -> None:
        for missing in ("outcome", "comparator", "bound"):
            constraint = {"outcome": "qps", "comparator": ">=", "bound": 1.0}
            del constraint[missing]

            error = fault(outcomes={"yield": "maximize"}, outcome_constraints=[constraint])

            assert error.field_path == f"outcome_constraints[0].{missing}"

    def test_the_outcome_of_a_constraint_is_not_blank(self) -> None:
        error = fault(
            outcomes={"yield": "maximize"},
            outcome_constraints=[{"outcome": "", "comparator": ">=", "bound": 1.0}],
        )

        assert error.field_path == "outcome_constraints[0].outcome"

    def test_the_second_constraint_that_is_wrong_is_found_by_its_position(self) -> None:
        error = fault(
            outcomes={"yield": "maximize"},
            outcome_constraints=[
                {"outcome": "qps", "comparator": ">=", "bound": 1.0},
                {"outcome": "latency", "comparator": "<=", "bound": "slow"},
            ],
        )

        assert error.field_path == "outcome_constraints[1].bound"


class TestTheAcceptedOutcomeKinds:
    def test_an_unregistered_kind_is_rejected_with_the_registered_ones_listed(self) -> None:
        error = fault(outcomes={"yield": "maximize"}, outcome_kinds=["continous"])

        assert error.field_path == "outcome_kinds"
        assert "['continous']" in error.message
        assert "binary, continuous" in error.message  # a typo shows next to what exists

    def test_at_least_one_kind_is_required(self) -> None:
        error = fault(outcomes={"yield": "maximize"}, outcome_kinds=[])

        assert error.field_path == "outcome_kinds"

    def test_several_registered_kinds_are_accepted(self) -> None:
        config = read(outcomes={"yield": "maximize"}, outcome_kinds=["continuous", "binary"])

        assert config.outcome_kinds == ("continuous", "binary")

    def test_a_json_list_is_read_as_the_tuple_the_schema_holds(self) -> None:
        assert read(
            outcomes={"yield": "maximize"}, outcome_kinds=["continuous"]
        ).outcome_kinds == ("continuous",)


class TestTheSchemaItself:
    def test_an_unknown_field_is_rejected_and_the_fields_are_listed(self) -> None:
        error = fault(outcomes=TWO, direction="maximize")

        assert error.field_path == "direction"
        assert "Unknown field" in error.message
        assert error.suggestion is not None
        for field in ("outcomes", "weights", "outcome_constraints", "outcome_kinds"):
            assert field in error.suggestion

    def test_the_type_is_pinned_to_the_registered_name(self) -> None:
        with pytest.raises(ValidationError):
            OptimizeGoalConfig(type="optimize_v2", outcomes=TWO)  # type: ignore[arg-type]

    def test_the_type_defaults_when_the_configuration_is_built_in_python(self) -> None:
        assert OptimizeGoalConfig(outcomes=TWO).type == "optimize_v1"

    def test_every_field_is_described(self) -> None:
        # The contract of a configuration schema (specification 10.8); the registry contract
        # test checks the same for every registered entry.
        for name, field in OptimizeGoalConfig.model_fields.items():
            assert field.description, name

    def test_a_schema_built_in_python_raises_a_validation_error_with_the_field_path(
        self,
    ) -> None:
        with pytest.raises(ValidationError) as caught:
            OptimizeGoalConfig(outcomes=TWO, weights={"yield": 1.0})

        assert caught.value.errors()[0]["loc"] == ("weights",)


class TestAnOutcomeConstraintOnItsOwn:
    def test_it_is_built_with_keywords_and_stores_what_it_is_given(self) -> None:
        constraint = OutcomeConstraint(outcome="qps", comparator=">=", bound=100)

        assert (constraint.outcome, constraint.comparator, constraint.bound) == (
            "qps",
            ">=",
            100.0,
        )
        assert isinstance(constraint.bound, float)

    def test_it_is_frozen(self) -> None:
        constraint = OutcomeConstraint(outcome="qps", comparator=">=", bound=100.0)

        with pytest.raises(ValidationError):
            constraint.bound = 1.0  # type: ignore[misc]

    def test_equal_constraints_are_equal_and_a_different_bound_is_not(self) -> None:
        one = OutcomeConstraint(outcome="qps", comparator=">=", bound=100.0)

        assert one == OutcomeConstraint(outcome="qps", comparator=">=", bound=100.0)
        assert one != OutcomeConstraint(outcome="qps", comparator=">=", bound=101.0)
        assert one != OutcomeConstraint(outcome="qps", comparator="<=", bound=100.0)

    def test_it_round_trips_through_a_dump(self) -> None:
        constraint = OutcomeConstraint(outcome="qps", comparator="<=", bound=-2.5)

        assert (
            OutcomeConstraint.model_validate(constraint.model_dump(mode="json")) == constraint
        )

    def test_every_field_is_described(self) -> None:
        for name, field in OutcomeConstraint.model_fields.items():
            assert field.description, name
