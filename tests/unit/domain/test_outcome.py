"""Unit tests for `Outcome`."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from planexs.core.ConfigError import ConfigError
from planexs.domain.Outcome import Outcome
from planexs.domain.outcome_kinds.BinaryKind import BinaryKind
from planexs.domain.outcome_kinds.ContinuousKind import ContinuousKind
from planexs.domain.outcome_kinds.OutcomeValueError import OutcomeValueError


class TestBuildingAnOutcome:
    def test_it_has_a_name_a_type_and_an_optional_description(self) -> None:
        outcome = Outcome(name="yield", type="continuous", description="Mass yield, in %.")

        assert outcome.name == "yield"
        assert outcome.type == "continuous"
        assert outcome.description == "Mass yield, in %."

    def test_the_description_is_optional(self) -> None:
        assert Outcome(name="phase", type="binary").description is None

    def test_it_is_built_with_keyword_arguments(self) -> None:
        with pytest.raises(TypeError):
            Outcome("yield", "continuous")  # type: ignore[misc]

    def test_it_says_nothing_about_a_direction_or_a_weight(self) -> None:
        # The direction to optimize belongs to the goal (ADR-0012): the same outcome serves
        # any goal.
        with pytest.raises(ValidationError, match="minimize"):
            Outcome(name="yield", type="continuous", minimize=True)  # type: ignore[call-arg]

    def test_an_unknown_field_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="extra"):
            Outcome(name="yield", type="continuous", unit="%")  # type: ignore[call-arg]

    @pytest.mark.parametrize("field", ["name", "type"])
    def test_a_field_that_is_not_a_string_is_refused(self, field: str) -> None:
        fields: dict[str, Any] = {"name": "yield", "type": "continuous"}
        fields[field] = 3

        with pytest.raises(ValidationError, match=field):
            Outcome(**fields)

    @pytest.mark.parametrize("field", ["name", "type"])
    def test_a_required_field_cannot_be_left_out(self, field: str) -> None:
        fields: dict[str, Any] = {"name": "yield", "type": "continuous"}
        del fields[field]

        with pytest.raises(ValidationError, match=field):
            Outcome(**fields)

    def test_it_is_immutable(self) -> None:
        outcome = Outcome(name="yield", type="continuous")

        with pytest.raises(ValidationError, match="frozen"):
            outcome.name = "other"  # type: ignore[misc]

    def test_two_equal_outcomes_are_equal_and_hash_alike(self) -> None:
        first = Outcome(name="yield", type="continuous")
        second = Outcome(name="yield", type="continuous")

        assert first == second
        assert hash(first) == hash(second)
        assert first != Outcome(name="yield", type="binary")
        assert first != Outcome(name="other", type="continuous")


class TestTheTypeIsACheckedKind:
    def test_an_unknown_type_is_an_error_that_lists_the_registered_kinds(self) -> None:
        with pytest.raises(ConfigError) as caught:
            Outcome(name="grade", type="ordinal")

        error = caught.value
        assert "Unknown outcome_kind type 'ordinal'" in error.message
        assert "Registered names: binary, continuous" in error.message
        assert error.field_path == "type"
        assert error.component == "outcome_kind"

    def test_a_type_that_differs_by_case_is_not_the_kind(self) -> None:
        with pytest.raises(ConfigError, match="Unknown outcome_kind type 'Continuous'"):
            Outcome(name="yield", type="Continuous")


class TestTheKindOfAnOutcome:
    def test_a_continuous_outcome_builds_the_continuous_kind(self) -> None:
        assert type(Outcome(name="yield", type="continuous").kind()) is ContinuousKind

    def test_a_binary_outcome_builds_the_binary_kind(self) -> None:
        assert type(Outcome(name="phase", type="binary").kind()) is BinaryKind

    def test_the_noise_rule_is_the_rule_of_the_kind(self) -> None:
        assert Outcome(name="yield", type="continuous").kind().allows_noise is True
        assert Outcome(name="phase", type="binary").kind().allows_noise is False

    def test_the_capability_tag_is_the_one_of_the_kind(self) -> None:
        assert Outcome(name="yield", type="continuous").kind().capability_tag == (
            "outcome_kind:continuous"
        )
        assert Outcome(name="phase", type="binary").kind().capability_tag == (
            "outcome_kind:binary"
        )


class TestValidatingAValue:
    def test_a_continuous_outcome_takes_a_finite_number(self) -> None:
        outcome = Outcome(name="yield", type="continuous")

        assert outcome.validateValue(73) == 73.0
        assert outcome.validateValue(0.25) == 0.25

    def test_a_binary_outcome_takes_a_label(self) -> None:
        outcome = Outcome(name="phase", type="binary")

        assert outcome.validateValue(True) == 1.0
        assert outcome.validateValue(0) == 0.0

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), "high"])
    def test_a_bad_continuous_value_is_refused_and_the_error_names_the_outcome(
        self, value: Any
    ) -> None:
        with pytest.raises(OutcomeValueError) as caught:
            Outcome(name="throughput", type="continuous").validateValue(value)

        assert "'throughput'" in caught.value.message
        assert caught.value.component == "outcome 'throughput'"

    @pytest.mark.parametrize("value", [2, 1.0, "yes", None, float("nan")])
    def test_a_bad_binary_value_is_refused_and_the_error_names_the_outcome(
        self, value: Any
    ) -> None:
        with pytest.raises(OutcomeValueError) as caught:
            Outcome(name="phase", type="binary").validateValue(value)

        assert "'phase'" in caught.value.message
        assert caught.value.component == "outcome 'phase'"

    def test_the_same_value_is_good_for_one_kind_and_bad_for_the_other(self) -> None:
        assert Outcome(name="a", type="continuous").validateValue(2) == 2.0
        with pytest.raises(OutcomeValueError):
            Outcome(name="a", type="binary").validateValue(2)


class TestSerializing:
    def test_to_dict_has_the_name_the_type_and_the_description(self) -> None:
        outcome = Outcome(name="yield", type="continuous", description="Mass yield.")

        assert outcome.toDict() == {
            "name": "yield",
            "type": "continuous",
            "description": "Mass yield.",
        }

    def test_to_dict_always_writes_the_description(self) -> None:
        assert Outcome(name="phase", type="binary").toDict() == {
            "name": "phase",
            "type": "binary",
            "description": None,
        }

    def test_to_dict_is_what_pydantic_dumps(self) -> None:
        outcome = Outcome(name="yield", type="continuous", description="Mass yield.")

        assert outcome.toDict() == outcome.model_dump(mode="json")
