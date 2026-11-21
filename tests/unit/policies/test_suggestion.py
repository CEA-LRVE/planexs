"""Unit tests for `Suggestion` (roadmap task P6-01, specification 5.9, 6).

What is tested: what a suggestion holds and defaults, that it is immutable and keeps deep
read-only copies, that it is validated as the trial it becomes, the phase tag, and that the
study can turn it into a trial without a further check of its shape.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any

import pytest
from pydantic import ValidationError

from planexs.domain.Trial import Trial
from planexs.policies.Suggestion import Suggestion


class TestWhatItHolds:
    def test_a_parameterization_and_nothing_else_is_a_suggestion(self) -> None:
        suggestion = Suggestion(parameters={"x": 0.5, "n": 3, "kind": "a", "flag": True})

        assert dict(suggestion.parameters) == {"x": 0.5, "n": 3, "kind": "a", "flag": True}
        assert dict(suggestion.metadata) == {}

    def test_the_metadata_is_kept_beside_the_parameters(self) -> None:
        suggestion = Suggestion(
            parameters={"x": 0.5}, metadata={"phase": "model", "acquisition_value": 1.25}
        )

        assert dict(suggestion.metadata) == {"phase": "model", "acquisition_value": 1.25}

    def test_it_is_built_with_keywords_only(self) -> None:
        with pytest.raises(TypeError):
            Suggestion({"x": 0.5})  # type: ignore[misc]

    def test_an_unknown_keyword_is_refused(self) -> None:
        with pytest.raises(TypeError, match="phase"):
            Suggestion(parameters={"x": 0.5}, phase="design")  # type: ignore[call-arg]

    def test_the_parameters_are_required(self) -> None:
        with pytest.raises(TypeError, match="parameters"):
            Suggestion()  # type: ignore[call-arg]

    def test_an_empty_parameterization_is_a_valid_suggestion_of_an_empty_space(self) -> None:
        assert dict(Suggestion(parameters={}).parameters) == {}

    def test_two_suggestions_with_the_same_content_are_equal(self) -> None:
        first = Suggestion(parameters={"x": 0.5}, metadata={"phase": "design"})
        second = Suggestion(parameters={"x": 0.5}, metadata={"phase": "design"})

        assert first == second
        assert first != Suggestion(parameters={"x": 0.6}, metadata={"phase": "design"})
        assert first != Suggestion(parameters={"x": 0.5})


class TestItIsImmutable:
    def test_no_attribute_can_be_assigned(self) -> None:
        suggestion = Suggestion(parameters={"x": 0.5})

        with pytest.raises(dataclasses.FrozenInstanceError):
            suggestion.parameters = {"x": 0.9}  # type: ignore[misc]
        with pytest.raises(dataclasses.FrozenInstanceError):
            suggestion.metadata = {}  # type: ignore[misc]

    def test_no_attribute_can_be_added_or_deleted(self) -> None:
        suggestion = Suggestion(parameters={"x": 0.5})

        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError, TypeError)):
            suggestion.extra = 1  # type: ignore[attr-defined]
        with pytest.raises(dataclasses.FrozenInstanceError):
            del suggestion.parameters  # type: ignore[misc]

    def test_it_has_no_dictionary_to_write_to(self) -> None:
        with pytest.raises(TypeError):
            vars(Suggestion(parameters={"x": 0.5}))

    def test_the_parameters_cannot_be_changed_through_the_suggestion(self) -> None:
        suggestion = Suggestion(parameters={"x": 0.5})

        with pytest.raises(TypeError):
            suggestion.parameters["x"] = 0.9  # type: ignore[index]
        with pytest.raises(TypeError):
            suggestion.parameters["y"] = 0.9  # type: ignore[index]
        assert dict(suggestion.parameters) == {"x": 0.5}

    def test_the_metadata_is_read_only_all_the_way_down(self) -> None:
        suggestion = Suggestion(
            parameters={"x": 0.5}, metadata={"info": {"scores": [1, 2]}, "tags": ["a"]}
        )

        with pytest.raises(TypeError):
            suggestion.metadata["info"]["scores"] = []  # type: ignore[index]
        with pytest.raises(TypeError):
            suggestion.metadata["new"] = 1  # type: ignore[index]
        # A JSON array becomes a tuple: it has no `append`.
        assert not hasattr(suggestion.metadata["tags"], "append")
        assert suggestion.metadata["info"]["scores"] == (1, 2)

    def test_it_keeps_copies_so_the_policy_cannot_change_it_afterwards(self) -> None:
        parameters: dict[str, Any] = {"x": 0.5}
        metadata: dict[str, Any] = {"info": {"scores": [1, 2]}}
        suggestion = Suggestion(parameters=parameters, metadata=metadata)

        parameters["x"] = 0.9
        parameters["y"] = 1.0
        metadata["info"]["scores"].append(3)
        metadata["phase"] = "late"

        assert dict(suggestion.parameters) == {"x": 0.5}
        assert suggestion.metadata["info"]["scores"] == (1, 2)
        assert "phase" not in suggestion.metadata

    def test_a_suggestion_can_be_built_from_the_read_only_mappings_of_another(self) -> None:
        first = Suggestion(parameters={"x": 0.5}, metadata={"info": {"a": [1]}})

        second = Suggestion(parameters=first.parameters, metadata=first.metadata)

        assert second == first


class TestItIsValidatedAsTheTrialItBecomes:
    @pytest.mark.parametrize("value", [0.5, 3, True, "a", -2.5e300, 0, False, ""])
    def test_the_four_types_of_a_trial_value_are_accepted(self, value: Any) -> None:
        assert Suggestion(parameters={"p": value}).parameters["p"] == value

    @pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
    def test_a_number_that_is_not_finite_is_refused(self, value: float) -> None:
        with pytest.raises(ValidationError):
            Suggestion(parameters={"p": value})

    @pytest.mark.parametrize("value", [[1.0], (1.0,), {"a": 1}, None, object()])
    def test_a_value_of_another_type_is_refused(self, value: Any) -> None:
        with pytest.raises(ValidationError, match="parameters"):
            Suggestion(parameters={"p": value})

    def test_the_error_names_the_parameter_at_fault(self) -> None:
        with pytest.raises(ValidationError) as caught:
            Suggestion(parameters={"fine": 1.0, "bad": [1]})

        first = caught.value.errors()[0]
        assert first["loc"][:2] == ("parameters", "bad")

    def test_a_name_that_is_not_a_string_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            Suggestion(parameters={1: 0.5})  # type: ignore[dict-item]

    def test_parameters_that_are_not_a_mapping_are_refused(self) -> None:
        with pytest.raises(ValidationError):
            Suggestion(parameters=[("x", 0.5)])  # type: ignore[arg-type]

    @pytest.mark.parametrize("value", [math.nan, math.inf])
    def test_a_number_that_is_not_finite_in_the_metadata_is_refused(
        self, value: float
    ) -> None:
        with pytest.raises(ValidationError):
            Suggestion(parameters={"x": 0.5}, metadata={"score": value})
        with pytest.raises(ValidationError):
            Suggestion(parameters={"x": 0.5}, metadata={"nested": {"scores": [1.0, value]}})

    def test_metadata_that_is_not_json_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="metadata"):
            Suggestion(parameters={"x": 0.5}, metadata={"model": object()})

    def test_metadata_that_is_not_a_mapping_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            Suggestion(parameters={"x": 0.5}, metadata=["phase"])  # type: ignore[arg-type]


class TestThePhase:
    def test_it_is_the_phase_the_policy_tagged(self) -> None:
        assert Suggestion(parameters={}, metadata={"phase": "design"}).phase == "design"
        assert Suggestion(parameters={}, metadata={"phase": "model"}).phase == "model"

    def test_it_is_none_when_the_policy_did_not_tag_one(self) -> None:
        assert Suggestion(parameters={"x": 0.5}).phase is None
        assert Suggestion(parameters={}, metadata={"other": "design"}).phase is None

    @pytest.mark.parametrize("value", [1, None, ["design"], {"name": "design"}, True])
    def test_a_phase_that_is_not_a_string_is_not_one(self, value: Any) -> None:
        assert Suggestion(parameters={}, metadata={"phase": value}).phase is None


class TestTheStudyCanRecordIt:
    def test_a_trial_is_made_from_it_without_a_further_check_of_its_shape(self) -> None:
        suggestion = Suggestion(
            parameters={"x": 0.5, "kind": "a"}, metadata={"phase": "design", "info": [1, 2]}
        )

        trial = Trial(id=4, parameters=suggestion.parameters, metadata=suggestion.metadata)

        assert dict(trial.parameters) == {"x": 0.5, "kind": "a"}
        assert trial.metadata["phase"] == "design"
        assert trial.state.value == "pending"

    @pytest.mark.parametrize(
        "parameters",
        [{"x": 0.5}, {"x": 1, "y": "b", "z": False}, {}],
    )
    def test_what_a_suggestion_accepts_a_trial_accepts(
        self, parameters: dict[str, Any]
    ) -> None:
        suggestion = Suggestion(parameters=parameters, metadata={"phase": "p"})

        assert Trial(id=0, parameters=suggestion.parameters, metadata=suggestion.metadata)
