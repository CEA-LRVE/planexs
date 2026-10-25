"""Unit tests for `toConfigError` and `toValidationError`."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

from planexs.core.ConfigError import ConfigError
from planexs.core.configErrors import toConfigError, toValidationError


class Inner(BaseModel):
    """A model nested inside `Outer`."""

    model_config = ConfigDict(extra="forbid")

    b: int = 0


class Outer(BaseModel):
    """A model with a list of nested models, a plain field and a rule across fields."""

    model_config = ConfigDict(extra="forbid")

    items: list[Inner] = []
    inner: Inner = Inner()
    n: int = 0
    tags: list[int] = []

    @model_validator(mode="after")
    def _evenOnly(self) -> Outer:
        if self.n % 2:
            raise ValueError("n must be even")
        return self


def validationErrorOf(data: Any, schema: type[BaseModel] = Outer) -> ValidationError:
    """The pydantic error that validating `data` against `schema` raises."""
    with pytest.raises(ValidationError) as caught:
        schema.model_validate(data)
    return caught.value


def configErrorOf(data: Any, label: str = "the outer") -> ConfigError:
    """The `ConfigError` that `toConfigError` makes of the failure of `data` for `Outer`."""
    return toConfigError(validationErrorOf(data), label=label, schema=Outer)


class TestToConfigErrorMessage:
    def test_a_single_problem_is_one_sentence(self) -> None:
        error = configErrorOf({"n": "x"})
        assert error.message == (
            "Invalid configuration for the outer: n: Input should be a valid integer, unable "
            "to parse string as an integer (got 'x')."
        )

    def test_the_label_names_the_component_of_the_error(self) -> None:
        error = configErrorOf({"n": "x"}, label="surrogate 'single_task_gp_v1'")
        assert error.component == "surrogate 'single_task_gp_v1'"
        assert "Invalid configuration for surrogate 'single_task_gp_v1':" in error.message

    def test_a_message_that_ends_with_a_period_does_not_get_a_second_one(self) -> None:
        class Ends(BaseModel):
            n: int = 0

            @model_validator(mode="after")
            def _fail(self) -> Ends:
                raise ValueError("this ends with a period.")

        error = toConfigError(validationErrorOf({}, Ends), label="x", schema=Ends)
        assert (
            error.message
            == "Invalid configuration for x: Value error, this ends with a period."
        )

    def test_several_problems_are_listed_one_per_line(self) -> None:
        error = configErrorOf({"n": "x", "tags": [1, "y"], "bogus": 1})
        lines = error.message.splitlines()
        assert lines[0] == "Invalid configuration for the outer: 3 problems."
        assert [line.startswith("  - ") for line in lines[1:]] == [True, True, True]
        assert lines[1].startswith("  - n: ")
        assert lines[2].startswith("  - tags[1]: ")
        assert lines[3] == "  - bogus: Unknown field"

    def test_the_field_path_is_the_one_of_the_first_problem(self) -> None:
        assert configErrorOf({"n": "x", "bogus": 1}).field_path == "n"

    def test_the_value_that_was_given_is_shown(self) -> None:
        assert "(got 'x')" in configErrorOf({"n": "x"}).message

    def test_a_long_value_is_shortened(self) -> None:
        error = configErrorOf({"n": "x" * 10_000})
        assert len(error.message) < 1_000

    def test_a_missing_field_does_not_show_the_mapping_it_is_missing_from(self) -> None:
        class Needs(BaseModel):
            level: float

        error = toConfigError(validationErrorOf({}, Needs), label="x", schema=Needs)
        assert error.message == "Invalid configuration for x: level: Field required."

    def test_an_unknown_field_is_reported_by_its_name_only(self) -> None:
        error = configErrorOf({"bogus": [1, 2, 3]})
        assert error.message == "Invalid configuration for the outer: bogus: Unknown field."


class TestToConfigErrorFieldPath:
    @pytest.mark.parametrize(
        "data, path",
        [
            ({"n": "x"}, "n"),
            ({"inner": {"b": "x"}}, "inner.b"),
            ({"items": [{"b": "x"}]}, "items[0].b"),
            ({"items": [{}, {"b": 1}, {"b": "x"}]}, "items[2].b"),
            ({"items": ["x"]}, "items[0]"),
            ({"tags": ["x"]}, "tags[0]"),
            ({"tags": [1, 2, "x"]}, "tags[2]"),
            ({"inner": {"zzz": 1}}, "inner.zzz"),
        ],
        ids=lambda value: value if isinstance(value, str) else "",
    )
    def test_the_path_is_dotted_with_list_positions_in_brackets(
        self, data: dict[str, Any], path: str
    ) -> None:
        assert configErrorOf(data).field_path == path

    def test_a_rule_across_fields_has_no_path(self) -> None:
        error = configErrorOf({"n": 3})
        assert error.field_path is None
        assert (
            error.message
            == "Invalid configuration for the outer: Value error, n must be even."
        )


class TestToConfigErrorSuggestion:
    def test_an_unknown_field_of_the_model_suggests_its_fields(self) -> None:
        error = configErrorOf({"nn": 1})
        assert error.suggestion == (
            "check the spelling; the fields of the outer are: items, inner, n, tags"
        )

    def test_an_unknown_field_of_a_nested_model_suggests_nothing(self) -> None:
        # The fields to choose from would be the nested model's, not the model's own.
        assert configErrorOf({"inner": {"zzz": 1}}).suggestion is None

    def test_other_problems_suggest_nothing(self) -> None:
        assert configErrorOf({"n": "x"}).suggestion is None


class TestToValidationError:
    def test_it_is_a_validation_error_with_one_problem(self) -> None:
        error = toValidationError(ConfigError("Unknown kernel type 'x'."), {"type": "x"})
        assert isinstance(error, ValidationError)
        assert error.error_count() == 1

    def test_the_message_is_kept(self) -> None:
        error = toValidationError(ConfigError("Unknown kernel type 'x'."), {"type": "x"})
        assert error.errors()[0]["msg"] == "Unknown kernel type 'x'."

    def test_braces_in_a_message_are_kept_as_they_are(self) -> None:
        message = "Registered names: {a}, {b}; got {'type': 1}."
        error = toValidationError(ConfigError(message), {})
        assert error.errors()[0]["msg"] == message

    def test_the_location_is_the_field_path_of_the_error(self) -> None:
        error = toValidationError(ConfigError("x", field_path="type"), {"nu": 1})
        assert error.errors()[0]["loc"] == ("type",)

    def test_an_error_without_a_field_path_is_located_at_the_field_itself(self) -> None:
        assert toValidationError(ConfigError("x"), []).errors()[0]["loc"] == ()

    def test_the_value_is_kept_as_the_input(self) -> None:
        value = {"type": "x", "nu": 1}
        assert toValidationError(ConfigError("x"), value).errors()[0]["input"] is value

    def test_the_component_of_the_error_is_the_title(self) -> None:
        error = toValidationError(ConfigError("x", component="kernel"), {})
        assert error.title == "kernel"
        assert toValidationError(ConfigError("x"), {}).title == "component"


class TestTheTwoConversionsTogether:
    def test_a_selection_problem_is_reported_like_any_other_with_its_path(self) -> None:
        converted = toValidationError(
            ConfigError("Unknown kernel type 'x'.", field_path="type"), {"type": "x"}
        )
        error = toConfigError(converted, label="surrogate", schema=Outer)
        assert error.field_path == "type"
        # A sentence of its own: the value is not added after it, nor a second period.
        assert (
            error.message
            == "Invalid configuration for surrogate: type: Unknown kernel type 'x'."
        )

    def test_a_selection_problem_raised_in_a_validator_keeps_its_location(self) -> None:
        class Holder(BaseModel):
            @model_validator(mode="before")
            @classmethod
            def _select(cls, data: Any) -> Any:
                raise toValidationError(
                    ConfigError("Unknown kernel type 'x'.", field_path="type"), data
                )

        error = toConfigError(validationErrorOf({"a": 1}, Holder), label="x", schema=Holder)
        assert error.message == "Invalid configuration for x: type: Unknown kernel type 'x'."
