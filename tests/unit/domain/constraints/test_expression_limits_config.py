"""Unit tests for `ExpressionLimitsConfig` and for the limits `validateExpression` applies.

Roadmap task P3-06 (specification 3.4, 10.8): the two numbers that bound the size of an
expression, 1000 characters and 200 levels of nesting, used to be module constants that the
caller could not reach. They are now the fields of a configuration schema, which
`validateExpression` takes as an argument and a `NonlinearConstraint` carries as `limits`.

The two limits are tested one at a time, with an expression that exceeds only that one: a
single long name is long but not deep, and a chain of unary minus signs is deep but short. A
sum of many terms is both (a left-leaning tree is as deep as it has terms), so it would hide
which limit refused it.
"""

from __future__ import annotations

import importlib
from typing import Any

import pytest
from pydantic import ValidationError

from planexs.core.ComponentConfig import ComponentConfig
from planexs.core.ConfigError import ConfigError
from planexs.core.validateConfig import validateConfig
from planexs.domain.constraints.ExpressionLimitsConfig import ExpressionLimitsConfig
from planexs.domain.constraints.NonlinearConstraintConfig import NonlinearConstraintConfig
from planexs.domain.constraints.validateExpression import validateExpression

#: The limits the validator applied before they became settings. Written here a second time,
#: on purpose: a change of a default in the schema must fail a test and be noticed, because it
#: changes which expressions every study without explicit limits accepts.
OLD_MAX_LENGTH = 1000
OLD_MAX_DEPTH = 200


def longName(length: int) -> str:
    """A single parameter name of `length` characters: long as asked, and not deep at all."""
    return "a" * length


def nested(levels: int) -> str:
    """`levels` unary minus signs before `x`: the name sits `levels` levels below the root."""
    return "-" * levels + "x"


class TestTheSchema:
    def test_the_defaults_are_the_values_the_validator_always_used(self) -> None:
        limits = ExpressionLimitsConfig()

        assert (limits.max_length, limits.max_depth) == (OLD_MAX_LENGTH, OLD_MAX_DEPTH)

    def test_it_is_a_plain_options_schema_and_not_a_component(self) -> None:
        assert not issubclass(ExpressionLimitsConfig, ComponentConfig)
        assert "type" not in ExpressionLimitsConfig.model_fields

    def test_every_field_is_described_and_the_description_states_the_default(self) -> None:
        for name, info in ExpressionLimitsConfig.model_fields.items():
            assert info.description and info.description.strip(), name
            assert f"The default is {info.default}" in info.description, name

    def test_it_is_frozen(self) -> None:
        limits = ExpressionLimitsConfig()

        with pytest.raises(ValidationError, match="frozen"):
            limits.max_length = 5  # type: ignore[misc]

    def test_an_unknown_field_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="max_depht"):
            ExpressionLimitsConfig.model_validate({"max_depht": 5})

    @pytest.mark.parametrize(
        ("field", "value"), [("max_length", 0), ("max_length", -5), ("max_depth", -1)]
    )
    def test_a_limit_below_its_minimum_is_refused(self, field: str, value: int) -> None:
        with pytest.raises(ValidationError, match=field):
            ExpressionLimitsConfig(**{field: value})

    @pytest.mark.parametrize("value", [1.5, "many", None])
    @pytest.mark.parametrize("field", ["max_length", "max_depth"])
    def test_a_limit_that_is_not_a_whole_number_is_refused(
        self, field: str, value: Any
    ) -> None:
        with pytest.raises(ValidationError, match=field):
            ExpressionLimitsConfig(**{field: value})

    def test_a_bad_field_is_reported_with_its_full_path_in_the_stored_form(self) -> None:
        stored = {
            "type": "nonlinear",
            "expression": "x * y",
            "comparator": "<=",
            "bound": 1.0,
            "limits": {"max_depth": -1},
        }

        with pytest.raises(ConfigError) as caught:
            validateConfig(NonlinearConstraintConfig, stored)

        assert caught.value.field_path == "limits.max_depth"


class TestTheDefaultsReproduceTheOldLimits:
    """The boundary of each default, to the character and to the level."""

    def test_a_text_of_exactly_the_old_length_is_accepted(self) -> None:
        expression = longName(OLD_MAX_LENGTH)

        assert validateExpression(expression) == {expression}

    def test_one_character_more_is_refused(self) -> None:
        with pytest.raises(ValueError, match="limited"):
            validateExpression(longName(OLD_MAX_LENGTH + 1))

    def test_a_nesting_of_exactly_the_old_depth_is_accepted(self) -> None:
        assert validateExpression(nested(OLD_MAX_DEPTH)) == {"x"}

    def test_one_level_more_is_refused(self) -> None:
        with pytest.raises(ValueError, match="deeply"):
            validateExpression(nested(OLD_MAX_DEPTH + 1))

    @pytest.mark.parametrize("expression", ["x * y", "sqrt(x) + log(y)", nested(50)])
    def test_no_limits_none_and_the_default_object_are_the_same(self, expression: str) -> None:
        assert (
            validateExpression(expression)
            == validateExpression(expression, None)
            == validateExpression(expression, ExpressionLimitsConfig())
        )


class TestCustomLimits:
    def test_a_larger_max_length_accepts_a_text_the_defaults_refuse(self) -> None:
        expression = longName(1500)
        with pytest.raises(ValueError, match="limited"):
            validateExpression(expression)

        assert validateExpression(expression, ExpressionLimitsConfig(max_length=1500)) == {
            expression
        }

    def test_max_length_is_inclusive(self) -> None:
        limits = ExpressionLimitsConfig(max_length=1500)

        assert validateExpression(longName(1500), limits)
        with pytest.raises(ValueError, match="limited"):
            validateExpression(longName(1501), limits)

    def test_a_larger_max_depth_accepts_a_nesting_the_defaults_refuse(self) -> None:
        with pytest.raises(ValueError, match="deeply"):
            validateExpression(nested(250))

        assert validateExpression(nested(250), ExpressionLimitsConfig(max_depth=250)) == {"x"}

    def test_max_depth_is_inclusive(self) -> None:
        limits = ExpressionLimitsConfig(max_depth=250)

        assert validateExpression(nested(250), limits)
        with pytest.raises(ValueError, match="deeply"):
            validateExpression(nested(251), limits)

    def test_a_smaller_max_length_refuses_what_the_defaults_accept(self) -> None:
        limits = ExpressionLimitsConfig(max_length=10)

        assert validateExpression("x + y + z") == {"x", "y", "z"}  # 9 characters
        assert validateExpression("x + y + z", limits) == {"x", "y", "z"}
        with pytest.raises(ValueError, match="limited"):
            validateExpression("x + y + z + w", limits)  # 13 characters

    def test_a_smaller_max_depth_refuses_what_the_defaults_accept(self) -> None:
        # `x + y + z` leans to the left: ((x + y) + z), so its names sit two levels down.
        assert validateExpression("x + y") == {"x", "y"}
        assert validateExpression("x + y", ExpressionLimitsConfig(max_depth=1)) == {"x", "y"}
        assert validateExpression("x + y + z", ExpressionLimitsConfig(max_depth=2))
        with pytest.raises(ValueError, match="deeply"):
            validateExpression("x + y + z", ExpressionLimitsConfig(max_depth=1))

    def test_a_depth_of_zero_allows_only_a_bare_name(self) -> None:
        limits = ExpressionLimitsConfig(max_depth=0)

        assert validateExpression("x", limits) == {"x"}
        assert validateExpression("(x)", limits) == {"x"}  # parentheses that group nothing
        for expression in ["-x", "x + y", "sqrt(x)"]:
            with pytest.raises(ValueError, match="deeply"):
                validateExpression(expression, limits)

    def test_the_length_is_that_of_the_text_without_the_whitespace_around_it(self) -> None:
        limits = ExpressionLimitsConfig(max_length=10)

        assert validateExpression("  \n" + longName(10) + "\t  ", limits)
        with pytest.raises(ValueError, match="limited"):
            validateExpression("  " + longName(11) + "  ", limits)

    def test_the_two_limits_are_independent(self) -> None:
        # A long name is not deep, so no depth limit refuses it; a short chain is not long.
        assert validateExpression(longName(500), ExpressionLimitsConfig(max_depth=0))
        assert validateExpression(nested(50), ExpressionLimitsConfig(max_length=60))

    def test_the_limits_of_one_call_are_not_remembered_by_the_next(self) -> None:
        validateExpression(longName(1500), ExpressionLimitsConfig(max_length=1500))

        with pytest.raises(ValueError, match="limited"):
            validateExpression(longName(1500))

    def test_the_errors_name_the_setting_and_the_numbers(self) -> None:
        with pytest.raises(
            ValueError,
            match=r"limited to 12 characters \(`max_length` of its limits\); got 13",
        ):
            validateExpression(longName(13), ExpressionLimitsConfig(max_length=12))
        with pytest.raises(ValueError, match=r"more than 1 level\(s\) \(`max_depth` of its"):
            validateExpression("x + y + z", ExpressionLimitsConfig(max_depth=1))


class TestWhenTheStackRunsOutFirst:
    """The walk is recursive, so a `max_depth` near the interpreter's limit must not leak a
    `RecursionError`: before the limit was a setting, 200 kept the walk far from it.
    """

    def test_a_stack_overflow_is_the_same_refusal(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        module = importlib.import_module("planexs.domain.constraints.validateExpression")

        def overflow(*arguments: object, **keywords: object) -> None:
            raise RecursionError("maximum recursion depth exceeded")

        monkeypatch.setattr(module, "_visit", overflow)

        with pytest.raises(ValueError, match=r"too deeply to be checked.*max_depth") as caught:
            validateExpression("x + y", ExpressionLimitsConfig(max_depth=10**6))

        assert isinstance(caught.value.__cause__, RecursionError)

    @pytest.mark.parametrize("terms", [1500, 3000])
    def test_a_real_overflow_is_a_value_error_and_never_a_raw_recursion_error(
        self, terms: int
    ) -> None:
        # Far past the interpreter's default recursion limit (1000), with limits that would
        # let the depth check pass: only the guard stands between this and a `RecursionError`.
        limits = ExpressionLimitsConfig(max_length=10**6, max_depth=10**6)

        with pytest.raises(ValueError):
            validateExpression("+".join(["x"] * terms), limits)
