"""Unit tests for the checks that a parameter type makes of a value (P2-10)."""

from __future__ import annotations

import math
from typing import Any

import pytest

from planexs import Boolean, Categorical, Choice, Derived, Fixed, Integer, Range, Real
from planexs.core.PlanexsError import PlanexsError
from planexs.domain.ParameterizationCheckConfig import ParameterizationCheckConfig
from planexs.domain.parameters.isNumber import isNumber
from planexs.domain.parameters.Parameter import Parameter
from planexs.domain.parameters.ParameterValueError import ParameterValueError


class TestIsNumber:
    @pytest.mark.parametrize("value", [0, 1, -3, 2.5, -0.0, 10**30, 1e300])
    def test_an_int_or_float_is_a_number(self, value: object) -> None:
        assert isNumber(value) is True

    @pytest.mark.parametrize(
        "value",
        [
            True,
            False,
            "1",
            "abc",
            None,
            [1.0],
            (1,),
            complex(1, 0),
            math.nan,
            math.inf,
            -math.inf,
        ],
    )
    def test_anything_else_is_not(self, value: object) -> None:
        assert isNumber(value) is False

    def test_an_int_too_large_for_a_float_is_not(self) -> None:
        assert isNumber(10**400) is False


class TestParameterValueError:
    def test_it_is_a_planexs_error_and_a_value_error(self) -> None:
        error = ParameterValueError("x", component="parameter 'a'", field_path="a")

        assert isinstance(error, PlanexsError)
        assert isinstance(error, ValueError)
        assert error.field_path == "a"


class TestRange:
    def test_inside_and_on_both_bounds(self) -> None:
        parameter = Real("x", 0.0, 1.0)

        for value in (0.0, 0.5, 1.0, 1, 0):
            parameter.validateValue(value)

    @pytest.mark.parametrize("value", [-0.1, 1.1, -1e300])
    def test_outside_the_bounds(self, value: float) -> None:
        with pytest.raises(ParameterValueError, match=r"'x'.*outside the bounds 0.0 to 1.0"):
            Real("x", 0.0, 1.0).validateValue(value)

    @pytest.mark.parametrize("value", ["0.5", True, None, math.nan, math.inf, [0.5], 10**400])
    def test_a_value_that_is_not_a_finite_number(self, value: object) -> None:
        with pytest.raises(ParameterValueError, match=r"not a finite number"):
            Real("x", 0.0, 1.0).validateValue(value)

    def test_the_scaling_does_not_matter(self) -> None:
        parameter = Real("rate", 1e-3, 1.0, scaling="log")

        parameter.validateValue(0.5)
        with pytest.raises(ParameterValueError, match="outside the bounds"):
            parameter.validateValue(0.0)

    @pytest.mark.parametrize("value", [1, 5, 10, 3.0])
    def test_an_integer_range_takes_whole_numbers(self, value: float) -> None:
        Integer("n", 1, 10).validateValue(value)

    @pytest.mark.parametrize("value", [2.5, 9.999])
    def test_an_integer_range_refuses_a_fraction(self, value: float) -> None:
        with pytest.raises(ParameterValueError, match="not a whole number"):
            Integer("n", 1, 10).validateValue(value)

    def test_integer_bounds_that_are_not_whole_numbers(self) -> None:
        parameter = Range("n", (0.5, 3.5), parameter_type="int")

        parameter.validateValue(1)
        parameter.validateValue(3)
        with pytest.raises(ParameterValueError, match="outside the bounds"):
            parameter.validateValue(0)
        with pytest.raises(ParameterValueError, match="outside the bounds"):
            parameter.validateValue(4)

    def test_the_bounds_are_judged_before_the_whole_number_rule(self) -> None:
        with pytest.raises(ParameterValueError, match="outside the bounds"):
            Integer("n", 1, 10).validateValue(99.5)

    @pytest.mark.parametrize("value", [0.0, 0.1, 0.3, 0.7, 1.0, 0.1 * 3, 0.30000000000000004])
    def test_a_value_on_the_grid_of_a_step_is_accepted(self, value: float) -> None:
        Real("x", 0.0, 1.0, step_size=0.1).validateValue(value)

    @pytest.mark.parametrize("value", [0.05, 0.35, 0.99])
    def test_a_value_between_grid_points_is_refused(self, value: float) -> None:
        with pytest.raises(ParameterValueError, match=r"not on the grid of step 0.1"):
            Real("x", 0.0, 1.0, step_size=0.1).validateValue(value)

    def test_the_grid_starts_at_the_lower_bound(self) -> None:
        parameter = Real("x", 2.0, 4.0, step_size=0.5)

        parameter.validateValue(3.5)
        with pytest.raises(ParameterValueError, match="not on the grid"):
            parameter.validateValue(3.25)

    def test_the_last_grid_point_may_fall_short_of_the_upper_bound(self) -> None:
        parameter = Real("x", 0.0, 1.0, step_size=0.3)  # 0.0 0.3 0.6 0.9

        parameter.validateValue(0.9)
        with pytest.raises(ParameterValueError, match="not on the grid"):
            parameter.validateValue(1.0)

    def test_the_grid_tolerance_is_a_setting(self) -> None:
        parameter = Real("x", 0.0, 1.0, step_size=0.1)
        rounded = 0.30000000000000004  # 3 * 0.1 in floating point: off the grid by 4e-16 steps
        strict = ParameterizationCheckConfig(grid_tolerance=0.0)
        loose = ParameterizationCheckConfig(grid_tolerance=0.5)

        parameter.validateValue(rounded)  # the default tolerance absorbs the rounding
        with pytest.raises(ParameterValueError, match="not on the grid"):
            parameter.validateValue(rounded, strict)  # exact asks for the impossible
        parameter.validateValue(0.34, loose)  # 3.4 steps, within half a step of 3
        with pytest.raises(ParameterValueError, match="not on the grid"):
            parameter.validateValue(0.34)

    def test_a_value_exactly_on_the_grid_is_accepted_with_no_tolerance_at_all(self) -> None:
        strict = ParameterizationCheckConfig(grid_tolerance=0.0)
        parameter = Real("x", 0.0, 1.0, step_size=0.5)

        for value in (0.0, 0.5, 1.0):
            parameter.validateValue(value, strict)

    def test_options_default_to_the_defaults_of_the_config(self) -> None:
        parameter = Real("x", 0.0, 1.0, step_size=0.1)

        parameter.validateValue(0.3 + 5e-11)  # 5e-10 steps: inside the default 1e-9
        with pytest.raises(ParameterValueError):
            parameter.validateValue(0.3 + 5e-9)  # 5e-8 steps: outside it

    def test_the_error_names_the_parameter_the_value_and_a_fix(self) -> None:
        with pytest.raises(ParameterValueError) as caught:
            Real("temperature", 20.0, 80.0).validateValue(95.0)

        error = caught.value
        assert "temperature" in error.message and "95.0" in error.message
        assert error.component == "parameter 'temperature'"
        assert error.field_path == "temperature"
        assert error.suggestion is not None and "20.0" in error.suggestion


class TestChoice:
    def test_a_value_of_the_choice_is_accepted(self) -> None:
        parameter = Categorical("solvent", ["water", "ethanol"])

        parameter.validateValue("water")
        parameter.validateValue("ethanol")

    @pytest.mark.parametrize("value", ["methanol", "Water", 1, None, True, ["water"]])
    def test_anything_else_is_refused_and_the_values_are_listed(self, value: object) -> None:
        with pytest.raises(ParameterValueError, match=r"\['water', 'ethanol'\]"):
            Categorical("solvent", ["water", "ethanol"]).validateValue(value)

    def test_true_is_not_the_integer_one(self) -> None:
        # Python makes True == 1, so `value in values` would accept it.
        with pytest.raises(ParameterValueError):
            Choice("n", [0, 1], "int").validateValue(True)

    def test_one_is_not_the_boolean_true(self) -> None:
        with pytest.raises(ParameterValueError):
            Boolean("flag").validateValue(1)

    def test_a_boolean_choice_takes_booleans(self) -> None:
        Boolean("flag").validateValue(True)
        Boolean("flag").validateValue(False)

    @pytest.mark.parametrize("value", [1, 1.0, 3, 3.0])
    def test_a_whole_float_matches_the_integer_level_it_equals(self, value: float) -> None:
        Choice("n", [1, 2, 3], "int").validateValue(value)

    def test_a_string_is_not_a_number(self) -> None:
        with pytest.raises(ParameterValueError):
            Choice("n", [1, 2, 3], "int").validateValue("1")

    def test_float_levels(self) -> None:
        parameter = Choice("ratio", [0.25, 0.5, 1.0], "float")

        parameter.validateValue(0.5)
        parameter.validateValue(1)
        with pytest.raises(ParameterValueError):
            parameter.validateValue(0.75)

    def test_nan_is_never_a_level(self) -> None:
        with pytest.raises(ParameterValueError):
            Choice("ratio", [0.25, 0.5], "float").validateValue(math.nan)

    def test_a_fixed_parameter_accepts_only_its_value(self) -> None:
        parameter = Fixed("scale", 3)

        parameter.validateValue(3)
        with pytest.raises(ParameterValueError):
            parameter.validateValue(4)

    def test_the_options_are_not_used(self) -> None:
        Boolean("flag").validateValue(True, ParameterizationCheckConfig(grid_tolerance=0.0))


class TestDerived:
    @pytest.mark.parametrize(
        ("parameter_type", "good", "bad"),
        [
            ("float", [0.5, 3, -2.0], ["1", True, None, math.nan, math.inf]),
            ("int", [3, 3.0, -2], [2.5, "3", True, math.nan]),
            ("bool", [True, False], [1, 0, "True", None]),
            ("str", ["a", ""], [1, True, None]),
        ],
    )
    def test_the_value_has_the_type_of_the_expression(
        self, parameter_type: Any, good: list[object], bad: list[object]
    ) -> None:
        parameter = Derived("d", "a + b", parameter_type)

        for value in good:
            parameter.validateValue(value)
        for value in bad:
            with pytest.raises(ParameterValueError, match="derived parameter 'd'"):
                parameter.validateValue(value)

    def test_the_error_says_that_the_parameter_can_be_left_out(self) -> None:
        with pytest.raises(ParameterValueError) as caught:
            Derived("d", "a + b", "float").validateValue("oops")

        assert (
            caught.value.suggestion is not None
            and "leave the parameter out" in caught.value.suggestion
        )

    def test_the_value_is_not_recomputed(self) -> None:
        # Documented: only the type is checked. A wrong number of the right type passes.
        Derived("d", "a + b", "float").validateValue(12345.0)


class TestComputedAndActivations:
    def test_only_a_derived_parameter_is_computed(self) -> None:
        assert Derived("d", "a + b", "float").is_computed is True
        assert Real("x", 0.0, 1.0).is_computed is False
        assert Categorical("c", ["a", "b"]).is_computed is False

    def test_a_choice_activates_the_parameters_of_its_values(self) -> None:
        choice = Categorical(
            "model", ["linear", "tree"], dependent_parameters={"tree": ["depth", "leaves"]}
        )

        assert dict(choice.activations()) == {"tree": ["depth", "leaves"]}

    def test_a_choice_without_dependents_activates_nothing(self) -> None:
        assert dict(Categorical("c", ["a", "b"]).activations()) == {}

    @pytest.mark.parametrize(
        "parameter",
        [Real("x", 0.0, 1.0), Derived("d", "a + b", "float")],
        ids=["range", "derived"],
    )
    def test_other_types_activate_nothing(self, parameter: Parameter) -> None:
        assert dict(parameter.activations()) == {}


class TestTheContract:
    def test_a_type_that_does_not_say_how_to_check_a_value_cannot_be_built(self) -> None:
        class Lazy(Parameter):
            def _validate_default(self) -> None:
                return None

            def toDict(self) -> dict[str, Any]:
                return {"type": "lazy", "name": self.name}

        with pytest.raises(TypeError, match="validateValue"):
            Lazy(name="lazy")  # type: ignore[abstract]
