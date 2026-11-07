"""Unit tests for the values of the `binary` outcome kind."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import numpy as np
import pytest

from planexs.core.PlanexsError import PlanexsError
from planexs.domain.outcome_kinds.BinaryKind import BinaryKind
from planexs.domain.outcome_kinds.OutcomeValueError import OutcomeValueError

KIND = BinaryKind()


class TestAGoodValue:
    @pytest.mark.parametrize(
        ("value", "stored"),
        [
            (True, 1.0),
            (False, 0.0),
            (1, 1.0),
            (0, 0.0),
            (np.True_, 1.0),
            (np.False_, 0.0),
            (np.bool_(True), 1.0),
            (np.int64(1), 1.0),
            (np.int64(0), 0.0),
            (np.int8(1), 1.0),
            (np.uint8(0), 0.0),
            (np.array(True), 1.0),
        ],
        ids=lambda v: type(v).__name__ + ":" + repr(v),
    )
    def test_a_bool_or_the_integer_zero_or_one_is_accepted_and_stored_as_zero_or_one(
        self, value: Any, stored: float
    ) -> None:
        result = KIND.validateValue(value, "phase")

        assert result == stored
        assert type(result) is float

    def test_a_comparison_gives_a_label(self) -> None:
        # What `x > 0` gives on a NumPy number is a NumPy boolean, not a `bool`.
        assert KIND.validateValue(np.float64(3.0) > 0, "phase") == 1.0
        assert KIND.validateValue(np.float64(-3.0) > 0, "phase") == 0.0


class TestABadValue:
    @pytest.mark.parametrize("value", [2, -1, 3, 10, 100, np.int64(2), np.int8(-1)])
    def test_an_integer_other_than_zero_or_one_is_refused(self, value: Any) -> None:
        with pytest.raises(OutcomeValueError, match="not a label"):
            KIND.validateValue(value, "phase")

    def test_two_is_refused_and_named(self) -> None:
        with pytest.raises(OutcomeValueError) as caught:
            KIND.validateValue(2, "phase")

        assert (
            "The value 2 of the binary outcome 'phase' is not a label" in caught.value.message
        )

    @pytest.mark.parametrize(
        "value",
        [0.0, 1.0, 0.5, -1.0, np.float64(1.0), np.float32(0.0), Decimal(1)],
        ids=lambda v: type(v).__name__ + ":" + repr(v),
    )
    def test_a_float_is_refused_even_when_it_is_zero_or_one(self, value: Any) -> None:
        with pytest.raises(OutcomeValueError, match="not a label"):
            KIND.validateValue(value, "phase")

    @pytest.mark.parametrize(
        "value",
        [
            "1",
            "0",
            "true",
            "True",
            "yes",
            "",
            b"1",
            None,
            float("nan"),
            float("inf"),
            [1],
            (True,),
            {"label": True},
            1 + 0j,
            object(),
        ],
        ids=lambda v: type(v).__name__ + ":" + repr(v)[:12],
    )
    def test_what_is_not_a_label_is_refused(self, value: Any) -> None:
        with pytest.raises(OutcomeValueError, match="not a label"):
            KIND.validateValue(value, "phase")

    @pytest.mark.parametrize(
        "value",
        [np.array([True]), np.array([True, False]), np.array([1]), np.array([[True]])],
        ids=["bool[1]", "bool[2]", "int[1]", "bool[1x1]"],
    )
    def test_an_array_is_refused_whatever_its_size(self, value: Any) -> None:
        # An array of one element could be mistaken for a scalar, and one of several elements
        # has no truth value: both are reported as a value that is not a label.
        with pytest.raises(OutcomeValueError, match="not a label"):
            KIND.validateValue(value, "phase")

    def test_the_advice_for_a_float_is_to_cast_it(self) -> None:
        with pytest.raises(OutcomeValueError) as caught:
            KIND.validateValue(1.0, "phase")

        assert "int(value)" in str(caught.value.suggestion)


class TestTheError:
    def test_it_names_the_outcome_and_the_type_of_the_value(self) -> None:
        with pytest.raises(OutcomeValueError) as caught:
            KIND.validateValue("liquid", "phase")

        error = caught.value
        assert "'phase'" in error.message
        assert "'liquid'" in error.message
        assert "got str" in error.message
        assert error.component == "outcome 'phase'"

    def test_it_is_a_planexs_error_and_a_value_error(self) -> None:
        with pytest.raises(OutcomeValueError) as caught:
            KIND.validateValue(2, "phase")

        assert isinstance(caught.value, PlanexsError)
        assert isinstance(caught.value, ValueError)


class TestTheKindItself:
    def test_no_noise_standard_deviation_is_meaningful(self) -> None:
        assert KIND.allows_noise is False

    def test_its_capability_tag(self) -> None:
        assert KIND.capability_tag == "outcome_kind:binary"
