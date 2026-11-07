"""Unit tests for the values of the `continuous` outcome kind."""

from __future__ import annotations

import math
from decimal import Decimal
from fractions import Fraction
from typing import Any

import numpy as np
import pytest

from planexs.core.PlanexsError import PlanexsError
from planexs.domain.outcome_kinds.ContinuousKind import ContinuousKind
from planexs.domain.outcome_kinds.OutcomeValueError import OutcomeValueError

KIND = ContinuousKind()


class TestAGoodValue:
    @pytest.mark.parametrize(
        ("value", "stored"),
        [
            (0, 0.0),
            (3, 3.0),
            (-7, -7.0),
            (0.0, 0.0),
            (-0.0, -0.0),
            (2.5, 2.5),
            (-1e-300, -1e-300),
            (1.7976931348623157e308, 1.7976931348623157e308),
            (0.1 + 0.2, 0.1 + 0.2),
            (10**20, 1e20),
            (Fraction(1, 2), 0.5),
            (np.float64(2.5), 2.5),
            (np.float32(0.5), 0.5),
            (np.int64(7), 7.0),
            (np.int8(-3), -3.0),
        ],
    )
    def test_a_finite_number_is_accepted_and_stored_as_a_float(
        self, value: Any, stored: float
    ) -> None:
        result = KIND.validateValue(value, "yield")

        assert result == stored
        assert type(result) is float

    def test_a_float_is_stored_exactly(self) -> None:
        value = 0.1 + 0.2

        assert KIND.validateValue(value, "yield") == value


class TestABadValue:
    @pytest.mark.parametrize(
        "value",
        [
            float("nan"),
            float("inf"),
            float("-inf"),
            np.float64("nan"),
            np.float32("inf"),
            math.nan,
            math.inf,
        ],
        ids=["nan", "inf", "-inf", "np-nan", "np-inf32", "math.nan", "math.inf"],
    )
    def test_nan_and_the_infinities_are_refused(self, value: Any) -> None:
        with pytest.raises(OutcomeValueError, match="not finite"):
            KIND.validateValue(value, "yield")

    @pytest.mark.parametrize(
        "value",
        [
            "1.0",
            "nan",
            "",
            b"1",
            None,
            True,
            False,
            np.True_,
            [1.0],
            (1.0,),
            {"value": 1.0},
            1 + 2j,
            Decimal("1.5"),
            np.array([1.0]),
            object(),
        ],
        ids=lambda v: type(v).__name__ + ":" + repr(v)[:12],
    )
    def test_what_is_not_a_number_is_refused(self, value: Any) -> None:
        with pytest.raises(OutcomeValueError, match="not a number"):
            KIND.validateValue(value, "yield")

    def test_a_bool_is_refused_because_it_is_a_label_and_not_a_number(self) -> None:
        with pytest.raises(OutcomeValueError, match=r"True.*not a number \(got bool\)"):
            KIND.validateValue(True, "yield")

    def test_an_integer_too_large_for_a_float_is_refused(self) -> None:
        with pytest.raises(OutcomeValueError, match="too large to be a float"):
            KIND.validateValue(10**400, "yield")


class TestTheError:
    def test_it_names_the_outcome_and_the_value(self) -> None:
        with pytest.raises(OutcomeValueError) as caught:
            KIND.validateValue("high", "throughput")

        error = caught.value
        assert "'throughput'" in error.message
        assert "'high'" in error.message
        assert error.component == "outcome 'throughput'"
        assert error.suggestion

    def test_it_is_a_planexs_error_and_a_value_error(self) -> None:
        with pytest.raises(OutcomeValueError) as caught:
            KIND.validateValue(float("nan"), "yield")

        assert isinstance(caught.value, PlanexsError)
        assert isinstance(caught.value, ValueError)

    def test_a_nan_is_reported_with_advice_to_fail_the_trial_instead(self) -> None:
        with pytest.raises(OutcomeValueError) as caught:
            KIND.validateValue(float("nan"), "yield")

        assert "failed" in str(caught.value.suggestion)


class TestTheKindItself:
    def test_a_noise_standard_deviation_is_meaningful(self) -> None:
        assert KIND.allows_noise is True

    def test_its_capability_tag(self) -> None:
        assert KIND.capability_tag == "outcome_kind:continuous"

    def test_it_keeps_no_state_between_values(self) -> None:
        assert KIND.validateValue(1, "a") == 1.0
        with pytest.raises(OutcomeValueError):
            KIND.validateValue(float("nan"), "a")
        assert KIND.validateValue(2, "a") == 2.0
