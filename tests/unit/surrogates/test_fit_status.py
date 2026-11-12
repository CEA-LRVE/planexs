"""Unit tests for `FitStatus` (roadmap task P3-07, specification 5.6, 3.11)."""

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import ValidationError

from planexs.surrogates.FitStatus import FitStatus


class TestAFitStatus:
    def test_it_holds_the_count_the_flag_and_the_warnings(self) -> None:
        status = FitStatus(n_observations=3, is_low_data=True, warnings=("too few points",))

        assert status.n_observations == 3
        assert status.is_low_data is True
        assert status.warnings == ("too few points",)

    def test_there_are_no_warnings_unless_there_is_something_to_say(self) -> None:
        assert FitStatus(n_observations=20, is_low_data=False).warnings == ()

    def test_zero_observations_is_a_count_like_another(self) -> None:
        assert FitStatus(n_observations=0, is_low_data=True).n_observations == 0

    def test_a_list_of_warnings_is_read_as_a_tuple(self) -> None:
        status = FitStatus(n_observations=3, is_low_data=True, warnings=["a", "b"])  # type: ignore[arg-type]

        assert status.warnings == ("a", "b")

    def test_it_is_frozen(self) -> None:
        status = FitStatus(n_observations=3, is_low_data=True)

        with pytest.raises(ValidationError, match="frozen"):
            status.n_observations = 4

    def test_equal_statuses_are_equal_and_hashable(self) -> None:
        first = FitStatus(n_observations=3, is_low_data=True, warnings=("w",))
        second = FitStatus(n_observations=3, is_low_data=True, warnings=("w",))

        assert first == second
        assert hash(first) == hash(second)
        assert first != FitStatus(n_observations=4, is_low_data=True, warnings=("w",))

    def test_it_round_trips_through_json(self) -> None:
        status = FitStatus(n_observations=3, is_low_data=True, warnings=("w", "x"))

        text = status.model_dump_json()

        assert json.loads(text) == {
            "n_observations": 3,
            "is_low_data": True,
            "warnings": ["w", "x"],
        }
        assert FitStatus.model_validate_json(text) == status


class TestAnInvalidFitStatusIsRefused:
    @pytest.mark.parametrize(
        ("fields", "location"),
        [
            ({"is_low_data": True}, "n_observations"),
            ({"n_observations": 3}, "is_low_data"),
            ({"n_observations": -1, "is_low_data": True}, "n_observations"),
            ({"n_observations": 3.0, "is_low_data": True}, "n_observations"),
            ({"n_observations": "3", "is_low_data": True}, "n_observations"),
            ({"n_observations": True, "is_low_data": True}, "n_observations"),
            ({"n_observations": 3, "is_low_data": 1}, "is_low_data"),
            ({"n_observations": 3, "is_low_data": "yes"}, "is_low_data"),
            ({"n_observations": 3, "is_low_data": True, "extra": 1}, "extra"),
        ],
        ids=[
            "no-count",
            "no-flag",
            "negative-count",
            "float-count",
            "string-count",
            "bool-count",
            "int-flag",
            "string-flag",
            "unknown-field",
        ],
    )
    def test_a_field_that_is_missing_or_not_of_its_type(
        self, fields: dict[str, Any], location: str
    ) -> None:
        with pytest.raises(ValidationError) as caught:
            FitStatus(**fields)

        assert [error["loc"] for error in caught.value.errors()] == [(location,)]

    @pytest.mark.parametrize("message", ["", "   ", "\n"])
    def test_a_blank_warning_says_nothing_and_is_refused(self, message: str) -> None:
        with pytest.raises(ValidationError, match=r"warnings\[1\] is blank"):
            FitStatus(n_observations=3, is_low_data=True, warnings=("fine", message))

    def test_a_warning_that_is_not_text_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            FitStatus(n_observations=3, is_low_data=True, warnings=(1,))  # type: ignore[arg-type]
