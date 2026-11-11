"""Unit tests for the tolerances of the parameterization check (P2-10, specification 3.4)."""

from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from planexs.domain.ParameterizationCheckConfig import ParameterizationCheckConfig


class TestDefaults:
    def test_the_defaults_are_those_the_library_uses_elsewhere(self) -> None:
        config = ParameterizationCheckConfig()

        assert config.grid_tolerance == 1e-9  # what `Range` applies to its own default
        assert (
            config.constraint_tolerance == 1e-6
        )  # what `EncodedConstraints.isFeasible` accepts

    def test_every_field_is_described(self) -> None:
        for name, field in ParameterizationCheckConfig.model_fields.items():
            assert field.description and len(field.description) > 40, name


class TestValidation:
    @pytest.mark.parametrize("field", ["grid_tolerance", "constraint_tolerance"])
    @pytest.mark.parametrize("value", [-1e-9, -1.0, math.nan, math.inf, -math.inf])
    def test_a_tolerance_is_a_finite_number_that_is_not_negative(
        self, field: str, value: float
    ) -> None:
        with pytest.raises(ValidationError, match=field):
            ParameterizationCheckConfig(**{field: value})

    @pytest.mark.parametrize("field", ["grid_tolerance", "constraint_tolerance"])
    def test_zero_is_allowed_and_asks_for_exact_values(self, field: str) -> None:
        assert getattr(ParameterizationCheckConfig(**{field: 0.0}), field) == 0.0

    def test_an_unknown_field_is_an_error_not_a_silent_no_op(self) -> None:
        with pytest.raises(ValidationError, match="grid_tolerence"):
            ParameterizationCheckConfig(grid_tolerence=1e-3)  # type: ignore[call-arg]

    def test_it_is_frozen(self) -> None:
        config = ParameterizationCheckConfig()

        with pytest.raises(ValidationError, match="frozen"):
            config.grid_tolerance = 0.5  # type: ignore[misc]


class TestSerialization:
    def test_it_round_trips_through_json(self) -> None:
        config = ParameterizationCheckConfig(grid_tolerance=1e-7, constraint_tolerance=1e-4)

        rebuilt = ParameterizationCheckConfig.model_validate_json(config.model_dump_json())

        assert rebuilt == config
