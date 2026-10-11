"""Unit tests for `Range`, `Real`, and `Integer`."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from planexs import Integer, Real
from planexs.domain.parameters.Range import Range


class TestRange:
    def test_bounds_and_type_are_stored(self) -> None:
        parameter = Range("x", (0.0, 1.0), parameter_type="float")
        assert parameter.bounds == (0.0, 1.0)
        assert parameter.parameter_type == "float"

    def test_inverted_bounds_are_rejected(self) -> None:
        with pytest.raises(ValidationError, match="strictly less than"):
            Range("x", (1.0, 0.0))

    def test_log_scaling_requires_strictly_positive_lower_bound(self) -> None:
        with pytest.raises(ValidationError, match="log scaling"):
            Range("x", (0.0, 10.0), scaling="log")

    def test_default_outside_bounds_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="falls outside"):
            Range("x", (0.0, 1.0), default=5.0)

    def test_default_on_step_grid_is_accepted(self) -> None:
        parameter = Range("x", (0.0, 10.0), step_size=2.5, default=5.0)
        assert parameter.default == 5.0

    def test_default_off_step_grid_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="step_size"):
            Range("x", (0.0, 10.0), step_size=2.5, default=4.0)

    def test_to_ax_kwargs_excludes_default(self) -> None:
        parameter = Range("x", (0.0, 1.0), default=0.5)
        kwargs = parameter.toAxKwargs()
        assert "default" not in kwargs
        assert kwargs == {
            "name": "x",
            "bounds": (0.0, 1.0),
            "parameter_type": "float",
            "step_size": None,
            "scaling": None,
        }


class TestReal:
    def test_is_sugar_for_a_float_range(self) -> None:
        parameter = Real("temperature", 20.0, 120.0)
        assert isinstance(parameter, Range)
        assert parameter.bounds == (20.0, 120.0)
        assert parameter.parameter_type == "float"


class TestInteger:
    def test_is_sugar_for_an_int_range(self) -> None:
        parameter = Integer("num_layers", 1, 8)
        assert isinstance(parameter, Range)
        assert parameter.bounds == (1.0, 8.0)
        assert parameter.parameter_type == "int"

    def test_step_size_is_forwarded(self) -> None:
        parameter = Integer("n", 0, 100, step_size=5)
        assert parameter.step_size == 5.0
