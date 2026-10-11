"""Unit tests for `Fixed` and `Derived`."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from planexs import Derived, Fixed
from planexs.domain.parameters.Choice import Choice


class TestFixed:
    def test_infers_bool_type_before_int(self) -> None:
        # bool is a subclass of int in Python: this would misclassify as
        # "int" if the isinstance checks were not ordered/narrowed correctly.
        parameter = Fixed("debug", True)
        assert parameter.parameter_type == "bool"
        assert parameter.values == [True]

    def test_infers_int_type(self) -> None:
        parameter = Fixed("batch_size", 32)
        assert parameter.parameter_type == "int"

    def test_infers_float_type(self) -> None:
        parameter = Fixed("tolerance", 1e-6)
        assert parameter.parameter_type == "float"

    def test_infers_str_type(self) -> None:
        parameter = Fixed("device", "cpu")
        assert parameter.parameter_type == "str"

    def test_default_equals_value(self) -> None:
        parameter = Fixed("device", "cpu")
        assert parameter.default == "cpu"

    def test_is_a_single_value_choice(self) -> None:
        parameter = Fixed("device", "cpu")
        assert isinstance(parameter, Choice)
        assert parameter.values == ["cpu"]


class TestDerived:
    def test_stores_expression_and_type(self) -> None:
        parameter = Derived("total", "a + b", parameter_type="float")
        assert parameter.expression == "a + b"
        assert parameter.parameter_type == "float"
        assert parameter.default is None

    def test_empty_expression_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="empty expression"):
            Derived("total", "   ", parameter_type="float")

    def test_to_ax_kwargs_uses_expression_str_field_name(self) -> None:
        parameter = Derived("total", "a + b", parameter_type="float")
        kwargs = parameter.toAxKwargs()
        assert kwargs["expression_str"] == "a + b"
        assert "expression" not in kwargs
