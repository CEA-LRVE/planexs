"""Unit tests for `toAxDerivedKwargs`.

Pure dictionary mapping over planexs' own `Derived`: does not import `ax.*`, so these tests
run without a real Ax install.
"""

from __future__ import annotations

from planexs import Derived
from planexs.backends.ax.toAxDerivedKwargs import toAxDerivedKwargs


class TestToAxDerivedKwargs:
    def test_uses_expression_str_field_name(self) -> None:
        parameter = Derived("total", "a + b", parameter_type="float")
        kwargs = toAxDerivedKwargs(parameter)
        assert kwargs["expression_str"] == "a + b"
        assert "expression" not in kwargs

    def test_has_exactly_the_fields_of_the_ax_config(self) -> None:
        kwargs = toAxDerivedKwargs(Derived("total", "a + b", parameter_type="int"))
        assert kwargs == {"name": "total", "expression_str": "a + b", "parameter_type": "int"}
