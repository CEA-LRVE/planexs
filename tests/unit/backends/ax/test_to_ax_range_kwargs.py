"""Unit tests for `toAxRangeKwargs`.

Pure dictionary mapping over planexs' own `Range`: does not import `ax.*`, so these tests
run without a real Ax install.
"""

from __future__ import annotations

from planexs import Integer, Real
from planexs.backends.ax.toAxRangeKwargs import toAxRangeKwargs
from planexs.domain.parameters.Range import Range


class TestToAxRangeKwargs:
    def test_excludes_default(self) -> None:
        parameter = Range("x", (0.0, 1.0), default=0.5)
        kwargs = toAxRangeKwargs(parameter)
        assert "default" not in kwargs
        assert kwargs == {
            "name": "x",
            "bounds": (0.0, 1.0),
            "parameter_type": "float",
            "step_size": None,
            "scaling": None,
        }

    def test_step_size_scaling_and_integer_type_are_passed_through(self) -> None:
        parameter = Integer("n", 1, 64, step_size=1, scaling="log")
        assert toAxRangeKwargs(parameter) == {
            "name": "n",
            "bounds": (1.0, 64.0),
            "parameter_type": "int",
            "step_size": 1.0,
            "scaling": "log",
        }

    def test_a_sugar_subclass_maps_like_its_base_class(self) -> None:
        sugar = Real("x", 0.0, 1.0)
        base = Range("x", (0.0, 1.0), parameter_type="float")
        assert toAxRangeKwargs(sugar) == toAxRangeKwargs(base)
