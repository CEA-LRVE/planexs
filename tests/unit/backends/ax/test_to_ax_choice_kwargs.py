"""Unit tests for `toAxChoiceKwargs`.

Pure dictionary mapping over planexs' own `Choice`: does not import `ax.*`, so these tests
run without a real Ax install.
"""

from __future__ import annotations

from planexs import Categorical, Fixed
from planexs.backends.ax.toAxChoiceKwargs import toAxChoiceKwargs
from planexs.domain.parameters.Choice import Choice


class TestToAxChoiceKwargs:
    def test_excludes_default(self) -> None:
        parameter = Choice(
            "solvent", ["water", "ethanol"], parameter_type="str", default="water"
        )
        kwargs = toAxChoiceKwargs(parameter)
        assert "default" not in kwargs
        assert kwargs["values"] == ["water", "ethanol"]

    def test_has_exactly_the_fields_of_the_ax_config(self) -> None:
        kwargs = toAxChoiceKwargs(Choice("c", ["a", "b"], parameter_type="str"))
        assert set(kwargs) == {
            "name",
            "values",
            "parameter_type",
            "is_ordered",
            "dependent_parameters",
        }

    def test_ordering_and_dependent_parameters_are_passed_through(self) -> None:
        parameter = Categorical(
            "model_type", ["mlp", "cnn"], dependent_parameters={"mlp": ["hidden_units"]}
        )
        kwargs = toAxChoiceKwargs(parameter)
        assert kwargs["is_ordered"] is False
        assert kwargs["dependent_parameters"] == {"mlp": ["hidden_units"]}

    def test_a_fixed_parameter_maps_to_a_single_value_choice(self) -> None:
        kwargs = toAxChoiceKwargs(Fixed("device", "cpu"))
        assert kwargs["values"] == ["cpu"]
        assert kwargs["parameter_type"] == "str"
