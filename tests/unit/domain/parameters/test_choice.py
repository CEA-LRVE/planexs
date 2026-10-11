"""Unit tests for `Choice`, `Categorical`, and `Boolean`."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from planexs import Boolean, Categorical
from planexs.domain.parameters.Choice import Choice


class TestChoice:
    def test_values_and_type_are_stored(self) -> None:
        parameter = Choice("solvent", ["water", "ethanol"], parameter_type="str")
        assert parameter.values == ["water", "ethanol"]
        assert parameter.is_ordered is None

    def test_empty_values_are_rejected(self) -> None:
        with pytest.raises(ValidationError, match="empty"):
            Choice("solvent", [], parameter_type="str")

    def test_duplicate_values_are_rejected(self) -> None:
        with pytest.raises(ValidationError, match="duplicate"):
            Choice("solvent", ["water", "water"], parameter_type="str")

    def test_default_not_in_values_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="not one of"):
            Choice("solvent", ["water", "ethanol"], parameter_type="str", default="toluene")

    def test_default_in_values_is_accepted(self) -> None:
        parameter = Choice(
            "solvent", ["water", "ethanol"], parameter_type="str", default="water"
        )
        assert parameter.default == "water"

    def test_dependent_parameters_keyed_on_unknown_value_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="dependent_parameters"):
            Choice(
                "model_type",
                ["mlp", "cnn"],
                parameter_type="str",
                dependent_parameters={"transformer": ["num_heads"]},
            )

    def test_dependent_parameters_keyed_on_known_values_is_accepted(self) -> None:
        parameter = Choice(
            "model_type",
            ["mlp", "cnn"],
            parameter_type="str",
            dependent_parameters={"mlp": ["hidden_units"], "cnn": ["num_filters"]},
        )
        assert parameter.dependent_parameters == {
            "mlp": ["hidden_units"],
            "cnn": ["num_filters"],
        }

    def test_to_ax_kwargs_excludes_default(self) -> None:
        parameter = Choice(
            "solvent", ["water", "ethanol"], parameter_type="str", default="water"
        )
        kwargs = parameter.toAxKwargs()
        assert "default" not in kwargs
        assert kwargs["values"] == ["water", "ethanol"]


class TestCategorical:
    def test_is_sugar_for_an_unordered_str_choice(self) -> None:
        parameter = Categorical("solvent", ["water", "ethanol", "toluene"])
        assert isinstance(parameter, Choice)
        assert parameter.parameter_type == "str"
        assert parameter.is_ordered is False


class TestBoolean:
    def test_is_sugar_for_a_true_false_choice(self) -> None:
        parameter = Boolean("use_catalyst", default=True)
        assert isinstance(parameter, Choice)
        assert parameter.values == [True, False]
        assert parameter.parameter_type == "bool"
        assert parameter.default is True
