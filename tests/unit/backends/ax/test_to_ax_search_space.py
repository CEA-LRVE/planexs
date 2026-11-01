"""Unit tests for `toAxSearchSpace`.

Requires a real `ax-platform` install (it maps to `ax.api.configs` objects
directly); skipped gracefully if `ax` is not importable, so the rest of the
suite still runs in an environment where the full ML stack has not been
installed yet.
"""

from __future__ import annotations

import pytest

pytest.importorskip("ax")

from ax.api.configs import ChoiceParameterConfig, DerivedParameterConfig, RangeParameterConfig

from planexs import (
    Boolean,
    Categorical,
    Derived,
    Fixed,
    Integer,
    LinearConstraint,
    Real,
    SearchSpace,
)
from planexs.backends.ax.toAxParameterConstraintString import toAxParameterConstraintString
from planexs.backends.ax.toAxSearchSpace import toAxSearchSpace


class TestToAxSearchSpace:
    def test_range_maps_to_range_parameter_config(self) -> None:
        search_space = SearchSpace(parameters=[Real("temperature", 20.0, 120.0)])
        (ax_parameter,) = toAxSearchSpace(search_space)
        assert isinstance(ax_parameter, RangeParameterConfig)
        assert ax_parameter.name == "temperature"
        assert ax_parameter.bounds == (20.0, 120.0)
        assert ax_parameter.parameter_type == "float"

    def test_categorical_maps_to_choice_parameter_config(self) -> None:
        search_space = SearchSpace(parameters=[Categorical("solvent", ["water", "ethanol"])])
        (ax_parameter,) = toAxSearchSpace(search_space)
        assert isinstance(ax_parameter, ChoiceParameterConfig)
        assert ax_parameter.values == ["water", "ethanol"]
        assert ax_parameter.is_ordered is False

    def test_fixed_maps_to_a_single_value_choice_parameter_config(self) -> None:
        search_space = SearchSpace(parameters=[Fixed("device", "cpu")])
        (ax_parameter,) = toAxSearchSpace(search_space)
        assert isinstance(ax_parameter, ChoiceParameterConfig)
        assert ax_parameter.values == ["cpu"]

    def test_derived_maps_to_derived_parameter_config(self) -> None:
        search_space = SearchSpace(
            parameters=[
                Real("a", 0.0, 1.0),
                Real("b", 0.0, 1.0),
                Derived("total", "a + b", parameter_type="float"),
            ]
        )
        ax_parameters = toAxSearchSpace(search_space)
        derived = [p for p in ax_parameters if isinstance(p, DerivedParameterConfig)]
        assert len(derived) == 1
        assert derived[0].expression_str == "a + b"

    def test_dependent_parameters_are_passed_through(self) -> None:
        search_space = SearchSpace(
            parameters=[
                Categorical(
                    "model_type",
                    ["mlp", "cnn"],
                    dependent_parameters={"mlp": ["hidden_units"], "cnn": ["num_filters"]},
                ),
                Real("hidden_units", 8, 512),
                Real("num_filters", 8, 256),
            ]
        )
        (model_type_config, *_rest) = toAxSearchSpace(search_space)
        assert model_type_config.dependent_parameters == {
            "mlp": ["hidden_units"],
            "cnn": ["num_filters"],
        }


class TestReloadedSearchSpace:
    @staticmethod
    def _mixedSpace() -> SearchSpace:
        return SearchSpace(
            parameters=[
                Real("a", 0.0, 1.0, default=0.25),
                Real("b", 0.0, 1.0),
                Integer("n", 1, 8),
                Categorical(
                    "model_type",
                    ["mlp", "cnn"],
                    dependent_parameters={"mlp": ["n"]},
                ),
                Boolean("flag", default=False),
                Derived("total", "a + b", parameter_type="float"),
            ],
            constraints=[LinearConstraint({"a": 1.0, "b": 1.0}, bound=1.0, comparator="<=")],
        )

    def test_reloaded_space_maps_to_the_same_ax_parameters(self) -> None:
        original = self._mixedSpace()
        reloaded = SearchSpace.model_validate_json(original.model_dump_json())

        assert toAxSearchSpace(reloaded) == toAxSearchSpace(original)

    def test_reloaded_space_maps_to_the_same_ax_constraint_strings(self) -> None:
        original = self._mixedSpace()
        reloaded = SearchSpace.model_validate_json(original.model_dump_json())

        def strings(space: SearchSpace) -> list[str]:
            return [
                toAxParameterConstraintString(constraint)
                for constraint in space.constraints
                if isinstance(constraint, LinearConstraint)
            ]

        assert strings(reloaded) == strings(original)
        assert strings(original) == ["a + b <= 1.0"]
