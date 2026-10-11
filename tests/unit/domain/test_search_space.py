"""Unit tests for `SearchSpace`."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from planexs import Boolean, Categorical, Derived, Integer, LinearConstraint, Real, SearchSpace


class TestSearchSpace:
    def test_parameter_names(self) -> None:
        search_space = SearchSpace(parameters=[Real("x", 0.0, 1.0), Real("y", 0.0, 1.0)])
        assert search_space.parameter_names == ["x", "y"]

    def test_duplicate_parameter_names_are_rejected_at_construction(self) -> None:
        with pytest.raises(ValidationError, match="duplicate"):
            SearchSpace(parameters=[Real("x", 0.0, 1.0), Real("x", 2.0, 3.0)])

    def test_add_parameter_appends(self) -> None:
        search_space = SearchSpace(parameters=[Real("x", 0.0, 1.0)])
        search_space.addParameter(Real("y", 0.0, 1.0))
        assert search_space.parameter_names == ["x", "y"]

    def test_add_parameter_rejects_name_collision(self) -> None:
        search_space = SearchSpace(parameters=[Real("x", 0.0, 1.0)])
        with pytest.raises(ValueError, match="already has a parameter"):
            search_space.addParameter(Real("x", 2.0, 3.0))

    def test_add_constraint_appends(self) -> None:
        search_space = SearchSpace(parameters=[Real("x", 0.0, 1.0), Real("y", 0.0, 1.0)])
        constraint = LinearConstraint({"x": 1.0, "y": 1.0}, bound=1.0, comparator="<=")

        search_space.addConstraint(constraint)

        assert search_space.constraints == [constraint]

    def test_constructor_accepts_constraints_directly(self) -> None:
        constraint = LinearConstraint({"x": 1.0}, bound=1.0, comparator="<=")
        search_space = SearchSpace(parameters=[Real("x", 0.0, 1.0)], constraints=[constraint])
        assert search_space.constraints == [constraint]

    def test_no_constraints_defaults_to_empty_list(self) -> None:
        search_space = SearchSpace(parameters=[Real("x", 0.0, 1.0)])
        assert search_space.constraints == []


class TestSearchSpaceSerialization:
    """`parameters`/`constraints` are lists of abstract base types: a naive
    Pydantic dump lost the subclass fields and a reload could not rebuild
    them. These tests pin the explicit, type-tagged round trip.
    """

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

    def test_dump_keeps_subclass_fields(self) -> None:
        dumped = self._mixedSpace().model_dump(mode="json")

        first = dumped["parameters"][0]
        assert first["kind"] == "range"
        assert first["bounds"] == [0.0, 1.0]
        assert dumped["constraints"][0]["kind"] == "linear"

    def test_json_round_trip_preserves_every_parameter(self) -> None:
        original = self._mixedSpace()
        reloaded = SearchSpace.model_validate_json(original.model_dump_json())

        assert reloaded.parameter_names == original.parameter_names
        by_name = {p.name: p for p in reloaded.parameters}
        assert by_name["a"].bounds == (0.0, 1.0)
        assert by_name["a"].default == 0.25
        assert by_name["n"].parameter_type == "int"
        assert by_name["model_type"].dependent_parameters == {"mlp": ["n"]}
        assert by_name["flag"].default is False
        assert by_name["total"].expression == "a + b"

    def test_json_round_trip_preserves_constraints(self) -> None:
        original = self._mixedSpace()
        reloaded = SearchSpace.model_validate_json(original.model_dump_json())

        assert reloaded.constraints == original.constraints

    def test_reloaded_space_maps_to_the_same_ax_strings(self) -> None:
        original = self._mixedSpace()
        reloaded = SearchSpace.model_validate_json(original.model_dump_json())

        assert [c.toAxParameterConstraintString() for c in reloaded.constraints] == [
            c.toAxParameterConstraintString() for c in original.constraints
        ]
        assert [p.toAxKwargs() for p in reloaded.parameters if hasattr(p, "toAxKwargs")] == [
            p.toAxKwargs() for p in original.parameters if hasattr(p, "toAxKwargs")
        ]
