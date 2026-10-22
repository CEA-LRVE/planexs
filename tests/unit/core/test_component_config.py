"""Unit tests for `ComponentConfig` (specification 5.12, 10.8)."""

from __future__ import annotations

import json
from typing import Any, Literal

import pytest
from pydantic import BaseModel, Field, ValidationError

from planexs.core.ComponentConfig import ComponentConfig


class GpConfig(ComponentConfig):
    """A concrete configuration, shaped like the ones real components will register."""

    type: Literal["gp_v1"] = "gp_v1"
    min_observations: int = Field(default=2, description="Smallest number of observations.")
    bounds: tuple[float, float] = Field(default=(0.0, 1.0), description="A pair of bounds.")


class TestBaseClass:
    def test_it_is_a_pydantic_model(self) -> None:
        assert issubclass(ComponentConfig, BaseModel)

    def test_the_type_is_kept_as_given(self) -> None:
        assert ComponentConfig(type="gp_v1").type == "gp_v1"

    def test_the_type_is_required(self) -> None:
        with pytest.raises(ValidationError) as caught:
            ComponentConfig()  # type: ignore[call-arg]
        (problem,) = caught.value.errors()
        assert problem["type"] == "missing"
        assert problem["loc"] == ("type",)

    @pytest.mark.parametrize("bad", [3, None, ["gp_v1"], 1.5], ids=repr)
    def test_the_type_must_be_a_string(self, bad: Any) -> None:
        with pytest.raises(ValidationError) as caught:
            ComponentConfig(type=bad)
        assert [problem["loc"] for problem in caught.value.errors()] == [("type",)]

    def test_unknown_fields_are_rejected(self) -> None:
        with pytest.raises(ValidationError) as caught:
            ComponentConfig(type="gp_v1", bogus=1)  # type: ignore[call-arg]
        (problem,) = caught.value.errors()
        assert problem["type"] == "extra_forbidden"
        assert problem["loc"] == ("bogus",)

    def test_every_field_has_a_description(self) -> None:
        # Specification 10.8: a description on every field, the `type` field included.
        for name, field in ComponentConfig.model_fields.items():
            assert field.description, f"field {name!r} has no description"
            assert field.description.strip() == field.description

    def test_the_only_field_of_the_base_class_is_the_type(self) -> None:
        assert list(ComponentConfig.model_fields) == ["type"]


class TestSubclass:
    def test_it_inherits_the_rejection_of_unknown_fields(self) -> None:
        with pytest.raises(ValidationError) as caught:
            GpConfig(min_observation=3)  # type: ignore[call-arg]
        (problem,) = caught.value.errors()
        assert problem["type"] == "extra_forbidden"
        assert problem["loc"] == ("min_observation",)

    def test_it_adds_its_own_fields_to_the_type(self) -> None:
        assert list(GpConfig.model_fields) == ["type", "min_observations", "bounds"]

    def test_defaults_are_the_ones_of_the_schema(self) -> None:
        config = GpConfig()
        assert (config.type, config.min_observations, config.bounds) == (
            "gp_v1",
            2,
            (0.0, 1.0),
        )

    def test_a_pinned_type_rejects_another_name(self) -> None:
        with pytest.raises(ValidationError) as caught:
            GpConfig.model_validate({"type": "other_v1"})
        (problem,) = caught.value.errors()
        assert problem["type"] == "literal_error"
        assert problem["loc"] == ("type",)

    def test_a_field_of_the_wrong_type_is_rejected(self) -> None:
        with pytest.raises(ValidationError) as caught:
            GpConfig.model_validate({"type": "gp_v1", "min_observations": "three"})
        assert [problem["loc"] for problem in caught.value.errors()] == [("min_observations",)]

    def test_a_subclass_that_leaves_the_type_alone_still_requires_it(self) -> None:
        class Plain(ComponentConfig):
            level: float = Field(default=0.5, description="A level.")

        with pytest.raises(ValidationError) as caught:
            Plain()  # type: ignore[call-arg]
        assert [problem["loc"] for problem in caught.value.errors()] == [("type",)]
        assert Plain(type="plain_v1").level == 0.5


class TestPythonJsonAndMappingMeaning:
    """Specification 10.8: Python, JSON and YAML (plain data) give the same meaning."""

    def test_a_dump_validates_back_to_an_equal_configuration(self) -> None:
        config = GpConfig(min_observations=5, bounds=(0.25, 0.75))
        assert GpConfig.model_validate(config.model_dump()) == config

    def test_a_json_round_trip_gives_an_equal_configuration(self) -> None:
        config = GpConfig(min_observations=5, bounds=(0.25, 0.75))
        text = json.dumps(config.model_dump(mode="json"))
        # JSON has no tuples: the bounds come back as a list and are read as a tuple.
        assert json.loads(text)["bounds"] == [0.25, 0.75]
        assert GpConfig.model_validate(json.loads(text)) == config

    def test_the_dump_names_the_component(self) -> None:
        assert GpConfig().model_dump()["type"] == "gp_v1"
