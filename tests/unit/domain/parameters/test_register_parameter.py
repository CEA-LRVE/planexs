"""Unit tests for `registerParameter`, with a toy parameter type registered inside the tests.

The toy type is an angle that wraps around: a type that the three built-in ones cannot
express, defined here the way a plugin would define it (specification 1.3). It is registered
in a copy of the `parameter` kind (the `isolatedParameterTable` fixture), because a registered
name is permanent and must not outlive the test that registers it.
"""

from __future__ import annotations

import json
from typing import Any, Literal

import pytest
from pydantic import Field, ValidationError
from pydantic_core import PydanticSerializationError

from planexs import Choice, Range, SearchSpace
from planexs.core.ComponentConfig import ComponentConfig
from planexs.core.ConfigError import ConfigError
from planexs.core.KindTable import KindTable
from planexs.core.RegistryError import RegistryError
from planexs.domain.parameters.Parameter import Parameter
from planexs.domain.parameters.parameterFromDict import parameterFromDict
from planexs.domain.parameters.parameterRegistry import parameterRegistry
from planexs.domain.parameters.RangeConfig import RangeConfig
from planexs.domain.parameters.registerParameter import registerParameter

#: The registry of the real table, taken when this module is collected: no test in this file
#: may leave anything in it.
REAL_REGISTRY = parameterRegistry()
REAL_NAMES = ["choice", "derived", "range"]


def defineCyclic(register: bool = True) -> tuple[type[Parameter], type[ComponentConfig]]:
    """Defines the toy type (a schema and a class), and registers it unless told not to.

    It is defined inside a function so that it is registered when the test calls it, with the
    isolated table already swapped in, and not when the module is imported.
    """

    class CyclicConfig(ComponentConfig):
        type: Literal["cyclic"] = Field(default="cyclic", description="Always 'cyclic'.")
        name: str = Field(description="The parameter's name.")
        period: float = Field(description="The length of one turn, in the unit of the angle.")
        default: float | None = Field(default=None, description="The optional default.")

    class Cyclic(Parameter):
        period: float

        def __init__(self, name: str, period: float, default: float | None = None) -> None:
            super().__init__(name=name, period=period, default=default)

        def _validate_default(self) -> None:
            if self.period <= 0.0:
                raise ValueError(f"Cyclic {self.name!r} needs a positive period.")
            if self.default is not None and not 0.0 <= float(self.default) < self.period:
                raise ValueError(f"Cyclic {self.name!r} has a default outside one turn.")

        def toDict(self) -> dict[str, Any]:
            return {
                "type": "cyclic",
                "name": self.name,
                "period": self.period,
                "default": self.default,
            }

    if register:
        registerParameter(
            "cyclic", config=CyclicConfig, description="An angle that wraps around."
        )(Cyclic)
    return Cyclic, CyclicConfig


class TestAToyTypeRegisteredInTheTest:
    def test_it_is_registered_next_to_the_built_in_types(
        self, isolatedParameterTable: KindTable
    ) -> None:
        defineCyclic()

        assert parameterRegistry().names() == ["choice", "cyclic", "derived", "range"]
        assert parameterRegistry() is not REAL_REGISTRY

    def test_it_is_read_from_its_stored_form(self, isolatedParameterTable: KindTable) -> None:
        cyclic, _ = defineCyclic()

        parameter = parameterFromDict({"type": "cyclic", "name": "angle", "period": 360.0})

        assert type(parameter) is cyclic
        assert parameter.name == "angle"

    def test_it_round_trips_inside_a_search_space(
        self, isolatedParameterTable: KindTable
    ) -> None:
        cyclic, _ = defineCyclic()
        space = SearchSpace(
            parameters=[
                Range("x", (0.0, 1.0)),
                cyclic("angle", period=360.0, default=90.0),  # type: ignore[call-arg]
                Choice("c", ["a", "b"], parameter_type="str"),
            ]
        )

        document = space.model_dump_json()
        reloaded = SearchSpace.model_validate_json(document)

        assert [p["type"] for p in json.loads(document)["parameters"]] == [
            "range",
            "cyclic",
            "choice",
        ]
        assert reloaded == space
        assert type(reloaded.parameters[1]) is cyclic

    def test_a_field_of_the_toy_type_is_refused_with_its_path(
        self, isolatedParameterTable: KindTable
    ) -> None:
        defineCyclic()

        with pytest.raises(ConfigError) as caught:
            parameterFromDict({"type": "cyclic", "name": "angle"})

        assert caught.value.field_path == "period"

    def test_the_class_still_validates_what_the_schema_cannot(
        self, isolatedParameterTable: KindTable
    ) -> None:
        defineCyclic()

        with pytest.raises(ValidationError, match="positive period"):
            parameterFromDict({"type": "cyclic", "name": "angle", "period": -1.0})

    def test_it_cannot_be_read_where_it_is_not_registered(
        self, isolatedParameterTable: KindTable
    ) -> None:
        defineCyclic(register=False)
        document = '{"parameters": [{"type": "cyclic", "name": "a", "period": 360.0}]}'

        with pytest.raises(ConfigError) as caught:
            SearchSpace.model_validate_json(document)

        assert "Unknown parameter type 'cyclic'" in caught.value.message
        assert "Registered names: choice, derived, range." in caught.value.message

    def test_it_does_not_outlive_the_test(self) -> None:
        # Runs after the tests above (same file) and uses no fixture: the real registry must
        # hold exactly the built-in types, whatever the earlier tests registered.
        assert REAL_REGISTRY.names() == REAL_NAMES
        assert parameterRegistry() is REAL_REGISTRY


class TestASpaceIsRefusedWhenItCouldNotBeReadBack:
    def test_a_type_that_nobody_registered_cannot_be_saved(
        self, isolatedParameterTable: KindTable
    ) -> None:
        cyclic, _ = defineCyclic(register=False)
        space = SearchSpace(parameters=[cyclic("angle", period=360.0)])  # type: ignore[call-arg]

        with pytest.raises(
            PydanticSerializationError, match="Registered names: choice, derived"
        ):
            space.model_dump()
        with pytest.raises(PydanticSerializationError, match="Unknown parameter 'cyclic'"):
            space.model_dump_json()

    def test_a_form_without_a_type_entry_cannot_be_saved(
        self, isolatedParameterTable: KindTable
    ) -> None:
        class Untagged(Parameter):
            def _validate_default(self) -> None:
                return None

            def toDict(self) -> dict[str, Any]:
                return {"name": self.name}

        space = SearchSpace(parameters=[Untagged(name="u")])

        with pytest.raises(PydanticSerializationError, match="wrote no 'type' entry"):
            space.model_dump()

    def test_the_real_types_are_saved_as_before(self) -> None:
        space = SearchSpace(parameters=[Range("x", (0.0, 1.0))])

        assert space.model_dump(mode="json")["parameters"][0]["type"] == "range"


class TestTheDecorator:
    def test_it_returns_the_class_unchanged_and_registers_its_builder(
        self, isolatedParameterTable: KindTable
    ) -> None:
        cyclic, config = defineCyclic()
        entry = parameterRegistry().lookup("cyclic")

        assert entry.builder == cyclic.fromConfig
        assert entry.config_schema is config
        assert entry.description == "An angle that wraps around."
        assert entry.needs == frozenset()
        assert entry.supports == frozenset()

    def test_the_default_builder_passes_the_fields_to_the_constructor(
        self, isolatedParameterTable: KindTable
    ) -> None:
        cyclic, config = defineCyclic()

        built = cyclic.fromConfig(config.model_validate({"name": "a", "period": 10.0}))  # type: ignore[attr-defined]

        assert built.toDict() == {
            "type": "cyclic",
            "name": "a",
            "period": 10.0,
            "default": None,
        }

    def test_a_class_that_is_not_a_parameter_is_refused(
        self, isolatedParameterTable: KindTable
    ) -> None:
        _, config = defineCyclic(register=False)

        with pytest.raises(RegistryError, match="not a subclass of Parameter"):
            registerParameter("cyclic", config=config, description="d")(int)  # type: ignore[type-var]

        assert "cyclic" not in parameterRegistry()

    def test_a_schema_that_is_not_pinned_to_the_name_is_refused(
        self, isolatedParameterTable: KindTable
    ) -> None:
        cyclic, config = defineCyclic(register=False)

        with pytest.raises(RegistryError) as caught:
            registerParameter("angle", config=config, description="d")(cyclic)

        assert caught.value.field_path == "type"
        assert "pinned" in caught.value.message
        assert "angle" not in parameterRegistry()

    def test_a_name_that_is_taken_is_refused(self, isolatedParameterTable: KindTable) -> None:
        cyclic, _ = defineCyclic(register=False)

        with pytest.raises(RegistryError, match="already registered"):
            registerParameter("range", config=RangeConfig, description="d")(cyclic)

        assert parameterRegistry().lookup("range").builder == Range.fromConfig

    def test_a_name_that_is_not_snake_case_is_refused(
        self, isolatedParameterTable: KindTable
    ) -> None:
        cyclic, config = defineCyclic(register=False)

        with pytest.raises(RegistryError):
            registerParameter("Cyclic", config=config, description="d")(cyclic)

    def test_a_capability_tag_is_checked_against_the_vocabulary(
        self, isolatedParameterTable: KindTable
    ) -> None:
        cyclic, config = defineCyclic(register=False)

        with pytest.raises(RegistryError, match="space:cyclic"):
            registerParameter(
                "cyclic",
                config=config,
                description="d",
                needs=frozenset({"space:cyclic"}),
            )(cyclic)

        assert "cyclic" not in parameterRegistry()

    def test_a_builder_that_does_not_return_a_parameter_is_caught_when_reading(
        self, isolatedParameterTable: KindTable
    ) -> None:
        cyclic, config = defineCyclic(register=False)

        class Broken(cyclic):  # type: ignore[valid-type, misc]
            @classmethod
            def fromConfig(cls, config: ComponentConfig) -> Any:
                return 3

        registerParameter("cyclic", config=config, description="d")(Broken)

        with pytest.raises(RegistryError, match="which is not a Parameter"):
            parameterFromDict({"type": "cyclic", "name": "a", "period": 1.0})
