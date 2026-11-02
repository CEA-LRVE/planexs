"""Unit tests for the `parameter` registry and for reading a stored parameter."""

from __future__ import annotations

import copy
from typing import ClassVar

import pytest
from pydantic import ValidationError

from planexs import Categorical, Choice, Derived, Range, Real, SearchSpace
from planexs.core.ComponentConfig import ComponentConfig
from planexs.core.ConfigError import ConfigError
from planexs.core.KindTable import KindTable
from planexs.domain.parameters.ChoiceConfig import ChoiceConfig
from planexs.domain.parameters.DerivedConfig import DerivedConfig
from planexs.domain.parameters.parameterFromDict import parameterFromDict
from planexs.domain.parameters.parameterRegistry import parameterRegistry
from planexs.domain.parameters.RangeConfig import RangeConfig

REGISTERED = {
    "range": (Range, RangeConfig),
    "choice": (Choice, ChoiceConfig),
    "derived": (Derived, DerivedConfig),
}


class TestTheRegisteredTypes:
    def test_the_kind_holds_the_three_types_of_the_specification(self) -> None:
        assert parameterRegistry().names() == ["choice", "derived", "range"]

    def test_the_kind_is_in_the_process_wide_table(self) -> None:
        assert "parameter" in KindTable.default().kinds()
        assert KindTable.default().registry("parameter") is parameterRegistry()

    @pytest.mark.parametrize("name", sorted(REGISTERED))
    def test_an_entry_names_its_schema_and_its_builder(self, name: str) -> None:
        cls, schema = REGISTERED[name]
        entry = parameterRegistry().lookup(name)

        assert entry.name == name
        assert entry.config_schema is schema
        assert entry.builder == cls.fromConfig
        assert entry.description.strip() and "\n" not in entry.description

    @pytest.mark.parametrize("name", sorted(REGISTERED))
    def test_a_schema_is_pinned_to_the_name_it_is_registered_under(self, name: str) -> None:
        _, schema = REGISTERED[name]

        assert schema.model_fields["type"].default == name
        assert issubclass(schema, ComponentConfig)
        with pytest.raises(ValidationError):
            schema.model_validate({"type": "other", "name": "x"})

    def test_a_type_declares_no_capability_tag_yet(self) -> None:
        # The tags of the search space (`space:categorical`, ...) are introduced with the
        # optimizers that need them (roadmap P5-11), and the facts of a problem are mapped
        # to needs by P6-05. Until then a parameter type says nothing about capabilities.
        for name in REGISTERED:
            entry = parameterRegistry().lookup(name)
            assert entry.needs == frozenset()
            assert entry.supports == frozenset()

    def test_the_convenience_constructors_are_not_registered(self) -> None:
        for sugar in ("real", "integer", "categorical", "boolean", "fixed"):
            assert sugar not in parameterRegistry()

    @pytest.mark.parametrize(
        ("parameter", "name"),
        [
            (Range("x", (0.0, 1.0)), "range"),
            (Real("x", 0.0, 1.0), "range"),
            (Choice("c", ["a", "b"], parameter_type="str"), "choice"),
            (Categorical("c", ["a", "b"]), "choice"),
            (Derived("d", "a + 1", parameter_type="float"), "derived"),
        ],
    )
    def test_the_tag_a_parameter_writes_is_a_registered_name(
        self, parameter: Range | Choice | Derived, name: str
    ) -> None:
        assert parameter.toDict()["type"] == name
        assert name in parameterRegistry()


class TestTheSchemasStayInStepWithTheClasses:
    """The fields of a schema are declared a second time, next to the class they build.

    That duplication is what lets a stored form carry the field paths of its errors and the
    descriptions a configuration file is read against, so it is guarded here: a field added
    to a class and forgotten in its schema fails these tests, and so does a default that
    differs between the two.
    """

    # The fields of a class that its schema leaves out on purpose.
    NOT_STORED: ClassVar[dict[str, set[str]]] = {
        "range": set(),
        "choice": set(),
        "derived": {"default"},
    }

    @pytest.mark.parametrize("name", sorted(REGISTERED))
    def test_a_schema_has_the_fields_of_its_class(self, name: str) -> None:
        cls, schema = REGISTERED[name]

        stored = set(schema.model_fields) - {"type"}
        assert stored == set(cls.model_fields) - self.NOT_STORED[name]

    @pytest.mark.parametrize("name", sorted(REGISTERED))
    def test_a_shared_field_has_the_same_default_and_the_same_need_to_be_given(
        self, name: str
    ) -> None:
        cls, schema = REGISTERED[name]

        for field in set(schema.model_fields) - {"type"}:
            in_class = cls.model_fields[field]
            in_schema = schema.model_fields[field]
            assert in_schema.is_required() == in_class.is_required(), field
            if not in_class.is_required():
                assert in_schema.default == in_class.default, field

    @pytest.mark.parametrize("name", sorted(REGISTERED))
    def test_every_field_of_a_schema_is_described(self, name: str) -> None:
        _, schema = REGISTERED[name]

        for field, info in schema.model_fields.items():
            assert info.description and info.description.strip(), field


class TestReadingAStoredParameter:
    def test_an_unknown_type_is_an_error_that_lists_the_registered_types(self) -> None:
        with pytest.raises(ConfigError) as caught:
            parameterFromDict({"type": "mystery", "name": "x"})

        error = caught.value
        assert "Unknown parameter type 'mystery'" in error.message
        assert "Registered names: choice, derived, range" in error.message
        assert error.field_path == "type"
        assert error.component == "parameter"

    def test_a_missing_type_is_an_error_that_lists_the_registered_types(self) -> None:
        with pytest.raises(ConfigError, match="no 'type' field") as caught:
            parameterFromDict({"name": "x", "bounds": [0.0, 1.0]})

        assert "Registered names: choice, derived, range" in caught.value.message

    def test_a_document_of_the_old_format_is_refused_and_says_why(self) -> None:
        # Before the registry the tag was written under `kind`. A document of that format is
        # not read (roadmap D-2: no compatibility with the 0.1.0 files), and the error says
        # what is missing instead of failing somewhere deeper.
        with pytest.raises(ConfigError, match="no 'type' field"):
            parameterFromDict({"kind": "range", "name": "x", "bounds": [0.0, 1.0]})

    def test_a_type_that_is_not_a_string_is_refused(self) -> None:
        with pytest.raises(ConfigError, match="must be a string"):
            parameterFromDict({"type": 3, "name": "x"})

    def test_what_is_not_a_mapping_is_refused(self) -> None:
        with pytest.raises(ConfigError, match="must be a mapping"):
            parameterFromDict(["range", "x"])  # type: ignore[arg-type]

    def test_an_unknown_field_is_refused_with_its_path(self) -> None:
        with pytest.raises(ConfigError) as caught:
            parameterFromDict({"type": "range", "name": "x", "bounds": [0.0, 1.0], "step": 2})

        assert caught.value.field_path == "step"
        assert "Unknown field" in caught.value.message

    def test_a_field_of_the_wrong_type_is_refused_with_its_path(self) -> None:
        with pytest.raises(ConfigError) as caught:
            parameterFromDict({"type": "range", "name": "x", "bounds": "zero to one"})

        assert caught.value.field_path == "bounds"

    def test_a_missing_field_is_refused_with_its_path(self) -> None:
        with pytest.raises(ConfigError) as caught:
            parameterFromDict({"type": "choice", "name": "c", "values": ["a", "b"]})

        assert caught.value.field_path == "parameter_type"

    def test_the_schema_of_another_type_does_not_read_the_form(self) -> None:
        with pytest.raises(ConfigError, match="Unknown field"):
            parameterFromDict({"type": "derived", "name": "d", "bounds": [0.0, 1.0]})

    def test_what_a_schema_cannot_say_alone_is_still_refused_by_the_class(self) -> None:
        with pytest.raises(ValidationError, match="inverted"):
            parameterFromDict({"type": "range", "name": "x", "bounds": [1.0, 0.0]})
        with pytest.raises(ValidationError, match="not one of its own values"):
            parameterFromDict(
                {
                    "type": "choice",
                    "name": "c",
                    "values": ["a"],
                    "parameter_type": "str",
                    "default": "z",
                }
            )

    def test_the_mapping_that_is_read_is_not_modified(self) -> None:
        data = {
            "type": "choice",
            "name": "c",
            "values": [1, 2],
            "parameter_type": "int",
            "dependent_parameters": [[1, ["a"]]],
        }
        before = copy.deepcopy(data)

        parameterFromDict(data)

        assert data == before


class TestASearchSpaceReadsThroughTheRegistry:
    def test_an_unknown_type_in_a_stored_space_is_an_error_that_lists_the_registered_types(
        self,
    ) -> None:
        document = '{"parameters": [{"type": "mystery", "name": "x"}], "constraints": []}'

        with pytest.raises(ConfigError, match="Registered names: choice, derived, range"):
            SearchSpace.model_validate_json(document)

    def test_an_unknown_type_given_in_python_is_the_same_error(self) -> None:
        with pytest.raises(ConfigError, match="Unknown parameter type 'mystery'"):
            SearchSpace(parameters=[{"type": "mystery", "name": "x"}])  # type: ignore[list-item]

    def test_a_space_saved_with_the_old_tag_is_refused(self) -> None:
        document = (
            '{"parameters": [{"kind": "range", "name": "x", "bounds": [0, 1]}], '
            '"constraints": []}'
        )

        with pytest.raises(ConfigError, match="no 'type' field"):
            SearchSpace.model_validate_json(document)

    def test_a_dumped_space_tags_its_parameters_with_their_type(self) -> None:
        space = SearchSpace(
            parameters=[Real("a", 0.0, 1.0), Categorical("c", ["u", "v"])],
        )

        dumped = space.model_dump(mode="json")

        assert [parameter["type"] for parameter in dumped["parameters"]] == ["range", "choice"]
        assert all("kind" not in parameter for parameter in dumped["parameters"])
