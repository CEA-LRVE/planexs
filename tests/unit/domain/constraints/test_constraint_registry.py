"""Unit tests for the `constraint` registry and for reading a stored constraint."""

from __future__ import annotations

import copy
import json
from typing import ClassVar

import pytest
from pydantic import ValidationError

from planexs import (
    Categorical,
    Constraint,
    LinearConstraint,
    NonlinearConstraint,
    Range,
    SearchSpace,
)
from planexs.core.ComponentConfig import ComponentConfig
from planexs.core.ConfigError import ConfigError
from planexs.core.KindTable import KindTable
from planexs.domain.constraints.constraintFromDict import constraintFromDict
from planexs.domain.constraints.constraintRegistry import constraintRegistry
from planexs.domain.constraints.LinearConstraintConfig import LinearConstraintConfig
from planexs.domain.constraints.NonlinearConstraintConfig import NonlinearConstraintConfig

REGISTERED: dict[str, tuple[type[Constraint], type[ComponentConfig]]] = {
    "linear": (LinearConstraint, LinearConstraintConfig),
    "nonlinear": (NonlinearConstraint, NonlinearConstraintConfig),
}


class TestTheRegisteredTypes:
    def test_the_kind_holds_the_two_types_of_the_specification(self) -> None:
        assert constraintRegistry().names() == ["linear", "nonlinear"]

    def test_the_kind_is_in_the_process_wide_table(self) -> None:
        assert "constraint" in KindTable.default().kinds()
        assert KindTable.default().registry("constraint") is constraintRegistry()

    @pytest.mark.parametrize("name", sorted(REGISTERED))
    def test_an_entry_names_its_schema_and_its_builder(self, name: str) -> None:
        cls, schema = REGISTERED[name]
        entry = constraintRegistry().lookup(name)

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
            schema.model_validate({"type": "other"})

    def test_a_type_declares_no_capability_tag_yet(self) -> None:
        # The tags of the constraints (`constraints:linear`, `constraints:equality`,
        # `constraints:nonlinear`) are introduced with the optimizers that need them (roadmap
        # P5-11), and the facts of a problem are mapped to needs by P6-05. Until then a
        # constraint type says nothing about capabilities.
        for name in REGISTERED:
            entry = constraintRegistry().lookup(name)
            assert entry.needs == frozenset()
            assert entry.supports == frozenset()

    @pytest.mark.parametrize(
        ("constraint", "name"),
        [
            (LinearConstraint({"a": 1.0}, bound=1.0, comparator="<="), "linear"),
            (NonlinearConstraint("a * b", "<=", 5.0), "nonlinear"),
        ],
    )
    def test_the_tag_a_constraint_writes_is_a_registered_name(
        self, constraint: Constraint, name: str
    ) -> None:
        assert constraint.toDict()["type"] == name
        assert name in constraintRegistry()


class TestTheSchemasStayInStepWithTheClasses:
    """The fields of a schema are declared a second time, next to the class they build.

    That duplication is what lets a stored form carry the field paths of its errors and the
    descriptions a configuration file is read against, so it is guarded here: a field added
    to a class and forgotten in its schema fails these tests, and so does a default that
    differs between the two.
    """

    # The fields of a class that its schema leaves out on purpose.
    NOT_STORED: ClassVar[dict[str, set[str]]] = {"linear": set(), "nonlinear": set()}

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


# One example of every shape of every registered type, written so that a lossy round trip
# would show: floats that have no short decimal form, negative and fractional coefficients,
# every comparator, and expressions that use a function, a power and a unary minus.
EXACT_EXAMPLES: list[Constraint] = [
    LinearConstraint({"a": 1.0}, bound=1.0, comparator="<="),
    LinearConstraint({"a": 1.0, "b": 1.0, "c": 1.0}, bound=1.0, comparator="="),
    LinearConstraint({"a": 2.0, "b": -1.0}, bound=5.0, comparator=">="),
    LinearConstraint({"a": 0.1 + 0.2, "b": 1e-9, "c": -1e9}, bound=0.1 + 0.7, comparator="<="),
    LinearConstraint({"b": 1.0, "a": 3.0}, bound=-2.5, comparator=">="),
    NonlinearConstraint("var * x ** z", "<=", 50.0),
    NonlinearConstraint("x * y", ">=", 2.0),
    NonlinearConstraint("sqrt(x) + log(y) - 1.5", "<=", 0.1 + 0.2),
    NonlinearConstraint("-x ** 2 + max(a, b)", ">=", -1e-9),
]


class TestTheRoundTripIsExact:
    @pytest.mark.parametrize(
        "original",
        EXACT_EXAMPLES,
        ids=lambda c: f"{c.toDict()['type']}-{c.toDict()['comparator']}",
    )
    def test_a_constraint_comes_back_equal_and_of_the_same_type(
        self, original: Constraint
    ) -> None:
        reloaded = constraintFromDict(json.loads(json.dumps(original.toDict())))

        assert type(reloaded) is type(original)
        assert reloaded == original
        assert reloaded.toDict() == original.toDict()

    @pytest.mark.parametrize(
        "original",
        EXACT_EXAMPLES,
        ids=lambda c: f"{c.toDict()['type']}-{c.toDict()['comparator']}",
    )
    def test_a_second_round_trip_changes_nothing(self, original: Constraint) -> None:
        once = constraintFromDict(json.loads(json.dumps(original.toDict())))
        twice = constraintFromDict(json.loads(json.dumps(once.toDict())))

        assert json.dumps(twice.toDict(), sort_keys=True) == json.dumps(
            original.toDict(), sort_keys=True
        )

    def test_the_order_of_the_coefficients_survives(self) -> None:
        original = LinearConstraint({"b": 1.0, "a": 3.0}, bound=1.0, comparator="<=")

        reloaded = constraintFromDict(json.loads(json.dumps(original.toDict())))

        assert isinstance(reloaded, LinearConstraint)
        assert list(reloaded.coefficients) == ["b", "a"]

    def test_a_whole_search_space_comes_back_equal(self) -> None:
        space = SearchSpace(
            parameters=[
                Range("a", (0.0, 1.0)),
                Range("b", (0.0, 1.0)),
                Range("c", (0.5, 4.0)),
                Categorical("k", ["u", "v"]),
            ],
            constraints=list(EXACT_EXAMPLES),
        )

        reloaded = SearchSpace.model_validate_json(space.model_dump_json())

        assert reloaded.constraints == space.constraints
        assert [type(c) for c in reloaded.constraints] == [type(c) for c in EXACT_EXAMPLES]


class TestReadingAStoredConstraint:
    def test_an_unknown_type_is_an_error_that_lists_the_registered_types(self) -> None:
        with pytest.raises(ConfigError) as caught:
            constraintFromDict({"type": "quadratic", "expression": "x * y"})

        error = caught.value
        assert "Unknown constraint type 'quadratic'" in error.message
        assert "Registered names: linear, nonlinear" in error.message
        assert error.field_path == "type"
        assert error.component == "constraint"

    def test_a_missing_type_is_an_error_that_lists_the_registered_types(self) -> None:
        with pytest.raises(ConfigError, match="no 'type' field") as caught:
            constraintFromDict({"expression": "x * y", "comparator": "<=", "bound": 1.0})

        assert "Registered names: linear, nonlinear" in caught.value.message

    def test_a_document_of_the_old_format_is_refused_and_says_why(self) -> None:
        # Before the registry the tag was written under `kind`. A document of that format is
        # not read (roadmap D-2: no compatibility with the 0.1.0 files), and the error says
        # what is missing instead of failing somewhere deeper.
        with pytest.raises(ConfigError, match="no 'type' field"):
            constraintFromDict(
                {"kind": "linear", "coefficients": {"a": 1.0}, "bound": 1.0, "comparator": "="}
            )

    def test_a_type_that_is_not_a_string_is_refused(self) -> None:
        with pytest.raises(ConfigError, match="must be a string"):
            constraintFromDict({"type": 3})

    def test_what_is_not_a_mapping_is_refused(self) -> None:
        with pytest.raises(ConfigError, match="must be a mapping"):
            constraintFromDict(["linear"])  # type: ignore[arg-type]

    def test_an_unknown_field_is_refused_with_its_path(self) -> None:
        with pytest.raises(ConfigError) as caught:
            constraintFromDict(
                {
                    "type": "linear",
                    "coefficients": {"a": 1.0},
                    "bound": 1.0,
                    "comparator": "<=",
                    "tolerance": 0.1,
                }
            )

        assert caught.value.field_path == "tolerance"
        assert "Unknown field" in caught.value.message

    def test_a_field_of_the_wrong_type_is_refused_with_its_path(self) -> None:
        with pytest.raises(ConfigError) as caught:
            constraintFromDict(
                {"type": "linear", "coefficients": [1.0], "bound": 1.0, "comparator": "<="}
            )

        assert caught.value.field_path == "coefficients"

    def test_a_comparator_that_the_type_does_not_have_is_refused(self) -> None:
        # An exact nonlinear equality is a set of measure zero: only the linear type has "=".
        with pytest.raises(ConfigError) as caught:
            constraintFromDict(
                {"type": "nonlinear", "expression": "x * y", "comparator": "=", "bound": 1.0}
            )

        assert caught.value.field_path == "comparator"

    def test_a_missing_field_is_refused_with_its_path(self) -> None:
        with pytest.raises(ConfigError) as caught:
            constraintFromDict({"type": "nonlinear", "expression": "x * y", "bound": 1.0})

        assert caught.value.field_path == "comparator"

    def test_the_schema_of_another_type_does_not_read_the_form(self) -> None:
        with pytest.raises(ConfigError, match="Unknown field"):
            constraintFromDict(
                {"type": "linear", "expression": "x * y", "bound": 1.0, "comparator": "<="}
            )

    def test_what_a_schema_cannot_say_alone_is_still_refused_by_the_class(self) -> None:
        with pytest.raises(ValidationError, match="at least one coefficient"):
            constraintFromDict(
                {"type": "linear", "coefficients": {}, "bound": 1.0, "comparator": "<="}
            )
        with pytest.raises(ValidationError, match="must be finite"):
            constraintFromDict(
                {
                    "type": "nonlinear",
                    "expression": "x * y",
                    "comparator": "<=",
                    "bound": float("inf"),
                }
            )
        with pytest.raises(ValidationError, match="not allowed"):
            constraintFromDict(
                {"type": "nonlinear", "expression": "x ^ 2", "comparator": "<=", "bound": 1.0}
            )

    def test_the_mapping_that_is_read_is_not_modified(self) -> None:
        data = {
            "type": "linear",
            "coefficients": {"a": 1.0, "b": 2.0},
            "bound": 3.0,
            "comparator": "<=",
        }
        before = copy.deepcopy(data)

        constraintFromDict(data)

        assert data == before


class TestASearchSpaceReadsThroughTheRegistry:
    PARAMETERS = '[{"type": "range", "name": "x", "bounds": [0.0, 1.0]}]'

    def test_an_unknown_type_in_a_stored_space_is_an_error_that_lists_the_registered_types(
        self,
    ) -> None:
        document = (
            f'{{"parameters": {self.PARAMETERS}, '
            '"constraints": [{"type": "quadratic", "expression": "x * y"}]}'
        )

        with pytest.raises(ConfigError, match="Registered names: linear, nonlinear"):
            SearchSpace.model_validate_json(document)

    def test_an_unknown_type_given_in_python_is_the_same_error(self) -> None:
        with pytest.raises(ConfigError, match="Unknown constraint type 'quadratic'"):
            SearchSpace(
                parameters=[Range("x", (0.0, 1.0))],
                constraints=[{"type": "quadratic"}],  # type: ignore[list-item]
            )

    def test_a_space_saved_with_the_old_tag_is_refused(self) -> None:
        document = (
            f'{{"parameters": {self.PARAMETERS}, "constraints": [{{"kind": "linear", '
            '"coefficients": {"x": 1.0}, "bound": 1.0, "comparator": "<="}]}'
        )

        with pytest.raises(ConfigError, match="no 'type' field"):
            SearchSpace.model_validate_json(document)

    def test_a_dumped_space_tags_its_constraints_with_their_type(self) -> None:
        space = SearchSpace(
            parameters=[Range("x", (0.0, 1.0)), Range("y", (0.0, 1.0))],
            constraints=[
                LinearConstraint({"x": 1.0}, bound=1.0, comparator="<="),
                NonlinearConstraint("x * y", ">=", 0.1),
            ],
        )

        dumped = space.model_dump(mode="json")

        assert [c["type"] for c in dumped["constraints"]] == ["linear", "nonlinear"]
        assert all("kind" not in c for c in dumped["constraints"])
        assert all("kind" not in p for p in dumped["parameters"])
