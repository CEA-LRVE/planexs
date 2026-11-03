"""Unit tests for `registerConstraint`, with a toy constraint type registered in the tests.

The toy type is an ordering between two parameters (`first <= second`): a type that the two
built-in ones cannot express, defined here the way a plugin would define it (specification
1.3). It is registered in a copy of the domain's kinds (the `isolatedDomainKinds` fixture),
because a registered name is permanent and must not outlive the test that registers it.
"""

from __future__ import annotations

import json
from typing import Any, Literal

import pytest
from pydantic import Field
from pydantic_core import PydanticSerializationError

from planexs import LinearConstraint, Range, SearchSpace
from planexs.core.ComponentConfig import ComponentConfig
from planexs.core.ConfigError import ConfigError
from planexs.core.KindTable import KindTable
from planexs.core.RegistryError import RegistryError
from planexs.domain.constraints.Constraint import Constraint
from planexs.domain.constraints.constraintFromDict import constraintFromDict
from planexs.domain.constraints.constraintRegistry import constraintRegistry
from planexs.domain.constraints.LinearConstraintConfig import LinearConstraintConfig
from planexs.domain.constraints.registerConstraint import registerConstraint

#: The registry of the real table, taken when this module is collected: no test in this file
#: may leave anything in it.
REAL_REGISTRY = constraintRegistry()
REAL_NAMES = ["linear", "nonlinear"]


def defineOrdering(register: bool = True) -> tuple[type[Constraint], type[ComponentConfig]]:
    """Defines the toy type (a schema and a class), and registers it unless told not to.

    It is defined inside a function so that it is registered when the test calls it, with the
    isolated table already swapped in, and not when the module is imported.
    """

    class OrderingConfig(ComponentConfig):
        type: Literal["ordering"] = Field(default="ordering", description="Always 'ordering'.")
        first: str = Field(description="The parameter that must not exceed the second.")
        second: str = Field(description="The parameter that the first must not exceed.")

    class Ordering(Constraint):
        first: str
        second: str

        def __init__(self, first: str, second: str) -> None:
            super().__init__(first=first, second=second)

        def toDict(self) -> dict[str, Any]:
            return {"type": "ordering", "first": self.first, "second": self.second}

    if register:
        registerConstraint(
            "ordering",
            config=OrderingConfig,
            description="One parameter stays below another.",
        )(Ordering)
    return Ordering, OrderingConfig


class TestAToyTypeRegisteredInTheTest:
    def test_it_is_registered_next_to_the_built_in_types(
        self, isolatedDomainKinds: KindTable
    ) -> None:
        defineOrdering()

        assert constraintRegistry().names() == ["linear", "nonlinear", "ordering"]
        assert constraintRegistry() is not REAL_REGISTRY

    def test_it_is_read_from_its_stored_form(self, isolatedDomainKinds: KindTable) -> None:
        ordering, _ = defineOrdering()

        constraint = constraintFromDict({"type": "ordering", "first": "a", "second": "b"})

        assert type(constraint) is ordering
        assert constraint.toDict() == {"type": "ordering", "first": "a", "second": "b"}

    def test_it_round_trips_inside_a_search_space(
        self, isolatedDomainKinds: KindTable
    ) -> None:
        ordering, _ = defineOrdering()
        space = SearchSpace(
            parameters=[Range("a", (0.0, 1.0)), Range("b", (0.0, 1.0))],
            constraints=[
                LinearConstraint({"a": 1.0, "b": 1.0}, bound=1.5, comparator="<="),
                ordering("a", "b"),  # type: ignore[call-arg]
            ],
        )

        document = space.model_dump_json()
        reloaded = SearchSpace.model_validate_json(document)

        assert [c["type"] for c in json.loads(document)["constraints"]] == [
            "linear",
            "ordering",
        ]
        assert reloaded == space
        assert type(reloaded.constraints[1]) is ordering

    def test_a_field_of_the_toy_type_is_refused_with_its_path(
        self, isolatedDomainKinds: KindTable
    ) -> None:
        defineOrdering()

        with pytest.raises(ConfigError) as caught:
            constraintFromDict({"type": "ordering", "first": "a"})

        assert caught.value.field_path == "second"

    def test_it_cannot_be_read_where_it_is_not_registered(
        self, isolatedDomainKinds: KindTable
    ) -> None:
        defineOrdering(register=False)
        document = (
            '{"parameters": [], "constraints": '
            '[{"type": "ordering", "first": "a", "second": "b"}]}'
        )

        with pytest.raises(ConfigError) as caught:
            SearchSpace.model_validate_json(document)

        assert "Unknown constraint type 'ordering'" in caught.value.message
        assert "Registered names: linear, nonlinear." in caught.value.message

    def test_it_does_not_outlive_the_test(self) -> None:
        # Runs after the tests above (same file) and uses no fixture: the real registry must
        # hold exactly the built-in types, whatever the earlier tests registered.
        assert REAL_REGISTRY.names() == REAL_NAMES
        assert constraintRegistry() is REAL_REGISTRY


class TestASpaceIsRefusedWhenItCouldNotBeReadBack:
    def test_a_type_that_nobody_registered_cannot_be_saved(
        self, isolatedDomainKinds: KindTable
    ) -> None:
        ordering, _ = defineOrdering(register=False)
        space = SearchSpace(
            parameters=[Range("a", (0.0, 1.0)), Range("b", (0.0, 1.0))],
            constraints=[ordering("a", "b")],  # type: ignore[call-arg]
        )

        with pytest.raises(PydanticSerializationError, match="Registered names: linear"):
            space.model_dump()
        with pytest.raises(PydanticSerializationError, match="Unknown constraint 'ordering'"):
            space.model_dump_json()

    def test_a_form_without_a_type_entry_cannot_be_saved(
        self, isolatedDomainKinds: KindTable
    ) -> None:
        class Untagged(Constraint):
            def toDict(self) -> dict[str, Any]:
                return {}

        space = SearchSpace(parameters=[Range("a", (0.0, 1.0))], constraints=[Untagged()])

        with pytest.raises(PydanticSerializationError, match="wrote no 'type' entry"):
            space.model_dump()

    def test_the_real_types_are_saved_as_before(self) -> None:
        space = SearchSpace(
            parameters=[Range("a", (0.0, 1.0))],
            constraints=[LinearConstraint({"a": 1.0}, bound=1.0, comparator="<=")],
        )

        assert space.model_dump(mode="json")["constraints"][0]["type"] == "linear"


class TestTheDecorator:
    def test_it_returns_the_class_unchanged_and_registers_its_builder(
        self, isolatedDomainKinds: KindTable
    ) -> None:
        ordering, config = defineOrdering()
        entry = constraintRegistry().lookup("ordering")

        assert entry.builder == ordering.fromConfig
        assert entry.config_schema is config
        assert entry.description == "One parameter stays below another."
        assert entry.needs == frozenset()
        assert entry.supports == frozenset()

    def test_the_default_builder_passes_the_fields_to_the_constructor(
        self, isolatedDomainKinds: KindTable
    ) -> None:
        ordering, config = defineOrdering()

        built = ordering.fromConfig(config.model_validate({"first": "a", "second": "b"}))

        assert built.toDict() == {"type": "ordering", "first": "a", "second": "b"}

    def test_a_class_that_is_not_a_constraint_is_refused(
        self, isolatedDomainKinds: KindTable
    ) -> None:
        _, config = defineOrdering(register=False)

        with pytest.raises(RegistryError, match="not a subclass of Constraint"):
            registerConstraint("ordering", config=config, description="d")(int)  # type: ignore[type-var]

        assert "ordering" not in constraintRegistry()

    def test_a_schema_that_is_not_pinned_to_the_name_is_refused(
        self, isolatedDomainKinds: KindTable
    ) -> None:
        ordering, config = defineOrdering(register=False)

        with pytest.raises(RegistryError) as caught:
            registerConstraint("precedence", config=config, description="d")(ordering)

        assert caught.value.field_path == "type"
        assert "pinned" in caught.value.message
        assert "precedence" not in constraintRegistry()

    def test_a_name_that_is_taken_is_refused(self, isolatedDomainKinds: KindTable) -> None:
        ordering, _ = defineOrdering(register=False)

        with pytest.raises(RegistryError, match="already registered"):
            registerConstraint("linear", config=LinearConstraintConfig, description="d")(
                ordering
            )

        assert constraintRegistry().lookup("linear").builder == LinearConstraint.fromConfig

    def test_a_name_that_is_not_snake_case_is_refused(
        self, isolatedDomainKinds: KindTable
    ) -> None:
        ordering, config = defineOrdering(register=False)

        with pytest.raises(RegistryError):
            registerConstraint("Ordering", config=config, description="d")(ordering)

    def test_a_capability_tag_is_checked_against_the_vocabulary(
        self, isolatedDomainKinds: KindTable
    ) -> None:
        ordering, config = defineOrdering(register=False)

        with pytest.raises(RegistryError, match="constraints:ordering"):
            registerConstraint(
                "ordering",
                config=config,
                description="d",
                needs=frozenset({"constraints:ordering"}),
            )(ordering)

        assert "ordering" not in constraintRegistry()

    def test_a_builder_that_does_not_return_a_constraint_is_caught_when_reading(
        self, isolatedDomainKinds: KindTable
    ) -> None:
        ordering, config = defineOrdering(register=False)

        class Broken(ordering):  # type: ignore[valid-type, misc]
            @classmethod
            def fromConfig(cls, config: ComponentConfig) -> Any:
                return 3

        registerConstraint("ordering", config=config, description="d")(Broken)

        with pytest.raises(RegistryError, match="which is not a Constraint"):
            constraintFromDict({"type": "ordering", "first": "a", "second": "b"})

    def test_what_the_class_cannot_check_alone_is_left_to_it(
        self, isolatedDomainKinds: KindTable
    ) -> None:
        # The schema only says that the fields are strings: the class decides what a value
        # means, and the error of its constructor comes out of the builder as it is.
        ordering, config = defineOrdering(register=False)

        class Strict(ordering):  # type: ignore[valid-type, misc]
            def __init__(self, first: str, second: str) -> None:
                if first == second:
                    raise ValueError("an ordering needs two different parameters")
                super().__init__(first, second)

        registerConstraint("ordering", config=config, description="d")(Strict)

        with pytest.raises(ValueError, match="two different parameters"):
            constraintFromDict({"type": "ordering", "first": "a", "second": "a"})
