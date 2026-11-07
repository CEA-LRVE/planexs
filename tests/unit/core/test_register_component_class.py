"""Unit tests for `registerComponentClass`, on a registry of a kind made for the tests."""

from __future__ import annotations

from operator import attrgetter
from typing import Any, Literal

import pytest
from pydantic import Field

from planexs.core.CapabilityVocabulary import CapabilityVocabulary
from planexs.core.ComponentConfig import ComponentConfig
from planexs.core.ComponentEntry import ComponentEntry
from planexs.core.registerComponentClass import registerComponentClass
from planexs.core.Registry import Registry
from planexs.core.RegistryError import RegistryError


class Widget:
    """The contract class of the test kind."""

    @classmethod
    def fromConfig(cls, config: ComponentConfig) -> Widget:
        return cls()


class Sprocket(Widget):
    """A concrete widget."""


class SprocketConfig(ComponentConfig):
    type: Literal["sprocket"] = Field(default="sprocket", description="Always 'sprocket'.")


class UnpinnedConfig(ComponentConfig):
    type: Literal["gear"] = Field(default="gear", description="Always 'gear'.")


def newRegistry() -> Registry[ComponentEntry]:
    vocabulary = CapabilityVocabulary()
    vocabulary.register("demo:tag", "A tag of the tests.")
    return Registry("widget", vocabulary)


def register(
    registry: Registry[ComponentEntry], cls: Any = Sprocket, **overrides: Any
) -> None:
    arguments: dict[str, Any] = {
        "base": Widget,
        "name": "sprocket",
        "config": SprocketConfig,
        "description": "A sprocket for tests.",
        "needs": frozenset(),
        "supports": frozenset(),
        "builder_of": attrgetter("fromConfig"),
    }
    arguments.update(overrides)
    registerComponentClass(registry, cls, **arguments)


class TestARegistrationThatHonoursTheContract:
    def test_the_entry_holds_what_it_was_given(self) -> None:
        registry = newRegistry()

        register(
            registry,
            needs=frozenset({"demo:tag"}),
            supports=frozenset({"demo:tag"}),
        )

        entry = registry.lookup("sprocket")
        assert entry.name == "sprocket"
        assert entry.config_schema is SprocketConfig
        assert entry.description == "A sprocket for tests."
        assert entry.needs == frozenset({"demo:tag"})
        assert entry.supports == frozenset({"demo:tag"})

    def test_the_builder_is_taken_from_the_class(self) -> None:
        registry = newRegistry()

        register(registry)

        assert registry.lookup("sprocket").builder == Sprocket.fromConfig

    def test_the_builder_is_given_by_the_caller_and_not_by_a_fixed_name(self) -> None:
        class Gizmo(Widget):
            @classmethod
            def make(cls, config: ComponentConfig) -> str:
                return "made"

        registry = newRegistry()

        register(registry, Gizmo, builder_of=attrgetter("make"))

        assert registry.lookup("sprocket").builder == Gizmo.make

    def test_the_kind_is_named_in_the_messages_after_the_registry(self) -> None:
        with pytest.raises(RegistryError) as caught:
            register(newRegistry(), int)

        assert "as the widget type 'sprocket'" in caught.value.message
        assert caught.value.component == "widget 'sprocket'"


class TestARegistrationThatBreaksTheContract:
    def test_a_class_that_does_not_derive_from_the_contract_is_refused(self) -> None:
        registry = newRegistry()

        with pytest.raises(RegistryError, match="not a subclass of Widget") as caught:
            register(registry, int)

        assert "derive the class from Widget" in str(caught.value.suggestion)
        assert registry.names() == []

    def test_what_is_not_a_class_is_refused(self) -> None:
        with pytest.raises(RegistryError, match="not a subclass of Widget"):
            register(newRegistry(), Sprocket())

    def test_a_class_that_has_no_builder_is_refused_as_not_honouring_the_contract(
        self,
    ) -> None:
        # The builder is looked up after the class is checked: `int` is reported as what it
        # is, a class that is not a widget, and not through the error of a missing attribute.
        with pytest.raises(RegistryError, match="not a subclass of Widget"):
            register(newRegistry(), int, builder_of=attrgetter("nothing_of_the_kind"))

    def test_a_schema_that_is_not_pinned_to_the_name_is_refused(self) -> None:
        registry = newRegistry()

        with pytest.raises(RegistryError) as caught:
            register(registry, config=UnpinnedConfig)

        assert caught.value.field_path == "type"
        assert "defaults to 'gear'" in caught.value.message
        assert "Literal['sprocket'] = 'sprocket'" in str(caught.value.suggestion)
        assert registry.names() == []

    def test_a_name_that_is_taken_is_refused_and_leaves_the_first_entry(self) -> None:
        registry = newRegistry()
        register(registry)

        with pytest.raises(RegistryError, match="already registered"):
            register(registry, description="Another sprocket.")

        assert registry.lookup("sprocket").description == "A sprocket for tests."

    def test_a_name_that_is_not_snake_case_is_refused(self) -> None:
        class CapitalConfig(ComponentConfig):
            type: Literal["Sprocket"] = Field(default="Sprocket", description="Always.")

        registry = newRegistry()

        with pytest.raises(RegistryError, match="not a valid component name"):
            register(registry, name="Sprocket", config=CapitalConfig)

        assert registry.names() == []

    def test_a_tag_that_is_not_in_the_vocabulary_is_refused_and_nothing_is_registered(
        self,
    ) -> None:
        registry = newRegistry()

        with pytest.raises(RegistryError, match="demo:unknown"):
            register(registry, needs=frozenset({"demo:unknown"}))

        assert registry.names() == []
