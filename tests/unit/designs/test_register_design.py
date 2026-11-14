"""Unit tests for `registerDesign` (roadmap task P4-06, specification 5.12, 6, 10.6).

A fake design stands for a plugin's: the decorator is tested on its own. Every test that
registers one uses `isolatedDesignKind`, because a registered name is permanent and must never
reach the real registry.

The tags used are those that exist today (`outcome_kind:*`, registered by the domain): the
tags that describe a space or its constraints arrive with the optimizers (roadmap task P5-11).
"""

from __future__ import annotations

from typing import Any, Literal

import pytest
from design_fakes import RecordingDesign, defineFake, fakeConfig, makeContext
from pydantic import Field

import planexs.designs  # noqa: F401
from planexs.core.ComponentConfig import ComponentConfig
from planexs.core.ConfigError import ConfigError
from planexs.core.KindTable import KindTable
from planexs.core.RegistryError import RegistryError
from planexs.core.resolveComponent import resolveComponent
from planexs.core.toResolvedConfig import toResolvedConfig
from planexs.designs.AbstractDesign import AbstractDesign
from planexs.designs.designRegistry import designRegistry
from planexs.designs.registerDesign import registerDesign


class TestRegistering:
    def test_the_entry_holds_everything_the_decorator_was_given(
        self, isolatedDesignKind: KindTable
    ) -> None:
        design = defineFake(
            "plugin_v1",
            needs=frozenset({"outcome_kind:continuous"}),
            supports=frozenset({"outcome_kind:binary"}),
            description="A plugin design for tests.",
        )

        entry = designRegistry().lookup("plugin_v1")

        assert entry.name == "plugin_v1"
        assert entry.description == "A plugin design for tests."
        assert entry.needs == {"outcome_kind:continuous"}
        assert entry.supports == {"outcome_kind:binary"}
        assert entry.config_schema.model_fields["type"].default == "plugin_v1"
        assert entry.builder == design.fromConfig

    def test_the_class_is_returned_unchanged(self, isolatedDesignKind: KindTable) -> None:
        design = defineFake()

        assert issubclass(design, AbstractDesign)
        assert design.__name__ == "FakeDesign"

    def test_a_design_that_declares_no_tag_is_the_normal_case(
        self, isolatedDesignKind: KindTable
    ) -> None:
        # A design proposes points that satisfy whatever constraints it is given: it needs
        # nothing from a study and supports nothing for it.
        defineFake()

        entry = designRegistry().lookup("fake_design_v1")

        assert entry.needs == frozenset()
        assert entry.supports == frozenset()

    def test_a_plugin_design_lives_next_to_the_built_in_one(
        self, isolatedDesignKind: KindTable
    ) -> None:
        defineFake("a_v1")

        assert designRegistry().names() == ["a_v1", "sobol_v1"]

    def test_registering_builds_nothing(self, isolatedDesignKind: KindTable) -> None:
        # A builder is free of side effects at import and registration time (specification 6).
        calls: list[str] = []

        @registerDesign("lazy_v1", config=fakeConfig("lazy_v1"), description="Lazy.")
        class Lazy(RecordingDesign):
            def __init__(self, config: Any) -> None:
                calls.append("init")
                super().__init__(config)

        assert calls == []

    def test_the_entry_passes_the_contract_of_specification_6(
        self, isolatedDesignKind: KindTable
    ) -> None:
        defineFake("plugin_v1", supports=frozenset({"outcome_kind:binary"}))
        entry = designRegistry().lookup("plugin_v1")
        vocabulary = isolatedDesignKind.vocabulary

        assert designRegistry().isValidName(entry.name)
        assert issubclass(entry.config_schema, ComponentConfig)
        assert all(
            field.description and field.description.strip()
            for field in entry.config_schema.model_fields.values()
        )
        assert all(tag in vocabulary for tag in entry.needs | entry.supports)
        assert callable(entry.builder)
        assert entry.description.splitlines() == [entry.description]


class TestFromAConfigurationToAPoint:
    """The path that a study will take: a mapping, the registry, a design, a context."""

    def test_a_mapping_resolves_to_the_entry_and_a_validated_configuration(
        self, isolatedDesignKind: KindTable
    ) -> None:
        defineFake("plugin_v1")

        resolved = resolveComponent(designRegistry(), {"type": "plugin_v1", "n_points": 9})

        assert resolved.entry.name == "plugin_v1"
        assert resolved.config.model_dump() == {"type": "plugin_v1", "n_points": 9}

    def test_the_entry_builder_makes_the_design_and_the_design_proposes(
        self, isolatedDesignKind: KindTable
    ) -> None:
        defineFake("plugin_v1")
        resolved = resolveComponent(designRegistry(), {"type": "plugin_v1", "n_points": 6})

        design = resolved.entry.builder(resolved.config)
        points = design.propose(makeContext())

        assert isinstance(design, AbstractDesign)
        assert design.config is resolved.config
        assert points.shape[0] == 6

    def test_the_resolved_configuration_has_every_default_written_out(
        self, isolatedDesignKind: KindTable
    ) -> None:
        defineFake("plugin_v1")

        resolved = resolveComponent(designRegistry(), {"type": "plugin_v1"})

        assert toResolvedConfig(resolved.config) == {"type": "plugin_v1", "n_points": 4}

    def test_an_unknown_field_is_refused_with_its_name(
        self, isolatedDesignKind: KindTable
    ) -> None:
        defineFake("plugin_v1")

        with pytest.raises(ConfigError, match="n_point") as caught:
            resolveComponent(designRegistry(), {"type": "plugin_v1", "n_point": 9})

        assert "design 'plugin_v1'" in caught.value.message

    def test_an_unknown_design_lists_the_registered_ones(
        self, isolatedDesignKind: KindTable
    ) -> None:
        defineFake("a_v1")

        with pytest.raises(ConfigError, match="Registered names: a_v1, sobol_v1") as caught:
            resolveComponent(designRegistry(), {"type": "c_v1"})

        assert caught.value.field_path == "type"


class TestARegistrationThatIsRefused:
    def assertNothingWasAdded(self, table: KindTable) -> None:
        assert table.registry("design").names() == ["sobol_v1"]

    @pytest.mark.parametrize(
        "name", ["Upper", "1st_v1", "with space", "trailing_", "", "a__b"]
    )
    def test_an_invalid_name(self, isolatedDesignKind: KindTable, name: str) -> None:
        with pytest.raises(RegistryError, match="not a valid component name"):
            defineFake(name)

        self.assertNothingWasAdded(isolatedDesignKind)

    def test_a_name_that_is_taken_is_never_replaced(
        self, isolatedDesignKind: KindTable
    ) -> None:
        with pytest.raises(RegistryError, match="already registered") as caught:
            defineFake("sobol_v1", description="An imposter.")

        assert "permanent identifier" in str(caught.value)
        assert designRegistry().lookup("sobol_v1").description != "An imposter."

    def test_a_class_that_is_not_a_design(self, isolatedDesignKind: KindTable) -> None:
        class NotADesign:
            """Not a subclass of the contract."""

        with pytest.raises(RegistryError, match="not a subclass of AbstractDesign") as caught:
            registerDesign("plain_v1", config=fakeConfig("plain_v1"), description="Plain.")(
                NotADesign  # type: ignore[type-var]
            )

        assert caught.value.component == "design 'plain_v1'"
        self.assertNothingWasAdded(isolatedDesignKind)

    def test_a_schema_whose_type_is_not_pinned_to_the_name(
        self, isolatedDesignKind: KindTable
    ) -> None:
        class OtherConfig(ComponentConfig):
            """A schema pinned to another name."""

            type: Literal["other_v1"] = Field(default="other_v1", description="The name.")

        with pytest.raises(RegistryError, match="pinned to the registered name") as caught:
            registerDesign("plugin_v1", config=OtherConfig, description="Mismatch.")(
                RecordingDesign
            )

        assert caught.value.field_path == "type"
        self.assertNothingWasAdded(isolatedDesignKind)

    @pytest.mark.parametrize("description", ["", "   ", "Two\nlines."])
    def test_a_description_that_is_not_one_line(
        self, isolatedDesignKind: KindTable, description: str
    ) -> None:
        with pytest.raises(RegistryError, match="description"):
            defineFake(description=description)

        self.assertNothingWasAdded(isolatedDesignKind)

    def test_a_tag_that_is_not_in_the_vocabulary(self, isolatedDesignKind: KindTable) -> None:
        with pytest.raises(RegistryError, match="Unknown capability tag") as caught:
            defineFake(supports=frozenset({"outcome_kind:continous"}))

        assert caught.value.field_path == "supports"
        assert "outcome_kind:continuous" in caught.value.message
        self.assertNothingWasAdded(isolatedDesignKind)

    def test_a_needs_that_is_a_single_string_and_not_a_collection(
        self, isolatedDesignKind: KindTable
    ) -> None:
        with pytest.raises(RegistryError, match="needs"):
            defineFake(needs="outcome_kind:continuous")  # type: ignore[arg-type]

        self.assertNothingWasAdded(isolatedDesignKind)
