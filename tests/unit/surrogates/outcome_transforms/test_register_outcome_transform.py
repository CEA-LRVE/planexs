"""Unit tests for `registerOutcomeTransform` (roadmap task P3-12, specification 5.12, 6, 10.6).

A fake transform stands for a plugin's: the contract of the kind is tested on its own, beside
the real transforms. Every test that registers one uses `isolatedOutcomeTransformKind`, because
a registered name is permanent and must never reach the real registry.
"""

from __future__ import annotations

from typing import Any, Literal

import pytest
from botorch.models.transforms.outcome import Standardize
from outcome_transform_fakes import RecordingTransformBuilder, defineFake, fakeConfig
from pydantic import Field

import planexs.surrogates  # noqa: F401
from planexs.core.ComponentConfig import ComponentConfig
from planexs.core.KindTable import KindTable
from planexs.core.RegistryError import RegistryError
from planexs.core.resolveComponent import resolveComponent
from planexs.core.toResolvedConfig import toResolvedConfig
from planexs.surrogates.outcome_transforms.AbstractOutcomeTransformBuilder import (
    AbstractOutcomeTransformBuilder,
)
from planexs.surrogates.outcome_transforms.outcomeTransformRegistry import (
    outcomeTransformRegistry,
)
from planexs.surrogates.outcome_transforms.registerOutcomeTransform import (
    registerOutcomeTransform,
)


class TestRegistering:
    def test_the_entry_holds_everything_the_decorator_was_given(
        self, isolatedOutcomeTransformKind: KindTable
    ) -> None:
        builder = defineFake(
            "winsorize_v1",
            needs=frozenset({"noise:known"}),
            supports=frozenset({"outputs:multi"}),
            description="A transform for tests.",
        )

        entry = outcomeTransformRegistry().lookup("winsorize_v1")

        assert entry.name == "winsorize_v1"
        assert entry.description == "A transform for tests."
        assert entry.needs == {"noise:known"}
        assert entry.supports == {"outputs:multi"}
        assert entry.config_schema.model_fields["type"].default == "winsorize_v1"
        assert entry.builder == builder.fromConfig

    def test_the_class_is_returned_unchanged(
        self, isolatedOutcomeTransformKind: KindTable
    ) -> None:
        builder = defineFake()

        assert issubclass(builder, AbstractOutcomeTransformBuilder)
        assert builder.__name__ == "FakeOutcomeTransformBuilder"

    def test_a_transform_that_declares_no_tag_is_the_normal_case(
        self, isolatedOutcomeTransformKind: KindTable
    ) -> None:
        defineFake()

        entry = outcomeTransformRegistry().lookup("fake_transform_v1")

        assert entry.needs == frozenset() and entry.supports == frozenset()

    def test_a_plugin_transform_lives_next_to_the_built_in_ones(
        self, isolatedOutcomeTransformKind: KindTable
    ) -> None:
        defineFake("a_plugin_v1")

        assert {"standardize_v1", "log_v1", "none_v1", "a_plugin_v1"} <= set(
            outcomeTransformRegistry().names()
        )

    def test_registering_builds_nothing(self, isolatedOutcomeTransformKind: KindTable) -> None:
        # A builder is free of side effects at import and registration time (specification 6).
        builder = defineFake()

        assert builder.__dict__.get("received") is None
        assert outcomeTransformRegistry().lookup("fake_transform_v1").builder == (
            builder.fromConfig
        )


class TestFromAConfigurationToATransform:
    """What a surrogate's configuration does with a transform: name it, and get one."""

    def test_a_mapping_resolves_to_the_entry_and_a_validated_configuration(
        self, isolatedOutcomeTransformKind: KindTable
    ) -> None:
        defineFake()

        resolved = resolveComponent(
            outcomeTransformRegistry(), {"type": "fake_transform_v1", "scale": 3.0}
        )

        assert resolved.entry.name == "fake_transform_v1"
        assert resolved.config.model_dump() == {"type": "fake_transform_v1", "scale": 3.0}

    def test_the_entry_builder_makes_the_builder_which_makes_the_transform(
        self, isolatedOutcomeTransformKind: KindTable
    ) -> None:
        defineFake()
        resolved = resolveComponent(outcomeTransformRegistry(), {"type": "fake_transform_v1"})

        builder = resolved.entry.builder(resolved.config)
        transform = builder.build(2)

        assert isinstance(builder, RecordingTransformBuilder)
        assert builder.config is resolved.config
        assert isinstance(transform, Standardize)
        assert builder.received == [2]

    def test_the_resolved_configuration_has_every_default_written_out(
        self, isolatedOutcomeTransformKind: KindTable
    ) -> None:
        defineFake()

        resolved = resolveComponent(outcomeTransformRegistry(), {"type": "fake_transform_v1"})

        assert toResolvedConfig(resolved.config) == {"type": "fake_transform_v1", "scale": 1.0}

    def test_an_unknown_field_is_refused_with_its_name(
        self, isolatedOutcomeTransformKind: KindTable
    ) -> None:
        defineFake()

        with pytest.raises(Exception, match="scal"):
            resolveComponent(
                outcomeTransformRegistry(), {"type": "fake_transform_v1", "scal": 3.0}
            )


class TestARegistrationThatIsRefused:
    """Nothing is registered, and the registry is left as it was."""

    @pytest.mark.parametrize(
        "name", ["Standardize", "1standardize", "standardize v1", "", "_s"]
    )
    def test_an_invalid_name(self, isolatedOutcomeTransformKind: KindTable, name: str) -> None:
        before = outcomeTransformRegistry().names()

        with pytest.raises(RegistryError, match="not a valid component name"):
            # The schema is pinned to the invalid name itself: the decorator checks the pin
            # before it asks the registry, and it is the registry that judges the name.
            registerOutcomeTransform(
                name, config=fakeConfig(name), description="A transform."
            )(RecordingTransformBuilder)

        assert outcomeTransformRegistry().names() == before

    def test_a_name_that_is_taken_is_never_replaced(
        self, isolatedOutcomeTransformKind: KindTable
    ) -> None:
        taken = outcomeTransformRegistry().lookup("standardize_v1")

        with pytest.raises(RegistryError, match="already registered"):
            defineFake("standardize_v1")

        assert outcomeTransformRegistry().lookup("standardize_v1") is taken

    def test_a_class_that_is_not_an_outcome_transform(
        self, isolatedOutcomeTransformKind: KindTable
    ) -> None:
        class NotATransform:
            pass

        with pytest.raises(
            RegistryError, match="not a subclass of AbstractOutcomeTransformBuilder"
        ):
            registerOutcomeTransform(
                "fake_transform_v1",
                config=fakeConfig("fake_transform_v1"),
                description="A transform.",
            )(NotATransform)  # type: ignore[type-var]

        assert "fake_transform_v1" not in outcomeTransformRegistry()

    def test_a_schema_whose_type_is_not_pinned_to_the_name(
        self, isolatedOutcomeTransformKind: KindTable
    ) -> None:
        class WrongType(ComponentConfig):
            type: Literal["something_else_v1"] = Field(
                default="something_else_v1", description="The registered name."
            )

        with pytest.raises(RegistryError, match="pinned to the registered name"):
            registerOutcomeTransform(
                "fake_transform_v1", config=WrongType, description="A transform."
            )(RecordingTransformBuilder)

        assert "fake_transform_v1" not in outcomeTransformRegistry()

    @pytest.mark.parametrize("description", ["", "   ", "two\nlines"])
    def test_a_description_that_is_not_one_line(
        self, isolatedOutcomeTransformKind: KindTable, description: str
    ) -> None:
        with pytest.raises(RegistryError, match="no usable description"):
            defineFake(description=description)

        assert "fake_transform_v1" not in outcomeTransformRegistry()

    def test_a_tag_that_is_not_in_the_vocabulary(
        self, isolatedOutcomeTransformKind: KindTable
    ) -> None:
        with pytest.raises(RegistryError, match="Unknown capability tag"):
            defineFake(supports=frozenset({"transform:made_up"}))

        assert "fake_transform_v1" not in outcomeTransformRegistry()

    def test_a_needs_that_is_a_single_string_and_not_a_collection(
        self, isolatedOutcomeTransformKind: KindTable
    ) -> None:
        bad: Any = "noise:known"

        with pytest.raises(RegistryError, match="not a collection of tag strings"):
            defineFake(needs=bad)

        assert "fake_transform_v1" not in outcomeTransformRegistry()
