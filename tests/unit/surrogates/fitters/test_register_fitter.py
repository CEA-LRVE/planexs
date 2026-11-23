"""Unit tests for `registerFitter` (roadmap task P3-13, specification 5.12, 6, 10.6).

A fake fitter stands for a plugin's: the contract of the kind is tested on its own, beside the
real fitter. Every test that registers one uses `isolatedFitterKind`, because a registered name
is permanent and must never reach the real registry.
"""

from __future__ import annotations

from typing import Any, Literal

import pytest
from fitter_fakes import RecordingFitter, defineFake, fakeConfig
from pydantic import Field

import planexs.surrogates  # noqa: F401
from planexs.core.ComponentConfig import ComponentConfig
from planexs.core.KindTable import KindTable
from planexs.core.RegistryError import RegistryError
from planexs.core.resolveComponent import resolveComponent
from planexs.core.toResolvedConfig import toResolvedConfig
from planexs.surrogates.fitters.AbstractFitter import AbstractFitter
from planexs.surrogates.fitters.fitterRegistry import fitterRegistry
from planexs.surrogates.fitters.registerFitter import registerFitter


class TestRegistering:
    def test_the_entry_holds_everything_the_decorator_was_given(
        self, isolatedFitterKind: KindTable
    ) -> None:
        fitter = defineFake(
            "adam_v1",
            needs=frozenset({"posterior:gaussian"}),
            supports=frozenset({"noise:known"}),
            description="A fitter for tests.",
        )

        entry = fitterRegistry().lookup("adam_v1")

        assert entry.name == "adam_v1"
        assert entry.description == "A fitter for tests."
        assert entry.needs == {"posterior:gaussian"}
        assert entry.supports == {"noise:known"}
        assert entry.config_schema.model_fields["type"].default == "adam_v1"
        assert entry.builder == fitter.fromConfig

    def test_the_class_is_returned_unchanged(self, isolatedFitterKind: KindTable) -> None:
        fitter = defineFake()

        assert issubclass(fitter, AbstractFitter)
        assert fitter.__name__ == "FakeFitter"

    def test_a_fitter_that_declares_no_tag_is_allowed(
        self, isolatedFitterKind: KindTable
    ) -> None:
        defineFake()

        entry = fitterRegistry().lookup("fake_fitter_v1")

        assert entry.needs == frozenset() and entry.supports == frozenset()

    def test_a_plugin_fitter_lives_next_to_the_built_in_one(
        self, isolatedFitterKind: KindTable
    ) -> None:
        defineFake("a_plugin_v1")

        assert {"map_v1", "a_plugin_v1"} <= set(fitterRegistry().names())

    def test_registering_fits_nothing(self, isolatedFitterKind: KindTable) -> None:
        # A fitter is free of side effects at import and registration time (specification 6).
        fitter = defineFake()

        assert fitter.__dict__.get("received") is None
        assert fitterRegistry().lookup("fake_fitter_v1").builder == fitter.fromConfig


class TestFromAConfigurationToAFitter:
    """What a surrogate's configuration does with a fitter: name it, and get one."""

    def test_a_mapping_resolves_to_the_entry_and_a_validated_configuration(
        self, isolatedFitterKind: KindTable
    ) -> None:
        defineFake()

        resolved = resolveComponent(fitterRegistry(), {"type": "fake_fitter_v1", "scale": 3.0})

        assert resolved.entry.name == "fake_fitter_v1"
        assert resolved.config.model_dump() == {"type": "fake_fitter_v1", "scale": 3.0}

    def test_the_entry_builder_makes_the_fitter(self, isolatedFitterKind: KindTable) -> None:
        defineFake()
        resolved = resolveComponent(fitterRegistry(), {"type": "fake_fitter_v1"})

        fitter = resolved.entry.builder(resolved.config)

        assert isinstance(fitter, RecordingFitter)
        assert fitter.config is resolved.config

    def test_the_resolved_configuration_has_every_default_written_out(
        self, isolatedFitterKind: KindTable
    ) -> None:
        defineFake()

        resolved = resolveComponent(fitterRegistry(), {"type": "fake_fitter_v1"})

        assert toResolvedConfig(resolved.config) == {"type": "fake_fitter_v1", "scale": 1.0}

    def test_an_unknown_field_is_refused_with_its_name(
        self, isolatedFitterKind: KindTable
    ) -> None:
        defineFake()

        with pytest.raises(Exception, match="scal"):
            resolveComponent(fitterRegistry(), {"type": "fake_fitter_v1", "scal": 3.0})


class TestARegistrationThatIsRefused:
    """Nothing is registered, and the registry is left as it was."""

    @pytest.mark.parametrize("name", ["Map", "1map", "map v1", "", "_map"])
    def test_an_invalid_name(self, isolatedFitterKind: KindTable, name: str) -> None:
        before = fitterRegistry().names()

        with pytest.raises(RegistryError, match="not a valid component name"):
            # The schema is pinned to the invalid name itself: the decorator checks the pin
            # before it asks the registry, and it is the registry that judges the name.
            registerFitter(name, config=fakeConfig(name), description="A fitter.")(
                RecordingFitter
            )

        assert fitterRegistry().names() == before

    def test_a_name_that_is_taken_is_never_replaced(
        self, isolatedFitterKind: KindTable
    ) -> None:
        taken = fitterRegistry().lookup("map_v1")

        with pytest.raises(RegistryError, match="already registered"):
            defineFake("map_v1")

        assert fitterRegistry().lookup("map_v1") is taken

    def test_a_class_that_is_not_a_fitter(self, isolatedFitterKind: KindTable) -> None:
        class NotAFitter:
            pass

        with pytest.raises(RegistryError, match="not a subclass of AbstractFitter"):
            registerFitter(
                "fake_fitter_v1", config=fakeConfig("fake_fitter_v1"), description="A fitter."
            )(NotAFitter)  # type: ignore[type-var]

        assert "fake_fitter_v1" not in fitterRegistry()

    def test_a_schema_whose_type_is_not_pinned_to_the_name(
        self, isolatedFitterKind: KindTable
    ) -> None:
        class WrongType(ComponentConfig):
            type: Literal["something_else_v1"] = Field(
                default="something_else_v1", description="The registered name."
            )

        with pytest.raises(RegistryError, match="pinned to the registered name"):
            registerFitter("fake_fitter_v1", config=WrongType, description="A fitter.")(
                RecordingFitter
            )

        assert "fake_fitter_v1" not in fitterRegistry()

    @pytest.mark.parametrize("description", ["", "   ", "two\nlines"])
    def test_a_description_that_is_not_one_line(
        self, isolatedFitterKind: KindTable, description: str
    ) -> None:
        with pytest.raises(RegistryError, match="no usable description"):
            defineFake(description=description)

        assert "fake_fitter_v1" not in fitterRegistry()

    def test_a_tag_that_is_not_in_the_vocabulary(self, isolatedFitterKind: KindTable) -> None:
        with pytest.raises(RegistryError, match="Unknown capability tag"):
            defineFake(needs=frozenset({"fitter:made_up"}))

        assert "fake_fitter_v1" not in fitterRegistry()

    def test_a_needs_that_is_a_single_string_and_not_a_collection(
        self, isolatedFitterKind: KindTable
    ) -> None:
        bad: Any = "posterior:gaussian"

        with pytest.raises(RegistryError, match="not a collection of tag strings"):
            defineFake(needs=bad)

        assert "fake_fitter_v1" not in fitterRegistry()
