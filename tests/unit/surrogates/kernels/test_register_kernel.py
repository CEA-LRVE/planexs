"""Unit tests for `registerKernel` (roadmap task P3-10, specification 5.12, 6, 10.6).

A fake kernel stands for a plugin's: the contract of the kind is tested on its own, beside the
real kernels. Every test that registers one uses `isolatedKernelKind`, because a registered
name is permanent and must never reach the real registry.
"""

from __future__ import annotations

from typing import Any, Literal

import pytest
from gpytorch.kernels import LinearKernel
from kernel_fakes import RecordingKernelBuilder, defineFake, fakeConfig
from pydantic import Field

import planexs.surrogates  # noqa: F401
from planexs.core.ComponentConfig import ComponentConfig
from planexs.core.KindTable import KindTable
from planexs.core.RegistryError import RegistryError
from planexs.core.resolveComponent import resolveComponent
from planexs.core.toResolvedConfig import toResolvedConfig
from planexs.surrogates.kernels.AbstractKernelBuilder import AbstractKernelBuilder
from planexs.surrogates.kernels.kernelRegistry import kernelRegistry
from planexs.surrogates.kernels.registerKernel import registerKernel


class TestRegistering:
    def test_the_entry_holds_everything_the_decorator_was_given(
        self, isolatedKernelKind: KindTable
    ) -> None:
        builder = defineFake(
            "periodic_v1",
            needs=frozenset({"noise:known"}),
            supports=frozenset({"outputs:multi"}),
            description="A kernel for tests.",
        )

        entry = kernelRegistry().lookup("periodic_v1")

        assert entry.name == "periodic_v1"
        assert entry.description == "A kernel for tests."
        assert entry.needs == {"noise:known"}
        assert entry.supports == {"outputs:multi"}
        assert entry.config_schema.model_fields["type"].default == "periodic_v1"
        assert entry.builder == builder.fromConfig

    def test_the_class_is_returned_unchanged(self, isolatedKernelKind: KindTable) -> None:
        builder = defineFake()

        assert issubclass(builder, AbstractKernelBuilder)
        assert builder.__name__ == "FakeKernelBuilder"

    def test_a_kernel_that_declares_no_tag_is_the_normal_case(
        self, isolatedKernelKind: KindTable
    ) -> None:
        defineFake()

        entry = kernelRegistry().lookup("fake_kernel_v1")

        assert entry.needs == frozenset() and entry.supports == frozenset()

    def test_a_plugin_kernel_lives_next_to_the_built_in_ones(
        self, isolatedKernelKind: KindTable
    ) -> None:
        defineFake("a_plugin_v1")

        assert {"matern_v1", "rbf_v1", "a_plugin_v1"} <= set(kernelRegistry().names())

    def test_registering_builds_nothing(self, isolatedKernelKind: KindTable) -> None:
        # A builder is free of side effects at import and registration time (specification 6).
        builder = defineFake()

        assert builder.__dict__.get("received") is None
        assert kernelRegistry().lookup("fake_kernel_v1").builder == builder.fromConfig


class TestFromAConfigurationToAKernel:
    """What a surrogate's configuration does with a kernel: name it, and get one."""

    def test_a_mapping_resolves_to_the_entry_and_a_validated_configuration(
        self, isolatedKernelKind: KindTable
    ) -> None:
        defineFake()

        resolved = resolveComponent(kernelRegistry(), {"type": "fake_kernel_v1", "scale": 3.0})

        assert resolved.entry.name == "fake_kernel_v1"
        assert resolved.config.model_dump() == {"type": "fake_kernel_v1", "scale": 3.0}

    def test_the_entry_builder_makes_the_builder_which_makes_the_kernel(
        self, isolatedKernelKind: KindTable
    ) -> None:
        defineFake()
        resolved = resolveComponent(kernelRegistry(), {"type": "fake_kernel_v1"})

        builder = resolved.entry.builder(resolved.config)
        kernel = builder.build(3)

        assert isinstance(builder, RecordingKernelBuilder)
        assert builder.config is resolved.config
        assert isinstance(kernel, LinearKernel)
        assert builder.received == [3]

    def test_the_resolved_configuration_has_every_default_written_out(
        self, isolatedKernelKind: KindTable
    ) -> None:
        defineFake()

        resolved = resolveComponent(kernelRegistry(), {"type": "fake_kernel_v1"})

        assert toResolvedConfig(resolved.config) == {"type": "fake_kernel_v1", "scale": 1.0}

    def test_an_unknown_field_is_refused_with_its_name(
        self, isolatedKernelKind: KindTable
    ) -> None:
        defineFake()

        with pytest.raises(Exception, match="scal"):
            resolveComponent(kernelRegistry(), {"type": "fake_kernel_v1", "scal": 3.0})


class TestARegistrationThatIsRefused:
    """Nothing is registered, and the registry is left as it was."""

    @pytest.mark.parametrize("name", ["Matern", "1matern", "matern v1", "", "_matern"])
    def test_an_invalid_name(self, isolatedKernelKind: KindTable, name: str) -> None:
        before = kernelRegistry().names()

        with pytest.raises(RegistryError, match="not a valid component name"):
            # The schema is pinned to the invalid name itself: the decorator checks the pin
            # before it asks the registry, and it is the registry that judges the name.
            registerKernel(name, config=fakeConfig(name), description="A kernel.")(
                RecordingKernelBuilder
            )

        assert kernelRegistry().names() == before

    def test_a_name_that_is_taken_is_never_replaced(
        self, isolatedKernelKind: KindTable
    ) -> None:
        taken = kernelRegistry().lookup("matern_v1")

        with pytest.raises(RegistryError, match="already registered"):
            defineFake("matern_v1")

        assert kernelRegistry().lookup("matern_v1") is taken

    def test_a_class_that_is_not_a_kernel(self, isolatedKernelKind: KindTable) -> None:
        class NotAKernel:
            pass

        with pytest.raises(RegistryError, match="not a subclass of AbstractKernelBuilder"):
            registerKernel(
                "fake_kernel_v1", config=fakeConfig("fake_kernel_v1"), description="A kernel."
            )(NotAKernel)  # type: ignore[type-var]

        assert "fake_kernel_v1" not in kernelRegistry()

    def test_a_schema_whose_type_is_not_pinned_to_the_name(
        self, isolatedKernelKind: KindTable
    ) -> None:
        class WrongType(ComponentConfig):
            type: Literal["something_else_v1"] = Field(
                default="something_else_v1", description="The registered name."
            )

        with pytest.raises(RegistryError, match="pinned to the registered name"):
            registerKernel("fake_kernel_v1", config=WrongType, description="A kernel.")(
                RecordingKernelBuilder
            )

        assert "fake_kernel_v1" not in kernelRegistry()

    @pytest.mark.parametrize("description", ["", "   ", "two\nlines"])
    def test_a_description_that_is_not_one_line(
        self, isolatedKernelKind: KindTable, description: str
    ) -> None:
        with pytest.raises(RegistryError, match="no usable description"):
            defineFake(description=description)

        assert "fake_kernel_v1" not in kernelRegistry()

    def test_a_tag_that_is_not_in_the_vocabulary(self, isolatedKernelKind: KindTable) -> None:
        with pytest.raises(RegistryError, match="Unknown capability tag"):
            defineFake(supports=frozenset({"kernel:made_up"}))

        assert "fake_kernel_v1" not in kernelRegistry()

    def test_a_needs_that_is_a_single_string_and_not_a_collection(
        self, isolatedKernelKind: KindTable
    ) -> None:
        bad: Any = "noise:known"

        with pytest.raises(RegistryError, match="not a collection of tag strings"):
            defineFake(needs=bad)

        assert "fake_kernel_v1" not in kernelRegistry()
