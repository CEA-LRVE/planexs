"""Unit tests for `selectComponent`: which component does a configuration mapping name?"""

from __future__ import annotations

from collections.abc import Callable
from types import MappingProxyType
from typing import Any

import pytest
from nested_config_schemas import MaternConfig, RbfConfig

from planexs.core.ComponentConfig import ComponentConfig
from planexs.core.ComponentEntry import ComponentEntry
from planexs.core.ConfigError import ConfigError
from planexs.core.KindTable import KindTable
from planexs.core.Registry import Registry
from planexs.core.RegistryError import RegistryError
from planexs.core.selectComponent import selectComponent


@pytest.fixture
def kernels(nestedKinds: KindTable) -> Registry[ComponentEntry]:
    """The `kernel` registry: `matern_v1` and `rbf_v1`."""
    registry: Registry[ComponentEntry] = nestedKinds.registry("kernel")
    return registry


def failureOf(registry: Registry[Any], data: Any) -> ConfigError:
    """The `ConfigError` that selecting a component for `data` raises."""
    with pytest.raises(ConfigError) as caught:
        selectComponent(registry, data)
    return caught.value


class TestSelection:
    def test_it_returns_the_registered_entry_and_its_schema(
        self, kernels: Registry[ComponentEntry]
    ) -> None:
        entry, schema = selectComponent(kernels, {"type": "matern_v1"})
        assert entry is kernels.lookup("matern_v1")
        assert schema is MaternConfig
        assert schema is entry.config_schema

    def test_each_name_selects_its_own_component(
        self, kernels: Registry[ComponentEntry]
    ) -> None:
        assert selectComponent(kernels, {"type": "rbf_v1"})[1] is RbfConfig

    def test_the_schema_is_a_component_config(self, kernels: Registry[ComponentEntry]) -> None:
        assert issubclass(selectComponent(kernels, {"type": "rbf_v1"})[1], ComponentConfig)

    def test_only_the_type_is_looked_at(self, kernels: Registry[ComponentEntry]) -> None:
        # Selecting decides which component the mapping is for: the settings are validated
        # later, against the schema that was selected.
        data = {"type": "matern_v1", "nu": "not a number", "bogus": 1}
        assert selectComponent(kernels, data)[1] is MaternConfig

    def test_any_mapping_is_accepted_and_none_is_modified(
        self, kernels: Registry[ComponentEntry]
    ) -> None:
        data = {"type": "rbf_v1", "ard": True}
        selectComponent(kernels, MappingProxyType(data))
        assert data == {"type": "rbf_v1", "ard": True}


class TestFailures:
    def test_a_value_that_is_not_a_mapping(self, kernels: Registry[ComponentEntry]) -> None:
        error = failureOf(kernels, ["matern_v1"])
        assert error.message == (
            "A kernel configuration must be a mapping with a 'type' key; got list."
        )
        assert error.component == "kernel"
        assert error.field_path is None

    def test_a_key_that_is_not_a_string(self, kernels: Registry[ComponentEntry]) -> None:
        error = failureOf(kernels, {"type": "rbf_v1", 1: 2})
        assert "must have string keys; got the key 1 (int)" in error.message
        assert error.field_path is None

    def test_a_missing_type(self, kernels: Registry[ComponentEntry]) -> None:
        error = failureOf(kernels, {"nu": 1.0})
        assert "has no 'type' field" in error.message
        assert "Registered names: matern_v1, rbf_v1." in error.message
        assert error.field_path == "type"
        assert error.component == "kernel"

    @pytest.mark.parametrize("given", [3, None, ["rbf_v1"], True], ids=repr)
    def test_a_type_that_is_not_a_string(
        self, kernels: Registry[ComponentEntry], given: Any
    ) -> None:
        error = failureOf(kernels, {"type": given})
        assert "must be a string naming a registered kernel" in error.message
        assert error.field_path == "type"

    def test_an_unknown_type_lists_the_registered_names(
        self, kernels: Registry[ComponentEntry]
    ) -> None:
        error = failureOf(kernels, {"type": "matren_v1"})
        assert error.message == (
            "Unknown kernel type 'matren_v1'. Registered names: matern_v1, rbf_v1."
        )
        assert error.field_path == "type"
        assert error.suggestion is not None

    def test_a_name_is_matched_exactly(self, kernels: Registry[ComponentEntry]) -> None:
        assert failureOf(kernels, {"type": "Matern_v1"}).field_path == "type"
        assert failureOf(kernels, {"type": " matern_v1"}).field_path == "type"

    def test_a_schema_that_is_not_a_component_config_is_a_registry_error(
        self, nestedKinds: KindTable, makeEntry: Callable[..., ComponentEntry]
    ) -> None:
        kernels = nestedKinds.registry("kernel")
        kernels.register(makeEntry("plain_v1"))
        with pytest.raises(RegistryError) as caught:
            selectComponent(kernels, {"type": "plain_v1"})
        assert caught.value.component == "kernel 'plain_v1'"
        assert caught.value.field_path == "config_schema"
        assert "DummyConfig" in caught.value.message
