"""Unit tests for `kindRegistry`."""

from __future__ import annotations

import pytest

from planexs.core.kindRegistry import kindRegistry
from planexs.core.KindTable import KindTable
from planexs.core.RegistryError import RegistryError


class TestKindRegistry:
    def test_it_returns_the_registry_of_the_kind_in_the_process_wide_table(
        self, isolatedKindTable: KindTable
    ) -> None:
        registry = isolatedKindTable.define("widget")

        assert kindRegistry("widget") is registry

    def test_it_is_looked_up_at_each_call_and_follows_a_swap_of_the_table(
        self, isolatedKindTable: KindTable, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        first = isolatedKindTable.define("widget")
        other = KindTable(isolatedKindTable.vocabulary)
        second = other.define("widget")

        assert kindRegistry("widget") is first
        monkeypatch.setattr("planexs.core.KindTable._DEFAULT_TABLE", other)
        assert kindRegistry("widget") is second
        assert kindRegistry("widget") is not first

    def test_a_kind_that_is_not_defined_is_an_error_that_lists_the_defined_kinds(
        self, isolatedKindTable: KindTable
    ) -> None:
        isolatedKindTable.define("widget")

        with pytest.raises(RegistryError) as caught:
            kindRegistry("gadget")

        assert "Unknown component kind 'gadget'" in caught.value.message
        assert "Defined kinds: widget." in caught.value.message
