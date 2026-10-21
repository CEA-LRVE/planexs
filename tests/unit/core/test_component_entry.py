"""Unit tests for `ComponentEntry` (specification 5.12, 6)."""

from __future__ import annotations

import dataclasses
import functools
from typing import Any

import pytest
from pydantic import BaseModel

from planexs.core.ComponentEntry import ComponentEntry
from planexs.core.PlanexsError import PlanexsError
from planexs.core.RegistryError import RegistryError


class Config(BaseModel):
    """A configuration schema."""


class OtherConfig(Config):
    """A subclass of a configuration schema."""


def build() -> None:
    """A builder."""


def make(**overrides: Any) -> ComponentEntry:
    fields: dict[str, Any] = {
        "name": "single_task_gp_v1",
        "builder": build,
        "config_schema": Config,
        "description": "An exact Gaussian process on one task.",
    }
    fields.update(overrides)
    return ComponentEntry(**fields)


BAD_DESCRIPTIONS = [
    pytest.param("", id="empty"),
    pytest.param(" ", id="space"),
    pytest.param("   \t ", id="whitespace"),
    pytest.param("\n", id="newline-only"),
    pytest.param("line one\nline two", id="two-lines"),
    pytest.param("trailing newline\n", id="trailing-newline"),
    pytest.param("\nleading newline", id="leading-newline"),
    pytest.param("carriage\rreturn", id="carriage-return"),
    pytest.param("windows\r\nline", id="crlf"),
    pytest.param("separator\u2028here", id="unicode-line-separator"),
    pytest.param(None, id="none"),
    pytest.param(3, id="int"),
]


class TestFields:
    def test_every_field_is_kept(self) -> None:
        entry = make(needs={"space:categorical"}, supports={"posterior:gaussian"})
        assert entry.name == "single_task_gp_v1"
        assert entry.builder is build
        assert entry.config_schema is Config
        assert entry.description == "An exact Gaussian process on one task."
        assert entry.needs == frozenset({"space:categorical"})
        assert entry.supports == frozenset({"posterior:gaussian"})

    def test_needs_and_supports_default_to_empty(self) -> None:
        entry = make()
        assert entry.needs == frozenset()
        assert entry.supports == frozenset()
        assert isinstance(entry.needs, frozenset)
        assert isinstance(entry.supports, frozenset)

    @pytest.mark.parametrize(
        "given",
        [
            pytest.param(["b:x", "a:y"], id="list"),
            pytest.param(("b:x", "a:y"), id="tuple"),
            pytest.param({"b:x", "a:y"}, id="set"),
            pytest.param(frozenset({"b:x", "a:y"}), id="frozenset"),
            pytest.param({"b:x": 1, "a:y": 2}, id="dict-keys"),
            pytest.param((tag for tag in ["b:x", "a:y"]), id="generator"),
            pytest.param(["b:x", "a:y", "b:x"], id="duplicates-collapse"),
        ],
    )
    def test_any_collection_of_tags_becomes_a_frozen_set(self, given: Any) -> None:
        entry = make(needs=given)
        assert entry.needs == frozenset({"a:y", "b:x"})
        assert isinstance(entry.needs, frozenset)
        assert entry.supports == frozenset()

    def test_a_generator_is_not_consumed_by_the_checks(self) -> None:
        entry = make(supports=(tag for tag in ["a:x", "b:y"]))
        assert entry.supports == frozenset({"a:x", "b:y"})

    def test_the_name_is_not_judged_here(self) -> None:
        # The name rule belongs to `Registry`, which owns the pattern.
        assert make(name="Not A Valid Name").name == "Not A Valid Name"

    def test_a_builder_can_be_a_function_a_class_a_lambda_or_a_partial(self) -> None:
        for builder in (build, Config, lambda: None, functools.partial(build)):
            assert make(builder=builder).builder is builder

    def test_a_schema_can_be_a_subclass_of_a_model(self) -> None:
        assert make(config_schema=OtherConfig).config_schema is OtherConfig


class TestImmutabilityAndEquality:
    @pytest.mark.parametrize("field", ["name", "builder", "config_schema", "description"])
    def test_a_field_cannot_be_reassigned(self, field: str) -> None:
        entry = make()
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(entry, field, "other")

    def test_needs_cannot_be_changed_in_place(self) -> None:
        entry = make(needs={"a:x"})
        with pytest.raises(AttributeError):
            entry.needs.add("b:y")  # type: ignore[attr-defined]

    def test_equal_entries_are_equal_and_hash_alike(self) -> None:
        first, second = make(needs=["a:x"]), make(needs=("a:x",))
        assert first == second
        assert hash(first) == hash(second)
        assert len({first, second}) == 1

    def test_entries_that_differ_are_not_equal(self) -> None:
        assert make(description="One.") != make(description="Two.")
        assert make(needs={"a:x"}) != make(supports={"a:x"})


class TestDescription:
    @pytest.mark.parametrize("description", BAD_DESCRIPTIONS)
    def test_an_entry_without_a_usable_description_is_rejected(self, description: Any) -> None:
        with pytest.raises(RegistryError, match="no usable description"):
            make(description=description)

    def test_the_error_names_the_component_and_the_field(self) -> None:
        with pytest.raises(RegistryError) as caught:
            make(name="matern_v1", description="")
        assert "'matern_v1'" in caught.value.message
        assert caught.value.component == "matern_v1"
        assert caught.value.field_path == "description"
        assert caught.value.suggestion is not None

    def test_a_missing_description_is_rejected(self) -> None:
        with pytest.raises(TypeError, match="description"):
            ComponentEntry(name="a", builder=build, config_schema=Config)  # type: ignore[call-arg]

    @pytest.mark.parametrize(
        "description",
        [
            "One line.",
            "x",
            "Matern kernel (nu = 2.5), ARD optional.",
            "Mat\u00e9rn \u03bd = 2.5",
        ],
    )
    def test_a_single_non_blank_line_is_accepted(self, description: str) -> None:
        assert make(description=description).description == description

    def test_a_rejection_is_a_planexs_error(self) -> None:
        with pytest.raises(PlanexsError):
            make(description="")


class TestBuilder:
    @pytest.mark.parametrize("builder", [3, "builder", None, [build], {"build": build}])
    def test_a_builder_that_is_not_callable_is_rejected(self, builder: Any) -> None:
        with pytest.raises(RegistryError, match="not callable") as caught:
            make(builder=builder)
        assert caught.value.field_path == "builder"
        assert caught.value.component == "single_task_gp_v1"


class TestConfigSchema:
    @pytest.mark.parametrize(
        "schema",
        [
            pytest.param(Config(), id="instance-of-a-model"),
            pytest.param({"type": "x"}, id="dict"),
            pytest.param(None, id="none"),
            pytest.param(int, id="class-that-is-not-a-model"),
            pytest.param(object, id="object"),
            pytest.param("Config", id="string"),
        ],
    )
    def test_a_schema_that_is_not_a_model_class_is_rejected(self, schema: Any) -> None:
        with pytest.raises(RegistryError, match="not a Pydantic model class") as caught:
            make(config_schema=schema)
        assert caught.value.field_path == "config_schema"
        assert caught.value.suggestion is not None


class TestTagCollections:
    @pytest.mark.parametrize("field", ["needs", "supports"])
    @pytest.mark.parametrize(
        "given",
        [
            pytest.param("posterior:gaussian", id="a-bare-string"),
            pytest.param(["a:x", 3], id="a-non-string-element"),
            pytest.param([None], id="none-element"),
            pytest.param(None, id="none"),
            pytest.param(3, id="int"),
            pytest.param([["a:x"]], id="nested-list"),
        ],
    )
    def test_something_that_is_not_a_collection_of_strings_is_rejected(
        self, field: str, given: Any
    ) -> None:
        with pytest.raises(RegistryError, match="not a collection of tag strings") as caught:
            make(**{field: given})
        assert caught.value.field_path == field
        assert caught.value.component == "single_task_gp_v1"

    def test_a_bare_string_is_not_split_into_characters(self) -> None:
        with pytest.raises(RegistryError):
            make(needs="ab")

    def test_the_tags_themselves_are_not_judged_here(self) -> None:
        # Whether a tag is well formed and known is the vocabulary's call, at registration.
        assert make(needs={"Not A Tag"}).needs == frozenset({"Not A Tag"})
