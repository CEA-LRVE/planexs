"""Unit tests for `resolveComponent` and `ResolvedComponent` (specification 5.12, 10.8)."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Any, Literal

import pytest
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from planexs.core.CapabilityVocabulary import CapabilityVocabulary
from planexs.core.ComponentConfig import ComponentConfig
from planexs.core.ComponentEntry import ComponentEntry
from planexs.core.ConfigError import ConfigError
from planexs.core.PlanexsError import PlanexsError
from planexs.core.Registry import Registry
from planexs.core.RegistryError import RegistryError
from planexs.core.resolveComponent import resolveComponent
from planexs.core.ResolvedComponent import ResolvedComponent


class KernelSettings(BaseModel):
    """A nested model, to check the paths of errors below the top level."""

    model_config = ConfigDict(extra="forbid")

    nu: float = 2.5


class GpConfig(ComponentConfig):
    """The configuration of the main test component."""

    type: Literal["single_task_gp_v1"] = "single_task_gp_v1"
    min_observations: int = Field(default=2, description="Smallest number of observations.")
    alpha: float = Field(default=0.5, description="A weight in [0, 1].")
    kernel: KernelSettings = Field(default_factory=KernelSettings, description="The kernel.")
    values: list[int] = Field(default_factory=list, description="Some integers.")
    bounds: tuple[float, float] = Field(default=(0.0, 1.0), description="A pair of bounds.")

    @field_validator("alpha")
    @classmethod
    def _alphaInRange(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"alpha must be in [0, 1]; got {value!r}.")
        return value


class ClassifierConfig(ComponentConfig):
    """The configuration of the second test component, with fields of its own."""

    type: Literal["variational_gp_classifier_v1"] = "variational_gp_classifier_v1"
    n_inducing: int = Field(default=16, description="Number of inducing points.")


class RbfConfig(ComponentConfig):
    """The configuration of a component of another kind (a kernel)."""

    type: Literal["rbf_v1"] = "rbf_v1"


class RequiredConfig(ComponentConfig):
    """A configuration with a field that has no default."""

    type: Literal["needs_a_level_v1"] = "needs_a_level_v1"
    level: float = Field(description="The level, with no default.")


class ConsistentConfig(ComponentConfig):
    """A configuration with a rule across fields (a model-level validator)."""

    type: Literal["ordered_bounds_v1"] = "ordered_bounds_v1"
    low: float = Field(default=0.0, description="Lower bound.")
    high: float = Field(default=1.0, description="Upper bound.")

    @model_validator(mode="after")
    def _ordered(self) -> ConsistentConfig:
        if self.low >= self.high:
            raise ValueError("low must be less than high")
        return self


class PinnedElsewhereConfig(ComponentConfig):
    """A schema whose `type` is pinned to a name other than the one it is registered under."""

    type: Literal["some_other_name_v1"] = "some_other_name_v1"


class PlainSchema(BaseModel):
    """A schema that is a Pydantic model but not a `ComponentConfig`."""

    model_config = ConfigDict(extra="forbid")


def builder() -> None:
    """A builder for test components."""


def makeComponent(name: str, schema: type[BaseModel] = GpConfig) -> ComponentEntry:
    """A complete entry for `name`, with `schema` as its configuration schema."""
    return ComponentEntry(
        name=name, builder=builder, config_schema=schema, description=f"The {name} component."
    )


@pytest.fixture
def registry(vocabulary: CapabilityVocabulary) -> Registry[ComponentEntry]:
    """A `surrogate` registry holding two components."""
    surrogates: Registry[ComponentEntry] = Registry("surrogate", vocabulary)
    surrogates.register(makeComponent("single_task_gp_v1"))
    surrogates.register(makeComponent("variational_gp_classifier_v1", ClassifierConfig))
    return surrogates


def failureOf(registry: Registry[Any], data: Any) -> ConfigError:
    """The `ConfigError` that resolving `data` raises (the test fails if there is none)."""
    with pytest.raises(ConfigError) as caught:
        resolveComponent(registry, data)
    return caught.value


class TestValidInput:
    def test_the_entry_and_the_configuration_come_back(
        self, registry: Registry[ComponentEntry]
    ) -> None:
        resolved = resolveComponent(
            registry, {"type": "single_task_gp_v1", "min_observations": 7, "alpha": 0.25}
        )
        assert isinstance(resolved, ResolvedComponent)
        assert resolved.entry is registry.lookup("single_task_gp_v1")
        assert isinstance(resolved.config, GpConfig)
        assert resolved.config.min_observations == 7
        assert resolved.config.alpha == 0.25

    def test_the_configuration_is_an_instance_of_the_schema_of_the_entry(
        self, registry: Registry[ComponentEntry]
    ) -> None:
        resolved = resolveComponent(registry, {"type": "single_task_gp_v1"})
        assert type(resolved.config) is resolved.entry.config_schema

    def test_the_type_is_part_of_the_validated_configuration(
        self, registry: Registry[ComponentEntry]
    ) -> None:
        resolved = resolveComponent(registry, {"type": "single_task_gp_v1"})
        assert resolved.config.type == "single_task_gp_v1"

    def test_defaults_come_from_the_schema(self, registry: Registry[ComponentEntry]) -> None:
        config = resolveComponent(registry, {"type": "single_task_gp_v1"}).config
        assert isinstance(config, GpConfig)
        assert (config.min_observations, config.alpha) == (2, 0.5)
        assert config.kernel == KernelSettings(nu=2.5)
        assert config.values == []

    def test_nested_values_are_validated_into_nested_models(
        self, registry: Registry[ComponentEntry]
    ) -> None:
        config = resolveComponent(
            registry, {"type": "single_task_gp_v1", "kernel": {"nu": 1.5}, "values": [1, 2]}
        ).config
        assert isinstance(config, GpConfig)
        assert config.kernel == KernelSettings(nu=1.5)
        assert config.values == [1, 2]

    def test_the_second_component_is_found_by_its_own_name(
        self, registry: Registry[ComponentEntry]
    ) -> None:
        resolved = resolveComponent(
            registry, {"type": "variational_gp_classifier_v1", "n_inducing": 8}
        )
        assert resolved.entry.name == "variational_gp_classifier_v1"
        assert isinstance(resolved.config, ClassifierConfig)
        assert resolved.config.n_inducing == 8

    def test_each_component_is_validated_against_its_own_schema(
        self, registry: Registry[ComponentEntry]
    ) -> None:
        # `min_observations` belongs to the first component's schema, not to the second's.
        resolveComponent(registry, {"type": "single_task_gp_v1", "min_observations": 3})
        error = failureOf(
            registry, {"type": "variational_gp_classifier_v1", "min_observations": 3}
        )
        assert error.field_path == "min_observations"
        assert error.suggestion is not None
        assert "n_inducing" in error.suggestion

    def test_a_json_array_is_read_as_a_tuple(self, registry: Registry[ComponentEntry]) -> None:
        # What a configuration file gives for a tuple field (specification 10.8).
        config = resolveComponent(
            registry, {"type": "single_task_gp_v1", "bounds": [0.25, 0.75]}
        ).config
        assert isinstance(config, GpConfig)
        assert config.bounds == (0.25, 0.75)

    def test_any_mapping_is_accepted(self, registry: Registry[ComponentEntry]) -> None:
        data = MappingProxyType({"type": "single_task_gp_v1", "min_observations": 4})
        config = resolveComponent(registry, data).config
        assert isinstance(config, GpConfig)
        assert config.min_observations == 4

    def test_the_input_mapping_is_not_modified(
        self, registry: Registry[ComponentEntry]
    ) -> None:
        data: dict[str, Any] = {"type": "single_task_gp_v1", "kernel": {"nu": 1.5}}
        resolveComponent(registry, data)
        assert data == {"type": "single_task_gp_v1", "kernel": {"nu": 1.5}}

    def test_resolving_again_gives_an_equal_configuration(
        self, registry: Registry[ComponentEntry]
    ) -> None:
        data = {"type": "single_task_gp_v1", "min_observations": 6}
        assert (
            resolveComponent(registry, data).config == resolveComponent(registry, data).config
        )

    def test_the_registry_is_left_as_it_was(self, registry: Registry[ComponentEntry]) -> None:
        before = registry.names()
        resolveComponent(registry, {"type": "single_task_gp_v1"})
        assert registry.names() == before

    def test_an_entry_of_a_subclass_keeps_its_own_type(
        self, vocabulary: CapabilityVocabulary
    ) -> None:
        @dataclasses.dataclass(frozen=True)
        class SurrogateEntry(ComponentEntry):
            outputs: int = 1

        surrogates: Registry[SurrogateEntry] = Registry("surrogate", vocabulary)
        surrogates.register(
            SurrogateEntry(
                name="single_task_gp_v1",
                builder=builder,
                config_schema=GpConfig,
                description="A surrogate.",
                outputs=3,
            )
        )
        resolved = resolveComponent(surrogates, {"type": "single_task_gp_v1"})
        assert isinstance(resolved.entry, SurrogateEntry)
        assert resolved.entry.outputs == 3


class TestMissingType:
    def test_it_is_a_config_error(self, registry: Registry[ComponentEntry]) -> None:
        error = failureOf(registry, {"min_observations": 3})
        assert isinstance(error, PlanexsError)
        assert not isinstance(error, RegistryError)

    def test_it_names_the_registry_and_the_field(
        self, registry: Registry[ComponentEntry]
    ) -> None:
        error = failureOf(registry, {"min_observations": 3})
        assert error.component == "surrogate"
        assert error.field_path == "type"
        assert "surrogate configuration has no 'type' field" in error.message

    def test_it_lists_the_registered_names(self, registry: Registry[ComponentEntry]) -> None:
        error = failureOf(registry, {})
        assert "single_task_gp_v1, variational_gp_classifier_v1" in str(error)

    def test_it_says_when_nothing_is_registered(
        self, vocabulary: CapabilityVocabulary
    ) -> None:
        error = failureOf(Registry("kernel", vocabulary), {})
        assert "Registered names: none." in error.message

    def test_a_type_of_none_is_not_a_missing_type(
        self, registry: Registry[ComponentEntry]
    ) -> None:
        # The key is there: `type: null` in a file is a wrong value, not an absent one.
        error = failureOf(registry, {"type": None})
        assert "must be a string" in error.message


class TestUnknownType:
    def test_it_is_a_config_error_and_not_a_registry_error(
        self, registry: Registry[ComponentEntry]
    ) -> None:
        error = failureOf(registry, {"type": "gp"})
        assert not isinstance(error, RegistryError)

    def test_it_lists_the_registered_names(self, registry: Registry[ComponentEntry]) -> None:
        error = failureOf(registry, {"type": "gp"})
        assert "'gp'" in error.message
        assert "single_task_gp_v1, variational_gp_classifier_v1" in error.message
        assert "single_task_gp_v1, variational_gp_classifier_v1" in str(error)

    def test_it_names_the_registry_and_the_field(
        self, registry: Registry[ComponentEntry]
    ) -> None:
        error = failureOf(registry, {"type": "gp"})
        assert error.component == "surrogate"
        assert error.field_path == "type"
        assert error.suggestion is not None

    def test_a_name_of_another_kind_is_unknown_here(
        self, vocabulary: CapabilityVocabulary, registry: Registry[ComponentEntry]
    ) -> None:
        kernels: Registry[ComponentEntry] = Registry("kernel", vocabulary)
        kernels.register(makeComponent("rbf_v1", RbfConfig))
        assert resolveComponent(kernels, {"type": "rbf_v1"}).entry.name == "rbf_v1"
        error = failureOf(registry, {"type": "rbf_v1"})
        assert error.component == "surrogate"
        assert "rbf_v1" in error.message

    def test_the_kind_of_the_registry_is_what_the_error_names(
        self, vocabulary: CapabilityVocabulary
    ) -> None:
        optimizers: Registry[ComponentEntry] = Registry("acquisition_optimizer", vocabulary)
        error = failureOf(optimizers, {"type": "nope_v1"})
        assert error.component == "acquisition_optimizer"
        assert "Unknown acquisition_optimizer type 'nope_v1'." in error.message
        assert "Registered names: none." in error.message

    @pytest.mark.parametrize("name", ["Single_Task_GP_v1", " single_task_gp_v1", ""], ids=repr)
    def test_a_name_is_matched_exactly(
        self, registry: Registry[ComponentEntry], name: str
    ) -> None:
        # Names are permanent identifiers: no case folding, no stripping.
        assert failureOf(registry, {"type": name}).field_path == "type"


class TestWrongType:
    @pytest.mark.parametrize(
        "name, shown",
        [
            (3, "3 (int)"),
            (None, "None (NoneType)"),
            (["a"], "['a'] (list)"),
            (True, "True (bool)"),
        ],
        ids=repr,
    )
    def test_a_type_that_is_not_a_string_is_reported_with_what_was_given(
        self, registry: Registry[ComponentEntry], name: Any, shown: str
    ) -> None:
        error = failureOf(registry, {"type": name})
        assert error.field_path == "type"
        assert error.component == "surrogate"
        assert f"got {shown}" in error.message

    def test_an_unhashable_type_does_not_crash_the_lookup(
        self, registry: Registry[ComponentEntry]
    ) -> None:
        assert failureOf(registry, {"type": {"a": 1}}).field_path == "type"


class TestNotAMapping:
    @pytest.mark.parametrize(
        "data", ["single_task_gp_v1", ["single_task_gp_v1"], None, 3, GpConfig()], ids=repr
    )
    def test_anything_but_a_mapping_is_rejected(
        self, registry: Registry[ComponentEntry], data: Any
    ) -> None:
        error = failureOf(registry, data)
        assert error.component == "surrogate"
        assert "must be a mapping" in error.message
        assert type(data).__name__ in error.message

    def test_a_key_that_is_not_a_string_is_rejected(
        self, registry: Registry[ComponentEntry]
    ) -> None:
        error = failureOf(registry, {"type": "single_task_gp_v1", 1: 2})
        assert "must have string keys" in error.message
        assert "1 (int)" in error.message
        assert error.component == "surrogate"


class TestUnknownField:
    def test_it_names_the_field(self, registry: Registry[ComponentEntry]) -> None:
        error = failureOf(registry, {"type": "single_task_gp_v1", "min_observation": 3})
        assert error.field_path == "min_observation"
        assert "min_observation: Unknown field" in error.message

    def test_it_names_the_component(self, registry: Registry[ComponentEntry]) -> None:
        error = failureOf(registry, {"type": "single_task_gp_v1", "bogus": 1})
        assert error.component == "surrogate 'single_task_gp_v1'"
        assert "surrogate 'single_task_gp_v1'" in error.message

    def test_it_suggests_the_fields_the_schema_defines(
        self, registry: Registry[ComponentEntry]
    ) -> None:
        error = failureOf(registry, {"type": "single_task_gp_v1", "bogus": 1})
        assert error.suggestion is not None
        for field in GpConfig.model_fields:
            assert field in error.suggestion

    def test_the_fields_suggested_are_those_of_the_schema_of_that_component(
        self, vocabulary: CapabilityVocabulary
    ) -> None:
        components: Registry[ComponentEntry] = Registry("goal", vocabulary)
        components.register(makeComponent("needs_a_level_v1", RequiredConfig))
        error = failureOf(components, {"type": "needs_a_level_v1", "level": 1.0, "bogus": 1})
        assert error.suggestion is not None
        assert "level" in error.suggestion
        assert "min_observations" not in error.suggestion

    def test_an_unknown_field_inside_a_nested_model_is_reported_by_its_path(
        self, registry: Registry[ComponentEntry]
    ) -> None:
        error = failureOf(registry, {"type": "single_task_gp_v1", "kernel": {"mu": 1.0}})
        assert error.field_path == "kernel.mu"
        # The top-level fields are not the ones to choose from here, so none are suggested.
        assert error.suggestion is None

    def test_the_type_itself_is_not_an_unknown_field(
        self, registry: Registry[ComponentEntry]
    ) -> None:
        # `type` is a field of every schema: resolving a mapping that has only it succeeds.
        assert resolveComponent(registry, {"type": "single_task_gp_v1"}).config.type


class TestWrongFieldType:
    def test_it_names_the_field_and_what_was_given(
        self, registry: Registry[ComponentEntry]
    ) -> None:
        error = failureOf(registry, {"type": "single_task_gp_v1", "min_observations": "three"})
        assert error.field_path == "min_observations"
        assert error.component == "surrogate 'single_task_gp_v1'"
        assert "min_observations: Input should be a valid integer" in error.message
        assert "(got 'three')" in error.message

    @pytest.mark.parametrize("given", [[1], {"a": 1}, None], ids=repr)
    def test_a_container_or_none_for_a_number_is_rejected(
        self, registry: Registry[ComponentEntry], given: Any
    ) -> None:
        error = failureOf(registry, {"type": "single_task_gp_v1", "min_observations": given})
        assert error.field_path == "min_observations"

    def test_a_wrong_type_deep_in_the_configuration_has_the_full_path(
        self, registry: Registry[ComponentEntry]
    ) -> None:
        error = failureOf(registry, {"type": "single_task_gp_v1", "kernel": {"nu": "smooth"}})
        assert error.field_path == "kernel.nu"
        assert "kernel.nu: Input should be a valid number" in error.message

    def test_a_list_position_is_written_in_brackets(
        self, registry: Registry[ComponentEntry]
    ) -> None:
        error = failureOf(registry, {"type": "single_task_gp_v1", "values": [1, 2, "x"]})
        assert error.field_path == "values[2]"

    def test_a_model_given_where_a_scalar_is_expected(
        self, registry: Registry[ComponentEntry]
    ) -> None:
        error = failureOf(registry, {"type": "single_task_gp_v1", "kernel": "matern"})
        assert error.field_path == "kernel"
        assert "(got 'matern')" in error.message

    def test_a_long_value_is_shortened_in_the_message(
        self, registry: Registry[ComponentEntry]
    ) -> None:
        given = "x" * 10_000
        error = failureOf(registry, {"type": "single_task_gp_v1", "min_observations": given})
        assert error.field_path == "min_observations"
        assert len(error.message) < len(given)


class TestOtherValidationFailures:
    def test_a_validator_message_is_kept(self, registry: Registry[ComponentEntry]) -> None:
        error = failureOf(registry, {"type": "single_task_gp_v1", "alpha": 7})
        assert error.field_path == "alpha"
        assert "alpha must be in [0, 1]; got 7.0." in error.message

    def test_a_missing_required_field_is_reported(
        self, vocabulary: CapabilityVocabulary
    ) -> None:
        components: Registry[ComponentEntry] = Registry("goal", vocabulary)
        components.register(makeComponent("needs_a_level_v1", RequiredConfig))
        error = failureOf(components, {"type": "needs_a_level_v1"})
        assert error.field_path == "level"
        assert "level: Field required" in error.message
        # The value to show would be the whole mapping, which says nothing about the field.
        assert "(got" not in error.message

    def test_a_rule_across_fields_has_no_field_path(
        self, vocabulary: CapabilityVocabulary
    ) -> None:
        components: Registry[ComponentEntry] = Registry("goal", vocabulary)
        components.register(makeComponent("ordered_bounds_v1", ConsistentConfig))
        error = failureOf(components, {"type": "ordered_bounds_v1", "low": 2.0, "high": 1.0})
        assert error.field_path is None
        assert error.component == "goal 'ordered_bounds_v1'"
        assert "low must be less than high" in error.message

    def test_a_schema_that_pins_another_name_fails_at_the_type(
        self, vocabulary: CapabilityVocabulary
    ) -> None:
        components: Registry[ComponentEntry] = Registry("goal", vocabulary)
        components.register(makeComponent("pinned_v1", PinnedElsewhereConfig))
        error = failureOf(components, {"type": "pinned_v1"})
        assert error.field_path == "type"
        assert "some_other_name_v1" in error.message

    def test_the_original_pydantic_error_is_kept_as_the_cause(
        self, registry: Registry[ComponentEntry]
    ) -> None:
        error = failureOf(registry, {"type": "single_task_gp_v1", "bogus": 1})
        assert isinstance(error.__cause__, ValidationError)


class TestSeveralProblems:
    DATA: Mapping[str, Any] = {
        "type": "single_task_gp_v1",
        "min_observations": "three",
        "kernel": {"nu": "x"},
        "bogus": 1,
    }

    def test_every_problem_is_listed_with_its_path(
        self, registry: Registry[ComponentEntry]
    ) -> None:
        message = failureOf(registry, self.DATA).message
        assert "3 problems" in message
        for path in ("min_observations:", "kernel.nu:", "bogus:"):
            assert path in message

    def test_the_field_path_is_the_first_problem(
        self, registry: Registry[ComponentEntry]
    ) -> None:
        assert failureOf(registry, self.DATA).field_path == "min_observations"

    def test_the_suggestion_is_still_given_for_the_unknown_field(
        self, registry: Registry[ComponentEntry]
    ) -> None:
        assert failureOf(registry, self.DATA).suggestion is not None

    def test_a_single_problem_is_not_called_a_list(
        self, registry: Registry[ComponentEntry]
    ) -> None:
        message = failureOf(registry, {"type": "single_task_gp_v1", "bogus": 1}).message
        assert "problems" not in message
        assert "\n" not in message


class TestBrokenRegistration:
    def test_a_schema_without_the_base_class_is_a_registry_error(
        self, vocabulary: CapabilityVocabulary
    ) -> None:
        components: Registry[ComponentEntry] = Registry("goal", vocabulary)
        components.register(makeComponent("plain_v1", PlainSchema))
        with pytest.raises(RegistryError) as caught:
            resolveComponent(components, {"type": "plain_v1"})
        error = caught.value
        assert error.component == "goal 'plain_v1'"
        assert error.field_path == "config_schema"
        assert "PlainSchema" in error.message
        assert error.suggestion is not None
        assert "ComponentConfig" in error.suggestion


class TestMessages:
    """Pins the wording of the errors a user reads (one snapshot per kind of failure)."""

    def test_missing_type(self, registry: Registry[ComponentEntry]) -> None:
        assert str(failureOf(registry, {})) == (
            "The surrogate configuration has no 'type' field, so it does not say which "
            "surrogate to use. Registered names: single_task_gp_v1, "
            "variational_gp_classifier_v1.\n"
            "Component: surrogate\n"
            "Field: type\n"
            "Suggestion: add a 'type' entry that names a registered component"
        )

    def test_unknown_type(self, registry: Registry[ComponentEntry]) -> None:
        assert str(failureOf(registry, {"type": "gp"})) == (
            "Unknown surrogate type 'gp'. Registered names: single_task_gp_v1, "
            "variational_gp_classifier_v1.\n"
            "Component: surrogate\n"
            "Field: type\n"
            "Suggestion: check the spelling, or import the module or plugin that registers it"
        )

    def test_unknown_field(self, registry: Registry[ComponentEntry]) -> None:
        error = failureOf(registry, {"type": "single_task_gp_v1", "min_observation": 3})
        assert str(error) == (
            "Invalid configuration for surrogate 'single_task_gp_v1': "
            "min_observation: Unknown field.\n"
            "Component: surrogate 'single_task_gp_v1'\n"
            "Field: min_observation\n"
            "Suggestion: check the spelling; the fields of surrogate 'single_task_gp_v1' are: "
            "type, min_observations, alpha, kernel, values, bounds"
        )

    def test_wrong_field_type(self, registry: Registry[ComponentEntry]) -> None:
        error = failureOf(registry, {"type": "single_task_gp_v1", "min_observations": "three"})
        assert str(error) == (
            "Invalid configuration for surrogate 'single_task_gp_v1': min_observations: "
            "Input should be a valid integer, unable to parse string as an integer "
            "(got 'three').\n"
            "Component: surrogate 'single_task_gp_v1'\n"
            "Field: min_observations"
        )


class TestResolvedComponent:
    def test_it_is_immutable(self, registry: Registry[ComponentEntry]) -> None:
        resolved = resolveComponent(registry, {"type": "single_task_gp_v1"})
        with pytest.raises(dataclasses.FrozenInstanceError):
            resolved.config = GpConfig()  # type: ignore[misc]

    def test_it_holds_exactly_the_entry_and_the_configuration(
        self, registry: Registry[ComponentEntry]
    ) -> None:
        resolved = resolveComponent(registry, {"type": "single_task_gp_v1"})
        assert [field.name for field in dataclasses.fields(resolved)] == ["entry", "config"]

    def test_the_builder_of_the_entry_is_the_registered_one(
        self, registry: Registry[ComponentEntry]
    ) -> None:
        resolved = resolveComponent(registry, {"type": "single_task_gp_v1"})
        assert isinstance(resolved.entry, ComponentEntry)
        factory: Callable[..., Any] = resolved.entry.builder
        assert factory is builder
