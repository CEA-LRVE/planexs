"""Unit tests for `ComponentOf`: a configuration field that holds another component."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Annotated, Literal

import pytest
from nested_config_schemas import (
    ComposedConfig,
    GpConfig,
    ImportanceConfig,
    LevelEstimateConfig,
    MaternConfig,
    RandomConfig,
    RbfConfig,
    SobolConfig,
    StudyConfig,
)
from pydantic import Field, ValidationError

from planexs.core.ComponentConfig import ComponentConfig
from planexs.core.ComponentEntry import ComponentEntry
from planexs.core.ComponentOf import ComponentOf
from planexs.core.ConfigError import ConfigError
from planexs.core.KindTable import KindTable
from planexs.core.RegistryError import RegistryError
from planexs.core.resolveComponent import resolveComponent

pytestmark = pytest.mark.usefixtures("nestedKinds")


class LateKernelConfig(ComponentConfig):
    """A kernel that a test registers after the other components."""

    type: Literal["late_kernel_v1"] = "late_kernel_v1"


def builder() -> None:
    """A builder for the components that the tests register themselves."""


def resolvePolicy(nestedKinds: KindTable, data: object) -> ComposedConfig:
    """The validated policy configuration for `data`, resolved through the policy registry."""
    config = resolveComponent(nestedKinds.registry("policy"), data).config  # type: ignore[arg-type]
    assert isinstance(config, ComposedConfig)
    return config


def failureOf(nestedKinds: KindTable, data: object) -> ConfigError:
    """The `ConfigError` that resolving the policy `data` raises."""
    with pytest.raises(ConfigError) as caught:
        resolveComponent(nestedKinds.registry("policy"), data)  # type: ignore[arg-type]
    return caught.value


def locationsOf(error: ValidationError) -> list[tuple[int | str, ...]]:
    """The locations of the problems of a pydantic validation error."""
    return [problem["loc"] for problem in error.errors()]


class TestSingleComponent:
    def test_a_mapping_is_resolved_into_the_schema_of_the_component_it_names(self) -> None:
        config = GpConfig.model_validate({"kernel": {"type": "rbf_v1", "ard": True}})
        assert isinstance(config.kernel, RbfConfig)
        assert config.kernel.ard is True

    def test_the_field_holds_a_configuration_object_and_not_the_mapping(self) -> None:
        config = GpConfig.model_validate({"kernel": {"type": "matern_v1"}})
        assert not isinstance(config.kernel, dict)
        assert isinstance(config.kernel, MaternConfig)

    def test_the_defaults_of_the_nested_schema_are_filled_in(self) -> None:
        kernel = GpConfig.model_validate({"kernel": {"type": "matern_v1"}}).kernel
        assert isinstance(kernel, MaternConfig)
        assert (kernel.nu, kernel.ard) == (2.5, True)

    def test_different_names_give_different_schemas(self) -> None:
        matern = GpConfig.model_validate({"kernel": {"type": "matern_v1"}}).kernel
        rbf = GpConfig.model_validate({"kernel": {"type": "rbf_v1"}}).kernel
        assert type(matern) is MaternConfig
        assert type(rbf) is RbfConfig

    def test_the_kind_of_the_field_chooses_the_registry(self) -> None:
        # `matern_v1` is a kernel: the `design` field of a policy does not know it.
        with pytest.raises(ValidationError) as caught:
            ComposedConfig.model_validate({"design": {"type": "matern_v1"}})
        assert locationsOf(caught.value) == [("design", "type")]

    def test_json_input_has_the_same_meaning_as_a_mapping(self) -> None:
        text = '{"kernel": {"type": "matern_v1", "nu": 1.5}}'
        assert GpConfig.model_validate_json(text) == GpConfig.model_validate(json.loads(text))

    def test_equal_input_gives_equal_configurations(self) -> None:
        data = {"kernel": {"type": "matern_v1", "nu": 1.5}}
        assert GpConfig.model_validate(data) == GpConfig.model_validate(data)
        assert GpConfig.model_validate(data) != GpConfig.model_validate(
            {"kernel": {"type": "matern_v1", "nu": 0.5}}
        )


class TestDefaultComponent:
    def test_an_omitted_field_gets_its_default_component(self) -> None:
        assert GpConfig().kernel == MaternConfig()

    def test_a_default_is_resolved_through_the_registry_like_any_value(self) -> None:
        config = ComposedConfig()
        assert type(config.design) is SobolConfig
        assert type(config.surrogate) is GpConfig

    def test_a_default_component_has_its_own_defaults_filled_in_to_any_depth(self) -> None:
        config = ComposedConfig()
        assert config.design.n_points == 8
        assert type(config.surrogate.kernel) is MaternConfig
        assert config.surrogate.kernel.nu == 2.5

    def test_each_configuration_gets_its_own_default_object(self) -> None:
        assert GpConfig().kernel is not GpConfig().kernel

    def test_a_given_value_replaces_the_default(self) -> None:
        config = ComposedConfig.model_validate({"design": {"type": "random_v1"}})
        assert type(config.design) is RandomConfig
        assert config.design.n_points == 4


class TestListOfComponents:
    def test_each_element_is_resolved_in_order(self) -> None:
        config = ComposedConfig.model_validate(
            {
                "analyses": [
                    {"type": "level_estimate_v1", "level": 0.1},
                    {"type": "importance_v1"},
                    {"type": "level_estimate_v1"},
                ]
            }
        )
        assert [type(item) for item in config.analyses] == [
            LevelEstimateConfig,
            ImportanceConfig,
            LevelEstimateConfig,
        ]
        assert config.analyses[0].level == 0.1
        assert config.analyses[2].level == 0.5

    def test_the_defaults_of_each_element_are_filled_in(self) -> None:
        config = ComposedConfig.model_validate({"analyses": [{"type": "importance_v1"}]})
        assert config.analyses[0].n_samples == 64

    def test_the_list_may_be_empty(self) -> None:
        assert ComposedConfig.model_validate({"analyses": []}).analyses == []
        assert ComposedConfig().analyses == []

    def test_a_list_in_a_model_that_is_not_a_component(self) -> None:
        study = StudyConfig.model_validate(
            {"policy": {"type": "composed_v1"}, "stopping": [{"type": "max_trials_v1"}] * 2}
        )
        assert len(study.stopping) == 2
        assert study.stopping[1].n_trials == 50


class TestOptionalComponent:
    def test_none_is_accepted(self) -> None:
        assert ComposedConfig.model_validate({"fallback": None}).fallback is None
        assert ComposedConfig().fallback is None

    def test_a_mapping_is_resolved(self) -> None:
        config = ComposedConfig.model_validate({"fallback": {"type": "random_v1"}})
        assert type(config.fallback) is RandomConfig


class TestConfigurationObjects:
    """A configuration built in Python holds the objects of its nested components."""

    def test_an_object_is_accepted_as_it_is(self) -> None:
        kernel = MaternConfig(nu=1.5)
        assert GpConfig(kernel=kernel).kernel is kernel

    def test_a_configuration_can_be_built_entirely_in_python(self) -> None:
        config = ComposedConfig(
            design=RandomConfig(n_points=3),
            surrogate=GpConfig(kernel=RbfConfig(ard=True)),
            analyses=[ImportanceConfig(n_samples=8), LevelEstimateConfig()],
        )
        assert config.surrogate.kernel.ard is True
        assert config.design.n_points == 3
        assert len(config.analyses) == 2

    def test_objects_and_mappings_can_be_mixed(self) -> None:
        config = ComposedConfig.model_validate(
            {
                "design": RandomConfig(n_points=3),
                "surrogate": {"type": "single_task_gp_v1", "kernel": RbfConfig()},
            }
        )
        assert type(config.design) is RandomConfig
        assert type(config.surrogate.kernel) is RbfConfig

    def test_an_object_whose_type_is_not_registered_in_the_kind_is_rejected(self) -> None:
        with pytest.raises(ValidationError) as caught:
            GpConfig(kernel=SobolConfig())
        assert locationsOf(caught.value) == [("kernel", "type")]
        assert "Unknown kernel type 'sobol_v1'" in str(caught.value)

    def test_an_object_of_another_class_than_the_registered_schema_is_rejected(self) -> None:
        class Impostor(ComponentConfig):
            type: Literal["matern_v1"] = "matern_v1"

        with pytest.raises(ValidationError) as caught:
            GpConfig(kernel=Impostor())
        assert locationsOf(caught.value) == [("kernel",)]
        assert "Impostor is not the configuration schema of the kernel 'matern_v1'" in str(
            caught.value
        )
        assert "MaternConfig" in str(caught.value)

    def test_an_object_of_a_subclass_of_the_registered_schema_is_accepted(self) -> None:
        class Tuned(MaternConfig):
            scale: float = Field(default=3.0, description="An extra setting.")

        kernel = Tuned()
        assert GpConfig(kernel=kernel).kernel is kernel

    def test_an_object_in_a_list_is_checked_like_a_mapping(self) -> None:
        with pytest.raises(ValidationError) as caught:
            ComposedConfig(analyses=[ImportanceConfig(), MaternConfig()])
        assert locationsOf(caught.value) == [("analyses", 1, "type")]


class TestDepth:
    def test_three_levels_below_a_model_that_is_not_a_component(self) -> None:
        study = StudyConfig.model_validate(
            {
                "policy": {
                    "type": "composed_v1",
                    "surrogate": {
                        "type": "single_task_gp_v1",
                        "kernel": {"type": "rbf_v1", "ard": True},
                    },
                }
            }
        )
        assert type(study.policy) is ComposedConfig
        assert type(study.policy.surrogate) is GpConfig
        assert type(study.policy.surrogate.kernel) is RbfConfig
        assert study.policy.surrogate.kernel.ard is True

    def test_the_levels_not_given_are_defaults_all_the_way_down(self) -> None:
        study = StudyConfig.model_validate({"policy": {"type": "composed_v1"}})
        assert study.policy.surrogate.kernel == MaternConfig()


class TestLocationsOfProblems:
    """What pydantic reports: the locations are merged through every level."""

    def test_a_problem_in_a_nested_component_is_located_below_its_field(self) -> None:
        with pytest.raises(ValidationError) as caught:
            GpConfig.model_validate({"kernel": {"type": "matern_v1", "nu": "smooth"}})
        assert locationsOf(caught.value) == [("kernel", "nu")]

    def test_the_location_goes_through_every_level(self) -> None:
        with pytest.raises(ValidationError) as caught:
            StudyConfig.model_validate(
                {
                    "policy": {
                        "type": "composed_v1",
                        "surrogate": {"type": "single_task_gp_v1", "kernel": {"nu": "x"}},
                    }
                }
            )
        # The kernel has no `type`: that is the problem, at its own path.
        assert locationsOf(caught.value) == [("policy", "surrogate", "kernel", "type")]

    def test_a_list_position_is_part_of_the_location(self) -> None:
        with pytest.raises(ValidationError) as caught:
            ComposedConfig.model_validate(
                {
                    "analyses": [
                        {"type": "importance_v1"},
                        {"type": "level_estimate_v1", "bogus": 1},
                    ]
                }
            )
        assert locationsOf(caught.value) == [("analyses", 1, "bogus")]


class TestErrorsOfAResolvedComponent:
    """The `ConfigError` of `resolveComponent` carries the full path from the policy."""

    def test_a_bad_value_two_levels_down(self, nestedKinds: KindTable) -> None:
        error = failureOf(
            nestedKinds,
            {
                "type": "composed_v1",
                "surrogate": {
                    "type": "single_task_gp_v1",
                    "kernel": {"type": "matern_v1", "nu": "smooth"},
                },
            },
        )
        assert error.field_path == "surrogate.kernel.nu"
        assert "surrogate.kernel.nu: Input should be a valid number" in error.message
        assert "(got 'smooth')" in error.message

    def test_a_bad_value_in_a_list_element(self, nestedKinds: KindTable) -> None:
        error = failureOf(
            nestedKinds,
            {
                "type": "composed_v1",
                "analyses": [
                    {"type": "importance_v1"},
                    {"type": "level_estimate_v1", "level": "high"},
                ],
            },
        )
        assert error.field_path == "analyses[1].level"

    def test_an_unknown_field_in_a_nested_component(self, nestedKinds: KindTable) -> None:
        error = failureOf(
            nestedKinds,
            {"type": "composed_v1", "surrogate": {"type": "single_task_gp_v1", "bogus": 1}},
        )
        assert error.field_path == "surrogate.bogus"
        assert "surrogate.bogus: Unknown field" in error.message
        # The fields suggested would be those of the policy, not of the surrogate.
        assert error.suggestion is None

    def test_an_unknown_type_names_the_kind_and_the_registered_names(
        self, nestedKinds: KindTable
    ) -> None:
        error = failureOf(
            nestedKinds,
            {
                "type": "composed_v1",
                "surrogate": {"type": "single_task_gp_v1", "kernel": {"type": "matren_v1"}},
            },
        )
        assert error.field_path == "surrogate.kernel.type"
        assert "Unknown kernel type 'matren_v1'. Registered names: matern_v1, rbf_v1." in (
            error.message
        )
        # The value is quoted by the message already: it is not added a second time.
        assert "(got" not in error.message

    def test_a_missing_type_is_reported_at_the_type_of_the_nested_component(
        self, nestedKinds: KindTable
    ) -> None:
        error = failureOf(
            nestedKinds,
            {
                "type": "composed_v1",
                "surrogate": {"type": "single_task_gp_v1", "kernel": {"nu": 1.0}},
            },
        )
        assert error.field_path == "surrogate.kernel.type"
        assert "kernel configuration has no 'type' field" in error.message

    def test_a_type_that_is_not_a_string(self, nestedKinds: KindTable) -> None:
        error = failureOf(nestedKinds, {"type": "composed_v1", "design": {"type": 3}})
        assert error.field_path == "design.type"
        assert "must be a string naming a registered design; got 3 (int)" in error.message

    @pytest.mark.parametrize("given", ["single_task_gp_v1", ["single_task_gp_v1"], None, 3])
    def test_a_value_that_is_not_a_mapping_is_reported_at_the_field(
        self, nestedKinds: KindTable, given: object
    ) -> None:
        error = failureOf(nestedKinds, {"type": "composed_v1", "surrogate": given})
        assert error.field_path == "surrogate"
        assert "A surrogate configuration must be a mapping" in error.message
        assert type(given).__name__ in error.message

    def test_an_element_that_is_not_a_mapping(self, nestedKinds: KindTable) -> None:
        error = failureOf(nestedKinds, {"type": "composed_v1", "analyses": ["importance_v1"]})
        assert error.field_path == "analyses[0]"

    def test_the_error_names_the_component_that_was_resolved(
        self, nestedKinds: KindTable
    ) -> None:
        error = failureOf(nestedKinds, {"type": "composed_v1", "design": {"type": "x"}})
        assert error.component == "policy 'composed_v1'"

    def test_every_problem_below_and_beside_is_listed_and_the_first_is_the_path(
        self, nestedKinds: KindTable
    ) -> None:
        error = failureOf(
            nestedKinds,
            {
                "type": "composed_v1",
                "design": {"type": "sobol_v1", "n_points": "many"},
                "surrogate": {
                    "type": "single_task_gp_v1",
                    "kernel": {"type": "matern_v1", "nu": "x"},
                },
                "analyses": [{"type": "nope_v1"}],
                "bounds": "wide",
            },
        )
        assert "4 problems" in error.message
        for path in (
            "design.n_points:",
            "surrogate.kernel.nu:",
            "analyses[0].type:",
            "bounds:",
        ):
            assert path in error.message
        assert error.field_path == "design.n_points"

    def test_a_single_selection_problem_does_not_end_with_two_periods(
        self, nestedKinds: KindTable
    ) -> None:
        error = failureOf(nestedKinds, {"type": "composed_v1", "design": {"type": "x"}})
        assert error.message.endswith("Registered names: random_v1, sobol_v1.")

    def test_the_original_error_is_the_cause(self, nestedKinds: KindTable) -> None:
        error = failureOf(nestedKinds, {"type": "composed_v1", "design": {"type": "x"}})
        assert isinstance(error.__cause__, ValidationError)


class TestDefectsOfTheDeclaration:
    """A wrong declaration is not a wrong configuration: it is a `RegistryError`."""

    def test_a_kind_that_is_not_defined(self, nestedKinds: KindTable) -> None:
        class Broken(ComponentConfig):
            type: Literal["broken_v1"] = "broken_v1"
            part: Annotated[ComponentConfig, ComponentOf("no_such_kind")] = Field(
                description="A part."
            )

        with pytest.raises(RegistryError) as caught:
            Broken.model_validate({"part": {"type": "x"}})
        assert "Unknown component kind 'no_such_kind'" in caught.value.message
        assert "kernel" in caught.value.message  # the defined kinds are listed

    def test_a_component_registered_with_a_schema_that_is_not_a_component_config(
        self, nestedKinds: KindTable, makeEntry: Callable[..., ComponentEntry]
    ) -> None:
        # `makeEntry` registers a plain pydantic model as the schema.
        nestedKinds.registry("kernel").register(makeEntry("plain_v1"))
        with pytest.raises(RegistryError) as caught:
            GpConfig.model_validate({"kernel": {"type": "plain_v1"}})
        assert caught.value.component == "kernel 'plain_v1'"
        assert "DummyConfig" in caught.value.message


class TestKindLookup:
    def test_the_kind_may_be_defined_after_the_schema(
        self, isolatedKindTable: KindTable
    ) -> None:
        class Late(ComponentConfig):
            type: Literal["late_v1"] = "late_v1"
            part: Annotated[ComponentConfig, ComponentOf("late_kind")] = Field(
                description="A part."
            )

        with pytest.raises(RegistryError):
            Late.model_validate({"part": {"type": "rbf_v1"}})

        # The kind and its components come into being later: the same class now validates.
        isolatedKindTable.define("late_kind").register(
            ComponentEntry("rbf_v1", builder, RbfConfig, "A late kernel.")
        )
        assert type(Late.model_validate({"part": {"type": "rbf_v1"}}).part) is RbfConfig

    def test_a_component_registered_after_a_first_validation_is_found(
        self, nestedKinds: KindTable
    ) -> None:
        assert (
            type(GpConfig.model_validate({"kernel": {"type": "rbf_v1"}}).kernel) is RbfConfig
        )
        nestedKinds.registry("kernel").register(
            ComponentEntry("late_kernel_v1", builder, LateKernelConfig, "A late kernel.")
        )
        # Nothing was kept from the earlier lookup: the registry is read at each validation.
        config = GpConfig.model_validate({"kernel": {"type": "late_kernel_v1"}})
        assert type(config.kernel) is LateKernelConfig


class TestSerialization:
    """A component field is dumped as the component it holds, not as `ComponentConfig`."""

    def test_a_nested_component_is_dumped_with_its_own_fields(self) -> None:
        config = GpConfig.model_validate({"kernel": {"type": "matern_v1", "nu": 1.5}})
        assert config.model_dump() == {
            "type": "single_task_gp_v1",
            "min_observations": 2,
            "kernel": {"type": "matern_v1", "nu": 1.5, "ard": True},
        }

    def test_the_json_mode_gives_the_same_fields(self) -> None:
        config = GpConfig.model_validate({"kernel": {"type": "rbf_v1"}})
        assert config.model_dump(mode="json") == config.model_dump()

    def test_the_json_text_gives_the_same_fields(self) -> None:
        config = GpConfig.model_validate({"kernel": {"type": "matern_v1", "nu": 1.5}})
        assert json.loads(config.model_dump_json()) == config.model_dump()

    def test_a_default_component_is_dumped_in_full(self) -> None:
        assert GpConfig().model_dump()["kernel"] == {
            "type": "matern_v1",
            "nu": 2.5,
            "ard": True,
        }

    def test_every_element_of_a_list_is_dumped_in_full(self) -> None:
        config = ComposedConfig.model_validate(
            {
                "analyses": [
                    {"type": "importance_v1"},
                    {"type": "level_estimate_v1", "level": 0.2},
                ]
            }
        )
        assert config.model_dump()["analyses"] == [
            {"type": "importance_v1", "n_samples": 64},
            {"type": "level_estimate_v1", "level": 0.2},
        ]

    def test_an_optional_component_is_dumped_in_full_or_as_none(self) -> None:
        given = ComposedConfig.model_validate({"fallback": {"type": "random_v1"}})
        assert given.model_dump()["fallback"] == {"type": "random_v1", "n_points": 4}
        assert ComposedConfig().model_dump()["fallback"] is None

    def test_the_fields_of_a_subclass_of_a_schema_are_dumped_too(self) -> None:
        class Tuned(MaternConfig):
            scale: float = Field(default=3.0, description="An extra setting.")

        assert GpConfig(kernel=Tuned()).model_dump()["kernel"]["scale"] == 3.0

    def test_a_dump_validates_back_into_an_equal_configuration(self) -> None:
        config = ComposedConfig.model_validate(
            {
                "surrogate": {
                    "type": "single_task_gp_v1",
                    "kernel": {"type": "rbf_v1", "ard": True},
                },
                "analyses": [{"type": "importance_v1", "n_samples": 8}],
                "fallback": {"type": "random_v1"},
            }
        )
        assert ComposedConfig.model_validate(config.model_dump()) == config
