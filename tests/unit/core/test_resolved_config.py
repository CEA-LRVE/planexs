"""Unit tests for `toResolvedConfig` and `validateConfig` (specification 3.4, 5.12)."""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, Literal

import pytest
from nested_config_schemas import (
    ComposedConfig,
    GpConfig,
    ImportanceConfig,
    LevelEstimateConfig,
    MaternConfig,
    RandomConfig,
    RbfConfig,
    StudyConfig,
)
from pydantic import Field, ValidationError

from planexs.core.ConfigError import ConfigError
from planexs.core.KindTable import KindTable
from planexs.core.resolveComponent import resolveComponent
from planexs.core.toResolvedConfig import toResolvedConfig
from planexs.core.validateConfig import validateConfig

pytestmark = pytest.mark.usefixtures("nestedKinds")

#: A study with components at every level, several settings left to their defaults.
STUDY: Mapping[str, Any] = {
    "name": "demo",
    "policy": {
        "type": "composed_v1",
        "design": {"type": "random_v1", "n_points": 5},
        "surrogate": {"type": "single_task_gp_v1", "kernel": {"type": "rbf_v1", "ard": True}},
        "analyses": [{"type": "importance_v1"}, {"type": "level_estimate_v1", "level": 0.2}],
    },
    "stopping": [{"type": "max_trials_v1", "n_trials": 30}],
}

#: The same study with almost everything left out: only what has no default is given.
MINIMAL_STUDY: Mapping[str, Any] = {"policy": {"type": "composed_v1"}}


def isPlainJson(value: object) -> bool:
    """Whether `value` is made only of dicts with string keys, lists, strings, numbers, bools
    and `None`, at every depth.
    """
    if isinstance(value, dict):
        return all(isinstance(key, str) and isPlainJson(item) for key, item in value.items())
    if isinstance(value, list):
        return all(isPlainJson(item) for item in value)
    return value is None or type(value) in (str, int, float, bool)


def failureOf(schema: type[Any], data: Any, **options: Any) -> ConfigError:
    """The `ConfigError` that validating `data` against `schema` raises."""
    with pytest.raises(ConfigError) as caught:
        validateConfig(schema, data, **options)
    return caught.value


class TestToResolvedConfig:
    def test_it_is_a_plain_dictionary(self) -> None:
        resolved = toResolvedConfig(validateConfig(StudyConfig, STUDY))
        assert type(resolved) is dict
        assert isPlainJson(resolved)

    def test_it_is_strict_json(self) -> None:
        resolved = toResolvedConfig(validateConfig(StudyConfig, STUDY))
        # `allow_nan=False` refuses anything that is not standard JSON.
        assert json.loads(json.dumps(resolved, allow_nan=False)) == resolved

    def test_the_settings_the_caller_left_out_are_written_with_their_defaults(self) -> None:
        resolved = toResolvedConfig(validateConfig(StudyConfig, STUDY))
        assert resolved["policy"]["bounds"] == [0.0, 1.0]
        assert resolved["policy"]["fallback"] is None
        assert resolved["policy"]["surrogate"]["min_observations"] == 2
        assert resolved["policy"]["analyses"][0]["n_samples"] == 64

    def test_a_default_component_is_written_in_full(self) -> None:
        resolved = toResolvedConfig(validateConfig(StudyConfig, MINIMAL_STUDY))
        assert resolved["policy"]["design"] == {"type": "sobol_v1", "n_points": 8}
        assert resolved["policy"]["surrogate"]["kernel"] == {
            "type": "matern_v1",
            "nu": 2.5,
            "ard": True,
        }

    def test_every_nested_component_has_its_type_written_out(self) -> None:
        policy = toResolvedConfig(validateConfig(StudyConfig, STUDY))["policy"]
        assert policy["type"] == "composed_v1"
        assert policy["design"]["type"] == "random_v1"
        assert policy["surrogate"]["type"] == "single_task_gp_v1"
        assert policy["surrogate"]["kernel"]["type"] == "rbf_v1"
        assert [item["type"] for item in policy["analyses"]] == [
            "importance_v1",
            "level_estimate_v1",
        ]

    def test_a_list_of_components_is_written_in_full_and_in_order(self) -> None:
        resolved = toResolvedConfig(validateConfig(StudyConfig, STUDY))
        assert resolved["stopping"] == [{"type": "max_trials_v1", "n_trials": 30}]
        assert resolved["policy"]["analyses"] == [
            {"type": "importance_v1", "n_samples": 64},
            {"type": "level_estimate_v1", "level": 0.2},
        ]

    def test_a_tuple_is_written_as_a_list(self) -> None:
        resolved = toResolvedConfig(ComposedConfig(bounds=(0.25, 0.75)))
        assert resolved["bounds"] == [0.25, 0.75]
        assert type(resolved["bounds"]) is list

    def test_the_fields_come_in_the_order_the_model_declares_them(self) -> None:
        resolved = toResolvedConfig(validateConfig(StudyConfig, STUDY))
        assert list(resolved) == ["name", "policy", "stopping"]
        assert list(resolved["policy"]) == [
            "type",
            "design",
            "surrogate",
            "analyses",
            "fallback",
            "bounds",
        ]

    def test_it_works_for_the_configuration_of_a_single_component(self) -> None:
        assert toResolvedConfig(GpConfig()) == {
            "type": "single_task_gp_v1",
            "min_observations": 2,
            "kernel": {"type": "matern_v1", "nu": 2.5, "ard": True},
        }

    def test_the_result_is_a_new_dictionary_each_time(self) -> None:
        study = validateConfig(StudyConfig, STUDY)
        first = toResolvedConfig(study)
        first["policy"]["analyses"].clear()
        first["policy"]["surrogate"]["kernel"]["nu"] = 99
        assert toResolvedConfig(study) == toResolvedConfig(validateConfig(StudyConfig, STUDY))
        assert len(study.policy.analyses) == 2

    def test_a_configuration_built_in_python_gives_the_same_dictionary(self) -> None:
        built = StudyConfig(
            name="demo",
            policy=ComposedConfig(
                design=RandomConfig(n_points=5),
                surrogate=GpConfig(kernel=RbfConfig(ard=True)),
                analyses=[ImportanceConfig(), LevelEstimateConfig(level=0.2)],
            ),
            stopping=[validateConfig(StudyConfig, STUDY).stopping[0]],
        )
        assert toResolvedConfig(built) == toResolvedConfig(validateConfig(StudyConfig, STUDY))


class TestRoundTrip:
    """Resolve, validate, resolve: the dictionary is a fixed point."""

    def test_resolve_then_validate_then_resolve_gives_the_same_dictionary(self) -> None:
        first = toResolvedConfig(validateConfig(StudyConfig, STUDY))
        second = toResolvedConfig(validateConfig(StudyConfig, first))
        assert second == first

    def test_it_is_also_a_fixed_point_from_a_minimal_configuration(self) -> None:
        first = toResolvedConfig(validateConfig(StudyConfig, MINIMAL_STUDY))
        assert toResolvedConfig(validateConfig(StudyConfig, first)) == first

    def test_the_rebuilt_configuration_equals_the_original(self) -> None:
        study = validateConfig(StudyConfig, STUDY)
        assert validateConfig(StudyConfig, toResolvedConfig(study)) == study

    def test_the_rebuilt_objects_are_those_of_the_nested_components(self) -> None:
        rebuilt = validateConfig(
            StudyConfig, toResolvedConfig(validateConfig(StudyConfig, STUDY))
        )
        assert type(rebuilt.policy) is ComposedConfig
        assert type(rebuilt.policy.surrogate.kernel) is RbfConfig
        assert [type(item) for item in rebuilt.policy.analyses] == [
            ImportanceConfig,
            LevelEstimateConfig,
        ]

    def test_a_component_is_rebuilt_through_its_registry_too(
        self, nestedKinds: KindTable
    ) -> None:
        policy = validateConfig(StudyConfig, STUDY).policy
        resolved = toResolvedConfig(policy)
        rebuilt = resolveComponent(nestedKinds.registry("policy"), resolved)
        assert rebuilt.config == policy
        assert toResolvedConfig(rebuilt.config) == resolved

    def test_the_dictionary_survives_json_dumps_and_loads_unchanged(self) -> None:
        resolved = toResolvedConfig(validateConfig(StudyConfig, STUDY))
        assert json.loads(json.dumps(resolved)) == resolved

    def test_the_dictionary_survives_json_for_a_minimal_configuration_too(self) -> None:
        resolved = toResolvedConfig(validateConfig(StudyConfig, MINIMAL_STUDY))
        assert json.loads(json.dumps(resolved)) == resolved

    def test_a_configuration_read_back_from_json_text_is_equal(self) -> None:
        study = validateConfig(StudyConfig, STUDY)
        text = json.dumps(toResolvedConfig(study), indent=2)
        assert validateConfig(StudyConfig, json.loads(text)) == study

    def test_a_dictionary_written_to_json_and_read_back_resolves_to_itself(self) -> None:
        text = json.dumps(toResolvedConfig(validateConfig(StudyConfig, STUDY)))
        again = toResolvedConfig(validateConfig(StudyConfig, json.loads(text)))
        assert json.dumps(again) == text


class TestOmittedDefaultsAreMaterialized:
    def test_the_dictionary_holds_a_default_the_caller_left_out(self) -> None:
        given = {"policy": {"type": "composed_v1"}}
        assert "name" not in given
        resolved = toResolvedConfig(validateConfig(StudyConfig, given))
        assert resolved["name"] == "study"
        assert resolved["stopping"] == []
        assert resolved["policy"]["surrogate"]["min_observations"] == 2

    def test_a_later_change_of_a_default_does_not_alter_a_stored_configuration(self) -> None:
        stored = toResolvedConfig(GpConfig())
        assert stored["min_observations"] == 2

        class ChangedDefault(GpConfig):
            min_observations: int = Field(default=5, description="The new default.")

        # Left to the new default, the setting follows it ...
        assert ChangedDefault().min_observations == 5
        # ... but a stored configuration has the value written out, so it keeps its meaning.
        assert validateConfig(ChangedDefault, stored).min_observations == 2


class TestValidateConfig:
    def test_it_rebuilds_a_model_that_is_not_a_component(self) -> None:
        study = validateConfig(StudyConfig, STUDY)
        assert type(study) is StudyConfig
        assert study.name == "demo"
        assert study.stopping[0].n_trials == 30

    def test_it_rebuilds_the_configuration_of_a_component(self) -> None:
        gp = validateConfig(
            GpConfig, {"type": "single_task_gp_v1", "kernel": {"type": "rbf_v1"}}
        )
        assert type(gp.kernel) is RbfConfig

    def test_it_accepts_a_configuration_with_defaults_left_out(self) -> None:
        study = validateConfig(StudyConfig, MINIMAL_STUDY)
        assert study.policy.surrogate.kernel == MaternConfig()

    def test_any_mapping_is_accepted(self) -> None:
        assert (
            validateConfig(StudyConfig, MappingProxyType(dict(MINIMAL_STUDY))).name == "study"
        )

    def test_the_input_is_not_modified(self) -> None:
        data = copy.deepcopy(dict(STUDY))
        validateConfig(StudyConfig, data)
        assert data == STUDY

    @pytest.mark.parametrize("data", [["policy"], None, "policy", 3], ids=repr)
    def test_a_value_that_is_not_a_mapping_is_rejected(self, data: Any) -> None:
        error = failureOf(StudyConfig, data)
        assert "must be a mapping" in error.message
        assert type(data).__name__ in error.message
        assert error.component == "StudyConfig"

    def test_the_error_names_the_model_by_default_and_the_label_when_given(self) -> None:
        assert failureOf(StudyConfig, {}).component == "StudyConfig"
        assert failureOf(StudyConfig, {}, label="the study").component == "the study"
        assert (
            "Invalid configuration for the study:"
            in failureOf(StudyConfig, {}, label="the study").message
        )

    def test_a_missing_required_field(self) -> None:
        error = failureOf(StudyConfig, {})
        assert error.field_path == "policy"
        assert "policy: Field required" in error.message

    def test_an_unknown_field_suggests_the_fields_of_the_model(self) -> None:
        error = failureOf(StudyConfig, {**MINIMAL_STUDY, "polcy": {}})
        assert error.field_path == "polcy"
        assert error.suggestion == (
            "check the spelling; the fields of StudyConfig are: name, policy, stopping"
        )

    def test_the_original_error_is_the_cause(self) -> None:
        assert isinstance(failureOf(StudyConfig, {}).__cause__, ValidationError)


class TestFullPathsOfErrors:
    def test_a_bad_value_two_levels_below_the_policy(self) -> None:
        # The path of the card: policy.surrogate.kernel.nu
        data = copy.deepcopy(dict(STUDY))
        data["policy"]["surrogate"]["kernel"] = {"type": "matern_v1", "nu": "smooth"}
        error = failureOf(StudyConfig, data)
        assert error.field_path == "policy.surrogate.kernel.nu"
        assert "policy.surrogate.kernel.nu: Input should be a valid number" in error.message
        assert error.component == "StudyConfig"

    def test_a_bad_value_in_the_list_of_a_component(self) -> None:
        data = copy.deepcopy(dict(STUDY))
        data["policy"]["analyses"][1]["level"] = "high"
        assert failureOf(StudyConfig, data).field_path == "policy.analyses[1].level"

    def test_a_bad_value_in_a_list_of_the_model(self) -> None:
        data = {**MINIMAL_STUDY, "stopping": [{"type": "max_trials_v1", "n_trials": "many"}]}
        assert failureOf(StudyConfig, data).field_path == "stopping[0].n_trials"

    def test_an_unknown_type_deep_in_the_configuration(self) -> None:
        data = copy.deepcopy(dict(STUDY))
        data["policy"]["surrogate"]["kernel"] = {"type": "matren_v1"}
        error = failureOf(StudyConfig, data)
        assert error.field_path == "policy.surrogate.kernel.type"
        assert "Unknown kernel type 'matren_v1'. Registered names: matern_v1, rbf_v1." in (
            error.message
        )

    def test_a_missing_type_deep_in_the_configuration(self) -> None:
        data = copy.deepcopy(dict(STUDY))
        del data["policy"]["design"]["type"]
        error = failureOf(StudyConfig, data)
        assert error.field_path == "policy.design.type"

    def test_the_path_of_a_component_resolved_on_its_own_is_relative_to_it(
        self, nestedKinds: KindTable
    ) -> None:
        data = copy.deepcopy(dict(STUDY["policy"]))
        data["surrogate"]["kernel"] = {"type": "matern_v1", "nu": "smooth"}
        with pytest.raises(ConfigError) as caught:
            resolveComponent(nestedKinds.registry("policy"), data)
        assert caught.value.field_path == "surrogate.kernel.nu"

    def test_the_wording_of_a_bad_nested_value(self) -> None:
        data = copy.deepcopy(dict(STUDY))
        data["policy"]["surrogate"]["kernel"] = {"type": "matern_v1", "nu": "smooth"}
        assert str(failureOf(StudyConfig, data)) == (
            "Invalid configuration for StudyConfig: policy.surrogate.kernel.nu: "
            "Input should be a valid number, unable to parse string as a number "
            "(got 'smooth').\n"
            "Component: StudyConfig\n"
            "Field: policy.surrogate.kernel.nu"
        )

    def test_the_wording_of_an_unknown_nested_type(self) -> None:
        data = copy.deepcopy(dict(STUDY))
        data["policy"]["surrogate"]["kernel"] = {"type": "matren_v1"}
        assert str(failureOf(StudyConfig, data)) == (
            "Invalid configuration for StudyConfig: policy.surrogate.kernel.type: Unknown "
            "kernel type 'matren_v1'. Registered names: matern_v1, rbf_v1.\n"
            "Component: StudyConfig\n"
            "Field: policy.surrogate.kernel.type"
        )


class TestSchemaThatPinsAnotherName:
    def test_a_resolved_configuration_for_the_wrong_component_is_rejected(self) -> None:
        resolved = toResolvedConfig(GpConfig())
        error = failureOf(ComposedConfig, resolved)
        assert error.field_path == "type"
        assert "composed_v1" in error.message

    def test_a_literal_pins_the_type_of_a_rebuilt_configuration(self) -> None:
        class Pinned(MaternConfig):
            type: Literal["matern_v1"] = "matern_v1"

        assert validateConfig(Pinned, {"type": "matern_v1"}).nu == 2.5
