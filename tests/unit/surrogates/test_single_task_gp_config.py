"""Unit tests for `SingleTaskGPConfig` (roadmap task P3-14, specification 3.4, 5.12, 10.8):
the defaults, what is validated and at which path, the nested components, and that the
configuration means the same in Python, JSON and YAML and is stored with every default written
out.
"""

from __future__ import annotations

import inspect
from typing import Any

import pytest
from pydantic import ValidationError

from planexs.api.BayesianOptimizer import MIN_TRIALS_FOR_MODEL
from planexs.core.ConfigError import ConfigError
from planexs.core.configFiles import dumpConfig, parseConfig
from planexs.core.resolveComponent import resolveComponent
from planexs.core.toResolvedConfig import toResolvedConfig
from planexs.core.validateConfig import validateConfig
from planexs.models.buildSurrogateModel import buildSurrogateModel
from planexs.surrogates.fitters.MapFitterConfig import MapFitterConfig
from planexs.surrogates.kernels.MaternKernelConfig import MaternKernelConfig
from planexs.surrogates.NoisePolicyConfig import NoisePolicyConfig
from planexs.surrogates.SingleTaskGPConfig import SingleTaskGPConfig
from planexs.surrogates.surrogateRegistry import surrogateRegistry


def make(**settings: Any) -> SingleTaskGPConfig:
    return validateConfig(SingleTaskGPConfig, settings)


class TestTheDefaults:
    def test_the_nested_components_default_to_the_ones_the_legacy_surrogate_hard_wired(
        self,
    ) -> None:
        config = SingleTaskGPConfig()

        assert config.kernel.type == "botorch_default_v1"
        assert config.likelihood.type == "gaussian_v1"
        assert [transform.type for transform in config.outcome_transform] == ["standardize_v1"]
        assert config.fitter == MapFitterConfig()

    def test_the_minimum_is_the_legacy_one(self) -> None:
        # Pinned against the legacy constant until it is deleted (roadmap task P11-01): a study
        # that sets nothing must still need the number of trials it needed.
        assert SingleTaskGPConfig().min_observations == MIN_TRIALS_FOR_MODEL == 2

    def test_the_low_data_threshold_is_the_legacy_one(self) -> None:
        legacy = inspect.signature(buildSurrogateModel).parameters[
            "minimum_points_for_free_fit"
        ]

        assert SingleTaskGPConfig().low_data_threshold == legacy.default == 5

    def test_the_noise_policy_is_the_default_of_its_own_schema(self) -> None:
        assert SingleTaskGPConfig().noise_policy == NoisePolicyConfig()

    def test_the_registered_schema_is_this_one_and_a_mapping_of_its_type_gets_every_default(
        self,
    ) -> None:
        resolved = resolveComponent(surrogateRegistry(), {"type": "single_task_gp_v1"})

        assert resolved.entry.config_schema is SingleTaskGPConfig
        assert resolved.config == SingleTaskGPConfig()


class TestWhatIsValidated:
    @pytest.mark.parametrize("value", [0, -3])
    def test_the_minimum_is_at_least_one_observation(self, value: int) -> None:
        with pytest.raises(ValidationError, match="min_observations"):
            SingleTaskGPConfig(min_observations=value)

    def test_the_threshold_may_be_zero_but_not_negative(self) -> None:
        assert SingleTaskGPConfig(low_data_threshold=0).low_data_threshold == 0
        with pytest.raises(ValidationError, match="low_data_threshold"):
            SingleTaskGPConfig(low_data_threshold=-1)

    def test_an_unknown_field_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="fitter_name"):
            SingleTaskGPConfig(fitter_name="map_v1")  # type: ignore[call-arg]

    def test_another_type_is_refused_even_without_a_registry(self) -> None:
        wrong: Any = "single_task_gp_v2"

        with pytest.raises(ValidationError, match="type"):
            SingleTaskGPConfig(type=wrong)

    @pytest.mark.parametrize(
        ("data", "path"),
        [
            ({"kernel": {"type": "nope_v1"}}, "kernel.type"),
            ({"kernel": {"type": "matern_v1", "nu": 3.0}}, "kernel.nu"),
            (
                {"likelihood": {"type": "gaussian_v1", "noise_lower_bound": -1.0}},
                "likelihood.noise_lower_bound",
            ),
            (
                {"outcome_transform": [{"type": "standardize_v1"}, {"type": "nope_v1"}]},
                "outcome_transform[1].type",
            ),
            (
                {"outcome_transform": [{"type": "standardize_v1", "min_std": 0.0}]},
                "outcome_transform[0].min_std",
            ),
            ({"fitter": {"type": "map_v1", "max_attempts": 0}}, "fitter.max_attempts"),
            ({"fitter": {"type": "map_v1", "method": "TNC"}}, "fitter.method"),
            ({"noise_policy": {"partial": "sometimes"}}, "noise_policy.partial"),
            ({"min_observations": 0}, "min_observations"),
        ],
    )
    def test_a_mistake_at_any_depth_is_reported_at_its_full_path(
        self, data: dict[str, Any], path: str
    ) -> None:
        with pytest.raises(ConfigError) as caught:
            make(**data)

        assert caught.value.field_path == path

    def test_the_error_for_an_unknown_nested_type_lists_the_registered_names(self) -> None:
        with pytest.raises(ConfigError) as caught:
            make(kernel={"type": "nope_v1"})

        message = caught.value.message
        assert all(name in message for name in ("botorch_default_v1", "matern_v1", "rbf_v1"))

    def test_an_empty_list_of_transforms_is_legal_and_means_no_rescaling(self) -> None:
        assert make(outcome_transform=[]).outcome_transform == []


class TestTheNestedComponentsAreResolved:
    def test_each_holds_the_validated_configuration_of_the_named_component(self) -> None:
        config = make(
            kernel={"type": "matern_v1", "nu": 1.5},
            fitter={"type": "map_v1", "maxiter": 50},
            outcome_transform=[
                {"type": "log_v1"},
                {"type": "standardize_v1", "min_std": 1e-6},
            ],
        )

        assert config.kernel == MaternKernelConfig(nu=1.5)
        assert config.fitter == MapFitterConfig(maxiter=50)
        assert [t.type for t in config.outcome_transform] == ["log_v1", "standardize_v1"]
        assert config.outcome_transform[1].model_dump()["min_std"] == 1e-6

    def test_a_configuration_built_in_python_is_accepted_if_it_fits_its_type(self) -> None:
        assert make(kernel=MaternKernelConfig(nu=0.5)).kernel == MaternKernelConfig(nu=0.5)


ONE_CONFIGURATION: dict[str, Any] = {
    "type": "single_task_gp_v1",
    "kernel": {"type": "matern_v1", "nu": 1.5, "ard": False},
    "likelihood": {
        "type": "gaussian_v1",
        "noise_lower_bound": 0.001,
        "min_observation_variance": 0.0001,
    },
    "outcome_transform": [{"type": "log_v1"}, {"type": "standardize_v1", "min_std": 0.001}],
    "fitter": {
        "type": "map_v1",
        "max_attempts": 3,
        "pick_best_of_all_attempts": True,
        "maxiter": 100,
    },
    "min_observations": 4,
    "low_data_threshold": 8,
    "noise_policy": {"partial": "error"},
}


class TestTheSameMeaningInPythonJsonAndYaml:
    @pytest.mark.parametrize("file_format", ["json", "yaml"])
    def test_a_file_means_what_the_python_mapping_means(self, file_format: Any) -> None:
        text = dumpConfig(ONE_CONFIGURATION, file_format)

        assert validateConfig(SingleTaskGPConfig, parseConfig(text, file_format)) == make(
            **ONE_CONFIGURATION
        )


class TestTheResolvedConfiguration:
    """A study stores its configuration with every default written out (specification 3.4)."""

    def test_a_mapping_that_says_only_its_type_is_written_out_in_full(self) -> None:
        resolved = toResolvedConfig(make())

        assert resolved["kernel"] == {"type": "botorch_default_v1"}
        assert resolved["likelihood"] == {
            "type": "gaussian_v1",
            "noise_lower_bound": None,
            "min_observation_variance": 1e-10,
        }
        assert resolved["outcome_transform"] == [{"type": "standardize_v1", "min_std": 1e-8}]
        assert resolved["fitter"] == {
            "type": "map_v1",
            "max_attempts": 5,
            "pick_best_of_all_attempts": False,
            "maxiter": None,
        }
        assert resolved["min_observations"] == 2
        assert resolved["low_data_threshold"] == 5
        assert resolved["noise_policy"] == {"partial": "ignore_with_warning"}

    def test_resolving_and_validating_again_is_a_fixed_point(self) -> None:
        resolved = toResolvedConfig(make(**ONE_CONFIGURATION))

        assert toResolvedConfig(validateConfig(SingleTaskGPConfig, resolved)) == resolved

    def test_the_resolved_configuration_survives_json(self) -> None:
        resolved = toResolvedConfig(make(**ONE_CONFIGURATION))

        assert parseConfig(dumpConfig(resolved, "json"), "json") == resolved

    def test_a_default_that_a_later_engine_changes_does_not_alter_a_stored_study(self) -> None:
        # The fitter's defaults are written in the resolved configuration, not left to BoTorch.
        resolved = toResolvedConfig(make(fitter={"type": "map_v1"}))

        assert resolved["fitter"]["max_attempts"] == 5
        assert resolved["fitter"]["pick_best_of_all_attempts"] is False
