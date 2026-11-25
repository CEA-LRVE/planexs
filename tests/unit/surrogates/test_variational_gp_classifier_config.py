"""Unit tests for `VariationalGPClassifierConfig` (roadmap task P3-16, specification 3.4,
10.8).

The defaults are pinned here, because a default is part of what a stored study means: changing
one changes every study that did not set it, which is a new name (specification 3.7).
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

import planexs.surrogates  # noqa: F401
from planexs.core.ConfigError import ConfigError
from planexs.core.resolveComponent import resolveComponent
from planexs.core.toResolvedConfig import toResolvedConfig
from planexs.core.validateConfig import validateConfig
from planexs.surrogates.fitters.VariationalElboFitterConfig import VariationalElboFitterConfig
from planexs.surrogates.kernels.BotorchDefaultKernelConfig import BotorchDefaultKernelConfig
from planexs.surrogates.surrogateRegistry import surrogateRegistry
from planexs.surrogates.VariationalGPClassifierConfig import VariationalGPClassifierConfig

NAME = "variational_gp_classifier_v1"


class TestTheDefaults:
    def test_the_defaults_are_written_down_in_the_schema(self) -> None:
        config = VariationalGPClassifierConfig()

        assert config.type == NAME
        assert config.n_inducing is None
        assert config.min_observations == 2
        assert config.low_data_threshold == 10

    def test_the_default_kernel_is_the_one_the_engine_chooses(self) -> None:
        config = VariationalGPClassifierConfig()

        assert config.kernel.type == "botorch_default_v1"
        assert isinstance(config.kernel, BotorchDefaultKernelConfig)

    def test_the_default_fitter_is_the_variational_elbo(self) -> None:
        config = VariationalGPClassifierConfig()

        assert config.fitter.type == "variational_elbo_v1"
        assert config.fitter == VariationalElboFitterConfig()

    def test_a_mapping_that_says_only_its_type_gets_every_default(self) -> None:
        resolved = resolveComponent(surrogateRegistry(), {"type": NAME})

        assert resolved.config == VariationalGPClassifierConfig()

    def test_a_mapping_that_sets_every_field_is_read_as_given(self) -> None:
        resolved = resolveComponent(
            surrogateRegistry(),
            {
                "type": NAME,
                "kernel": {"type": "matern_v1", "nu": 1.5},
                "fitter": {"type": "variational_elbo_v1", "n_iterations": 7},
                "n_inducing": 12,
                "min_observations": 4,
                "low_data_threshold": 25,
            },
        )

        config = resolved.config
        assert isinstance(config, VariationalGPClassifierConfig)
        assert config.kernel.type == "matern_v1"
        assert config.kernel.nu == 1.5  # type: ignore[attr-defined]
        assert config.fitter.n_iterations == 7  # type: ignore[attr-defined]
        assert (config.n_inducing, config.min_observations, config.low_data_threshold) == (
            12,
            4,
            25,
        )

    def test_the_resolved_configuration_holds_every_nested_default_and_round_trips(
        self,
    ) -> None:
        resolved = resolveComponent(surrogateRegistry(), {"type": NAME})

        written = toResolvedConfig(resolved.config)

        assert written["type"] == NAME
        assert written["kernel"]["type"] == "botorch_default_v1"
        assert written["fitter"]["type"] == "variational_elbo_v1"
        assert written["fitter"]["n_iterations"] == 100
        assert written["n_inducing"] is None
        again = resolveComponent(surrogateRegistry(), written)
        assert again.config == resolved.config


class TestEveryFieldIsDescribed:
    @pytest.mark.parametrize("name", list(VariationalGPClassifierConfig.model_fields))
    def test_with_a_description_that_says_more_than_the_name(self, name: str) -> None:
        field = VariationalGPClassifierConfig.model_fields[name]

        assert field.description
        assert len(field.description.split()) >= 10

    @pytest.mark.parametrize("name", ["n_inducing", "min_observations", "low_data_threshold"])
    def test_the_description_of_a_number_gives_its_unit_and_why_the_default_is_that(
        self, name: str
    ) -> None:
        description = VariationalGPClassifierConfig.model_fields[name].description or ""

        assert "in points" in description or "in observations" in description
        assert "default" in description.lower()

    def test_the_class_documents_every_attribute(self) -> None:
        documented = VariationalGPClassifierConfig.__doc__ or ""

        for name in VariationalGPClassifierConfig.model_fields:
            assert f"{name}:" in documented


class TestTheFieldsAreValidated:
    @pytest.mark.parametrize("field", ["n_inducing", "min_observations", "low_data_threshold"])
    @pytest.mark.parametrize("value", [0, -1])
    def test_a_count_is_at_least_one(self, field: str, value: int) -> None:
        with pytest.raises(ValidationError, match=field):
            VariationalGPClassifierConfig(**{field: value})

    @pytest.mark.parametrize("field", ["n_inducing", "min_observations", "low_data_threshold"])
    def test_a_count_is_a_whole_number(self, field: str) -> None:
        with pytest.raises(ValidationError, match=field):
            VariationalGPClassifierConfig(**{field: 2.5})

    @pytest.mark.parametrize("field", ["n_inducing", "min_observations", "low_data_threshold"])
    def test_one_is_enough(self, field: str) -> None:
        assert getattr(VariationalGPClassifierConfig(**{field: 1}), field) == 1

    def test_no_inducing_points_number_means_as_many_as_observations(self) -> None:
        assert VariationalGPClassifierConfig(n_inducing=None).n_inducing is None

    def test_the_type_is_the_name_of_this_surrogate_only(self) -> None:
        with pytest.raises(ValidationError, match="type"):
            VariationalGPClassifierConfig(type="variational_gp_classifier_v2")  # type: ignore[arg-type]

    def test_an_unknown_field_is_refused_with_its_name(self) -> None:
        with pytest.raises(ConfigError, match="n_inducing_points") as caught:
            resolveComponent(surrogateRegistry(), {"type": NAME, "n_inducing_points": 5})

        assert f"surrogate '{NAME}'" in caught.value.message

    def test_a_likelihood_is_not_a_field(self) -> None:
        # The likelihood is the identity of the surrogate (a Bernoulli one, read by a probit),
        # and another one is another surrogate: there is nothing to choose.
        assert "likelihood" not in VariationalGPClassifierConfig.model_fields
        with pytest.raises(ConfigError, match="likelihood"):
            resolveComponent(surrogateRegistry(), {"type": NAME, "likelihood": "gaussian_v1"})

    def test_an_error_names_the_field_as_every_configuration_does(self) -> None:
        with pytest.raises(ConfigError) as error:
            validateConfig(VariationalGPClassifierConfig, {"n_inducing": 0}, label="surrogate")

        assert error.value.field_path == "n_inducing"


class TestTheNestedComponents:
    def test_a_kernel_is_resolved_through_the_kernel_registry(self) -> None:
        config = VariationalGPClassifierConfig(kernel={"type": "rbf_v1"})  # type: ignore[arg-type]

        assert config.kernel.type == "rbf_v1"

    def test_the_settings_of_a_kernel_are_validated_by_its_own_schema(self) -> None:
        with pytest.raises(ValidationError, match="nu"):
            VariationalGPClassifierConfig(
                kernel={"type": "matern_v1", "nu": 7.0}  # type: ignore[arg-type]
            )

    @pytest.mark.parametrize(
        "kernel", [{"type": "no_such_kernel"}, {"type": "variational_elbo_v1"}, {}]
    )
    def test_a_kernel_that_is_not_registered_is_refused(self, kernel: dict[str, Any]) -> None:
        with pytest.raises(ConfigError):
            resolveComponent(surrogateRegistry(), {"type": NAME, "kernel": kernel})

    def test_a_fitter_is_resolved_through_the_fitter_registry(self) -> None:
        config = VariationalGPClassifierConfig(fitter={"type": "map_v1"})  # type: ignore[arg-type]

        assert config.fitter.type == "map_v1"

    def test_the_settings_of_a_fitter_are_validated_by_its_own_schema(self) -> None:
        with pytest.raises(ValidationError, match="n_iterations"):
            VariationalGPClassifierConfig(
                fitter={"type": "variational_elbo_v1", "n_iterations": 0}  # type: ignore[arg-type]
            )

    @pytest.mark.parametrize(
        "fitter", [{"type": "no_such_fitter"}, {"type": "rbf_v1"}, {"n_iterations": 5}]
    )
    def test_a_fitter_that_is_not_registered_or_not_named_is_refused(
        self, fitter: dict[str, Any]
    ) -> None:
        with pytest.raises(ConfigError):
            resolveComponent(surrogateRegistry(), {"type": NAME, "fitter": fitter})

    def test_an_error_in_a_nested_component_names_its_path(self) -> None:
        with pytest.raises(ConfigError) as caught:
            resolveComponent(
                surrogateRegistry(),
                {"type": NAME, "fitter": {"type": "variational_elbo_v1", "n_iterations": 0}},
            )

        assert caught.value.field_path is not None
        assert caught.value.field_path.startswith("fitter")
