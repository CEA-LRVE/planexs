"""Unit tests for `PredictionConfig` and `SamplingConfig` (roadmap task P3-09, specification
3.4, 10.8).
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from planexs.core.ConfigError import ConfigError
from planexs.core.validateConfig import validateConfig
from planexs.surrogates.PredictionConfig import PredictionConfig
from planexs.surrogates.SamplingConfig import SamplingConfig

OPTIONS: list[type[BaseModel]] = [PredictionConfig, SamplingConfig]


class TestTheOptions:
    def test_a_prediction_leaves_out_the_noise_by_default(self) -> None:
        # Specification 5.6: by default a prediction reports the uncertainty of the function.
        assert PredictionConfig().include_observation_noise is False

    def test_sampling_draws_256_noise_free_samples_by_default(self) -> None:
        options = SamplingConfig()

        assert options.n_samples == 256
        assert options.include_observation_noise is False

    @pytest.mark.parametrize("schema", OPTIONS)
    def test_every_field_is_described(self, schema: type[BaseModel]) -> None:
        for name, field in schema.model_fields.items():
            assert field.description, f"{schema.__name__}.{name} has no description"

    @pytest.mark.parametrize("schema", OPTIONS)
    def test_they_are_frozen(self, schema: type[BaseModel]) -> None:
        options = schema()

        with pytest.raises(ValidationError):
            options.include_observation_noise = True  # type: ignore[attr-defined]

    @pytest.mark.parametrize("schema", OPTIONS)
    def test_an_unknown_field_is_refused(self, schema: type[BaseModel]) -> None:
        with pytest.raises(ValidationError, match="extra"):
            schema(include_noise=True)

    @pytest.mark.parametrize("schema", OPTIONS)
    def test_the_noise_flag_must_be_a_boolean(self, schema: type[BaseModel]) -> None:
        with pytest.raises(ValidationError):
            schema(include_observation_noise="yes")

    @pytest.mark.parametrize("n_samples", [0, -3, 1.5, "10", True])
    def test_the_number_of_samples_is_a_whole_number_of_one_or_more(
        self, n_samples: Any
    ) -> None:
        with pytest.raises(ValidationError):
            SamplingConfig(n_samples=n_samples)

    def test_one_sample_is_enough(self) -> None:
        assert SamplingConfig(n_samples=1).n_samples == 1

    @pytest.mark.parametrize("schema", OPTIONS)
    def test_they_survive_json(self, schema: type[BaseModel]) -> None:
        options = schema(include_observation_noise=True)

        assert schema.model_validate_json(options.model_dump_json()) == options

    def test_an_error_names_the_field_as_every_configuration_does(self) -> None:
        with pytest.raises(ConfigError) as error:
            validateConfig(SamplingConfig, {"n_samples": 0}, label="sampling")

        assert error.value.field_path == "n_samples"
