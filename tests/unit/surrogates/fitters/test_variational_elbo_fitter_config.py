"""Unit tests for `VariationalElboFitterConfig` (roadmap task P3-15, specification 3.4,
10.8)."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from planexs.core.ConfigError import ConfigError
from planexs.core.resolveComponent import resolveComponent
from planexs.surrogates.fitters.fitterRegistry import fitterRegistry
from planexs.surrogates.fitters.VariationalElboFitterConfig import VariationalElboFitterConfig


class TestTheDefaults:
    def test_the_defaults_are_written_down_in_the_schema(self) -> None:
        # The engine has no default for them (the loop is this package's own), so they are
        # pinned here: the optimizer and the rate of the GPyTorch tutorial on non-Gaussian
        # likelihoods, and a budget of twice its 50 iterations. Changing one changes how every
        # stored study fits, which is a new name (specification 3.7).
        config = VariationalElboFitterConfig()

        assert config.type == "variational_elbo_v1"
        assert config.n_iterations == 100
        assert config.learning_rate == 0.1
        assert config.optimizer == "adam"

    def test_a_mapping_that_says_only_its_type_gets_every_default(self) -> None:
        resolved = resolveComponent(fitterRegistry(), {"type": "variational_elbo_v1"})

        assert resolved.config == VariationalElboFitterConfig()

    def test_a_mapping_that_sets_every_field_is_read_as_given(self) -> None:
        resolved = resolveComponent(
            fitterRegistry(),
            {
                "type": "variational_elbo_v1",
                "n_iterations": 7,
                "learning_rate": 0.003,
                "optimizer": "sgd",
            },
        )

        assert resolved.config == VariationalElboFitterConfig(
            n_iterations=7, learning_rate=0.003, optimizer="sgd"
        )


class TestTheFieldsAreValidated:
    @pytest.mark.parametrize("value", [0, -1])
    def test_n_iterations_is_at_least_one(self, value: int) -> None:
        with pytest.raises(ValidationError, match="n_iterations"):
            VariationalElboFitterConfig(n_iterations=value)

    def test_n_iterations_is_a_whole_number(self) -> None:
        with pytest.raises(ValidationError, match="n_iterations"):
            VariationalElboFitterConfig(n_iterations=2.5)  # type: ignore[arg-type]

    @pytest.mark.parametrize("value", [0.0, -0.1, float("inf"), float("nan")])
    def test_learning_rate_is_positive_and_finite(self, value: float) -> None:
        with pytest.raises(ValidationError, match="learning_rate"):
            VariationalElboFitterConfig(learning_rate=value)

    @pytest.mark.parametrize("value", ["lbfgs", "adamw", "Adam", ""])
    def test_an_optimizer_that_is_not_one_of_the_four_is_refused(self, value: str) -> None:
        wrong: Any = value

        with pytest.raises(ValidationError, match="optimizer"):
            VariationalElboFitterConfig(optimizer=wrong)

    def test_the_smallest_legal_values_are_accepted(self) -> None:
        config = VariationalElboFitterConfig(n_iterations=1, learning_rate=1e-12)

        assert (config.n_iterations, config.learning_rate) == (1, 1e-12)

    def test_a_setting_that_is_not_a_field_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="batch_size"):
            VariationalElboFitterConfig(batch_size=32)  # type: ignore[call-arg]

    def test_another_type_is_refused_even_without_a_registry(self) -> None:
        wrong: Any = "variational_elbo_v2"

        with pytest.raises(ValidationError, match="type"):
            VariationalElboFitterConfig(type=wrong)

    @pytest.mark.parametrize(
        ("data", "path"),
        [
            ({"n_iterations": 0}, "n_iterations"),
            ({"learning_rate": -1.0}, "learning_rate"),
            ({"optimizer": "lbfgs"}, "optimizer"),
            ({"momentum": 0.9}, "momentum"),
        ],
    )
    def test_the_resolver_reports_the_path_of_a_bad_field(
        self, data: dict[str, Any], path: str
    ) -> None:
        with pytest.raises(ConfigError) as caught:
            resolveComponent(fitterRegistry(), {"type": "variational_elbo_v1", **data})

        assert caught.value.field_path == path
