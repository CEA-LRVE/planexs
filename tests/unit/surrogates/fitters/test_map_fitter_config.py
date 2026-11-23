"""Unit tests for `MapFitterConfig` (roadmap task P3-13, specification 3.4, 10.8)."""

from __future__ import annotations

import inspect
from typing import Any

import pytest
from botorch import fit as engine
from pydantic import ValidationError

from planexs.core.ConfigError import ConfigError
from planexs.core.resolveComponent import resolveComponent
from planexs.surrogates.fitters.fitterRegistry import fitterRegistry
from planexs.surrogates.fitters.MapFitterConfig import MapFitterConfig


class TestTheDefaults:
    def test_the_defaults_are_the_engines_own(self) -> None:
        # The defaults are written down in the schema so that a later change of the engine does
        # not alter how a stored study fits (specification 3.4). This pins them against the
        # installed engine: a BoTorch that changes its own is noticed here, and the defaults
        # (and a new name, as specification 3.7 says) are then a decision, not an accident.
        parameters = inspect.signature(engine._fit_fallback).parameters
        defaults = MapFitterConfig()

        assert parameters["max_attempts"].default == defaults.max_attempts
        assert (
            parameters["pick_best_of_all_attempts"].default
            == defaults.pick_best_of_all_attempts
        )

    def test_the_optimizer_cap_is_left_to_the_engine_by_default(self) -> None:
        assert MapFitterConfig().maxiter is None

    def test_a_mapping_that_says_only_its_type_gets_every_default(self) -> None:
        resolved = resolveComponent(fitterRegistry(), {"type": "map_v1"})

        assert resolved.config == MapFitterConfig()


class TestTheFieldsAreValidated:
    @pytest.mark.parametrize("value", [0, -1])
    def test_max_attempts_is_at_least_one(self, value: int) -> None:
        with pytest.raises(ValidationError, match="max_attempts"):
            MapFitterConfig(max_attempts=value)

    @pytest.mark.parametrize("value", [0, -5])
    def test_maxiter_is_at_least_one_when_it_is_given(self, value: int) -> None:
        with pytest.raises(ValidationError, match="maxiter"):
            MapFitterConfig(maxiter=value)

    def test_one_attempt_and_one_iteration_are_legal_values(self) -> None:
        config = MapFitterConfig(max_attempts=1, maxiter=1)

        assert (config.max_attempts, config.maxiter) == (1, 1)

    def test_a_setting_that_is_not_a_field_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="timeout_sec"):
            MapFitterConfig(timeout_sec=10.0)  # type: ignore[call-arg]

    def test_another_type_is_refused_even_without_a_registry(self) -> None:
        wrong: Any = "map_v2"

        with pytest.raises(ValidationError, match="type"):
            MapFitterConfig(type=wrong)

    def test_the_resolver_reports_the_path_of_a_bad_field(self) -> None:
        with pytest.raises(ConfigError) as caught:
            resolveComponent(fitterRegistry(), {"type": "map_v1", "max_attempts": 0})

        assert caught.value.field_path == "max_attempts"
