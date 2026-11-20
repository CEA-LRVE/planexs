"""Unit tests for `SurrogateCache` (roadmap task P3-17, specification 5.6, 8).

The cache is a get-or-build keyed by the history version and the resolved configuration of the
surrogate. The card asks for four things: hit and miss counts, invalidation on a new trial, a
different configuration that rebuilds, and a builder that is not called twice for the same key.
The rest is what the design adds to them: one entry only, a builder that fails, and the
arguments being checked.

A builder is stood for by a counting callable that makes a new `Surrogate` over a fake model
each time (`makeSurrogate`), so that no Gaussian process is fitted and "built again" is
something a test can see: it is a new object, and the count of calls goes up.
"""

from __future__ import annotations

import gc
import logging
import weakref
from typing import Annotated, Any, Literal

import numpy as np
import pytest
from posterior_fakes import makeSurrogate
from pydantic import Field

import planexs.surrogates  # noqa: F401  (defines the `kernel` kind that the stand-in nests)
from planexs.core.ComponentConfig import ComponentConfig
from planexs.core.ComponentOf import ComponentOf
from planexs.core.ConfigError import ConfigError
from planexs.core.ModelNotReadyError import ModelNotReadyError
from planexs.surrogates.Surrogate import Surrogate
from planexs.surrogates.SurrogateCache import SurrogateCache


class StandInSurrogateConfig(ComponentConfig):
    """What a surrogate's configuration looks like to the cache: settings of its own and a
    nested component, with defaults that the resolved configuration writes out.
    """

    type: Literal["stand_in_v1"] = Field(default="stand_in_v1", description="The name.")
    min_observations: int = Field(default=3, ge=1, description="A setting of the surrogate.")
    noise: float = Field(default=0.5, description="Another setting of the surrogate.")
    kernel: Annotated[ComponentConfig, ComponentOf("kernel")] = Field(
        default_factory=lambda: {"type": "botorch_default_v1"},  # type: ignore[arg-type]
        validate_default=True,
        description="The kernel: a nested component.",
    )


class OtherStandInConfig(ComponentConfig):
    """Another surrogate: the settings of `StandInSurrogateConfig`, and another type."""

    type: Literal["other_v1"] = Field(default="other_v1", description="The name.")
    min_observations: int = Field(default=3, ge=1, description="A setting of the surrogate.")
    noise: float = Field(default=0.5, description="Another setting of the surrogate.")


class MappingStandInConfig(ComponentConfig):
    """A surrogate with a setting that is a mapping, such as a setting for each outcome."""

    type: Literal["mapping_stand_in_v1"] = Field(
        default="mapping_stand_in_v1", description="The name."
    )
    noise: dict[str, float] = Field(
        default_factory=dict, description="A noise for each outcome."
    )


class CountingBuilder:
    """Stands for the fitting of a surrogate: counts its calls, makes a new one each time."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self) -> Surrogate:
        self.calls += 1
        return makeSurrogate()


def config(**settings: Any) -> StandInSurrogateConfig:
    return StandInSurrogateConfig(**settings)


class TestAnEmptyCache:
    def test_it_has_had_no_hit_and_no_miss(self) -> None:
        cache = SurrogateCache()

        assert cache.hits == 0 and cache.misses == 0

    def test_two_caches_share_nothing(self) -> None:
        # No module-level state (specification 8): a study owns its cache.
        first, second = SurrogateCache(), SurrogateCache()
        builder = CountingBuilder()

        built = first.getOrBuild(0, config(), builder)
        again = second.getOrBuild(0, config(), builder)

        assert again is not built  # the second cache did not see the first one's entry
        assert (first.hits, first.misses) == (0, 1)
        assert (second.hits, second.misses) == (0, 1)
        assert builder.calls == 2


class TestHitAndMissCounts:
    def test_the_first_call_is_a_miss_and_builds(self) -> None:
        cache, builder = SurrogateCache(), CountingBuilder()

        cache.getOrBuild(0, config(), builder)

        assert (cache.hits, cache.misses) == (0, 1)
        assert builder.calls == 1

    def test_a_call_for_the_same_key_is_a_hit_and_builds_nothing(self) -> None:
        cache, builder = SurrogateCache(), CountingBuilder()
        cache.getOrBuild(0, config(), builder)

        cache.getOrBuild(0, config(), builder)

        assert (cache.hits, cache.misses) == (1, 1)
        assert builder.calls == 1

    def test_the_counts_add_up_over_a_run_of_asks_and_tells(self) -> None:
        # A study's life: several asks for each state of the history, then a new trial.
        cache, builder = SurrogateCache(), CountingBuilder()
        versions = [0, 0, 1, 1, 1, 2, 3, 3]

        for version in versions:
            cache.getOrBuild(version, config(), builder)

        assert (cache.hits, cache.misses) == (4, 4)
        assert cache.hits + cache.misses == len(versions)
        assert builder.calls == cache.misses  # the builder is called once for each miss

    def test_the_counts_are_read_without_touching_the_cache(self) -> None:
        cache, builder = SurrogateCache(), CountingBuilder()
        cache.getOrBuild(0, config(), builder)

        for _ in range(3):
            assert (cache.hits, cache.misses) == (0, 1)

        assert builder.calls == 1


class TestTheBuilderIsNotCalledTwiceForTheSameKey:
    def test_many_calls_build_once_and_return_the_very_same_surrogate(self) -> None:
        # What `Study.surrogate()` relies on (roadmap P6-14): the policy and the caller are
        # given one and the same object.
        cache, builder = SurrogateCache(), CountingBuilder()

        answers = [cache.getOrBuild(7, config(), builder) for _ in range(50)]

        assert builder.calls == 1
        assert all(answer is answers[0] for answer in answers)
        assert isinstance(answers[0], Surrogate)
        assert (cache.hits, cache.misses) == (49, 1)

    def test_the_surrogate_returned_the_first_time_is_the_one_the_builder_made(self) -> None:
        cache = SurrogateCache()
        made = makeSurrogate()

        returned = cache.getOrBuild(0, config(), lambda: made)

        assert returned is made

    def test_two_equal_configurations_are_one_key_whatever_the_objects(self) -> None:
        cache, builder = SurrogateCache(), CountingBuilder()
        first, second = config(noise=0.25), config(noise=0.25)
        assert first is not second

        built = cache.getOrBuild(0, first, builder)

        assert cache.getOrBuild(0, second, builder) is built
        assert builder.calls == 1

    def test_a_setting_left_out_and_given_as_its_default_are_one_configuration(self) -> None:
        # The key is the resolved configuration: every default written out.
        cache, builder = SurrogateCache(), CountingBuilder()
        implicit = config()
        explicit = config(min_observations=3, noise=0.5, kernel={"type": "botorch_default_v1"})

        built = cache.getOrBuild(0, implicit, builder)

        assert cache.getOrBuild(0, explicit, builder) is built
        assert builder.calls == 1

    def test_a_configuration_with_a_nan_is_still_the_same_configuration(self) -> None:
        # Comparing the objects, or dictionaries of them, would never find a `nan` equal to
        # itself, and would build at every call.
        cache, builder = SurrogateCache(), CountingBuilder()

        built = cache.getOrBuild(0, config(noise=float("nan")), builder)

        assert cache.getOrBuild(0, config(noise=float("nan")), builder) is built
        assert builder.calls == 1


class TestTheOrderOfTheEntriesOfAMappingIsNotPartOfTheKey:
    def test_two_equal_mappings_written_in_another_order_are_one_configuration(self) -> None:
        # A dictionary dumps its entries in the order they were put in, and the two below are
        # equal: the snapshot has sorted keys, so the same configuration is not fitted twice.
        cache, builder = SurrogateCache(), CountingBuilder()
        first = MappingStandInConfig(noise={"yield": 0.1, "cost": 0.2})
        second = MappingStandInConfig(noise={"cost": 0.2, "yield": 0.1})
        assert first == second and list(first.noise) != list(second.noise)

        built = cache.getOrBuild(0, first, builder)

        assert cache.getOrBuild(0, second, builder) is built
        assert builder.calls == 1

    def test_a_different_value_in_the_mapping_is_still_a_different_configuration(self) -> None:
        cache, builder = SurrogateCache(), CountingBuilder()
        cache.getOrBuild(0, MappingStandInConfig(noise={"yield": 0.1}), builder)

        cache.getOrBuild(0, MappingStandInConfig(noise={"yield": 0.2}), builder)

        assert builder.calls == 2


class TestANewTrialInvalidates:
    def test_a_new_history_version_builds_a_new_surrogate(self) -> None:
        cache, builder = SurrogateCache(), CountingBuilder()
        before = cache.getOrBuild(0, config(), builder)

        after = cache.getOrBuild(1, config(), builder)

        assert after is not before
        assert builder.calls == 2
        assert (cache.hits, cache.misses) == (0, 2)

    def test_the_new_surrogate_is_then_served_from_the_cache(self) -> None:
        cache, builder = SurrogateCache(), CountingBuilder()
        cache.getOrBuild(0, config(), builder)
        after = cache.getOrBuild(1, config(), builder)

        assert cache.getOrBuild(1, config(), builder) is after
        assert builder.calls == 2

    @pytest.mark.parametrize("version", [1, 2, 41])
    def test_any_other_version_is_another_key(self, version: int) -> None:
        cache, builder = SurrogateCache(), CountingBuilder()
        cache.getOrBuild(0, config(), builder)

        cache.getOrBuild(version, config(), builder)

        assert builder.calls == 2

    def test_a_key_is_equal_or_not_and_a_lower_version_is_another_key_too(self) -> None:
        # The cache does not order versions: it trusts the caller that a version that
        # differs is a history that differs.
        cache, builder = SurrogateCache(), CountingBuilder()

        for version in (5, 3, 5):
            cache.getOrBuild(version, config(), builder)

        assert (cache.hits, cache.misses) == (0, 3)

    def test_a_numpy_integer_is_a_version_like_any_other(self) -> None:
        cache, builder = SurrogateCache(), CountingBuilder()
        built = cache.getOrBuild(3, config(), builder)

        assert cache.getOrBuild(np.int64(3), config(), builder) is built
        assert cache.getOrBuild(np.int32(4), config(), builder) is not built
        assert builder.calls == 2


class TestADifferentConfigurationRebuilds:
    @pytest.mark.parametrize(
        "settings",
        [
            {"noise": 0.75},
            {"min_observations": 4},
            {"kernel": {"type": "rbf_v1"}},
            {"kernel": {"type": "matern_v1"}},
            {"kernel": {"type": "matern_v1", "nu": 1.5}},
            {"kernel": {"type": "matern_v1", "ard": False}},
        ],
        ids=["noise", "min_observations", "rbf", "matern", "nested-nu", "nested-ard"],
    )
    def test_a_change_of_a_setting_or_of_a_nested_one_is_a_new_key(
        self, settings: dict[str, Any]
    ) -> None:
        cache, builder = SurrogateCache(), CountingBuilder()
        built = cache.getOrBuild(0, config(), builder)

        rebuilt = cache.getOrBuild(0, config(**settings), builder)

        assert rebuilt is not built
        assert builder.calls == 2
        assert (cache.hits, cache.misses) == (0, 2)

    def test_the_same_nested_component_with_the_same_settings_is_not_a_change(self) -> None:
        cache, builder = SurrogateCache(), CountingBuilder()
        nested = {"type": "matern_v1", "nu": 1.5}
        built = cache.getOrBuild(0, config(kernel=nested), builder)

        assert cache.getOrBuild(0, config(kernel=dict(nested)), builder) is built
        assert builder.calls == 1

    def test_another_type_of_surrogate_with_the_same_settings_is_a_new_key(self) -> None:
        cache, builder = SurrogateCache(), CountingBuilder()
        built = cache.getOrBuild(0, StandInSurrogateConfig(), builder)

        rebuilt = cache.getOrBuild(0, OtherStandInConfig(), builder)

        assert rebuilt is not built
        assert builder.calls == 2

    def test_a_configuration_changed_in_place_after_the_build_is_a_new_key(self) -> None:
        # A configuration is a mutable object. The cache keeps a snapshot of what it was when
        # the surrogate was built, so the same object, changed, is a different configuration.
        cache, builder = SurrogateCache(), CountingBuilder()
        settings = config()
        built = cache.getOrBuild(0, settings, builder)

        settings.noise = 0.9
        rebuilt = cache.getOrBuild(0, settings, builder)

        assert rebuilt is not built
        assert builder.calls == 2

    def test_a_nested_component_changed_in_place_is_a_new_key_too(self) -> None:
        cache, builder = SurrogateCache(), CountingBuilder()
        settings = config(kernel={"type": "matern_v1", "nu": 2.5})
        built = cache.getOrBuild(0, settings, builder)

        settings.kernel.nu = 0.5  # type: ignore[attr-defined]
        rebuilt = cache.getOrBuild(0, settings, builder)

        assert rebuilt is not built
        assert builder.calls == 2

    def test_a_new_version_and_a_new_configuration_together_are_one_miss(self) -> None:
        cache, builder = SurrogateCache(), CountingBuilder()
        cache.getOrBuild(0, config(), builder)

        cache.getOrBuild(1, config(noise=0.75), builder)

        assert (cache.hits, cache.misses) == (0, 2)
        assert builder.calls == 2


class TestOnlyTheLatestEntryIsKept:
    def test_going_back_to_a_key_that_was_replaced_is_a_miss(self) -> None:
        cache, builder = SurrogateCache(), CountingBuilder()
        first = cache.getOrBuild(0, config(), builder)
        cache.getOrBuild(1, config(), builder)

        again = cache.getOrBuild(0, config(), builder)

        assert again is not first
        assert builder.calls == 3
        assert (cache.hits, cache.misses) == (0, 3)

    def test_two_configurations_asked_in_turn_for_one_history_rebuild_every_time(self) -> None:
        # The price of keeping one entry, which the card chooses: a caller that alternates
        # between two configurations for the same history (the policy's own and an explicit
        # one, roadmap P6-14) fits at every call.
        cache, builder = SurrogateCache(), CountingBuilder()

        for settings in (config(), config(noise=0.75), config(), config(noise=0.75)):
            cache.getOrBuild(0, settings, builder)

        assert (cache.hits, cache.misses) == (0, 4)

    def test_the_replaced_surrogate_is_released(self) -> None:
        # "Only the latest entry is kept", in memory too: nothing holds the old model once a
        # newer one has replaced it, so a long study does not pile up fitted models.
        cache, builder = SurrogateCache(), CountingBuilder()
        old = weakref.ref(cache.getOrBuild(0, config(), builder))
        assert old() is not None

        cache.getOrBuild(1, config(), builder)
        gc.collect()

        assert old() is None

    def test_the_latest_surrogate_is_kept_alive_by_the_cache(self) -> None:
        cache, builder = SurrogateCache(), CountingBuilder()
        latest = weakref.ref(cache.getOrBuild(0, config(), builder))

        gc.collect()

        assert latest() is not None
        assert cache.getOrBuild(0, config(), builder) is latest()


class TestABuilderThatFails:
    """A failure is not a surrogate: it is raised as it was, never stored."""

    def test_the_error_of_the_builder_is_raised_unchanged(self) -> None:
        # A policy tells "not enough data" from any other error by the type of the exception
        # (roadmap P6-04), so the cache must not wrap it.
        cache = SurrogateCache()
        error = ModelNotReadyError("fewer than 2 observations")

        def failing() -> Surrogate:
            raise error

        with pytest.raises(ModelNotReadyError) as caught:
            cache.getOrBuild(0, config(), failing)

        assert caught.value is error

    def test_a_failure_is_not_stored_and_the_next_call_tries_again(self) -> None:
        cache, builder = SurrogateCache(), CountingBuilder()
        attempts = {"count": 0}

        def failsOnce() -> Surrogate:
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise ModelNotReadyError("not yet")
            return builder()

        with pytest.raises(ModelNotReadyError):
            cache.getOrBuild(0, config(), failsOnce)
        built = cache.getOrBuild(0, config(), failsOnce)

        assert attempts["count"] == 2
        assert cache.getOrBuild(0, config(), failsOnce) is built  # now it is held
        assert attempts["count"] == 2

    def test_a_failed_call_is_a_miss_that_called_the_builder(self) -> None:
        cache = SurrogateCache()
        calls = {"count": 0}

        def failing() -> Surrogate:
            calls["count"] += 1
            raise ModelNotReadyError("not yet")

        for _ in range(3):
            with pytest.raises(ModelNotReadyError):
                cache.getOrBuild(0, config(), failing)

        assert (cache.hits, cache.misses) == (0, 3)
        assert calls["count"] == cache.misses

    def test_the_entry_that_was_held_survives_a_failure_for_another_key(self) -> None:
        cache, builder = SurrogateCache(), CountingBuilder()
        held = cache.getOrBuild(0, config(), builder)

        def failing() -> Surrogate:
            raise RuntimeError("the fit did not converge")

        with pytest.raises(RuntimeError, match="did not converge"):
            cache.getOrBuild(1, config(), failing)

        assert cache.getOrBuild(0, config(), builder) is held  # still right for its own key
        assert builder.calls == 1

    def test_an_interrupt_leaves_the_cache_as_it_was(self) -> None:
        cache, builder = SurrogateCache(), CountingBuilder()
        held = cache.getOrBuild(0, config(), builder)

        def interrupted() -> Surrogate:
            raise KeyboardInterrupt

        with pytest.raises(KeyboardInterrupt):
            cache.getOrBuild(1, config(), interrupted)

        assert cache.getOrBuild(0, config(), builder) is held


class TestTheArgumentsAreChecked:
    @pytest.mark.parametrize("bad", [-1, -40, True, False, 1.5, 2.0, None, "1", [1]])
    def test_a_history_version_that_is_not_a_whole_number_of_zero_or_more(
        self, bad: Any
    ) -> None:
        cache, builder = SurrogateCache(), CountingBuilder()

        with pytest.raises(ConfigError) as caught:
            cache.getOrBuild(bad, config(), builder)

        error = caught.value
        assert "whole number of zero or more" in error.message
        assert repr(bad) in error.message
        assert error.component == "surrogate cache"
        assert error.field_path == "history_version"
        assert "changes whenever a trial changes state" in (error.suggestion or "")

    def test_a_bad_call_builds_nothing_and_counts_for_nothing(self) -> None:
        cache, builder = SurrogateCache(), CountingBuilder()

        with pytest.raises(ConfigError):
            cache.getOrBuild(-1, config(), builder)

        assert builder.calls == 0
        assert (cache.hits, cache.misses) == (0, 0)

    def test_a_bad_call_is_refused_even_when_the_cache_holds_an_entry(self) -> None:
        cache, builder = SurrogateCache(), CountingBuilder()
        cache.getOrBuild(0, config(), builder)

        with pytest.raises(ConfigError):
            cache.getOrBuild(True, config(), builder)  # `True` is not version 1, nor 0

        assert (cache.hits, cache.misses) == (0, 1)

    @pytest.mark.parametrize(
        "bad",
        [{"type": "stand_in_v1"}, None, "stand_in_v1", 3],
        ids=["mapping", "none", "string", "number"],
    )
    def test_a_configuration_that_is_not_a_validated_component_configuration(
        self, bad: Any
    ) -> None:
        cache, builder = SurrogateCache(), CountingBuilder()

        with pytest.raises(ConfigError) as caught:
            cache.getOrBuild(0, bad, builder)

        error = caught.value
        assert "must be a validated component configuration" in error.message
        assert type(bad).__name__ in error.message
        assert error.component == "surrogate cache"
        assert error.field_path == "config"
        assert "resolveComponent" in (error.suggestion or "")
        assert builder.calls == 0


class TestLogging:
    LOGGER = "planexs.surrogates.SurrogateCache"

    def test_a_miss_and_a_hit_are_logged_at_debug_with_the_version(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # Why was my surrogate fitted again? The answer is in the log, at a level that a
        # library does not show unless it is asked to (specification 8).
        cache, builder = SurrogateCache(), CountingBuilder()

        with caplog.at_level(logging.DEBUG, logger=self.LOGGER):
            cache.getOrBuild(4, config(), builder)
            cache.getOrBuild(4, config(), builder)

        messages = [record.getMessage() for record in caplog.records]
        assert messages == [
            "No surrogate for history version 4 and this configuration: building one.",
            "Reusing the surrogate fitted for history version 4.",
        ]
        assert all(record.levelno == logging.DEBUG for record in caplog.records)

    def test_nothing_is_logged_above_debug(self, caplog: pytest.LogCaptureFixture) -> None:
        cache, builder = SurrogateCache(), CountingBuilder()

        with caplog.at_level(logging.INFO, logger=self.LOGGER):
            cache.getOrBuild(0, config(), builder)
            cache.getOrBuild(0, config(), builder)

        assert caplog.records == []
