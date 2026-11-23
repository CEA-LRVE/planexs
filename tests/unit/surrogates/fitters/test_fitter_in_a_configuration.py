"""A fitter as the nested component of a configuration (roadmap task P3-13, specification
5.12, 3.8).

This is how `single_task_gp_v1` (roadmap P3-14) will hold its fitter: a field marked
`ComponentOf("fitter")`, with `map_v1` as its default, written in the schema and nowhere else.
The tests make a stand-in model with that one field, and check what a surrogate will rely on:
the fitter is picked by name, a missing field takes the default, a configuration from Python,
JSON and YAML means the same thing, the resolved configuration says everything, and a mistake
is reported at the full path.
"""

from __future__ import annotations

from typing import Annotated, Any

import pytest
from fitter_fakes import defineFake, generatorOf, toyModel
from pydantic import BaseModel, ConfigDict, Field

from planexs.core.ComponentConfig import ComponentConfig
from planexs.core.ComponentOf import ComponentOf
from planexs.core.ConfigError import ConfigError
from planexs.core.configFiles import dumpConfig, parseConfig
from planexs.core.KindTable import KindTable
from planexs.core.resolveComponent import resolveComponent
from planexs.core.toResolvedConfig import toResolvedConfig
from planexs.core.validateConfig import validateConfig
from planexs.surrogates.fitters.fitterRegistry import fitterRegistry
from planexs.surrogates.fitters.MapFitterConfig import MapFitterConfig


class StandInSurrogateConfig(BaseModel):
    """What a surrogate's configuration holds of a fitter: one nested component, with a default
    that is a mapping, so that the schema does not import the fitter it names.
    """

    model_config = ConfigDict(extra="forbid")

    fitter: Annotated[ComponentConfig, ComponentOf("fitter")] = Field(
        default_factory=lambda: {"type": "map_v1"},  # type: ignore[arg-type,return-value]
        validate_default=True,
        description="The fitter of the surrogate: the component of kind 'fitter' to use.",
    )


def make(data: dict[str, Any]) -> StandInSurrogateConfig:
    return validateConfig(StandInSurrogateConfig, data)


class TestTheFitterIsPickedByName:
    @pytest.mark.parametrize(
        ("data", "expected"),
        [
            ({"fitter": {"type": "map_v1"}}, MapFitterConfig()),
            (
                {"fitter": {"type": "map_v1", "max_attempts": 3}},
                MapFitterConfig(max_attempts=3),
            ),
            ({"fitter": {"type": "map_v1", "maxiter": 50}}, MapFitterConfig(maxiter=50)),
        ],
        ids=["default", "attempts", "iterations"],
    )
    def test_the_field_holds_the_validated_configuration_of_the_named_fitter(
        self, data: dict[str, Any], expected: ComponentConfig
    ) -> None:
        assert make(data).fitter == expected

    def test_the_fitter_it_names_fits_through_the_registry(self) -> None:
        config = make({"fitter": {"type": "map_v1", "maxiter": 5}}).fitter
        model = toyModel()

        entry = fitterRegistry().lookup(config.type)
        entry.builder(config).fit(model, generator=generatorOf(0))

        assert not model.training

    def test_a_missing_fitter_takes_the_default_written_in_the_schema(self) -> None:
        assert make({}).fitter == MapFitterConfig()
        assert StandInSurrogateConfig().fitter == MapFitterConfig()

    def test_a_configuration_built_in_python_is_accepted_if_it_fits_its_type(self) -> None:
        assert make({"fitter": MapFitterConfig(max_attempts=2)}).fitter == MapFitterConfig(
            max_attempts=2
        )


class TestAMistakeIsReportedAtTheFullPath:
    def test_a_number_of_attempts_below_one(self) -> None:
        with pytest.raises(ConfigError) as caught:
            make({"fitter": {"type": "map_v1", "max_attempts": 0}})

        assert caught.value.field_path == "fitter.max_attempts"

    def test_a_cap_on_iterations_below_one(self) -> None:
        with pytest.raises(ConfigError) as caught:
            make({"fitter": {"type": "map_v1", "maxiter": 0}})

        assert caught.value.field_path == "fitter.maxiter"

    def test_an_option_of_the_engine_that_is_not_exposed_is_refused_and_not_ignored(
        self,
    ) -> None:
        # `method` is an option of the engine's routine that its batched optimizer ignores, so
        # it is not a setting: a configuration that names it is an error, never a silent no-op.
        with pytest.raises(ConfigError) as caught:
            make({"fitter": {"type": "map_v1", "method": "TNC"}})

        assert caught.value.field_path == "fitter.method"

    def test_an_unknown_fitter_lists_the_registered_ones(self) -> None:
        with pytest.raises(ConfigError) as caught:
            make({"fitter": {"type": "map_v2"}})

        error = caught.value
        assert error.field_path == "fitter.type"
        assert "map_v1" in error.message

    def test_a_fitter_without_a_type(self) -> None:
        with pytest.raises(ConfigError) as caught:
            make({"fitter": {"max_attempts": 3}})

        assert caught.value.field_path == "fitter.type"


ONE_CONFIGURATION = {
    "fitter": {
        "type": "map_v1",
        "max_attempts": 3,
        "pick_best_of_all_attempts": True,
        "maxiter": 200,
    }
}


class TestTheSameMeaningInPythonJsonAndYaml:
    @pytest.mark.parametrize("file_format", ["json", "yaml"])
    def test_a_file_means_what_the_python_mapping_means(self, file_format: Any) -> None:
        text = dumpConfig(ONE_CONFIGURATION, file_format)

        assert make(parseConfig(text, file_format)) == make(ONE_CONFIGURATION)

    def test_a_yaml_file_written_by_hand(self) -> None:
        text = (
            "fitter:\n  type: map_v1\n  max_attempts: 3\n"
            "  pick_best_of_all_attempts: true\n  maxiter: 200\n"
        )

        assert make(parseConfig(text, "yaml")).fitter == MapFitterConfig(
            max_attempts=3, pick_best_of_all_attempts=True, maxiter=200
        )


class TestTheResolvedConfiguration:
    """A study stores its configuration with every default written out (specification 3.4)."""

    def test_the_default_fitter_is_written_out_in_full(self) -> None:
        assert toResolvedConfig(make({})) == {
            "fitter": {
                "type": "map_v1",
                "max_attempts": 5,
                "pick_best_of_all_attempts": False,
                "maxiter": None,
            }
        }

    def test_resolving_and_validating_again_is_a_fixed_point(self) -> None:
        resolved = toResolvedConfig(make(ONE_CONFIGURATION))

        assert toResolvedConfig(make(resolved)) == resolved

    def test_the_resolved_configuration_survives_json(self) -> None:
        resolved = toResolvedConfig(make(ONE_CONFIGURATION))

        assert parseConfig(dumpConfig(resolved, "json"), "json") == resolved


class TestAFitterFromAPlugin:
    """The extension guarantee (specification 1.3): a fitter registered outside the package is
    picked by name exactly like the built-in one, with no change to planexs.
    """

    def test_a_registered_fitter_is_picked_by_name_in_the_same_field(
        self, isolatedFitterKind: KindTable
    ) -> None:
        fitter = defineFake("adam_v1")

        config = make({"fitter": {"type": "adam_v1", "scale": 3.0}}).fitter
        model = toyModel()
        built = fitterRegistry().lookup(config.type).builder(config)
        built.fit(model, generator=generatorOf(0))

        assert config.model_dump() == {"type": "adam_v1", "scale": 3.0}
        assert isinstance(built, fitter)
        assert built.received[0][0] is model  # type: ignore[attr-defined]

    def test_it_is_listed_in_the_error_for_an_unknown_name(
        self, isolatedFitterKind: KindTable
    ) -> None:
        defineFake("adam_v1")

        with pytest.raises(ConfigError, match="adam_v1"):
            make({"fitter": {"type": "nope_v1"}})

    def test_resolving_it_directly_gives_the_same_entry(
        self, isolatedFitterKind: KindTable
    ) -> None:
        defineFake("adam_v1")

        resolved = resolveComponent(fitterRegistry(), {"type": "adam_v1"})

        assert resolved.entry is fitterRegistry().lookup("adam_v1")
