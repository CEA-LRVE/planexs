"""A list of outcome transforms as the nested components of a configuration (roadmap task
P3-12, specification 5.12, 3.8).

This is how `single_task_gp_v1` (roadmap P3-14) will hold its transforms: a field that is a
list whose elements are marked `ComponentOf("outcome_transform")`, with `standardize_v1` as
the default, written in the schema and nowhere else. The tests make a stand-in model with that
one field, and check what a surrogate will rely on: the transforms are picked by name and keep
their order, a missing field takes the default and an empty list stays empty, a configuration
from Python, JSON and YAML means the same thing, the resolved configuration says everything, a
mistake is reported at the full path of the element, and the list builds the one transform that
a model is given.
"""

from __future__ import annotations

from typing import Annotated, Any

import pytest
from botorch.models.transforms.outcome import ChainedOutcomeTransform, Standardize
from outcome_transform_fakes import defineFake
from pydantic import BaseModel, ConfigDict, Field

import planexs.surrogates  # noqa: F401
from planexs.core.ComponentConfig import ComponentConfig
from planexs.core.ComponentOf import ComponentOf
from planexs.core.ConfigError import ConfigError
from planexs.core.configFiles import dumpConfig, parseConfig
from planexs.core.KindTable import KindTable
from planexs.core.toResolvedConfig import toResolvedConfig
from planexs.core.validateConfig import validateConfig
from planexs.surrogates.outcome_transforms.buildOutcomeTransform import buildOutcomeTransform
from planexs.surrogates.outcome_transforms.LogOutcomeTransformConfig import (
    LogOutcomeTransformConfig,
)
from planexs.surrogates.outcome_transforms.NoneOutcomeTransformConfig import (
    NoneOutcomeTransformConfig,
)
from planexs.surrogates.outcome_transforms.StandardizeOutcomeTransformConfig import (
    StandardizeOutcomeTransformConfig,
)


class StandInSurrogateConfig(BaseModel):
    """What a surrogate's configuration holds of its transforms: a list of nested components,
    with a default that is a list of mappings, so that the schema does not import the
    transform it names.
    """

    model_config = ConfigDict(extra="forbid")

    outcome_transform: list[Annotated[ComponentConfig, ComponentOf("outcome_transform")]] = (
        Field(
            default_factory=lambda: [{"type": "standardize_v1"}],  # type: ignore[list-item]
            validate_default=True,
            description="The outcome transforms of the surrogate, applied in order.",
        )
    )


def make(data: dict[str, Any]) -> StandInSurrogateConfig:
    return validateConfig(StandInSurrogateConfig, data)


def steps(data: dict[str, Any]) -> list[str]:
    return [step.type for step in make(data).outcome_transform]


class TestTheTransformsArePickedByNameAndKeepTheirOrder:
    @pytest.mark.parametrize(
        ("data", "expected"),
        [
            (
                {"outcome_transform": [{"type": "standardize_v1"}]},
                [StandardizeOutcomeTransformConfig()],
            ),
            ({"outcome_transform": [{"type": "log_v1"}]}, [LogOutcomeTransformConfig()]),
            ({"outcome_transform": [{"type": "none_v1"}]}, [NoneOutcomeTransformConfig()]),
            (
                {
                    "outcome_transform": [
                        {"type": "log_v1"},
                        {"type": "standardize_v1", "min_std": 0.5},
                    ]
                },
                [LogOutcomeTransformConfig(), StandardizeOutcomeTransformConfig(min_std=0.5)],
            ),
        ],
        ids=["standardize", "log", "none", "log-then-standardize"],
    )
    def test_the_field_holds_the_validated_configurations_in_the_order_given(
        self, data: dict[str, Any], expected: list[ComponentConfig]
    ) -> None:
        assert make(data).outcome_transform == expected

    def test_the_order_of_the_list_is_the_order_of_the_field(self) -> None:
        forward = steps(
            {"outcome_transform": [{"type": "log_v1"}, {"type": "standardize_v1"}]}
        )
        backward = steps(
            {"outcome_transform": [{"type": "standardize_v1"}, {"type": "log_v1"}]}
        )

        assert forward == ["log_v1", "standardize_v1"]
        assert backward == ["standardize_v1", "log_v1"]

    def test_a_missing_field_takes_the_default_written_in_the_schema(self) -> None:
        assert make({}).outcome_transform == [StandardizeOutcomeTransformConfig()]
        assert StandInSurrogateConfig().outcome_transform == [
            StandardizeOutcomeTransformConfig()
        ]

    def test_an_empty_list_stays_empty_and_is_not_replaced_by_the_default(self) -> None:
        # "No transform" is a choice that a configuration can make, and it is not the same as
        # saying nothing.
        assert make({"outcome_transform": []}).outcome_transform == []

    def test_a_configuration_built_in_python_is_accepted_if_it_fits_its_type(self) -> None:
        built = StandardizeOutcomeTransformConfig(min_std=0.5)

        assert make({"outcome_transform": [built]}).outcome_transform == [built]


class TestTheListBuildsTheTransformAModelIsGiven:
    def test_the_default_builds_the_standardize_that_botorch_builds_by_default(self) -> None:
        transform = buildOutcomeTransform(make({}).outcome_transform, 2)

        assert type(transform) is Standardize

    def test_a_list_builds_a_chain_in_the_order_given(self) -> None:
        data = {"outcome_transform": [{"type": "log_v1"}, {"type": "standardize_v1"}]}

        transform = buildOutcomeTransform(make(data).outcome_transform, 1)

        assert isinstance(transform, ChainedOutcomeTransform)
        assert list(transform.keys()) == ["0_log_v1", "1_standardize_v1"]

    @pytest.mark.parametrize(
        "listed", [[], [{"type": "none_v1"}], [{"type": "none_v1"}, {"type": "none_v1"}]]
    )
    def test_a_list_with_nothing_to_apply_builds_no_transform(self, listed: list[Any]) -> None:
        assert (
            buildOutcomeTransform(make({"outcome_transform": listed}).outcome_transform, 1)
            is None
        )


class TestAMistakeIsReportedAtTheFullPath:
    def test_a_bad_setting_of_the_second_transform_is_found_at_its_position(self) -> None:
        with pytest.raises(ConfigError) as caught:
            make(
                {
                    "outcome_transform": [
                        {"type": "log_v1"},
                        {"type": "standardize_v1", "min_std": 0.0},
                    ]
                }
            )

        assert caught.value.field_path == "outcome_transform[1].min_std"

    def test_a_setting_that_belongs_to_another_transform(self) -> None:
        with pytest.raises(ConfigError) as caught:
            make({"outcome_transform": [{"type": "log_v1", "min_std": 1e-8}]})

        assert caught.value.field_path == "outcome_transform[0].min_std"

    def test_an_unknown_transform_lists_the_registered_ones(self) -> None:
        with pytest.raises(ConfigError) as caught:
            make({"outcome_transform": [{"type": "standardize_v1"}, {"type": "box_cox_v1"}]})

        error = caught.value
        assert error.field_path == "outcome_transform[1].type"
        for name in ("log_v1", "none_v1", "standardize_v1"):
            assert name in error.message

    def test_a_transform_without_a_type(self) -> None:
        with pytest.raises(ConfigError) as caught:
            make({"outcome_transform": [{"min_std": 0.5}]})

        assert caught.value.field_path == "outcome_transform[0].type"

    def test_one_transform_where_a_list_is_expected(self) -> None:
        with pytest.raises(ConfigError) as caught:
            make({"outcome_transform": {"type": "standardize_v1"}})

        assert caught.value.field_path == "outcome_transform"


ONE_CONFIGURATION = {
    "outcome_transform": [
        {"type": "log_v1"},
        {"type": "none_v1"},
        {"type": "standardize_v1", "min_std": 1e-6},
    ]
}


class TestTheSameMeaningInPythonJsonAndYaml:
    @pytest.mark.parametrize("file_format", ["json", "yaml"])
    def test_a_file_means_what_the_python_mapping_means(self, file_format: Any) -> None:
        text = dumpConfig(ONE_CONFIGURATION, file_format)

        assert make(parseConfig(text, file_format)) == make(ONE_CONFIGURATION)

    def test_a_yaml_file_written_by_hand(self) -> None:
        text = (
            "outcome_transform:\n"
            "  - type: log_v1\n"
            "  - type: standardize_v1\n"
            "    min_std: 1.0e-6\n"
        )

        assert make(parseConfig(text, "yaml")).outcome_transform == [
            LogOutcomeTransformConfig(),
            StandardizeOutcomeTransformConfig(min_std=1e-6),
        ]


class TestTheResolvedConfiguration:
    """A study stores its configuration with every default written out (specification 3.4)."""

    def test_the_default_list_is_written_out_in_full(self) -> None:
        assert toResolvedConfig(make({})) == {
            "outcome_transform": [{"type": "standardize_v1", "min_std": 1e-8}]
        }

    def test_a_transform_that_leaves_settings_out_has_them_written_out(self) -> None:
        resolved = toResolvedConfig(
            make({"outcome_transform": [{"type": "log_v1"}, {"type": "standardize_v1"}]})
        )

        assert resolved == {
            "outcome_transform": [
                {"type": "log_v1"},
                {"type": "standardize_v1", "min_std": 1e-8},
            ]
        }

    def test_an_empty_list_is_written_out_as_empty(self) -> None:
        assert toResolvedConfig(make({"outcome_transform": []})) == {"outcome_transform": []}

    @pytest.mark.parametrize(
        "listed",
        [[], [{"type": "none_v1"}], [{"type": "log_v1"}, {"type": "standardize_v1"}]],
        ids=["empty", "none", "log-then-standardize"],
    )
    def test_resolving_and_validating_again_is_a_fixed_point(self, listed: list[Any]) -> None:
        resolved = toResolvedConfig(make({"outcome_transform": listed}))

        assert toResolvedConfig(make(resolved)) == resolved

    def test_the_resolved_configuration_survives_json(self) -> None:
        resolved = toResolvedConfig(make(ONE_CONFIGURATION))

        assert parseConfig(dumpConfig(resolved, "json"), "json") == resolved


class TestATransformFromAPlugin:
    """The extension guarantee (specification 1.3): a transform registered outside the package
    is picked by name exactly like a built-in one, with no change to planexs.
    """

    def test_a_registered_transform_is_picked_by_name_in_the_same_list(
        self, isolatedOutcomeTransformKind: KindTable
    ) -> None:
        defineFake("winsorize_v1")

        listed = make(
            {"outcome_transform": [{"type": "log_v1"}, {"type": "winsorize_v1", "scale": 3.0}]}
        ).outcome_transform

        assert listed[1].model_dump() == {"type": "winsorize_v1", "scale": 3.0}
        transform = buildOutcomeTransform(listed, 1)
        assert isinstance(transform, ChainedOutcomeTransform)
        assert list(transform.keys()) == ["0_log_v1", "1_winsorize_v1"]

    def test_it_is_listed_in_the_error_for_an_unknown_name(
        self, isolatedOutcomeTransformKind: KindTable
    ) -> None:
        defineFake("winsorize_v1")

        with pytest.raises(ConfigError, match="winsorize_v1"):
            make({"outcome_transform": [{"type": "nope_v1"}]})
