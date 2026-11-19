"""A kernel as the nested component of a configuration (roadmap task P3-10, specification
5.12, 3.8).

This is how `single_task_gp_v1` (roadmap P3-14) will hold its kernel: a field marked
`ComponentOf("kernel")`, with `botorch_default_v1` as its default, written in the schema and
nowhere else. The tests make a stand-in model with that one field, and check what a surrogate
will rely on: the kernel is picked by name, a missing field takes the default, a configuration
from Python, JSON and YAML means the same thing, the resolved configuration says everything,
and a mistake is reported at the full path.
"""

from __future__ import annotations

from typing import Annotated, Any

import pytest
from kernel_fakes import N_COLUMNS, defineFake
from pydantic import BaseModel, ConfigDict, Field

from planexs.core.ComponentConfig import ComponentConfig
from planexs.core.ComponentOf import ComponentOf
from planexs.core.ConfigError import ConfigError
from planexs.core.configFiles import dumpConfig, parseConfig
from planexs.core.KindTable import KindTable
from planexs.core.resolveComponent import resolveComponent
from planexs.core.toResolvedConfig import toResolvedConfig
from planexs.core.validateConfig import validateConfig
from planexs.surrogates.kernels.BotorchDefaultKernelConfig import BotorchDefaultKernelConfig
from planexs.surrogates.kernels.kernelRegistry import kernelRegistry
from planexs.surrogates.kernels.MaternKernelConfig import MaternKernelConfig
from planexs.surrogates.kernels.RBFKernelConfig import RBFKernelConfig


class StandInSurrogateConfig(BaseModel):
    """What a surrogate's configuration holds of a kernel: one nested component, with a default
    that is a mapping, so that the schema does not import the kernel it names.
    """

    model_config = ConfigDict(extra="forbid")

    kernel: Annotated[ComponentConfig, ComponentOf("kernel")] = Field(
        default_factory=lambda: {"type": "botorch_default_v1"},  # type: ignore[arg-type,return-value]
        validate_default=True,
        description="The kernel of the surrogate: the component of kind 'kernel' to use.",
    )


def make(data: dict[str, Any]) -> StandInSurrogateConfig:
    return validateConfig(StandInSurrogateConfig, data)


class TestTheKernelIsPickedByName:
    @pytest.mark.parametrize(
        ("data", "expected"),
        [
            ({"kernel": {"type": "botorch_default_v1"}}, BotorchDefaultKernelConfig()),
            ({"kernel": {"type": "matern_v1"}}, MaternKernelConfig()),
            ({"kernel": {"type": "matern_v1", "nu": 0.5}}, MaternKernelConfig(nu=0.5)),
            ({"kernel": {"type": "rbf_v1", "ard": False}}, RBFKernelConfig(ard=False)),
        ],
        ids=["default", "matern", "matern-rough", "rbf-shared"],
    )
    def test_the_field_holds_the_validated_configuration_of_the_named_kernel(
        self, data: dict[str, Any], expected: ComponentConfig
    ) -> None:
        assert make(data).kernel == expected

    def test_the_kernel_it_names_builds_through_the_registry(self) -> None:
        config = make({"kernel": {"type": "rbf_v1"}}).kernel

        entry = kernelRegistry().lookup(config.type)
        kernel = entry.builder(config).build(N_COLUMNS)

        assert type(kernel).__name__ == "RBFKernel"

    def test_a_missing_kernel_takes_the_default_written_in_the_schema(self) -> None:
        assert make({}).kernel == BotorchDefaultKernelConfig()
        assert StandInSurrogateConfig().kernel == BotorchDefaultKernelConfig()

    def test_a_configuration_built_in_python_is_accepted_if_it_fits_its_type(self) -> None:
        assert make({"kernel": MaternKernelConfig(nu=1.5)}).kernel == MaternKernelConfig(
            nu=1.5
        )


class TestAMistakeIsReportedAtTheFullPath:
    def test_an_unsupported_smoothness(self) -> None:
        with pytest.raises(ConfigError) as caught:
            make({"kernel": {"type": "matern_v1", "nu": 3.5}})

        assert caught.value.field_path == "kernel.nu"

    def test_a_setting_that_belongs_to_another_kernel(self) -> None:
        with pytest.raises(ConfigError) as caught:
            make({"kernel": {"type": "rbf_v1", "nu": 2.5}})

        assert caught.value.field_path == "kernel.nu"

    def test_an_unknown_kernel_lists_the_registered_ones(self) -> None:
        with pytest.raises(ConfigError) as caught:
            make({"kernel": {"type": "matern_v2"}})

        error = caught.value
        assert error.field_path == "kernel.type"
        for name in ("botorch_default_v1", "matern_v1", "rbf_v1"):
            assert name in error.message

    def test_a_kernel_without_a_type(self) -> None:
        with pytest.raises(ConfigError) as caught:
            make({"kernel": {"nu": 1.5}})

        assert caught.value.field_path == "kernel.type"


ONE_CONFIGURATION = {"kernel": {"type": "matern_v1", "nu": 1.5, "ard": False}}


class TestTheSameMeaningInPythonJsonAndYaml:
    @pytest.mark.parametrize("file_format", ["json", "yaml"])
    def test_a_file_means_what_the_python_mapping_means(self, file_format: Any) -> None:
        text = dumpConfig(ONE_CONFIGURATION, file_format)

        assert make(parseConfig(text, file_format)) == make(ONE_CONFIGURATION)

    def test_a_yaml_file_written_by_hand(self) -> None:
        text = "kernel:\n  type: matern_v1\n  nu: 1.5\n  ard: false\n"

        assert make(parseConfig(text, "yaml")).kernel == MaternKernelConfig(nu=1.5, ard=False)


class TestTheResolvedConfiguration:
    """A study stores its configuration with every default written out (specification 3.4)."""

    def test_the_default_kernel_is_written_out(self) -> None:
        assert toResolvedConfig(make({})) == {"kernel": {"type": "botorch_default_v1"}}

    def test_a_kernel_that_leaves_settings_out_has_them_written_out_in_full(self) -> None:
        resolved = toResolvedConfig(make({"kernel": {"type": "matern_v1"}}))

        assert resolved == {"kernel": {"type": "matern_v1", "nu": 2.5, "ard": True}}

    @pytest.mark.parametrize("kernel", ["botorch_default_v1", "matern_v1", "rbf_v1"])
    def test_resolving_and_validating_again_is_a_fixed_point(self, kernel: str) -> None:
        resolved = toResolvedConfig(make({"kernel": {"type": kernel}}))

        assert toResolvedConfig(make(resolved)) == resolved

    def test_the_resolved_configuration_survives_json(self) -> None:
        resolved = toResolvedConfig(make({"kernel": {"type": "rbf_v1"}}))

        assert parseConfig(dumpConfig(resolved, "json"), "json") == resolved


class TestAKernelFromAPlugin:
    """The extension guarantee (specification 1.3): a kernel registered outside the package is
    picked by name exactly like a built-in one, with no change to planexs.
    """

    def test_a_registered_kernel_is_picked_by_name_in_the_same_field(
        self, isolatedKernelKind: KindTable
    ) -> None:
        defineFake("periodic_v1")

        config = make({"kernel": {"type": "periodic_v1", "scale": 3.0}}).kernel
        kernel = kernelRegistry().lookup(config.type).builder(config).build(N_COLUMNS)

        assert config.model_dump() == {"type": "periodic_v1", "scale": 3.0}
        assert type(kernel).__name__ == "LinearKernel"

    def test_it_is_listed_in_the_error_for_an_unknown_name(
        self, isolatedKernelKind: KindTable
    ) -> None:
        defineFake("periodic_v1")

        with pytest.raises(ConfigError, match="periodic_v1"):
            make({"kernel": {"type": "nope_v1"}})

    def test_resolving_it_directly_gives_the_same_entry(
        self, isolatedKernelKind: KindTable
    ) -> None:
        defineFake("periodic_v1")

        resolved = resolveComponent(kernelRegistry(), {"type": "periodic_v1"})

        assert resolved.entry is kernelRegistry().lookup("periodic_v1")
