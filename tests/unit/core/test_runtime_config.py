"""Unit tests for `RuntimeConfig`: dtype, device and determinism (specification 8, 10.8)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, Field, ValidationError

import planexs.core
from planexs.core.ComponentConfig import ComponentConfig
from planexs.core.ConfigError import ConfigError
from planexs.core.configFiles import dumpConfig, parseConfig
from planexs.core.RuntimeConfig import RuntimeConfig
from planexs.core.toResolvedConfig import toResolvedConfig
from planexs.core.validateConfig import validateConfig

#: The defaults of decision D-12: double precision, the CPU, the deterministic mode off.
D12_DEFAULTS: dict[str, Any] = {
    "dtype": "float64",
    "device": "cpu",
    "deterministic_algorithms": False,
}

VALID_DEVICES = [
    "cpu",
    "cuda",
    "cuda:0",
    "cuda:12",
    "mps",
    "xpu",
    "xla:1",
    "privateuseone",
    "device_2:3",
]

#: Strings that are not device names, with what is wrong with each.
INVALID_DEVICES = [
    pytest.param("", id="empty"),
    pytest.param(" ", id="blank"),
    pytest.param("CPU", id="uppercase"),
    pytest.param("Cuda:0", id="mixed-case"),
    pytest.param("cuda:", id="colon-without-index"),
    pytest.param(":0", id="index-without-type"),
    pytest.param("cuda:-1", id="negative-index"),
    pytest.param("cuda:1.5", id="fractional-index"),
    pytest.param("cuda:a", id="letter-index"),
    pytest.param("cuda:0:1", id="two-indices"),
    pytest.param("cuda 0", id="space"),
    pytest.param(" cpu", id="leading-space"),
    pytest.param("cpu ", id="trailing-space"),
    pytest.param("cpu\n", id="trailing-newline"),
    pytest.param("0", id="starts-with-a-digit"),
    pytest.param("_cpu", id="starts-with-an-underscore"),
    pytest.param("cu-da", id="hyphen"),
    pytest.param("torch.device('cpu')", id="python-expression"),
]


def failureOf(data: Any) -> ConfigError:
    """The `ConfigError` that validating `data` as a `RuntimeConfig` raises."""
    with pytest.raises(ConfigError) as caught:
        validateConfig(RuntimeConfig, data)
    return caught.value


class StudyLikeConfig(BaseModel):
    """A study-level model that holds the runtime settings as one of its fields."""

    name: str = Field(default="study", description="The study's name.")
    runtime: RuntimeConfig = Field(
        default_factory=RuntimeConfig, description="The runtime settings of the study."
    )


class TestDefaults:
    def test_a_bare_configuration_has_the_defaults_of_decision_d12(self) -> None:
        config = RuntimeConfig()
        assert (config.dtype, config.device, config.deterministic_algorithms) == (
            "float64",
            "cpu",
            False,
        )

    def test_the_defaults_are_written_in_the_fields_of_the_schema(self) -> None:
        # Specification 3.4: a default is written in exactly one place, the schema field.
        written = {name: field.default for name, field in RuntimeConfig.model_fields.items()}
        assert written == D12_DEFAULTS

    def test_the_deterministic_flag_is_off_and_not_merely_falsy(self) -> None:
        assert RuntimeConfig().deterministic_algorithms is False

    def test_an_empty_mapping_validates_to_the_defaults(self) -> None:
        assert validateConfig(RuntimeConfig, {}) == RuntimeConfig()

    def test_setting_one_field_leaves_the_others_at_their_defaults(self) -> None:
        config = RuntimeConfig(dtype="float32")
        assert config.model_dump() == {**D12_DEFAULTS, "dtype": "float32"}

    def test_all_settings_can_be_given_at_once(self) -> None:
        config = RuntimeConfig(dtype="float32", device="cuda:1", deterministic_algorithms=True)
        assert config.model_dump() == {
            "dtype": "float32",
            "device": "cuda:1",
            "deterministic_algorithms": True,
        }


class TestDtype:
    @pytest.mark.parametrize("name", ["float32", "float64"])
    def test_the_two_documented_names_are_accepted(self, name: str) -> None:
        assert validateConfig(RuntimeConfig, {"dtype": name}).dtype == name

    @pytest.mark.parametrize(
        "bad",
        [
            "float16",
            "bfloat16",
            "double",
            "float",
            "Float64",
            "FLOAT32",
            "torch.float64",
            "numpy.float64",
            " float64",
            "float64 ",
            "",
        ],
    )
    def test_an_invalid_dtype_name_is_rejected_with_the_field_path(self, bad: str) -> None:
        error = failureOf({"dtype": bad})
        assert error.field_path == "dtype"
        assert error.component == "RuntimeConfig"
        # The message names the field, quotes what was given, and lists the accepted names.
        assert "dtype" in error.message
        assert repr(bad) in error.message
        assert "'float32'" in error.message
        assert "'float64'" in error.message

    @pytest.mark.parametrize("bad", [64, 32, 1.5, None, True, ["float64"]], ids=repr)
    def test_a_dtype_that_is_not_a_string_is_rejected_with_the_field_path(
        self, bad: Any
    ) -> None:
        assert failureOf({"dtype": bad}).field_path == "dtype"

    def test_a_dtype_rejected_in_python_is_a_pydantic_error_on_the_same_field(self) -> None:
        with pytest.raises(ValidationError) as caught:
            RuntimeConfig(dtype="float16")  # type: ignore[arg-type]
        assert [problem["loc"] for problem in caught.value.errors()] == [("dtype",)]


class TestDevice:
    @pytest.mark.parametrize("name", VALID_DEVICES)
    def test_a_device_name_in_torch_syntax_is_accepted_as_written(self, name: str) -> None:
        assert validateConfig(RuntimeConfig, {"device": name}).device == name

    @pytest.mark.parametrize("name", INVALID_DEVICES)
    def test_a_malformed_device_name_is_rejected_with_the_field_path(self, name: str) -> None:
        error = failureOf({"device": name})
        assert error.field_path == "device"
        # The message names the field and shows how to write a device.
        assert "device" in error.message
        assert "'cuda:1'" in error.message

    @pytest.mark.parametrize("bad", [0, 1.5, None, ["cpu"], True], ids=repr)
    def test_a_device_that_is_not_a_string_is_rejected_with_the_field_path(
        self, bad: Any
    ) -> None:
        assert failureOf({"device": bad}).field_path == "device"

    def test_a_device_that_may_not_exist_on_this_machine_is_still_accepted(self) -> None:
        # Names only: whether the device exists is for the encoding layer (P3-02), which owns
        # torch. This layer must not need it.
        assert RuntimeConfig(device="cuda:7").device == "cuda:7"

    def test_the_name_is_kept_exactly_and_not_normalized(self) -> None:
        # Nothing is trimmed or lowercased on the way in: a malformed name is rejected, not
        # repaired, so what a snapshot stores is what the caller wrote.
        assert RuntimeConfig(device="cuda:0").device == "cuda:0"
        assert failureOf({"device": "CUDA:0"}).field_path == "device"


class TestDeterministicAlgorithms:
    @pytest.mark.parametrize("value", [True, False])
    def test_a_boolean_is_accepted_as_given(self, value: bool) -> None:
        config = validateConfig(RuntimeConfig, {"deterministic_algorithms": value})
        assert config.deterministic_algorithms is value

    @pytest.mark.parametrize("bad", ["maybe", None, [True], {}, 2.5], ids=repr)
    def test_something_that_is_not_a_boolean_is_rejected_with_the_field_path(
        self, bad: Any
    ) -> None:
        assert failureOf({"deterministic_algorithms": bad}).field_path == (
            "deterministic_algorithms"
        )


class TestUnknownFields:
    def test_a_misspelled_setting_is_an_error_with_its_path(self) -> None:
        error = failureOf({"devise": "cpu"})
        assert error.field_path == "devise"
        assert "Unknown field" in error.message

    def test_the_suggestion_lists_the_fields_of_the_schema(self) -> None:
        suggestion = failureOf({"precision": "float64"}).suggestion
        assert suggestion is not None
        for name in D12_DEFAULTS:
            assert name in suggestion

    def test_a_type_field_is_unknown_because_this_is_not_a_component(self) -> None:
        error = failureOf({"type": "runtime_v1"})
        assert error.field_path == "type"
        assert "Unknown field" in error.message

    def test_several_problems_are_reported_together(self) -> None:
        error = failureOf({"dtype": "float16", "device": "CPU", "bogus": 1})
        for name in ("dtype", "device", "bogus"):
            assert name in error.message
        # `field_path` is the path of the first problem, in the order the model declares.
        assert error.field_path == "dtype"

    def test_a_configuration_that_is_not_a_mapping_is_a_config_error(self) -> None:
        with pytest.raises(ConfigError):
            validateConfig(RuntimeConfig, ["float64"])  # type: ignore[arg-type]


class TestTheSchema:
    def test_it_has_exactly_the_three_settings_of_the_card(self) -> None:
        assert list(RuntimeConfig.model_fields) == [
            "dtype",
            "device",
            "deterministic_algorithms",
        ]

    def test_every_field_has_a_description(self) -> None:
        # Specification 10.8: a description on every field.
        for name, field in RuntimeConfig.model_fields.items():
            assert field.description, f"field {name!r} has no description"
            assert field.description.strip() == field.description, name

    def test_every_description_says_what_the_default_is_and_why(self) -> None:
        # Specification 10.4: a description states why the default is what it is. Whether the
        # reason is a good one is for review; that one is given at all can be checked.
        for name, field in RuntimeConfig.model_fields.items():
            text = (field.description or "").lower()
            assert "the default is" in text, name
            assert repr(field.default).lower() in text, name

    def test_the_settings_are_names_and_flags_never_torch_objects(self) -> None:
        # The core imports no machine-learning library: the values must be plain data.
        for name, value in RuntimeConfig().model_dump().items():
            assert isinstance(value, (str, bool)), name

    def test_it_is_a_model_of_its_own_and_not_a_component_configuration(self) -> None:
        assert issubclass(RuntimeConfig, BaseModel)
        assert not issubclass(RuntimeConfig, ComponentConfig)
        assert "type" not in RuntimeConfig.model_fields

    def test_a_configuration_cannot_be_changed_once_built(self) -> None:
        config = RuntimeConfig()
        with pytest.raises(ValidationError) as caught:
            config.dtype = "float32"  # type: ignore[misc]
        assert caught.value.errors()[0]["type"] == "frozen_instance"
        assert config.dtype == "float64"

    def test_equal_settings_are_equal_and_hash_alike(self) -> None:
        first = RuntimeConfig(dtype="float32", device="cuda:0")
        second = RuntimeConfig(device="cuda:0", dtype="float32")
        assert first == second
        assert hash(first) == hash(second)
        assert first != RuntimeConfig(dtype="float32", device="cuda:1")

    def test_the_docstring_names_the_decision_and_the_names_only_rule(self) -> None:
        docstring = " ".join((RuntimeConfig.__doc__ or "").split())
        assert "D-12" in docstring
        assert "names only" in docstring.lower()


class TestResolvedConfiguration:
    def test_the_resolved_form_writes_out_every_default(self) -> None:
        assert toResolvedConfig(RuntimeConfig()) == D12_DEFAULTS

    def test_the_resolved_form_keeps_the_order_of_the_fields(self) -> None:
        assert list(toResolvedConfig(RuntimeConfig())) == list(RuntimeConfig.model_fields)

    def test_resolve_then_validate_then_resolve_is_a_fixed_point(self) -> None:
        config = RuntimeConfig(dtype="float32", device="cuda:1", deterministic_algorithms=True)
        resolved = toResolvedConfig(config)
        rebuilt = validateConfig(RuntimeConfig, resolved)
        assert rebuilt == config
        assert toResolvedConfig(rebuilt) == resolved

    def test_the_resolved_form_survives_json_unchanged(self) -> None:
        resolved = toResolvedConfig(
            RuntimeConfig(device="cuda:0", deterministic_algorithms=True)
        )
        assert json.loads(json.dumps(resolved)) == resolved

    def test_python_json_and_yaml_give_the_same_configuration(self) -> None:
        # Specification 10.8: identical meaning from Python, JSON and YAML.
        from_python = RuntimeConfig(
            dtype="float32", device="cuda:0", deterministic_algorithms=True
        )
        from_json = validateConfig(
            RuntimeConfig,
            parseConfig(
                '{"dtype": "float32", "device": "cuda:0", "deterministic_algorithms": true}',
                "json",
            ),
        )
        from_yaml = validateConfig(
            RuntimeConfig,
            parseConfig(
                "dtype: float32\ndevice: cuda:0\ndeterministic_algorithms: true\n", "yaml"
            ),
        )
        assert from_python == from_json == from_yaml

    @pytest.mark.parametrize("file_format", ["json", "yaml"])
    def test_dump_then_load_round_trips_in_both_file_formats(self, file_format: Any) -> None:
        config = RuntimeConfig(dtype="float32", device="cuda:3")
        text = dumpConfig(toResolvedConfig(config), file_format)
        assert validateConfig(RuntimeConfig, parseConfig(text, file_format)) == config

    def test_yaml_reads_a_device_with_an_index_as_a_string(self) -> None:
        # `cuda:0` must not be read as anything but the text it is.
        data = parseConfig("device: cuda:0\n", "yaml")
        assert data == {"device": "cuda:0"}


class TestInAStudyConfiguration:
    def test_a_missing_runtime_section_gets_the_defaults(self) -> None:
        study = validateConfig(StudyLikeConfig, {"name": "demo"})
        assert study.runtime == RuntimeConfig()

    def test_the_defaults_are_written_out_in_the_resolved_study(self) -> None:
        resolved = toResolvedConfig(validateConfig(StudyLikeConfig, {}))
        assert resolved["runtime"] == D12_DEFAULTS

    def test_a_partial_runtime_section_is_completed_with_the_defaults(self) -> None:
        study = validateConfig(StudyLikeConfig, {"runtime": {"dtype": "float32"}})
        assert study.runtime.model_dump() == {**D12_DEFAULTS, "dtype": "float32"}

    @pytest.mark.parametrize(
        "section, path",
        [
            ({"dtype": "float16"}, "runtime.dtype"),
            ({"device": "CPU"}, "runtime.device"),
            ({"deterministic_algorithms": "maybe"}, "runtime.deterministic_algorithms"),
            ({"precision": "float64"}, "runtime.precision"),
        ],
    )
    def test_an_error_carries_the_full_path_from_the_study(
        self, section: dict[str, Any], path: str
    ) -> None:
        with pytest.raises(ConfigError) as caught:
            validateConfig(StudyLikeConfig, {"runtime": section})
        assert caught.value.field_path == path
        assert path in caught.value.message

    def test_a_resolved_study_validates_back_into_an_equal_study(self) -> None:
        study = validateConfig(StudyLikeConfig, {"runtime": {"device": "cuda:2"}})
        resolved = toResolvedConfig(study)
        assert validateConfig(StudyLikeConfig, resolved) == study


#: Imports every module of `core` in a fresh interpreter and reports which machine-learning
#: libraries are loaded afterwards. `planexs/__init__.py` imports the whole library today, so
#: it is replaced by an empty package (as `test_rng_tree.py` does) and only `core` is loaded.
#: Roadmap task P2-12 makes the top-level import lazy and adds the same check on the real
#: package; until then this is the run-time evidence that `core` alone needs none of them.
SCRIPT = """
import importlib, json, pkgutil, sys, types

package = types.ModuleType("planexs")
package.__path__ = [sys.argv[1]]
sys.modules["planexs"] = package

import planexs.core

for module in pkgutil.iter_modules(planexs.core.__path__):
    importlib.import_module("planexs.core." + module.name)
for extra in sys.argv[2:]:
    importlib.import_module(extra)

forbidden = ("torch", "botorch", "gpytorch", "ax")
print(json.dumps({
    "forbidden": sorted({name.split(".")[0] for name in sys.modules} & set(forbidden)),
    "has_runtime_config": "planexs.core.RuntimeConfig" in sys.modules,
}))
"""


def importCoreInAFreshProcess(*extra_modules: str, path: Path | None = None) -> dict[str, Any]:
    """Imports all of `core` in a new interpreter and returns what the script reports.

    Args:
        *extra_modules: More modules to import after `core`.
        path: A directory to put first on the new interpreter's module search path.

    Returns:
        The decoded JSON the script printed.
    """
    source = str(Path(planexs.core.__file__).resolve().parent.parent)
    env = dict(os.environ)
    if path is not None:
        env["PYTHONPATH"] = os.pathsep.join(
            [str(path), *([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])]
        )
    result = subprocess.run(
        [sys.executable, "-c", SCRIPT, source, *extra_modules],
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )
    parsed: dict[str, Any] = json.loads(result.stdout)
    return parsed


class TestCoreStillImportsNoMachineLearningLibrary:
    def test_importing_all_of_core_loads_none_of_them(self) -> None:
        report = importCoreInAFreshProcess()
        assert report["forbidden"] == []

    def test_the_new_module_was_among_the_modules_imported(self) -> None:
        # Guards against the check passing because the module was never loaded.
        assert importCoreInAFreshProcess()["has_runtime_config"] is True

    def test_importing_the_module_alone_loads_none_of_them(self) -> None:
        report = importCoreInAFreshProcess("planexs.core.RuntimeConfig")
        assert report["forbidden"] == []

    def test_the_check_can_fail(self, tmp_path: Path) -> None:
        # A stand-in `torch` first on the search path proves that the script reports a
        # forbidden library when one is imported, so an empty list means something.
        (tmp_path / "torch.py").write_text("", encoding="utf-8")
        report = importCoreInAFreshProcess("torch", path=tmp_path)
        assert report["forbidden"] == ["torch"]
