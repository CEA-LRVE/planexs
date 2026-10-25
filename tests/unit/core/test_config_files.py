"""Unit tests for `parseConfig`, `readConfig`, `dumpConfig` and `writeConfig`."""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pytest
from nested_config_schemas import ComposedConfig, StudyConfig

from planexs.core.ConfigError import ConfigError
from planexs.core.configFiles import dumpConfig, parseConfig, readConfig, writeConfig
from planexs.core.KindTable import KindTable
from planexs.core.PlanexsError import PlanexsError
from planexs.core.resolveComponent import resolveComponent
from planexs.core.toResolvedConfig import toResolvedConfig
from planexs.core.validateConfig import validateConfig

#: The same policy configuration written by hand in both formats. The YAML has what a person
#: writes (comments, flow style, a number without a dot) and the JSON what a tool writes.
POLICY_YAML = """\
# A composed policy.
type: composed_v1
design:
  type: random_v1
  n_points: 5
surrogate:
  type: single_task_gp_v1
  kernel: {type: matern_v1, nu: 1.5e0}
analyses:
  - type: importance_v1
  - {type: level_estimate_v1, level: 1e-1}
bounds: [0.25, 0.75]
"""

POLICY_JSON = """\
{
  "type": "composed_v1",
  "design": {"type": "random_v1", "n_points": 5},
  "surrogate": {
    "type": "single_task_gp_v1",
    "kernel": {"type": "matern_v1", "nu": 1.5}
  },
  "analyses": [
    {"type": "importance_v1"},
    {"type": "level_estimate_v1", "level": 0.1}
  ],
  "bounds": [0.25, 0.75]
}
"""

#: What both texts mean.
POLICY: dict[str, Any] = {
    "type": "composed_v1",
    "design": {"type": "random_v1", "n_points": 5},
    "surrogate": {"type": "single_task_gp_v1", "kernel": {"type": "matern_v1", "nu": 1.5}},
    "analyses": [{"type": "importance_v1"}, {"type": "level_estimate_v1", "level": 0.1}],
    "bounds": [0.25, 0.75],
}

#: Values that are easy to damage on the way through a format.
AWKWARD: dict[str, Any] = {
    "yes_word": "yes",
    "number_word": "1.0",
    "date_word": "2020-01-01",
    "empty": "",
    "unicode": "é☃ 日本",
    "multiline": "first\nsecond",
    "quote": "say \"hi\" and 'bye'",
    "inf": math.inf,
    "minus_inf": -math.inf,
    "true": True,
    "none": None,
    "small": 1e-300,
    "large": 1.5e300,
    "int": 7,
    "list": [1, 2.5, {"k": "v"}, [3]],
    "nested": {"z": 1, "a": 2, "m": {"y": 0, "b": 1}},
}


def failureOf(call: Any, *arguments: Any, **options: Any) -> ConfigError:
    """The `ConfigError` that `call(*arguments, **options)` raises."""
    with pytest.raises(ConfigError) as caught:
        call(*arguments, **options)
    return caught.value


class TestParseConfig:
    def test_json_gives_a_dictionary(self) -> None:
        assert parseConfig(POLICY_JSON, "json") == POLICY

    def test_yaml_gives_a_dictionary(self) -> None:
        assert parseConfig(POLICY_YAML, "yaml") == POLICY

    def test_the_same_configuration_in_either_format_gives_the_same_dictionary(self) -> None:
        assert parseConfig(POLICY_YAML, "yaml") == parseConfig(POLICY_JSON, "json")

    def test_the_numbers_are_the_same_numbers(self) -> None:
        data = parseConfig(POLICY_YAML, "yaml")
        assert data["surrogate"]["kernel"]["nu"] == 1.5
        assert type(data["surrogate"]["kernel"]["nu"]) is float
        assert data["analyses"][1]["level"] == 0.1
        assert type(data["design"]["n_points"]) is int

    def test_an_empty_mapping_is_a_configuration(self) -> None:
        assert parseConfig("{}", "json") == {}
        assert parseConfig("{}", "yaml") == {}

    def test_a_byte_order_mark_is_ignored(self) -> None:
        assert parseConfig('\ufeff{"a": 1}', "json") == {"a": 1}
        assert parseConfig("\ufeffa: 1\n", "yaml") == {"a": 1}

    def test_a_new_dictionary_comes_back_each_time(self) -> None:
        first = parseConfig(POLICY_JSON, "json")
        first["design"]["n_points"] = 99
        assert parseConfig(POLICY_JSON, "json") == POLICY

    def test_json_allows_what_python_json_allows(self) -> None:
        data = parseConfig('{"a": Infinity, "b": -Infinity, "c": NaN}', "json")
        assert data["a"] == math.inf and data["b"] == -math.inf and math.isnan(data["c"])

    def test_yaml_dates_stay_strings_like_in_json(self) -> None:
        assert parseConfig("start: 2020-01-01\n", "yaml") == {"start": "2020-01-01"}

    def test_an_unknown_format_is_an_error(self) -> None:
        error = failureOf(parseConfig, "a = 1", "toml")
        assert error.message == "Unsupported configuration format 'toml'."
        assert error.suggestion == "use 'json' or 'yaml'"


class TestSyntaxErrors:
    def test_json_names_the_line_and_the_column(self) -> None:
        error = failureOf(parseConfig, '{"a": 1,\n "b": }', "json", source="study.json")
        assert str(error) == (
            "Cannot parse the configuration file 'study.json' as JSON (line 2, column 7): "
            "Expecting value."
        )

    def test_yaml_names_the_line_and_the_column(self) -> None:
        error = failureOf(parseConfig, "a: 1\nb: [1, 2\nc: 3\n", "yaml", source="study.yaml")
        assert str(error) == (
            "Cannot parse the configuration file 'study.yaml' as YAML (line 3, column 2): "
            "expected ',' or ']', but got ':' (while parsing a flow sequence)."
        )

    def test_yaml_reports_the_place_of_a_misplaced_colon(self) -> None:
        error = failureOf(parseConfig, "a: b: c\n", "yaml", source="x.yaml")
        assert "(line 1, column 5): mapping values are not allowed here." in error.message

    def test_yaml_names_the_place_of_a_character_it_does_not_allow(self) -> None:
        error = failureOf(parseConfig, "a: 1\nb: \x00\n", "yaml", source="nul.yaml")
        assert str(error) == (
            "Cannot parse the configuration file 'nul.yaml' as YAML (line 2, column 4): "
            "special characters are not allowed."
        )

    def test_the_place_of_such_a_character_counts_characters_and_not_bytes(self) -> None:
        error = failureOf(parseConfig, "a: 1\ncaf\u00e9: \u2603 \x07\n", "yaml")
        assert "(line 2, column 9)" in error.message

    def test_the_place_of_such_a_character_on_the_first_line(self) -> None:
        assert "(line 1, column 4)" in failureOf(parseConfig, "a: \x00", "yaml").message

    def test_json_refuses_a_control_character_with_its_place(self) -> None:
        error = failureOf(parseConfig, '{"a": "x\ty"}', "json")
        assert "(line 1, column 9)" in error.message
        assert "Invalid control character" in error.message

    def test_text_that_comes_from_no_file_says_so(self) -> None:
        assert "Cannot parse the configuration text as JSON (line 1, column 2)" in (
            failureOf(parseConfig, "{", "json").message
        )
        assert "Cannot parse the configuration text as YAML" in (
            failureOf(parseConfig, "a: [", "yaml").message
        )

    def test_an_error_is_a_config_error_and_a_planexs_error(self) -> None:
        error = failureOf(parseConfig, "{", "json")
        assert isinstance(error, PlanexsError)

    def test_the_parser_error_is_the_cause(self) -> None:
        assert isinstance(failureOf(parseConfig, "{", "json").__cause__, json.JSONDecodeError)

    def test_an_empty_json_text_is_a_syntax_error(self) -> None:
        assert (
            "(line 1, column 1): Expecting value" in failureOf(parseConfig, "", "json").message
        )

    def test_nesting_that_is_too_deep_is_an_error_and_not_a_crash(self) -> None:
        for file_format, text in (("json", "[" * 20000), ("yaml", "[" * 20000)):
            error = failureOf(parseConfig, text, file_format, source="deep.cfg")
            assert "it is nested too deeply" in error.message
            assert "'deep.cfg'" in error.message


class TestDuplicateKeys:
    def test_yaml_gives_the_line_of_both_keys(self) -> None:
        text = "type: composed_v1\ndesign:\n  n_points: 5\n  type: sobol_v1\n  n_points: 6\n"
        error = failureOf(parseConfig, text, "yaml", source="study.yaml")
        assert str(error) == (
            "Cannot parse the configuration file 'study.yaml' as YAML (line 5, column 3): "
            "found duplicate key 'n_points', first defined on line 3."
        )

    def test_json_names_the_key_and_the_file(self) -> None:
        text = '{"type": "composed_v1",\n "design": {"n_points": 5,\n "n_points": 6}}'
        error = failureOf(parseConfig, text, "json", source="study.json")
        # The standard library does not say where a duplicate is: there is no line.
        assert str(error) == (
            "Cannot parse the configuration file 'study.json' as JSON: "
            "found duplicate key 'n_points'."
        )

    def test_a_top_level_duplicate_in_json(self) -> None:
        assert (
            "found duplicate key 'a'"
            in failureOf(parseConfig, '{"a": 1, "a": 2}', "json").message
        )

    def test_a_duplicate_in_a_list_of_objects_is_found(self) -> None:
        assert (
            "duplicate key 'k'" in failureOf(parseConfig, '[{"k": 1, "k": 2}]', "json").message
        )
        error = failureOf(parseConfig, "items:\n  - k: 1\n    k: 2\n", "yaml")
        assert "duplicate key 'k', first defined on line 2" in error.message

    def test_the_same_key_in_different_objects_is_not_a_duplicate(self) -> None:
        text = '{"a": {"type": "x"}, "b": {"type": "y"}, "type": "z"}'
        assert parseConfig(text, "json")["b"] == {"type": "y"}
        assert parseConfig("a: {type: x}\nb: {type: y}\ntype: z\n", "yaml")["b"] == {
            "type": "y"
        }

    def test_the_last_value_is_not_silently_kept(self) -> None:
        # The behaviour of both parsers, left alone, would have given {"a": 2}.
        assert json.loads('{"a": 1, "a": 2}') == {"a": 2}
        with pytest.raises(ConfigError):
            parseConfig('{"a": 1, "a": 2}', "json")


class TestWhatTheTopLevelMayHold:
    @pytest.mark.parametrize("text", ["", "# only a comment\n", "null\n", "~\n"])
    def test_a_yaml_text_with_nothing_in_it_is_empty(self, text: str) -> None:
        error = failureOf(parseConfig, text, "yaml", source="empty.yaml")
        assert error.message == "The configuration file 'empty.yaml' is empty."
        assert error.suggestion is not None

    def test_json_null_is_empty(self) -> None:
        assert failureOf(parseConfig, "null", "json").message == (
            "The configuration text is empty."
        )

    @pytest.mark.parametrize(
        "text, kind",
        [("- a\n- b\n", "list"), ("plain text\n", "str"), ("3\n", "int"), ("true\n", "bool")],
    )
    def test_a_yaml_text_that_is_not_a_mapping(self, text: str, kind: str) -> None:
        error = failureOf(parseConfig, text, "yaml", source="x.yaml")
        assert error.message == (
            "The configuration file 'x.yaml' must hold a mapping from field names to values "
            f"at its top level; got {kind}."
        )

    @pytest.mark.parametrize(
        "text, kind", [("[1, 2]", "list"), ('"x"', "str"), ("2.5", "float")]
    )
    def test_a_json_text_that_is_not_an_object(self, text: str, kind: str) -> None:
        assert f"got {kind}." in failureOf(parseConfig, text, "json").message

    def test_two_yaml_documents_are_refused(self) -> None:
        error = failureOf(parseConfig, "a: 1\n---\nb: 2\n", "yaml", source="two.yaml")
        assert "(line 2, column 1)" in error.message
        assert "expected a single document" in error.message


class TestYamlStaysPlainData:
    def test_a_python_tag_is_refused_with_its_line(self) -> None:
        error = failureOf(parseConfig, "a: 1\nb: !!python/object/apply:os.getcwd []\n", "yaml")
        assert "(line 2, column 4)" in error.message
        assert "could not determine a constructor for the tag" in error.message

    def test_python_objects_are_never_built(self) -> None:
        for text in (
            "a: !!python/name:os.system\n",
            "a: !!python/object:collections.OrderedDict {}\n",
            "a: !!python/tuple [1, 2]\n",
        ):
            assert (
                "could not determine a constructor"
                in failureOf(parseConfig, text, "yaml").message
            )


class TestReadConfig:
    def write(self, directory: Path, name: str, text: str) -> Path:
        """Writes `text` to the file `name` of `directory` and returns its path."""
        path = directory / name
        path.write_text(text, encoding="utf-8")
        return path

    @pytest.mark.parametrize(
        "name", ["c.json", "c.yaml", "c.yml", "C.JSON", "c.YAML", "c.Yml"]
    )
    def test_the_format_follows_the_suffix(self, tmp_path: Path, name: str) -> None:
        text = POLICY_JSON if name.lower().endswith("json") else POLICY_YAML
        assert readConfig(self.write(tmp_path, name, text)) == POLICY

    def test_a_string_path_is_accepted(self, tmp_path: Path) -> None:
        assert readConfig(str(self.write(tmp_path, "c.yaml", POLICY_YAML))) == POLICY

    def test_the_json_file_and_the_yaml_file_give_the_same_dictionary(
        self, tmp_path: Path
    ) -> None:
        as_json = readConfig(self.write(tmp_path, "c.json", POLICY_JSON))
        as_yaml = readConfig(self.write(tmp_path, "c.yaml", POLICY_YAML))
        assert as_json == as_yaml

    def test_a_byte_order_mark_in_the_file_is_ignored(self, tmp_path: Path) -> None:
        path = tmp_path / "bom.json"
        path.write_bytes(b"\xef\xbb\xbf" + b'{"a": 1}')
        assert readConfig(path) == {"a": 1}

    def test_an_error_names_the_file_as_it_was_given(self, tmp_path: Path) -> None:
        path = self.write(tmp_path, "bad.yaml", "a: 1\nb: [1, 2\nc: 3\n")
        error = failureOf(readConfig, path)
        assert f"the configuration file '{path}' as YAML (line 3, column 2)" in error.message

    def test_a_syntax_error_in_a_file_has_its_line(self, tmp_path: Path) -> None:
        path = self.write(tmp_path, "bad.json", '{\n  "a": 1,\n  "b": ]\n}')
        error = failureOf(readConfig, path)
        assert "bad.json" in error.message
        assert "(line 3, column 8)" in error.message

    def test_a_duplicate_in_a_file_has_its_line(self, tmp_path: Path) -> None:
        path = self.write(tmp_path, "dup.yml", "a: 1\nb: 2\na: 3\n")
        error = failureOf(readConfig, path)
        assert "dup.yml" in error.message
        assert (
            "(line 3, column 1): found duplicate key 'a', first defined on line 1"
            in error.message
        )

    @pytest.mark.parametrize("name", ["c.txt", "c.toml", "c", "c.json.bak", ".json"])
    def test_a_suffix_that_is_not_a_format_is_an_error(
        self, tmp_path: Path, name: str
    ) -> None:
        path = self.write(tmp_path, name, "{}")
        error = failureOf(readConfig, path)
        assert "Cannot tell the format of the configuration file" in error.message
        assert str(path) in error.message
        assert error.suggestion == "give the file one of those suffixes"

    def test_a_file_that_does_not_exist(self, tmp_path: Path) -> None:
        path = tmp_path / "missing.yaml"
        error = failureOf(readConfig, path)
        assert error.message.startswith(f"Cannot read the configuration file '{path}': ")
        assert isinstance(error.__cause__, FileNotFoundError)

    def test_a_directory_is_not_a_file(self, tmp_path: Path) -> None:
        directory = tmp_path / "folder.yaml"
        directory.mkdir()
        error = failureOf(readConfig, directory)
        assert error.message.startswith(f"Cannot read the configuration file '{directory}': ")

    def test_a_file_that_is_not_utf_8(self, tmp_path: Path) -> None:
        path = tmp_path / "latin.json"
        path.write_bytes('{"name": "caf\u00e9"}'.encode("latin-1"))
        error = failureOf(readConfig, path)
        assert (
            error.message
            == f"Cannot read the configuration file '{path}': it is not valid UTF-8."
        )
        assert error.suggestion == "save the file as UTF-8"

    def test_a_file_with_nothing_in_it(self, tmp_path: Path) -> None:
        path = self.write(tmp_path, "empty.yaml", "")
        assert (
            failureOf(readConfig, path).message == f"The configuration file '{path}' is empty."
        )


class TestDumpConfig:
    def test_json_is_indented_ends_with_a_newline_and_keeps_the_order(self) -> None:
        text = dumpConfig({"z": 1, "a": {"y": [1, 2], "b": None}}, "json")
        assert text == (
            "{\n"
            '  "z": 1,\n'
            '  "a": {\n'
            '    "y": [\n'
            "      1,\n"
            "      2\n"
            "    ],\n"
            '    "b": null\n'
            "  }\n"
            "}\n"
        )

    def test_yaml_is_block_style_ends_with_a_newline_and_keeps_the_order(self) -> None:
        text = dumpConfig({"z": 1, "a": {"y": [1, 2], "b": None}}, "yaml")
        assert text == "z: 1\na:\n  y:\n  - 1\n  - 2\n  b: null\n"

    @pytest.mark.parametrize("file_format", ["json", "yaml"])
    def test_the_keys_are_not_sorted(self, file_format: str) -> None:
        text = dumpConfig({"b": 1, "a": 2, "c": {"y": 1, "x": 2}}, file_format)  # type: ignore[arg-type]
        assert text.index("b") < text.index("a") < text.index("c")
        assert text.index("y") < text.index("x")

    @pytest.mark.parametrize("file_format", ["json", "yaml"])
    def test_what_is_written_reads_back_as_an_equal_dictionary(self, file_format: str) -> None:
        text = dumpConfig(AWKWARD, file_format)  # type: ignore[arg-type]
        assert parseConfig(text, file_format) == AWKWARD  # type: ignore[arg-type]

    @pytest.mark.parametrize("file_format", ["json", "yaml"])
    def test_strings_that_look_like_other_values_stay_strings(self, file_format: str) -> None:
        data = parseConfig(dumpConfig(AWKWARD, file_format), file_format)  # type: ignore[arg-type]
        assert data["yes_word"] == "yes" and type(data["yes_word"]) is str
        assert data["number_word"] == "1.0" and type(data["number_word"]) is str
        assert data["date_word"] == "2020-01-01" and type(data["date_word"]) is str

    @pytest.mark.parametrize("file_format", ["json", "yaml"])
    def test_non_ascii_text_is_written_as_it_is(self, file_format: str) -> None:
        assert "é☃ 日本" in dumpConfig({"u": "é☃ 日本"}, file_format)  # type: ignore[arg-type]

    @pytest.mark.parametrize("file_format", ["json", "yaml"])
    def test_a_tuple_is_written_as_a_list(self, file_format: str) -> None:
        text = dumpConfig({"bounds": (0.25, 0.75)}, file_format)  # type: ignore[arg-type]
        assert parseConfig(text, file_format) == {"bounds": [0.25, 0.75]}  # type: ignore[arg-type]

    @pytest.mark.parametrize("file_format", ["json", "yaml"])
    def test_any_mapping_is_accepted_and_none_is_modified(self, file_format: str) -> None:
        data = {"a": {"b": 1}}
        dumpConfig(MappingProxyType(data), file_format)  # type: ignore[arg-type]
        assert data == {"a": {"b": 1}}

    @pytest.mark.parametrize("file_format", ["json", "yaml"])
    def test_something_that_is_not_plain_data_is_an_error(self, file_format: str) -> None:
        error = failureOf(dumpConfig, {"a": object()}, file_format)
        assert error.message.startswith(
            f"Cannot write the configuration as {file_format.upper()}: "
        )
        assert (
            error.suggestion
            == "write a resolved configuration, which `toResolvedConfig` returns"
        )

    def test_an_unknown_format_is_an_error(self) -> None:
        assert (
            failureOf(dumpConfig, {}, "toml").message
            == "Unsupported configuration format 'toml'."
        )


class TestWriteConfig:
    @pytest.mark.parametrize("name", ["c.json", "c.yaml", "c.yml", "C.JSON"])
    def test_the_file_holds_what_dump_config_writes(self, tmp_path: Path, name: str) -> None:
        path = tmp_path / name
        writeConfig(path, POLICY)
        file_format = "json" if name.lower().endswith("json") else "yaml"
        assert path.read_text(encoding="utf-8") == dumpConfig(POLICY, file_format)  # type: ignore[arg-type]

    @pytest.mark.parametrize("name", ["c.json", "c.yaml"])
    def test_what_is_written_reads_back_equal(self, tmp_path: Path, name: str) -> None:
        writeConfig(tmp_path / name, AWKWARD)
        assert readConfig(tmp_path / name) == AWKWARD

    def test_a_string_path_is_accepted(self, tmp_path: Path) -> None:
        writeConfig(str(tmp_path / "c.yaml"), POLICY)
        assert readConfig(tmp_path / "c.yaml") == POLICY

    def test_an_existing_file_is_replaced(self, tmp_path: Path) -> None:
        path = tmp_path / "c.json"
        writeConfig(path, {"old": "x" * 500})
        writeConfig(path, {"new": 1})
        assert readConfig(path) == {"new": 1}

    def test_the_file_is_utf_8(self, tmp_path: Path) -> None:
        path = tmp_path / "c.json"
        writeConfig(path, {"u": "é☃"})
        assert "é☃".encode() in path.read_bytes()

    def test_a_suffix_that_is_not_a_format_writes_nothing(self, tmp_path: Path) -> None:
        path = tmp_path / "c.txt"
        error = failureOf(writeConfig, path, POLICY)
        assert "Cannot tell the format of the configuration file" in error.message
        assert not path.exists()

    def test_something_that_is_not_plain_data_writes_nothing(self, tmp_path: Path) -> None:
        path = tmp_path / "c.yaml"
        failureOf(writeConfig, path, {"a": object()})
        assert not path.exists()

    def test_a_folder_that_does_not_exist(self, tmp_path: Path) -> None:
        path = tmp_path / "no" / "such" / "c.json"
        error = failureOf(writeConfig, path, POLICY)
        assert error.message.startswith(f"Cannot write the configuration file '{path}': ")
        assert isinstance(error.__cause__, FileNotFoundError)


class TestThroughTheResolver:
    """Files give plain dictionaries, and the resolver makes the validated objects."""

    def test_json_and_yaml_give_identical_objects(
        self, tmp_path: Path, nestedKinds: KindTable
    ) -> None:
        (tmp_path / "policy.json").write_text(POLICY_JSON, encoding="utf-8")
        (tmp_path / "policy.yaml").write_text(POLICY_YAML, encoding="utf-8")
        policies = nestedKinds.registry("policy")
        from_json = resolveComponent(policies, readConfig(tmp_path / "policy.json"))
        from_yaml = resolveComponent(policies, readConfig(tmp_path / "policy.yaml"))
        assert from_json.config == from_yaml.config
        assert from_json.entry is from_yaml.entry
        assert type(from_yaml.config) is ComposedConfig
        assert toResolvedConfig(from_json.config) == toResolvedConfig(from_yaml.config)

    def test_the_objects_are_the_ones_the_file_describes(
        self, tmp_path: Path, nestedKinds: KindTable
    ) -> None:
        (tmp_path / "policy.yaml").write_text(POLICY_YAML, encoding="utf-8")
        config = resolveComponent(
            nestedKinds.registry("policy"), readConfig(tmp_path / "policy.yaml")
        ).config
        assert isinstance(config, ComposedConfig)
        assert config.design.n_points == 5
        assert config.surrogate.kernel.nu == 1.5
        assert [item.type for item in config.analyses] == [
            "importance_v1",
            "level_estimate_v1",
        ]
        assert config.analyses[1].level == 0.1
        assert config.bounds == (0.25, 0.75)

    @pytest.mark.parametrize("name", ["study.json", "study.yaml"])
    def test_dump_then_load_gives_an_equal_configuration(
        self, tmp_path: Path, nestedKinds: KindTable, name: str
    ) -> None:
        study = validateConfig(
            StudyConfig,
            {
                "name": "demo",
                "policy": POLICY,
                "stopping": [{"type": "max_trials_v1", "n_trials": 30}],
            },
        )
        writeConfig(tmp_path / name, toResolvedConfig(study))
        assert validateConfig(StudyConfig, readConfig(tmp_path / name)) == study

    @pytest.mark.parametrize("name", ["study.json", "study.yaml"])
    def test_the_text_is_a_fixed_point_of_dump_load_dump(
        self, tmp_path: Path, nestedKinds: KindTable, name: str
    ) -> None:
        study = validateConfig(StudyConfig, {"policy": POLICY})
        first = tmp_path / name
        second = tmp_path / f"again_{name}"
        writeConfig(first, toResolvedConfig(study))
        again = validateConfig(StudyConfig, readConfig(first))
        writeConfig(second, toResolvedConfig(again))
        assert second.read_text(encoding="utf-8") == first.read_text(encoding="utf-8")

    def test_a_configuration_can_move_from_one_format_to_the_other(
        self, tmp_path: Path, nestedKinds: KindTable
    ) -> None:
        study = validateConfig(StudyConfig, {"policy": POLICY})
        writeConfig(tmp_path / "a.json", toResolvedConfig(study))
        writeConfig(tmp_path / "a.yaml", readConfig(tmp_path / "a.json"))
        writeConfig(tmp_path / "b.json", readConfig(tmp_path / "a.yaml"))
        assert (tmp_path / "b.json").read_text(encoding="utf-8") == (
            tmp_path / "a.json"
        ).read_text(encoding="utf-8")

    def test_the_resolved_defaults_are_in_the_file(
        self, tmp_path: Path, nestedKinds: KindTable
    ) -> None:
        study = validateConfig(StudyConfig, {"policy": {"type": "composed_v1"}})
        writeConfig(tmp_path / "study.yaml", toResolvedConfig(study))
        written = readConfig(tmp_path / "study.yaml")
        assert written["policy"]["surrogate"]["kernel"] == {
            "type": "matern_v1",
            "nu": 2.5,
            "ard": True,
        }
        assert written["name"] == "study"

    def test_a_problem_in_the_content_is_still_reported_by_the_resolver(
        self, tmp_path: Path, nestedKinds: KindTable
    ) -> None:
        text = POLICY_YAML.replace("nu: 1.5e0", "nu: smooth")
        (tmp_path / "policy.yaml").write_text(text, encoding="utf-8")
        with pytest.raises(ConfigError) as caught:
            resolveComponent(
                nestedKinds.registry("policy"), readConfig(tmp_path / "policy.yaml")
            )
        assert caught.value.field_path == "surrogate.kernel.nu"
