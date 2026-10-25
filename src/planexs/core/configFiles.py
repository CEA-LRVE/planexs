"""Reads and writes configurations as JSON and YAML, through plain dictionaries.

A configuration travels as a plain dictionary (dictionaries, lists, strings, numbers, booleans
and `None`): that is what these functions read from a file and write to one. Making objects of
it is the job of the resolver, as it is for a dictionary built in Python (specification 5.12)::

    data = readConfig("study.yaml")
    resolved = resolveComponent(registry, data)       # a component of a kind
    study = validateConfig(StudyConfig, data)         # any model that holds components

    writeConfig("resolved.json", toResolvedConfig(study))

Anything wrong with a file is a `ConfigError` that says which file and, when the format reports
it, the line and column.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import yaml

from planexs.core.ConfigError import ConfigError
from planexs.core.UniqueKeyYamlLoader import UniqueKeyYamlLoader

#: The formats a configuration can be read and written in.
ConfigFormat = Literal["json", "yaml"]

#: The format of a file, by the suffix of its name (lower case).
_FORMAT_OF_SUFFIX: dict[str, ConfigFormat] = {".json": "json", ".yaml": "yaml", ".yml": "yaml"}

_FORMAT_NAMES = {"json": "JSON", "yaml": "YAML"}


def parseConfig(
    text: str, file_format: ConfigFormat, *, source: str | None = None
) -> dict[str, Any]:
    """Reads a configuration from text, as a plain dictionary.

    JSON is read with the standard library, YAML with `UniqueKeyYamlLoader`, which is safe (it
    builds no object but plain data) and reads numbers as JSON does. In both formats a key
    written twice in one object is an error, and the same configuration written in either
    format gives the same dictionary. A leading byte order mark is ignored. JSON as Python
    reads it also allows `NaN`, `Infinity` and `-Infinity`.

    Args:
        text: The text of the configuration.
        file_format: `"json"` or `"yaml"`.
        source: The name of the file the text comes from, which the errors give. Left out for
            text that comes from nowhere in particular.

    Returns:
        The configuration: a dictionary, which may be empty.

    Raises:
        ConfigError: if `file_format` is neither of the two, the text is not valid in that
            format (with its line and column), a key is written twice (with its line in YAML),
            the nesting is too deep, or the text holds nothing or holds something that is not a
            mapping. The message names `source`.
    """
    _checkFormat(file_format)
    where = f"configuration file {source!r}" if source is not None else "configuration text"
    text = text.removeprefix("\ufeff")
    try:
        data = _parseJson(text, where) if file_format == "json" else _parseYaml(text, where)
    except RecursionError as error:
        raise ConfigError(
            f"Cannot parse the {where} as {_FORMAT_NAMES[file_format]}: "
            "it is nested too deeply.",
            suggestion="flatten the configuration",
        ) from error

    if data is None:
        raise ConfigError(
            f"The {where} is empty.",
            suggestion="write the configuration as a mapping from field names to values",
        )
    if not isinstance(data, dict):
        raise ConfigError(
            f"The {where} must hold a mapping from field names to values at its top level; "
            f"got {type(data).__name__}.",
            suggestion="write the configuration as a mapping from field names to values",
        )
    return data


def readConfig(path: str | Path) -> dict[str, Any]:
    """Reads a configuration file, as a plain dictionary.

    The format is chosen by the suffix of the name: `.json` for JSON, `.yaml` and `.yml` for
    YAML. The file is read as UTF-8. See `parseConfig` for how the text is read.

    Args:
        path: The file to read.

    Returns:
        The configuration: a dictionary, which may be empty.

    Raises:
        ConfigError: if the suffix is none of those, the file cannot be read or is not valid
            UTF-8, or what it holds is not a configuration (see `parseConfig`). The message
            names the file as it was given.
    """
    name = str(path)
    file_format = _formatOfFile(name)
    try:
        text = Path(path).read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise ConfigError(
            f"Cannot read the configuration file {name!r}: it is not valid UTF-8.",
            suggestion="save the file as UTF-8",
        ) from error
    except OSError as error:
        raise ConfigError(
            f"Cannot read the configuration file {name!r}: {error.strerror}."
        ) from error
    return parseConfig(text, file_format, source=name)


def dumpConfig(data: Mapping[str, Any], file_format: ConfigFormat) -> str:
    """Writes a configuration as text.

    The keys keep their order, which for a resolved configuration is the order in which the
    model declares its fields. Reading the text back with `parseConfig` gives an equal
    dictionary. JSON is indented, YAML is written in block style, and the text ends with a
    newline.

    Args:
        data: The configuration, made of plain values only: for example the result of
            `toResolvedConfig`.
        file_format: `"json"` or `"yaml"`.

    Returns:
        The text.

    Raises:
        ConfigError: if `file_format` is neither of the two, or `data` holds something that
            is not plain data.
    """
    _checkFormat(file_format)
    try:
        if file_format == "json":
            return json.dumps(dict(data), indent=2, ensure_ascii=False) + "\n"
        return yaml.safe_dump(
            dict(data), sort_keys=False, default_flow_style=False, allow_unicode=True
        )
    except (TypeError, ValueError, yaml.YAMLError) as error:
        raise ConfigError(
            f"Cannot write the configuration as {_FORMAT_NAMES[file_format]}: {error}",
            suggestion="write a resolved configuration, which `toResolvedConfig` returns",
        ) from error


def writeConfig(path: str | Path, data: Mapping[str, Any]) -> None:
    """Writes a configuration to a file, as UTF-8.

    The format is chosen by the suffix of the name, as `readConfig` does, and the file is
    replaced if it exists. See `dumpConfig` for how the text is written.

    Args:
        path: The file to write.
        data: The configuration, made of plain values only.

    Raises:
        ConfigError: if the suffix is none of `.json`, `.yaml` and `.yml`, `data` holds
            something that is not plain data, or the file cannot be written.
    """
    name = str(path)
    text = dumpConfig(data, _formatOfFile(name))
    try:
        Path(path).write_text(text, encoding="utf-8")
    except OSError as error:
        raise ConfigError(
            f"Cannot write the configuration file {name!r}: {error.strerror}."
        ) from error


def _checkFormat(file_format: str) -> None:
    """Checks a format name.

    Args:
        file_format: The name to check.

    Raises:
        ConfigError: if it is neither `"json"` nor `"yaml"`.
    """
    if file_format not in _FORMAT_NAMES:
        raise ConfigError(
            f"Unsupported configuration format {file_format!r}.",
            suggestion="use 'json' or 'yaml'",
        )


def _formatOfFile(name: str) -> ConfigFormat:
    """Tells the format of a configuration file from the suffix of its name.

    Args:
        name: The name of the file.

    Returns:
        `"json"` or `"yaml"`.

    Raises:
        ConfigError: if the suffix is none of `.json`, `.yaml` and `.yml`.
    """
    suffix = Path(name).suffix.lower()
    if suffix not in _FORMAT_OF_SUFFIX:
        raise ConfigError(
            f"Cannot tell the format of the configuration file {name!r}: "
            f"its suffix {suffix!r} is not .json, .yaml or .yml.",
            suggestion="give the file one of those suffixes",
        )
    return _FORMAT_OF_SUFFIX[suffix]


def _parseJson(text: str, where: str) -> object:
    """Reads JSON, refusing a key written twice in one object.

    Args:
        text: The JSON text.
        where: What the text is, for the errors.

    Returns:
        The value the text holds.

    Raises:
        ConfigError: on a syntax error or a duplicate key. The standard library does not say
            where a duplicate is, so that error has no line.
    """

    def rejectDuplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        """Builds an object from its key and value pairs, as `json` does by default."""
        entries: dict[str, Any] = {}
        for key, value in pairs:
            if key in entries:
                raise _parseError(where, "JSON", f"found duplicate key {key!r}")
            entries[key] = value
        return entries

    try:
        return json.loads(text, object_pairs_hook=rejectDuplicates)
    except json.JSONDecodeError as error:
        raise _parseError(where, "JSON", error.msg, error.lineno, error.colno) from error


def _parseYaml(text: str, where: str) -> object:
    """Reads YAML with `UniqueKeyYamlLoader`.

    Args:
        text: The YAML text.
        where: What the text is, for the errors.

    Returns:
        The value the text holds.

    Raises:
        ConfigError: on a syntax error, a duplicate key, a tag that is not plain data, a
            character that YAML does not allow, or a text of more than one document. The
            message has the line and column.
    """
    try:
        return yaml.load(text, Loader=UniqueKeyYamlLoader)  # a safe loader: plain data only
    except yaml.MarkedYAMLError as error:
        problem = error.problem or str(error)
        if error.context:
            problem = f"{problem} ({error.context})"
        mark = error.problem_mark or error.context_mark
        line, column = (mark.line + 1, mark.column + 1) if mark is not None else (None, None)
        raise _parseError(where, "YAML", problem, line, column) from error
    except yaml.reader.ReaderError as error:
        # A character that YAML does not allow (a NUL byte, for instance): the error gives its
        # offset in the text, which is turned into a line and a column.
        line = text.count("\n", 0, error.position) + 1
        column = error.position - text.rfind("\n", 0, error.position)
        raise _parseError(where, "YAML", error.reason, line, column) from error


def _parseError(
    where: str,
    format_name: str,
    problem: str,
    line: int | None = None,
    column: int | None = None,
) -> ConfigError:
    """Builds the error for a text that cannot be parsed.

    Args:
        where: What the text is, for example `configuration file 'study.yaml'`.
        format_name: `"JSON"` or `"YAML"`.
        problem: What is wrong, as the parser says it.
        line: The line, counted from 1, if the parser gave one.
        column: The column, counted from 1, if the parser gave one.

    Returns:
        The error.
    """
    position = f" (line {line}, column {column})" if line is not None else ""
    return ConfigError(f"Cannot parse the {where} as {format_name}{position}: {problem}.")
