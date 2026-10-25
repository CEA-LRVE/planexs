"""Conversions between pydantic validation errors and `ConfigError`.

Pydantic reports every problem of a configuration with the location of the offending field,
and merges the locations of nested models and lists as the errors bubble up: that is what gives
`policy.surrogate.kernel.nu` or `analyses[1].level` its full path. `toConfigError` writes such
an error as the `ConfigError` a user reads. `toValidationError` goes the other way for one kind
of problem, which is found by planexs and not by pydantic: a component field whose `type` names
nothing. Written as a pydantic error, it is located by the same merging as any other.
"""

from __future__ import annotations

import reprlib

from pydantic import BaseModel, ValidationError
from pydantic_core import ErrorDetails, InitErrorDetails, PydanticCustomError

from planexs.core.ConfigError import ConfigError

#: The pydantic error type of a problem with which component a configuration names. Its
#: message is a whole sentence, so the value that was given is not appended to it.
_SELECTION_ERROR = "component_selection"

#: Pydantic error types whose `input` is not the offending value: for a missing field it is the
#: whole mapping the field is missing from, an unknown field is reported by its name, and a
#: selection problem already quotes what it needs.
_NO_VALUE_TO_SHOW = frozenset({"missing", "extra_forbidden", _SELECTION_ERROR})


def toConfigError(
    error: ValidationError, *, label: str, schema: type[BaseModel]
) -> ConfigError:
    """Turns the validation errors of a configuration into one `ConfigError`.

    The field path of each problem is dotted, with a list position in brackets after the name
    it indexes, for example `kernel.nu` or `values[2]`. Below a component field it is the full
    path, because pydantic has already merged the locations of the nested validations.

    Args:
        error: The failure that pydantic raised.
        label: What the configuration is for, written the way the error should name it, for
            example `surrogate 'single_task_gp_v1'`.
        schema: The model the configuration was validated against.

    Returns:
        An error whose message lists every problem with its field path, whose `field_path` is
        the path of the first one (`None` if it is not about a field), and whose suggestion
        lists the fields of `schema` when one of its own fields (not a nested one) is unknown.
    """
    problems = error.errors(include_url=False)
    lines = [_describe(problem) for problem in problems]
    if len(lines) == 1:
        line = lines[0]
        message = (
            f"Invalid configuration for {label}: {line if line.endswith('.') else line + '.'}"
        )
    else:
        details = "\n".join(f"  - {line}" for line in lines)
        message = f"Invalid configuration for {label}: {len(lines)} problems.\n{details}"

    suggestion = None
    if any(p["type"] == "extra_forbidden" and len(p["loc"]) == 1 for p in problems):
        fields = ", ".join(schema.model_fields)
        suggestion = f"check the spelling; the fields of {label} are: {fields}"
    return ConfigError(
        message,
        component=label,
        field_path=_path(problems[0]["loc"]) or None,
        suggestion=suggestion,
    )


def toValidationError(error: ConfigError, value: object) -> ValidationError:
    """Writes a `ConfigError` about which component a field names as a pydantic error.

    Raised from a pydantic validator, the result is merged into the errors of the model that
    holds the field, so `toConfigError` reports it with its full path. The error keeps its
    message and its field path (relative to the component field); its `component` and
    `suggestion` hints are not carried over.

    Args:
        error: The error to convert, for example the one `selectComponent` raises.
        value: The value of the field, as it was given.

    Returns:
        A validation error with one problem, located at `error.field_path` (the field itself
        when the error has none).
    """
    location = (error.field_path,) if error.field_path else ()
    problem = InitErrorDetails(
        type=PydanticCustomError(_SELECTION_ERROR, "{message}", {"message": error.message}),
        loc=location,
        input=value,
    )
    return ValidationError.from_exception_data(error.component or "component", [problem])


def _describe(problem: ErrorDetails) -> str:
    """Describes one validation problem on a line: its field path, what is wrong and, where
    it helps, the value that was given.

    Args:
        problem: One entry of the errors that pydantic reports.

    Returns:
        The description, without a trailing period added.
    """
    path = _path(problem["loc"])
    text = "Unknown field" if problem["type"] == "extra_forbidden" else problem["msg"]
    if path and problem["type"] not in _NO_VALUE_TO_SHOW:
        text = f"{text} (got {reprlib.repr(problem['input'])})"
    return f"{path}: {text}" if path else text


def _path(location: tuple[int | str, ...]) -> str:
    """Writes a pydantic error location as a field path.

    Args:
        location: The location: field names and list positions, outermost first.

    Returns:
        The names joined by dots, a list position in brackets after the name it indexes, for
        example `kernel.nu` or `values[2]`. Empty for an error about the whole configuration.
    """
    path = ""
    for part in location:
        if isinstance(part, int):
            path += f"[{part}]"
        else:
            path += f".{part}" if path else part
    return path
