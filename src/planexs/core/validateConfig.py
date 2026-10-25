"""Validates a configuration mapping against a model, reporting failures as `ConfigError`."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from planexs.core.ConfigError import ConfigError
from planexs.core.configErrors import toConfigError

ModelT = TypeVar("ModelT", bound=BaseModel)


def validateConfig(
    schema: type[ModelT], data: Mapping[str, Any], *, label: str | None = None
) -> ModelT:
    """Rebuilds the validated configuration objects that `data` describes (specification 5.12).

    `data` is validated against `schema`, which fills in the defaults and rejects unknown
    fields. Every component field of `schema` (see `ComponentOf`) is resolved through the
    registry of its kind, to any depth, so the result holds the validated configuration object
    of each nested component and not the mapping it came from. This is the inverse of
    `toResolvedConfig`: a resolved configuration validates back into an equal configuration.
    It accepts any mapping that validates, so it also reads a configuration with defaults left
    out, and it works for any model, not only for the schema of one component: a study-level
    model that holds components is read the same way.

    Args:
        schema: The model to validate against, for example the configuration schema of a
            component, or a model with component fields.
        data: The configuration, for example a resolved configuration read back from JSON. It
            is not modified.
        label: What the configuration is for, written the way an error should name it, for
            example `surrogate 'single_task_gp_v1'`. Defaults to the name of `schema`.

    Returns:
        The validated configuration, an instance of `schema`.

    Raises:
        ConfigError: if `data` is not a mapping or does not validate. The message lists every
            problem with its full field path, for example `policy.surrogate.kernel.nu`, and
            `field_path` is the path of the first.
        RegistryError: if a component field names a kind that is not defined, or a component
            is registered with a schema that does not derive from `ComponentConfig`: defects
            of the declaration, not of the configuration.
    """
    name = label if label is not None else schema.__name__
    if not isinstance(data, Mapping):
        raise ConfigError(
            f"The configuration of {name} must be a mapping; got {type(data).__name__}.",
            component=name,
            suggestion="write the configuration as a mapping from field names to values",
        )
    try:
        return schema.model_validate(dict(data))
    except ValidationError as error:
        raise toConfigError(error, label=name, schema=schema) from error
