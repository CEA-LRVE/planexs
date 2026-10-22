"""The base class of the configuration of every registered component."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ComponentConfig(BaseModel):
    """The typed, validated settings of one component (specification 5.12, 10.8).

    Every registered component has a configuration schema that derives from this class. The
    `type` field holds the registered name of the component, so a configuration says on its
    own which component it is for: `{"type": "single_task_gp_v1", "min_observations": 3}` is a
    configuration of the surrogate registered as `single_task_gp_v1`. `resolveComponent` reads
    that name, finds the component in its registry, and validates the whole mapping against the
    schema the component registered.

    Unknown fields are rejected: a misspelled setting is an error, never a silent no-op. A
    subclass adds the settings of its component as fields. Each one carries a description that
    states its meaning, its unit and why its default is what it is, and the default is written
    there and nowhere else (specification 3.4, 10.8).

    A subclass should narrow `type` to a `Literal` of its registered name::

        class SingleTaskGPConfig(ComponentConfig):
            type: Literal["single_task_gp_v1"] = "single_task_gp_v1"

    Pydantic then rejects any other name even when the schema is validated directly, without
    going through a registry. The default is a convenience for building the configuration in
    Python: a configuration read from a file or a mapping must always give `type`, and
    `resolveComponent` refuses one that does not.

    Validation follows pydantic's default (lax) mode, so that a configuration built in Python,
    loaded from JSON and loaded from YAML has the same meaning (specification 10.8): a JSON
    array is accepted for a tuple field, for instance. A subclass that wants strict checking
    of a field can ask for it on that field.

    Attributes:
        type: The registered name of the component this configuration is for.
    """

    model_config = ConfigDict(extra="forbid")

    type: str = Field(
        description=(
            "The registered name of the component this configuration is for, for example "
            "'single_task_gp_v1'. It selects the component in the registry of its kind, and "
            "it has no default in a configuration read from a file or a mapping."
        ),
    )
