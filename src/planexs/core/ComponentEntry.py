"""The record a registry stores for one component."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from planexs.core.RegistryError import RegistryError


@dataclass(frozen=True)
class ComponentEntry:
    """Everything a registry holds about one registered component (specification 5.12, 6).

    An entry is built to be registered: `Registry.register` checks its name and its capability
    tags, because those checks need the registry and its vocabulary. What the entry can check
    by itself, it checks when it is built, so that an entry that exists is complete: it has a
    description, a callable builder and a configuration schema.

    Args:
        name: The registered name: lowercase snake_case with an optional `_vN` suffix.
        builder: What creates the native object of the component. Its signature belongs to the
            component's kind.
        config_schema: The Pydantic model that validates the component's configuration.
        description: One line saying what the component is.
        needs: The capability tags the component needs some participant to support.
        supports: The capability tags the component supports.

    Attributes:
        name: The registered name.
        builder: The component's builder.
        config_schema: The model of the component's configuration.
        description: The one-line description.
        needs: The needed tags, as a frozen set (any collection of strings is accepted).
        supports: The supported tags, as a frozen set (any collection of strings is accepted).

    Raises:
        RegistryError: if the description is blank or spans several lines, the builder is not
            callable, the schema is not a Pydantic model class, or `needs` or `supports` is a
            single string or holds something that is not a string.
    """

    name: str
    builder: Callable[..., Any]
    config_schema: type[BaseModel]
    description: str
    needs: frozenset[str] = field(default_factory=frozenset)
    supports: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        """Checks what an entry can check alone, and freezes `needs` and `supports`."""
        if (
            not isinstance(self.description, str)
            or not self.description.strip()
            or self.description.splitlines() != [self.description]
        ):
            raise RegistryError(
                f"Component {self.name!r} has no usable description: a component needs a "
                "description of one non-blank line.",
                component=str(self.name),
                field_path="description",
                suggestion="say in one line what the component is",
            )
        if not callable(self.builder):
            raise RegistryError(
                f"Component {self.name!r} has a builder that is not callable: "
                f"got {type(self.builder).__name__}.",
                component=str(self.name),
                field_path="builder",
                suggestion="pass the function or class that builds the component",
            )
        if not (
            isinstance(self.config_schema, type) and issubclass(self.config_schema, BaseModel)
        ):
            raise RegistryError(
                f"Component {self.name!r} has a configuration schema that is not a Pydantic "
                f"model class: got {self.config_schema!r}.",
                component=str(self.name),
                field_path="config_schema",
                suggestion="pass the model class itself, not an instance of it",
            )
        for field_name in ("needs", "supports"):
            tags = getattr(self, field_name)
            # Materialized once, so that a generator is not consumed by the check.
            items = (
                tuple(tags)
                if isinstance(tags, Iterable) and not isinstance(tags, str)
                else None
            )
            if items is None or not all(isinstance(tag, str) for tag in items):
                raise RegistryError(
                    f"Component {self.name!r} has {field_name} that is not a collection of "
                    f"tag strings: got {tags!r}.",
                    component=str(self.name),
                    field_path=field_name,
                    suggestion=f"write {field_name} as a collection such as ('group:name',)",
                )
            # Frozen dataclass: `object.__setattr__` is the sanctioned way to normalize.
            object.__setattr__(self, field_name, frozenset(items))
