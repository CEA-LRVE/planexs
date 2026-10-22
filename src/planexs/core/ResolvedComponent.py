"""What resolving a component configuration produces: the entry and its validated settings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

from planexs.core.ComponentConfig import ComponentConfig
from planexs.core.ComponentEntry import ComponentEntry

#: The type of entry a registry stores (`Registry` is generic over the same bound), so that
#: resolving against a registry of `ComponentEntry` subclasses keeps the subclass.
EntryT = TypeVar("EntryT", bound=ComponentEntry)


@dataclass(frozen=True)
class ResolvedComponent(Generic[EntryT]):
    """A component configuration after `resolveComponent` has resolved it (specification 5.12).

    It pairs what the registry knows about the component with the settings the caller gave:
    the registered entry, whose builder creates the native object, and the validated
    configuration to give that builder. The configuration is an instance of the entry's own
    schema, so every default is filled in and every field has been checked.

    Args:
        entry: The registry entry of the component the configuration names.
        config: The configuration, validated against `entry.config_schema`.

    Attributes:
        entry: The registry entry of the component.
        config: The validated configuration, an instance of `entry.config_schema`.
    """

    entry: EntryT
    config: ComponentConfig
