"""The table of registries: one registry per kind of component."""

from __future__ import annotations

from typing import Any

from planexs.core.CapabilityVocabulary import CapabilityVocabulary
from planexs.core.Registry import Registry
from planexs.core.RegistryError import RegistryError


class KindTable:
    """The table of registries, one per kind of component (specification 5.12).

    A kind (`surrogate`, `acquisition`, `goal`, ...) is defined once; its registry is where
    the components of that kind are registered. The table exists so that code that does not
    know the kinds in advance can iterate over them: the registry contract tests and the
    compatibility check, which looks for registered components that would meet a need.

    Every registry the table creates checks the capability tags of the entries registered in it
    against the table's vocabulary. The table owns its registries and nothing else, and a
    `KindTable` of your own shares no state with any other. Registries are created at import
    time by `defineKind`, which records them in the process-wide table returned by `default`.

    Args:
        vocabulary: The capability vocabulary handed to every registry the table creates.

    Attributes:
        vocabulary: The capability vocabulary handed to every registry the table creates.
    """

    def __init__(self, vocabulary: CapabilityVocabulary) -> None:
        """Creates an empty table."""
        self.vocabulary = vocabulary
        self._registries: dict[str, Registry[Any]] = {}

    @staticmethod
    def default() -> KindTable:
        """The process-wide table that `defineKind` records into.

        Its registries check tags against `CapabilityVocabulary.default()`.

        Returns:
            The same table on every call.
        """
        return _DEFAULT_TABLE

    def define(self, kind: str) -> Registry[Any]:
        """Defines a kind and creates its empty registry.

        Args:
            kind: The kind's name, in the same lowercase snake_case as a component name (see
                `Registry.isValidName`).

        Returns:
            The registry of the new kind, also reachable through `registry`.

        Raises:
            RegistryError: if the name is not valid or the kind is already defined. The table
                is left unchanged.
        """
        if not Registry.isValidName(kind):
            raise RegistryError(
                f"Cannot define {kind!r} as a component kind: it is not a valid name. A kind "
                "is named like a component: lowercase snake_case, starting with a letter.",
                component=str(kind),
                suggestion="use a name like 'acquisition_optimizer'",
            )
        if kind in self._registries:
            raise RegistryError(
                f"The component kind {kind!r} is already defined. "
                f"Defined kinds: {', '.join(self.kinds())}.",
                component=kind,
                suggestion="define each kind once, in the module that owns its registry",
            )
        registry: Registry[Any] = Registry(kind, self.vocabulary)
        self._registries[kind] = registry
        return registry

    def registry(self, kind: str) -> Registry[Any]:
        """Returns the registry of a defined kind.

        Args:
            kind: The kind's name.

        Returns:
            The registry created by `define`.

        Raises:
            RegistryError: if the kind is not defined. The message lists the defined kinds.
        """
        if kind not in self:
            listed = ", ".join(self.kinds()) or "none"
            raise RegistryError(
                f"Unknown component kind {kind!r}. Defined kinds: {listed}.",
                component=str(kind),
                suggestion="check the spelling, or import the module that defines the kind",
            )
        return self._registries[kind]

    def kinds(self) -> list[str]:
        """The names of the defined kinds, sorted: iterate over them to visit every registry.

        Returns:
            A new list: changing it does not change the table.
        """
        return sorted(self._registries)

    def reset(self) -> None:
        """Removes every kind, and with it every registry and every component in it.

        Meant for tests. Kinds and components registered at import time by modules that have
        already been imported are not registered again, so reset the process-wide table only
        in a test that has swapped it for a table of its own first; otherwise reset a
        `KindTable` created for the test. The vocabulary is not touched.
        """
        self._registries.clear()

    def __contains__(self, kind: object) -> bool:
        """Whether `kind` is defined (never raises, whatever `kind` is)."""
        return isinstance(kind, str) and kind in self._registries


#: The table behind `KindTable.default`. Private: callers go through `defineKind` to add to it
#: and through `KindTable.default` to read it.
_DEFAULT_TABLE = KindTable(CapabilityVocabulary.default())
