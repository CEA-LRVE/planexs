"""The registry of the components of one kind, keyed by name."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Generic, TypeVar

from planexs.core.ComponentEntry import ComponentEntry
from planexs.core.RegistryError import RegistryError

if TYPE_CHECKING:
    # Only the type is needed here. `CapabilityVocabulary` imports this module (for
    # `isValidName`), so importing it back at run time would be a cycle.
    from planexs.core.CapabilityVocabulary import CapabilityVocabulary

T = TypeVar("T", bound=ComponentEntry)

#: The one definition of a valid registered name (specification 5.12): lowercase snake_case,
#: starting with a letter, with single underscores between tokens of letters and digits. The
#: optional `_vN` version suffix (specification 3.7) is an ordinary token, so it needs no
#: pattern of its own. Matched with `fullmatch`: `$` would let a trailing newline through.
_NAME_PATTERN = re.compile(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)*")


class Registry(Generic[T]):
    """The table of the components of one kind (a surrogate, a kernel, ...), keyed by name.

    A registry stores `ComponentEntry` records (or a subclass of it, named by the type
    parameter). A name is a permanent identifier (specification 3.7): once registered it can
    neither be replaced nor removed, so a saved study keeps meaning what it meant. A name is
    lowercase snake_case with an optional `_vN` version suffix, for example
    `single_task_gp_v1`; see `isValidName`.

    When an entry is registered, the capability tags it declares are checked against the
    registry's vocabulary, so a typo in a `needs` or `supports` declaration fails at
    registration. Two registries never share state, even when they hold the same kind name.

    Args:
        kind: The name of the kind of component this registry holds, used in error messages.
        vocabulary: The capability vocabulary that the tags of registered entries are checked
            against. It is read at every registration, so tags added to it later are accepted.

    Attributes:
        kind: The name of the kind of component this registry holds.
        vocabulary: The vocabulary entries are checked against.
    """

    def __init__(self, kind: str, vocabulary: CapabilityVocabulary) -> None:
        """Creates an empty registry."""
        self.kind = kind
        self.vocabulary = vocabulary
        self._entries: dict[str, T] = {}

    @staticmethod
    def isValidName(name: object) -> bool:
        """Whether `name` is a valid registered name: lowercase snake_case, starting with a
        letter, made of letters, digits and single underscores, with no space, no leading or
        trailing underscore and no uppercase letter.

        Args:
            name: The candidate name. Anything that is not a string is not valid.

        Returns:
            `True` for a valid name.
        """
        return isinstance(name, str) and _NAME_PATTERN.fullmatch(name) is not None

    def register(self, entry: T) -> None:
        """Registers `entry` under its own name.

        The checks run in this order, and nothing is stored unless all of them pass: the entry
        is a `ComponentEntry`, its name is valid, the name is free, and every tag in its
        `needs` and then its `supports` (each in sorted order) is in the vocabulary.

        Args:
            entry: The component's record. Its `name` is the registered name.

        Raises:
            RegistryError: if any check fails. The registry is left unchanged. For a tag that
                is not in the vocabulary, the message lists the known tags of the same group.
        """
        if not isinstance(entry, ComponentEntry):
            raise RegistryError(
                f"A {self.kind} registry stores ComponentEntry records; got "
                f"{type(entry).__name__}.",
                component=self.kind,
                suggestion="wrap the component in a ComponentEntry before registering it",
            )
        name = entry.name
        if not self.isValidName(name):
            raise RegistryError(
                f"Cannot register {name!r} as a {self.kind}: not a valid component name. "
                "A name is lowercase snake_case: it starts with a letter and holds letters, "
                "digits and single underscores, with an optional version suffix such as _v1.",
                component=self.kind,
                suggestion="use a name like 'single_task_gp_v1'",
            )
        if name in self._entries:
            raise RegistryError(
                f"A {self.kind} named {name!r} is already registered.",
                component=self.kind,
                suggestion=(
                    "a registered name is a permanent identifier: choose another name, and "
                    "give a version that changes results a new suffix (for example _v2) "
                    "instead of replacing the old one"
                ),
            )
        for field_name, tags in (("needs", entry.needs), ("supports", entry.supports)):
            for tag in sorted(tags):
                self.vocabulary.check(
                    tag, component=f"{self.kind} {name!r}", field_path=field_name
                )
        self._entries[name] = entry

    def lookup(self, name: str) -> T:
        """Returns the entry registered under `name`.

        Args:
            name: The registered name.

        Returns:
            The entry stored by `register`.

        Raises:
            RegistryError: if nothing is registered under `name`. The message lists the
                registered names.
        """
        if name not in self:
            listed = ", ".join(self.names()) or "none"
            raise RegistryError(
                f"Unknown {self.kind} {name!r}. Registered names: {listed}.",
                component=self.kind,
                suggestion=(
                    "check the spelling, or import the module or plugin that registers it"
                ),
            )
        return self._entries[name]

    def names(self) -> list[str]:
        """The registered names, sorted.

        Returns:
            A new list: changing it does not change the registry.
        """
        return sorted(self._entries)

    def __contains__(self, name: object) -> bool:
        """Whether something is registered under `name` (never raises, whatever `name` is)."""
        return isinstance(name, str) and name in self._entries
