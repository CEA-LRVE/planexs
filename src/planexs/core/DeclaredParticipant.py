"""A participant whose tags are fixed when it is built."""

from __future__ import annotations

from dataclasses import dataclass, field

from planexs.core.CompatibilityError import CompatibilityError
from planexs.core.ComponentEntry import ComponentEntry


@dataclass(frozen=True)
class DeclaredParticipant:
    """A `Participant` whose needs and supports are stored values, fixed at construction.

    This is the common case: a problem whose facts imply some needs, a goal that supports a
    goal tag, a component taken as its registry entry declared it. A participant whose tags
    depend on its configuration can be any object with the three attributes of `Participant`.

    Args:
        label: Who this is, as it should appear in an error message.
        needs: The capability tags this participant needs some participant to support. Any
            collection of strings is accepted.
        supports: The capability tags this participant supports. Any collection of strings is
            accepted.

    Attributes:
        label: Who this is.
        needs: The needed tags, as a frozen set.
        supports: The supported tags, as a frozen set.

    Raises:
        CompatibilityError: if `needs` or `supports` is a single string, which would otherwise
            be read as a collection of one-character tags. Whether each tag is well formed and
            known is judged by `checkCompatibility`, against the vocabulary.
    """

    label: str
    needs: frozenset[str] = field(default_factory=frozenset)
    supports: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        """Freezes `needs` and `supports`, refusing a bare string."""
        for field_name in ("needs", "supports"):
            tags = getattr(self, field_name)
            if isinstance(tags, str):
                raise CompatibilityError(
                    f"Participant {self.label!r} has {field_name} given as the single string "
                    f"{tags!r}, not a collection of tags.",
                    component=str(self.label),
                    field_path=field_name,
                    suggestion=f"write {field_name} as a collection such as ('group:name',)",
                )
            # Frozen dataclass: `object.__setattr__` is the sanctioned way to normalize.
            object.__setattr__(self, field_name, frozenset(tags))

    @classmethod
    def fromEntry(cls, kind: str, entry: ComponentEntry) -> DeclaredParticipant:
        """The participant for a registered component, with the tags its entry declares.

        Args:
            kind: The kind of the component (`surrogate`, `acquisition`, ...).
            entry: The component's registry entry.

        Returns:
            A participant labelled like the registry's own error messages label the
            component, for example `surrogate 'single_task_gp_v1'`, with the entry's `needs`
            and `supports`.
        """
        return cls(label=f"{kind} {entry.name!r}", needs=entry.needs, supports=entry.supports)
