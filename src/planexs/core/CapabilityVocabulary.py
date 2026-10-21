"""The documented set of capability tags that components declare."""

from __future__ import annotations

from planexs.core.Registry import Registry
from planexs.core.RegistryError import RegistryError


class CapabilityVocabulary:
    """The documented set of known capability tags (specification 5.12).

    A capability tag is a short string naming a feature, such as `posterior:gaussian` or
    `space:categorical`. Components declare the tags they **need** and the tags they
    **support**, and the compatibility check matches the two. Because only tags from this
    vocabulary are accepted, a typo in a declaration is an error when the component is
    registered and not a silent mismatch later.

    A tag is `group:name`: two lowercase snake_case words (the rule of `Registry.isValidName`)
    joined by a colon. Every tag carries a one-line description, which is what makes the
    vocabulary documented. The vocabulary starts empty: the module that introduces a feature
    registers its tags, and a plugin may register its own.

    A `CapabilityVocabulary()` of your own shares no state with any other. The process-wide
    vocabulary that components are checked against is returned by `default`.
    """

    def __init__(self) -> None:
        """Creates an empty vocabulary."""
        self._descriptions: dict[str, str] = {}

    @staticmethod
    def default() -> CapabilityVocabulary:
        """The process-wide vocabulary, the one that registered components are checked against.

        Returns:
            The same vocabulary on every call.
        """
        return _DEFAULT_VOCABULARY

    @staticmethod
    def isValidTag(tag: object) -> bool:
        """Whether `tag` is well formed: `group:name`, both lowercase snake_case.

        Args:
            tag: The candidate. Anything that is not a string is not a tag.

        Returns:
            `True` if the tag is well formed. This says nothing about whether it is registered.
        """
        if not isinstance(tag, str):
            return False
        group, separator, name = tag.partition(":")
        return separator == ":" and Registry.isValidName(group) and Registry.isValidName(name)

    def register(self, tag: str, description: str) -> None:
        """Adds a tag to the vocabulary.

        Args:
            tag: A well-formed tag that is not registered yet.
            description: One non-blank line saying what the tag means.

        Raises:
            RegistryError: if the tag is malformed or already registered, or the description
                is blank or spans several lines. The vocabulary is left unchanged.
        """
        if not self.isValidTag(tag):
            raise self._malformed(tag, component=None, field_path=None)
        if (
            not isinstance(description, str)
            or not description.strip()
            or description.splitlines() != [description]
        ):
            raise RegistryError(
                f"Cannot register the capability tag {tag!r} without a description of one "
                "non-blank line.",
                field_path="description",
                suggestion="say in one line what the tag means",
            )
        if tag in self._descriptions:
            raise RegistryError(
                f"The capability tag {tag!r} is already registered: "
                f"{self._descriptions[tag]!r}.",
                suggestion="register each tag once, in the module that introduces the feature",
            )
        self._descriptions[tag] = description

    def check(
        self, tag: str, *, component: str | None = None, field_path: str | None = None
    ) -> None:
        """Checks that `tag` is well formed and registered.

        Args:
            tag: The tag to check.
            component: The component that declared the tag, passed on to the error as a hint.
            field_path: The field that holds the tag, passed on to the error as a hint.

        Raises:
            RegistryError: if the tag is malformed, or well formed but not registered. In the
                second case the message lists the registered tags of the same group (or the
                registered groups, if the group itself is unknown): that is where a typo shows.
        """
        if not self.isValidTag(tag):
            raise self._malformed(tag, component=component, field_path=field_path)
        if tag in self._descriptions:
            return
        group = tag.partition(":")[0]
        same_group = self.tags(group)
        if same_group:
            message = (
                f"Unknown capability tag {tag!r}. "
                f"Known tags of the group {group!r}: {', '.join(same_group)}."
            )
        else:
            message = (
                f"Unknown capability tag {tag!r}: no tag of the group {group!r} is "
                f"registered. Known groups: {', '.join(self.groups()) or 'none'}."
            )
        raise RegistryError(
            message,
            component=component,
            field_path=field_path,
            suggestion=(
                "check the spelling, or register the tag in the vocabulary, with a "
                "description, before a component declares it"
            ),
        )

    def describe(self, tag: str) -> str:
        """Returns the description a tag was registered with.

        Args:
            tag: A registered tag.

        Returns:
            The one-line description.

        Raises:
            RegistryError: if the tag is malformed or not registered.
        """
        self.check(tag)
        return self._descriptions[tag]

    def tags(self, group: str | None = None) -> list[str]:
        """The registered tags, sorted.

        Args:
            group: If given, only the tags of this group. A group with no tag gives an empty
                list.

        Returns:
            A new list: changing it does not change the vocabulary.
        """
        return sorted(
            tag
            for tag in self._descriptions
            if group is None or tag.partition(":")[0] == group
        )

    def groups(self) -> list[str]:
        """The groups that have at least one registered tag, sorted.

        Returns:
            A new list: changing it does not change the vocabulary.
        """
        return sorted({tag.partition(":")[0] for tag in self._descriptions})

    def __contains__(self, tag: object) -> bool:
        """Whether `tag` is registered (never raises, whatever `tag` is)."""
        return isinstance(tag, str) and tag in self._descriptions

    @staticmethod
    def _malformed(
        tag: object, *, component: str | None, field_path: str | None
    ) -> RegistryError:
        """The error for a tag that is not `group:name` in lowercase snake_case."""
        return RegistryError(
            f"{tag!r} is not a well-formed capability tag. A tag is group:name, both words in "
            "lowercase snake_case, for example 'posterior:gaussian'.",
            component=component,
            field_path=field_path,
            suggestion="write the tag as group:name in lowercase snake_case",
        )


#: The vocabulary behind `CapabilityVocabulary.default`. Private: callers register tags with
#: `CapabilityVocabulary.default().register(...)` and read it through `default`.
_DEFAULT_VOCABULARY = CapabilityVocabulary()
