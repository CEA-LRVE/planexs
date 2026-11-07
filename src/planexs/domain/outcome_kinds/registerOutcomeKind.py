"""The registration decorator of the `outcome_kind` kind."""

from __future__ import annotations

from collections.abc import Callable
from operator import attrgetter
from typing import TypeVar

from planexs.core.CapabilityVocabulary import CapabilityVocabulary
from planexs.core.ComponentConfig import ComponentConfig
from planexs.core.registerComponentClass import registerComponentClass
from planexs.core.RegistryError import RegistryError
from planexs.domain.outcome_kinds.OutcomeKind import OutcomeKind
from planexs.domain.outcome_kinds.outcomeKindRegistry import outcomeKindRegistry

OutcomeKindT = TypeVar("OutcomeKindT", bound=OutcomeKind)

#: The group of the capability tags that stand for an outcome kind.
_TAG_GROUP = "outcome_kind"


def registerOutcomeKind(
    name: str,
    *,
    config: type[ComponentConfig],
    description: str,
    needs: frozenset[str] = frozenset(),
    supports: frozenset[str] = frozenset(),
) -> Callable[[type[OutcomeKindT]], type[OutcomeKindT]]:
    """Registers an `OutcomeKind` subclass as an outcome kind, under a permanent name.

    Used as a class decorator, at import time (specification 5.12)::

        @registerOutcomeKind(
            "count", config=CountConfig, description="A count: a non-negative integer."
        )
        class CountKind(OutcomeKind):
            capability_tag: ClassVar[str] = "outcome_kind:count"
            allows_noise: ClassVar[bool] = False
            ...

    The registry entry holds the name, the configuration schema (the settings of the kind),
    the one-line description and the builder, which is the `fromConfig` of the class. The
    class is returned unchanged. The module that holds the decorated class must be imported by
    the `__init__` of its package, or the kind is silently unregistered (specification 5.12,
    10.6).

    The capability tag of the class is registered in the capability vocabulary of the registry
    at the same time, with the description of the kind, so that a surrogate can declare that it
    supports the kind and a problem that needs it. The entry itself declares no tag by default:
    a tag in the `supports` of an outcome kind would list the kind, among the registered
    components, as something that meets the need for it, which it does not (the participants
    that meet the need are the components of a policy).

    The name is the value of the `type` of an outcome. The `type` field of the schema must be
    pinned to that same name (`Literal["count"] = "count"`).

    Args:
        name: The registered name: lowercase snake_case with an optional `_vN` suffix. It is
            permanent: a change that alters results gets a new name (specification 3.7).
        config: The configuration schema of the kind, a `ComponentConfig` whose `type` field
            defaults to `name`.
        description: One line saying what the kind is. It is also the description of the
            capability tag.
        needs: The capability tags that the kind needs some participant to support.
        supports: The capability tags that the kind supports.

    Returns:
        The decorator.

    Raises:
        RegistryError: when the decorator is applied, if the class is not an `OutcomeKind`
            subclass, its `capability_tag` is not a well-formed tag of the group
            `outcome_kind` that no other kind has, its `allows_noise` is not a boolean, the
            `type` field of the schema is not pinned to `name`, or the entry is refused by the
            registry (an invalid or already taken name, a tag that is not in the vocabulary).
            Nothing is registered then: neither the kind nor its tag.
    """

    def decorate(cls: type[OutcomeKindT]) -> type[OutcomeKindT]:
        """Registers `cls` and its capability tag, and returns it unchanged."""
        registry = outcomeKindRegistry()
        tag: str | None = None
        # The tag is checked before anything is stored, so that a refusal leaves no kind
        # without its tag and no tag without its kind. A class that is not an outcome kind is
        # left to `registerComponentClass`, which reports it as such.
        if isinstance(cls, type) and issubclass(cls, OutcomeKind):
            tag = _checkedTag(cls, name, registry.vocabulary)
        registerComponentClass(
            registry,
            cls,
            base=OutcomeKind,
            name=name,
            config=config,
            description=description,
            needs=needs,
            supports=supports,
            builder_of=attrgetter("fromConfig"),
        )
        if tag is not None:
            registry.vocabulary.register(tag, description)
        return cls

    return decorate


def _checkedTag(cls: type[OutcomeKind], name: str, vocabulary: CapabilityVocabulary) -> str:
    """Checks the capability tag and the noise rule that a kind declares, and returns the tag.

    Args:
        cls: The class being registered.
        name: The name it is registered under, for the messages.
        vocabulary: The vocabulary that the tag is about to be registered in.

    Returns:
        The tag, which is well formed, of the group `outcome_kind`, and not registered yet.

    Raises:
        RegistryError: if `capability_tag` is missing, not a string, not a well-formed tag of
            the group `outcome_kind`, or already in the vocabulary, or if `allows_noise` is
            missing or not a boolean.
    """
    component = f"outcome_kind {name!r}"
    tag = getattr(cls, "capability_tag", None)
    if not isinstance(tag, str) or not CapabilityVocabulary.isValidTag(tag):
        raise RegistryError(
            f"The outcome kind {name!r} has no usable capability_tag: got {tag!r}.",
            component=component,
            field_path="capability_tag",
            suggestion=f"declare `capability_tag: ClassVar[str] = '{_TAG_GROUP}:{name}'`",
        )
    if tag.partition(":")[0] != _TAG_GROUP:
        raise RegistryError(
            f"The capability tag {tag!r} of the outcome kind {name!r} is not in the group "
            f"{_TAG_GROUP!r}, where the tags that stand for an outcome kind live.",
            component=component,
            field_path="capability_tag",
            suggestion=f"write it as '{_TAG_GROUP}:<name>'",
        )
    if tag in vocabulary:
        raise RegistryError(
            f"The capability tag {tag!r} of the outcome kind {name!r} is already registered: "
            f"{vocabulary.describe(tag)!r}.",
            component=component,
            field_path="capability_tag",
            suggestion="give each outcome kind a tag of its own",
        )
    if not isinstance(getattr(cls, "allows_noise", None), bool):
        raise RegistryError(
            f"The outcome kind {name!r} does not say whether a noise standard deviation is "
            f"meaningful for it: its allows_noise is {getattr(cls, 'allows_noise', None)!r}.",
            component=component,
            field_path="allows_noise",
            suggestion="declare `allows_noise: ClassVar[bool] = True` or `False`",
        )
    return tag
