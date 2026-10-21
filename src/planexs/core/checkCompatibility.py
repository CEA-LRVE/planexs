"""The compatibility check: does every need of a study have a supporter?"""

from __future__ import annotations

from collections.abc import Iterable

from planexs.core.CompatibilityError import CompatibilityError
from planexs.core.KindTable import KindTable
from planexs.core.Participant import Participant


def checkCompatibility(
    participants: Iterable[Participant], kinds: KindTable | None = None
) -> None:
    """Checks that every capability need of the participants is supported by some participant.

    This is the check run when a study is built (specification 3.6, 5.12), so that a
    combination that cannot work fails there, with an error that says how to fix it, and not
    deep inside an engine call. A need is met if at least one participant, whichever it is
    (including the one that has the need), lists the tag in its `supports`.

    The tags of every participant are first checked against the vocabulary of `kinds`, so a
    typo in a problem or goal contribution is reported as such, and not as an unmet need
    nobody could ever meet. Each participant's `needs` and `supports` are read once, when
    this function runs.

    Args:
        participants: The problem, the goal and the components of the policy, each seen through
            `Participant`.
        kinds: The kind table to look for alternatives in and whose vocabulary the tags are
            checked against. Defaults to the process-wide table, `KindTable.default()`.

    Raises:
        RegistryError: if a participant declares a tag that is malformed or not in the
            vocabulary. The first such tag is reported, taking participants in the order given,
            `needs` before `supports`, and tags in sorted order.
        CompatibilityError: if any need is unmet. The error lists every unmet need, not just
            the first, in sorted order. For each it gives the vocabulary's description of the
            tag, the participants that need it, and the registered components of any kind that
            declare they support it.
    """
    table = kinds if kinds is not None else KindTable.default()
    vocabulary = table.vocabulary

    snapshots: list[tuple[str, list[str], list[str]]] = []
    for participant in participants:
        label = participant.label
        needs = sorted(set(participant.needs), key=str)
        supports = sorted(set(participant.supports), key=str)
        for field_name, tags in (("needs", needs), ("supports", supports)):
            for tag in tags:
                vocabulary.check(tag, component=label, field_path=field_name)
        snapshots.append((label, needs, supports))

    supported = {tag for _, _, supports in snapshots for tag in supports}
    unmet: dict[str, list[str]] = {}
    for label, needs, _ in snapshots:
        for need in needs:
            if need not in supported:
                unmet.setdefault(need, []).append(label)
    if not unmet:
        return

    count = len(unmet)
    lines = [
        "The chosen components are not compatible: "
        + (f"{count} need is not met." if count == 1 else f"{count} needs are not met.")
    ]
    for need in sorted(unmet):
        supporters = _registeredSupporters(table, need)
        lines.append(f"Need {need!r} ({vocabulary.describe(need)})")
        lines.append(f"  needed by: {', '.join(unmet[need])}")
        lines.append(f"  supported by: {', '.join(supporters) or 'no registered component'}")
    raise CompatibilityError(
        "\n".join(lines),
        suggestion=(
            "choose components that support the needs above, or change the problem or the "
            "goal so that they are no longer needed"
        ),
    )


def _registeredSupporters(table: KindTable, tag: str) -> list[str]:
    """The registered components, of every kind, whose entry declares that it supports `tag`.

    Args:
        table: The kind table to look in.
        tag: The capability tag.

    Returns:
        One `kind 'name'` label per component, sorted by kind and then by name.
    """
    found: list[str] = []
    for kind in table.kinds():
        registry = table.registry(kind)
        for name in registry.names():
            if tag in registry.lookup(name).supports:
                found.append(f"{kind} {name!r}")
    return found
